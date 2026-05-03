"""
Phase 4: REST API
Exposes Phase 1-3 pipeline via FastAPI endpoints.
"""
import os
import sys
import shutil
import time
import yaml
import uvicorn
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from loguru import logger
from werkzeug.utils import secure_filename

# allow imports from src/
sys.path.insert(0, str(Path(__file__).parent))

from phase1_document_processing import DocumentProcessor
from phase2_semantic_search import SemanticSearchEngine
from phase3_llm_engine import LLMEngine

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LLM Document Processing API",
    description="Document ingestion, semantic search and LLM-powered Q&A",
    version="1.0.0",
)

# lazy-loaded singletons
_search_engine: Optional[SemanticSearchEngine] = None
_llm_engine: Optional[LLMEngine] = None

def get_search_engine() -> SemanticSearchEngine:
    global _search_engine
    if _search_engine is None:
        _search_engine = SemanticSearchEngine(str(CONFIG_PATH))
    return _search_engine

def get_llm_engine() -> LLMEngine:
    global _llm_engine
    if _llm_engine is None:
        _llm_engine = LLMEngine(str(CONFIG_PATH))
    return _llm_engine

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None

class SourceItem(BaseModel):
    document: str
    chunk_id: str
    page: Optional[int] = None
    section: Optional[str] = None
    similarity_score: float

class QueryResponse(BaseModel):
    query: str
    intent: str
    answer: str
    sources: List[SourceItem]
    latency_ms: float

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None

class SearchResult(BaseModel):
    rank: int
    text: str
    similarity_score: float
    file_name: str
    chunk_id: str
    page: Optional[int] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    latency_ms: float

class UploadResponse(BaseModel):
    filename: str
    chunks_created: int
    status: str

class ReindexResponse(BaseModel):
    status: str
    message: str

class StatsResponse(BaseModel):
    collection_name: str
    total_chunks: int
    embedding_model: str
    similarity_metric: str

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    cfg = load_config()
    return {
        "status": "ok",
        "model": cfg["llm"]["model"],
        "embedding_model": cfg["embeddings"]["model_name"],
        "version": cfg["project"]["version"],
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    t0 = time.time()
    try:
        engine = get_llm_engine()
        if req.top_k:
            engine.search_engine.top_k = req.top_k
        result = engine.query(req.query)
    except Exception as e:
        logger.error(f"/query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        query=result["query"],
        intent=result["intent"],
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
        latency_ms=round((time.time() - t0) * 1000, 2),
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    t0 = time.time()
    try:
        engine = get_search_engine()
        top_k = req.top_k or engine.top_k
        results = engine.semantic_search(req.query, top_k=top_k)
    except Exception as e:
        logger.error(f"/search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    formatted = [
        SearchResult(
            rank=r["rank"],
            text=r["text"],
            similarity_score=round(r["similarity_score"], 4),
            file_name=r["metadata"].get("file_name", "?"),
            chunk_id=r["metadata"].get("chunk_id", "?"),
            page=r["metadata"].get("page"),
        )
        for r in results
    ]
    return SearchResponse(
        query=req.query,
        results=formatted,
        latency_ms=round((time.time() - t0) * 1000, 2),
    )


@app.post("/upload", response_model=UploadResponse)
def upload(file: UploadFile = File(...)):
    cfg = load_config()
    raw_docs_path = Path(cfg["paths"]["raw_documents"])
    supported = set(cfg["document_processing"]["supported_formats"])

    safe_name = secure_filename(file.filename)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    suffix = Path(safe_name).suffix.lower()
    if suffix not in supported:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    dest = raw_docs_path / safe_name
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"Uploaded: {file.filename}")

    try:
        # Step 1: extract + chunk only the new file
        processor = DocumentProcessor(str(CONFIG_PATH))
        all_chunks = processor.process_all_documents()
        new_chunks = [c for c in all_chunks if c["file_name"] == safe_name]
        logger.info(f"New file produced {len(new_chunks)} chunks")

        # Step 2: embed + append only new chunks — no full rebuild
        engine = get_search_engine()
        inserted = engine.index_single_document(new_chunks)
    except Exception as e:
        logger.error(f"/upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    return UploadResponse(
        filename=safe_name,
        chunks_created=inserted,
        status="processed",
    )


@app.post("/reindex", response_model=ReindexResponse)
def reindex():
    try:
        engine = get_search_engine()
        engine.build_complete_search_index(force=True)
        # reset singletons so next request picks up fresh index
        global _search_engine, _llm_engine
        _search_engine = None
        _llm_engine = None
    except Exception as e:
        logger.error(f"/reindex error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return ReindexResponse(status="ok", message="Index rebuilt successfully.")


@app.get("/stats", response_model=StatsResponse)
def stats():
    try:
        engine = get_search_engine()
        s = engine.get_collection_stats()
        if "error" in s:
            raise HTTPException(status_code=503, detail=s["error"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return StatsResponse(**s)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = load_config()
    api_cfg = cfg.get("api", {})
    uvicorn.run(
        "phase4_api:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        reload=api_cfg.get("reload", False),
        workers=api_cfg.get("workers", 1),
    )
