"""
Phase 4 - REST API
"""
import sys
import asyncio
import shutil
import time
import re
import yaml
import uvicorn
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from phase1_document_processing import DocumentProcessor
from phase2_semantic_search import SemanticSearchEngine
from phase3_llm_engine import LLMEngine

load_dotenv()

CONFIG_PATH = PROJECT_ROOT / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """On every API start: wipe user_chunks collection and user_documents folder contents."""
    engine = get_search_engine()
    try:
        engine.chroma_client.delete_collection(engine.user_collection_name)
        logger.info("user_chunks cleared on startup.")
    except Exception:
        pass
    user_docs = PROJECT_ROOT / "data" / "user_documents"
    user_docs.mkdir(parents=True, exist_ok=True)
    for item in user_docs.iterdir():
        try:
            item.unlink() if item.is_file() else shutil.rmtree(item)
        except Exception as e:
            logger.warning(f"Could not delete {item.name}: {e}")
    logger.info("user_documents/ cleared on startup.")
    yield

app = FastAPI(title="InsureQ API", version="1.0.0", lifespan=lifespan)

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



# ── Pydantic models ───────────────────────────────────────────────────────────

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

# ── Endpoints ─────────────────────────────────────────────────────────────────

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
        results = engine.semantic_search(req.query, top_k=req.top_k or engine.top_k)
    except Exception as e:
        logger.error(f"/search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return SearchResponse(
        query=req.query,
        results=[
            SearchResult(
                rank=r["rank"],
                text=r["text"],
                similarity_score=round(r["similarity_score"], 4),
                file_name=r["metadata"].get("file_name", "?"),
                chunk_id=r["metadata"].get("chunk_id", "?"),
                page=r["metadata"].get("page"),
            )
            for r in results
        ],
        latency_ms=round((time.time() - t0) * 1000, 2),
    )


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    cfg = load_config()
    user_docs_path = PROJECT_ROOT / "data" / "user_documents"
    supported = set(cfg["document_processing"]["supported_formats"])

    safe_name = re.sub(r"[^\w.\-]", "_", Path(file.filename).name).strip("_")
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if Path(safe_name).suffix.lower() not in supported:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    dest = user_docs_path / safe_name
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            None, DocumentProcessor(str(CONFIG_PATH)).process_single_document, dest
        )
        inserted = await loop.run_in_executor(
            None, get_search_engine().index_user_document, chunks
        )
    except Exception as e:
        logger.error(f"/upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return UploadResponse(filename=safe_name, chunks_created=inserted, status="processed")


@app.post("/reindex", response_model=ReindexResponse)
async def reindex():
    try:
        engine = get_search_engine()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: engine.build_complete_search_index(force=True, force_embeddings=False)
        )
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
        s = get_search_engine().get_collection_stats()
        if "error" in s:
            raise HTTPException(status_code=503, detail=s["error"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return StatsResponse(**s)


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
