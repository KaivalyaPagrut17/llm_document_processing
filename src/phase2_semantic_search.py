"""
Phase 2 - Semantic Search
Embedding generation, ChromaDB management, and dual-collection search.

Collections:
  document_chunks  — 15 company docs, built once via phase2 CLI / reindex
  user_chunks      — user-uploaded docs, appended on each upload (never rebuilt)
"""
import json
import numpy as np
import chromadb
import yaml
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent


class SemanticSearchEngine:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = str(PROJECT_ROOT / "config.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        emb_cfg = self.config["embeddings"]
        logger.info(f"Loading embedding model: {emb_cfg['model_name']}")
        self.model = SentenceTransformer(emb_cfg["model_name"])

        self.rerank_enabled = self.config["semantic_search"].get("rerank", False)
        if self.rerank_enabled:
            reranker = self.config["semantic_search"].get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info(f"Loading reranker: {reranker}")
            self.reranker = CrossEncoder(reranker)

        db_path = PROJECT_ROOT / self.config["vector_db"]["persist_directory"]
        db_path.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(db_path))

        self.collection_name = self.config["vector_db"]["collection_name"]
        self.user_collection_name = self.config["vector_db"].get("user_collection_name", "user_chunks")

        self.top_k = self.config["semantic_search"]["top_k"]
        self.similarity_threshold = self.config["semantic_search"]["similarity_threshold"]

        self.processed_path = PROJECT_ROOT / self.config["paths"]["processed_data"]

        # ensure user_documents folder exists
        user_docs = PROJECT_ROOT / self.config["paths"].get("user_documents", "data/user_documents")
        user_docs.mkdir(parents=True, exist_ok=True)

    # ── Embedding helpers ─────────────────────────────────────────────────────

    def _encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.config["embeddings"]["batch_size"],
            normalize_embeddings=self.config["embeddings"]["normalize"],
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    def encode_query(self, query: str) -> np.ndarray:
        prefix = self.config["embeddings"].get(
            "query_instruction_prefix",
            self.config["embeddings"].get("instruction_prefix", ""),
        )
        return self._encode([prefix + query])[0]

    # ── Company index (built once) ────────────────────────────────────────────

    def generate_embeddings(self, force: bool = False) -> List[Dict[str, Any]]:
        """Embed chunked_documents.json → embedded_chunks.json. Skips if cache exists."""
        output_path = self.processed_path / "embedded_chunks.json"
        if output_path.exists() and not force:
            logger.info("Embeddings cache found, skipping generation.")
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)

        chunks_path = self.processed_path / "chunked_documents.json"
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        prefix = self.config["embeddings"].get("instruction_prefix", "")
        texts = [prefix + c["text"] for c in chunks]
        embeddings = self._encode(texts, show_progress=True)

        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.tolist()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        logger.success(f"Embeddings saved to {output_path}")
        return chunks

    def initialize_vector_database(self, force: bool = False) -> None:
        """Load embedded_chunks.json into ChromaDB company collection."""
        embedded_path = self.processed_path / "embedded_chunks.json"
        with open(embedded_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            if not force:
                logger.info("Company collection exists, skipping re-insertion.")
                return
            self.chroma_client.delete_collection(self.collection_name)
            logger.info(f"Deleted existing collection: {self.collection_name}")
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection(self.collection_name)
        logger.info(f"Created collection: {self.collection_name}")

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "file_name": c["file_name"],
                "chunk_id": c["chunk_id"],
                "doc_type": c.get("doc_type", "unknown"),
                "chunk_index": c.get("chunk_index", 0),
                "word_count": c.get("word_count", 0),
                **( {"page": c["page"]} if "page" in c else {}),
            }
            for c in chunks
        ]

        for i in tqdm(range(0, len(chunks), 100), desc="Inserting into ChromaDB"):
            end = min(i + 100, len(chunks))
            self.collection.add(
                ids=ids[i:end],
                embeddings=[c["embedding"] for c in chunks[i:end]],
                documents=[c["text"] for c in chunks[i:end]],
                metadatas=metadatas[i:end],
            )
        logger.success(f"Company collection ready: {len(chunks)} chunks")

    def build_complete_search_index(self, force: bool = False, force_embeddings: bool = False) -> None:
        """Full pipeline for company docs. force=True rebuilds ChromaDB. force_embeddings=True re-embeds."""
        logger.info("Building company search index...")
        self.generate_embeddings(force=force_embeddings)
        self.initialize_vector_database(force=force)
        stats = self.get_collection_stats()
        logger.success(f"Index ready — {stats['total_chunks']} total chunks")

    # ── User document indexing ────────────────────────────────────────────────

    def index_user_document(self, chunks: List[Dict[str, Any]]) -> int:
        """Embed and append chunks into user_chunks collection only. Fast — one doc at a time."""
        if not chunks:
            return 0

        try:
            user_col = self.chroma_client.get_collection(self.user_collection_name)
        except Exception:
            user_col = self.chroma_client.create_collection(self.user_collection_name)

        prefix = self.config["embeddings"].get("instruction_prefix", "")
        embeddings = self._encode([prefix + c["text"] for c in chunks])

        existing = user_col.count()
        ids = [f"user_chunk_{existing + i}" for i in range(len(chunks))]
        metadatas = [
            {
                "file_name": c["file_name"],
                "chunk_id": c["chunk_id"],
                "doc_type": c.get("doc_type", "unknown"),
                "chunk_index": c.get("chunk_index", 0),
                "word_count": c.get("word_count", 0),
                **( {"page": c["page"]} if "page" in c else {}),
            }
            for c in chunks
        ]

        for i in range(0, len(chunks), 100):
            end = min(i + 100, len(chunks))
            user_col.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end].tolist(),
                documents=[c["text"] for c in chunks[i:end]],
                metadatas=metadatas[i:end],
            )

        logger.success(f"Indexed {len(chunks)} chunks into '{self.user_collection_name}'")
        return len(chunks)

    # ── Search ────────────────────────────────────────────────────────────────

    def _query_collection(self, collection, query_embedding: np.ndarray, fetch_k: int) -> List[Dict]:
        try:
            n = min(fetch_k, collection.count())
            if n == 0:
                return []
            res = collection.query(
                query_embeddings=[query_embedding],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning(f"Query failed on {collection.name}: {e}")
            return []

        hits = []
        for i in range(len(res["documents"][0])):
            score = 1 - res["distances"][0][i]
            if score >= self.similarity_threshold:
                hits.append({
                    "text": res["documents"][0][i],
                    "metadata": res["metadatas"][0][i],
                    "similarity_score": score,
                })
        return hits

    def semantic_search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search both company and user collections, merge by score."""
        if top_k is None:
            top_k = self.top_k

        if not hasattr(self, "collection"):
            try:
                self.collection = self.chroma_client.get_collection(self.collection_name)
            except Exception as e:
                logger.error(f"Company collection not found: {e}")
                return []

        query_embedding = self.encode_query(query)
        fetch_k = top_k * 3 if self.rerank_enabled else top_k

        # company docs
        hits = self._query_collection(self.collection, query_embedding, fetch_k)

        # user docs (optional — silently skip if collection doesn't exist yet)
        try:
            user_col = self.chroma_client.get_collection(self.user_collection_name)
            hits.extend(self._query_collection(user_col, query_embedding, fetch_k))
        except Exception:
            pass

        # merge, sort, trim
        hits.sort(key=lambda x: x["similarity_score"], reverse=True)
        hits = hits[:fetch_k]

        if hits and self.rerank_enabled:
            pairs = [[query, h["text"]] for h in hits]
            scores = self.reranker.predict(pairs)
            for h, s in zip(hits, scores):
                h["rerank_score"] = float(s)
            hits.sort(key=lambda x: x["rerank_score"], reverse=True)
            hits = hits[:top_k]

        for i, h in enumerate(hits):
            h["rank"] = i + 1

        logger.info(f"Search returned {len(hits)} results for: '{query}'")
        return hits

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_collection_stats(self) -> Dict[str, Any]:
        if not hasattr(self, "collection"):
            try:
                self.collection = self.chroma_client.get_collection(self.collection_name)
            except Exception as e:
                return {"error": f"Collection not found: {e}"}

        total = self.collection.count()
        try:
            total += self.chroma_client.get_collection(self.user_collection_name).count()
        except Exception:
            pass

        return {
            "collection_name": self.collection_name,
            "total_chunks": total,
            "embedding_model": self.config["embeddings"]["model_name"],
            "similarity_metric": self.config["vector_db"]["similarity_metric"],
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild ChromaDB collection")
    parser.add_argument("--force-embeddings", action="store_true", help="Regenerate embeddings cache")
    args = parser.parse_args()

    engine = SemanticSearchEngine()
    engine.build_complete_search_index(force=args.force, force_embeddings=args.force_embeddings)


if __name__ == "__main__":
    main()
