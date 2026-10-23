"""ChromaDB Persistent Vector Store for FinAudit AI.

Provides persistent vector indexing across server restarts,
metadata filtering by ticker and date, and pluggable embedding functions.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from finaudit.rag.document_chunker import DocumentChunk

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHROMA_DIR = ROOT_DIR / "data" / "chroma_db"

COLLECTION_DISCLOSURES = "nse_disclosures"
COLLECTION_REPORTS = "financial_reports"
COLLECTION_NEWS = "news_and_commentary"
COLLECTION_FINDINGS = "research_findings"

ALL_COLLECTIONS = [
    COLLECTION_DISCLOSURES,
    COLLECTION_REPORTS,
    COLLECTION_NEWS,
    COLLECTION_FINDINGS,
]


class DeterministicFallbackEmbedding:
    """Lightweight 128-dimensional bag-of-words / sub-word embedding.
    Guarantees offline execution without network calls or external binary wheels.
    Uses hashlib for stable hash buckets across Python process restarts.
    """

    DIMENSION = 384

    def name(self) -> str:
        return "deterministic_fallback_v1"

    def __call__(self, input: Union[str, Sequence[str]]) -> List[List[float]]:
        if isinstance(input, str):
            texts = [input]
        else:
            texts = list(input)

        embeddings: List[List[float]] = []
        for text in texts:
            # Initialize with small, strictly positive dense baseline (prevents SIMD underflows in Rust HNSW)
            vec = [0.01 + (i % 17) * 0.001 for i in range(self.DIMENSION)]
            words = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
            for w in words:
                # Use stable md5 hash digest so index persists identically across restarts
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:8], 16) % self.DIMENSION
                vec[h] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [round(float(v / norm), 5) for v in vec]
            embeddings.append(vec)
        return embeddings


def get_embedding_function():
    """Create configured embedding function with graceful offline fallback."""
    provider = os.getenv("FINAUDIT_EMBEDDING_PROVIDER", "deterministic").lower().strip()
    if provider in ("fallback", "mock", "deterministic", "default"):
        return DeterministicFallbackEmbedding()

    try:
        from chromadb.utils import embedding_functions

        if provider in ("onnx", "chroma"):
            ef = embedding_functions.DefaultEmbeddingFunction()
            ef(["healthcheck"])
            return ef
        elif provider == "ollama":
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
            model_name = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
            ef = embedding_functions.OllamaEmbeddingFunction(
                url=f"{ollama_url}/api/embeddings",
                model_name=model_name,
            )
            ef(["healthcheck"])
            return ef
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=api_key,
                    model_name=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
                )
                ef(["healthcheck"])
                return ef
    except Exception as exc:
        logger.warning(
            "Primary embedding function initialization failed (%s); falling back to deterministic local embedding.",
            exc,
        )

    return DeterministicFallbackEmbedding()


class FinAuditVectorStore:
    """Manages persistent ChromaDB vector storage and queries for FinAudit AI."""

    def __init__(self, persist_dir: Optional[Union[str, Path]] = None):
        if persist_dir:
            self.persist_dir = Path(persist_dir).resolve()
        else:
            env_dir = os.getenv("FINAUDIT_CHROMA_DIR")
            self.persist_dir = Path(env_dir).resolve() if env_dir else DEFAULT_CHROMA_DIR

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.embedding_function = get_embedding_function()
        self._collections: Dict[str, Collection] = {}

    def _get_or_create_safe_collection(self, name: str) -> Collection:
        """Safely get existing collection, or create if it does not exist."""
        try:
            return self.client.get_collection(name=name)
        except Exception:
            try:
                return self.client.create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception:
                return self.client.get_or_create_collection(name=name)

    def _initialize_collections(self) -> None:
        """Ensure core collections exist with cosine similarity space."""
        for name in ALL_COLLECTIONS:
            self._collections[name] = self._get_or_create_safe_collection(name)

    def get_collection(self, name: str) -> Collection:
        """Retrieve collection by name, creating safely if missing."""
        if name not in self._collections:
            self._collections[name] = self._get_or_create_safe_collection(name)
        return self._collections[name]

    def has_document(
        self,
        collection_name: str,
        document_id: str,
        content_hash: Optional[str] = None,
    ) -> bool:
        """Check if a document (or specific content version) is already indexed."""
        col = self.get_collection(collection_name)
        where_filter = {"document_id": str(document_id)}
        if content_hash:
            where_filter = {
                "$and": [
                    {"document_id": str(document_id)},
                    {"content_hash": str(content_hash)},
                ]
            }

        try:
            res = col.get(where=where_filter, limit=1)
            return len(res.get("ids", [])) > 0
        except Exception:
            # Fallback simple search
            try:
                res = col.get(where={"document_id": str(document_id)}, limit=1)
                return len(res.get("ids", [])) > 0
            except Exception:
                return False

    def add_chunks(
        self,
        collection_name: str,
        chunks: List[DocumentChunk],
        replace_existing_doc: bool = True,
    ) -> int:
        """Add a list of DocumentChunk instances to the specified collection."""
        if not chunks:
            return 0

        col = self.get_collection(collection_name)

        # If replacing existing doc chunks, safely delete old occurrences by ID
        if replace_existing_doc:
            try:
                doc_ids = {c.document_id for c in chunks if c.document_id}
                for doc_id in doc_ids:
                    existing = col.get(where={"document_id": str(doc_id)})
                    if existing and existing.get("ids"):
                        col.delete(ids=existing["ids"])
            except Exception as exc:
                logger.debug("Safe cleanup of old doc versions skipped: %s", exc)

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.to_metadata() for c in chunks]
        embeddings = self.embedding_function(documents)

        # Chroma handles batches gracefully
        batch_size = 100
        total_added = 0
        for i in range(0, len(chunks), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            batch_embs = embeddings[i : i + batch_size]
            col.upsert(
                ids=batch_ids,
                embeddings=batch_embs,
                documents=batch_docs,
                metadatas=batch_meta,
            )
            total_added += len(batch_ids)

        return total_added

    def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform semantic similarity search over the specified collection."""
        col = self.get_collection(collection_name)
        if self._safe_collection_count(col) == 0:
            return []

        clean_query = (query_text or "").strip()
        if not clean_query:
            return []

        query_embs = self.embedding_function([clean_query])
        kwargs: Dict[str, Any] = {
            "query_embeddings": query_embs,
            "n_results": max(1, n_results),
        }
        if where_filter:
            kwargs["where"] = where_filter

        try:
            results = col.query(**kwargs)
        except Exception as exc:
            logger.warning("Chroma query with filter failed (%s); retrying without filter", exc)
            kwargs.pop("where", None)
            try:
                results = col.query(**kwargs)
            except Exception as inner_exc:
                logger.error("Chroma query failed: %s", inner_exc)
                return []

        formatted: List[Dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else [0.5] * len(ids)

        for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
            # Cosine distance in Chroma: 0 is identical, 2 is opposite.
            # Convert to similarity score in [0.0, 1.0]
            similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
            formatted.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": meta,
                    "similarity": round(similarity, 4),
                    "distance": round(dist, 4),
                    "collection": collection_name,
                }
            )

        return formatted

    def delete_by_document_id(self, collection_name: str, document_id: str) -> int:
        """Delete all chunks belonging to a document from a collection."""
        col = self.get_collection(collection_name)
        res = col.get(where={"document_id": str(document_id)})
        ids = res.get("ids", [])
        if ids:
            col.delete(ids=ids)
        return len(ids)

    def _safe_collection_count(self, col: Collection) -> int:
        """Safely get collection count via col.get() without invoking Rust _count hook."""
        try:
            res = col.get()
            return len(res.get("ids", []))
        except Exception:
            return 0

    def count(self, collection_name: Optional[str] = None) -> Union[int, Dict[str, int]]:
        """Return number of indexed chunks."""
        if collection_name:
            return self._safe_collection_count(self.get_collection(collection_name))
        return {name: self._safe_collection_count(self.get_collection(name)) for name in ALL_COLLECTIONS}

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic health and statistics for vector storage."""
        counts = self.count()
        total_chunks = sum(counts.values()) if isinstance(counts, dict) else counts
        return {
            "status": "ready",
            "persist_directory": str(self.persist_dir),
            "embedding_provider": os.getenv("FINAUDIT_EMBEDDING_PROVIDER", "default"),
            "total_chunks": total_chunks,
            "collection_counts": counts,
        }

    def reset(self) -> None:
        """Reset the database (clears all collections). Primarily for test teardown."""
        self.client.reset()
        self._collections.clear()
        self._initialize_collections()


_GLOBAL_VECTOR_STORE: Optional[FinAuditVectorStore] = None


def get_vector_store() -> FinAuditVectorStore:
    """Singleton factory for FinAuditVectorStore."""
    global _GLOBAL_VECTOR_STORE
    if _GLOBAL_VECTOR_STORE is None:
        _GLOBAL_VECTOR_STORE = FinAuditVectorStore()
    return _GLOBAL_VECTOR_STORE
