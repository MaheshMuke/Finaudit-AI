"""FinAudit AI Vector Database and RAG System.

Provides persistent vector indexing, semantic and hybrid retrieval,
document chunking, and agentic evidence grounding for Indian equities.
"""

from finaudit.rag.vector_store import FinAuditVectorStore, get_vector_store
from finaudit.rag.document_chunker import DocumentChunker, DocumentChunk
from finaudit.rag.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from finaudit.rag.ingestion_service import RAGIngestionService, get_ingestion_service

__all__ = [
    "FinAuditVectorStore",
    "get_vector_store",
    "DocumentChunker",
    "DocumentChunk",
    "HybridRetriever",
    "get_hybrid_retriever",
    "RAGIngestionService",
    "get_ingestion_service",
]
