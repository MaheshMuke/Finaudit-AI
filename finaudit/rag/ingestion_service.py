"""Document ingestion and incremental indexing pipeline for FinAudit AI RAG.

Connects to existing source adapters:
- NSE corporate disclosures and PDF attachments from SQLite table `nse_disclosures`.
- Annual reports and investor presentations from local PDF files (`PyMuPDF` / `pdfplumber`).
- Financial news articles from `NewsService`.
- Persistent research findings from `MemoryService`.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from finaudit import database
from finaudit.company_registry import get_company_registry
from finaudit.news_service import get_news_service
from finaudit.rag.document_chunker import DocumentChunk, DocumentChunker
from finaudit.rag.vector_store import (
    COLLECTION_DISCLOSURES,
    COLLECTION_REPORTS,
    COLLECTION_NEWS,
    COLLECTION_FINDINGS,
    FinAuditVectorStore,
    get_vector_store,
)

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


def extract_pages_from_pdf(pdf_path: Path) -> List[Tuple[int, str]]:
    """Extract page-by-page text using PyMuPDF or pdfplumber."""
    pages: List[Tuple[int, str]] = []
    if not pdf_path.exists():
        logger.warning("PDF path does not exist: %s", pdf_path)
        return []

    # Priority 1: PyMuPDF (fast, robust spatial parsing)
    if fitz is not None:
        try:
            doc = fitz.open(str(pdf_path))
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text = page.get_text("text") or ""
                if text.strip():
                    pages.append((page_idx + 1, text))
            doc.close()
            if pages:
                return pages
        except Exception as exc:
            logger.warning("PyMuPDF extraction failed for %s (%s); trying pdfplumber", pdf_path, exc)

    # Priority 2: pdfplumber
    if pdfplumber is not None:
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for idx, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append((idx + 1, text))
            if pages:
                return pages
        except Exception as exc:
            logger.error("pdfplumber extraction failed for %s: %s", pdf_path, exc)

    return pages


class RAGIngestionService:
    """Manages document text extraction and incremental indexing into ChromaDB."""

    def __init__(
        self,
        vector_store: Optional[FinAuditVectorStore] = None,
        chunker: Optional[DocumentChunker] = None,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.chunker = chunker or DocumentChunker()
        self.registry = get_company_registry()
        self.news_service = get_news_service()

    def index_nse_disclosure(
        self,
        disclosure: Dict[str, Any],
        force_reindex: bool = False,
    ) -> Dict[str, Any]:
        """Index a single NSE disclosure record from database into ChromaDB."""
        doc_id = str(disclosure.get("disclosure_id") or disclosure.get("document_id") or "")
        if not doc_id:
            return {"status": "skipped", "reason": "missing_document_id"}

        symbol = str(disclosure.get("symbol") or "").upper()
        company_name = str(disclosure.get("company_name") or "")
        if not company_name and symbol:
            rec = self.registry.get_by_symbol(symbol)
            company_name = rec.company_name if rec else symbol

        title = str(disclosure.get("title") or "NSE Disclosure")
        source_url = str(disclosure.get("source_url") or disclosure.get("attachment_url") or "")
        pub_date = str(disclosure.get("publication_date") or "")
        retrieved_at = str(disclosure.get("retrieved_at") or datetime.now(timezone.utc).isoformat())

        raw_text = str(disclosure.get("extracted_text") or "")
        local_path = disclosure.get("local_path")

        # If text is empty but local PDF exists, extract it
        if (not raw_text.strip() or len(raw_text.strip()) < 30) and local_path:
            p = Path(local_path)
            if p.exists() and p.suffix.lower() == ".pdf":
                pages = extract_pages_from_pdf(p)
                raw_text = "\n\n".join(t for _, t in pages)

        # Fall back to title + material facts if extracted text is unavailable
        if not raw_text.strip() or len(raw_text.strip()) < 20:
            facts = str(disclosure.get("material_facts") or "")
            raw_text = f"Title: {title}\nSymbol: {symbol}\nEvent: {disclosure.get('event_type')}\n{facts}"

        if len(raw_text.strip()) < 20:
            return {"status": "skipped", "reason": "empty_text", "doc_id": doc_id}

        c_hash = self.chunker.compute_content_hash(raw_text)

        # Check incremental skip
        if not force_reindex and self.vector_store.has_document(COLLECTION_DISCLOSURES, doc_id, c_hash):
            return {"status": "skipped_unchanged", "doc_id": doc_id, "content_hash": c_hash}

        metadata = {
            "document_id": doc_id,
            "company_name": company_name,
            "symbol": symbol,
            "document_type": "nse_announcement",
            "title": title,
            "source_url": source_url,
            "publication_date": pub_date,
            "retrieval_timestamp": retrieved_at,
            "content_hash": c_hash,
            "section_heading": str(disclosure.get("event_type") or "NSE Announcement"),
        }

        chunks = self.chunker.chunk_document(raw_text, metadata, page_number=1)
        if not chunks:
            return {"status": "skipped", "reason": "no_valid_chunks", "doc_id": doc_id}

        added = self.vector_store.add_chunks(COLLECTION_DISCLOSURES, chunks, replace_existing_doc=True)
        return {
            "status": "indexed",
            "doc_id": doc_id,
            "chunks_added": added,
            "collection": COLLECTION_DISCLOSURES,
            "content_hash": c_hash,
        }

    def index_pdf_document(
        self,
        pdf_path: Union[str, Path],
        metadata: Dict[str, Any],
        collection_name: str = COLLECTION_REPORTS,
        force_reindex: bool = False,
    ) -> Dict[str, Any]:
        """Extract and index a multi-page PDF document preserving page-level provenance."""
        path = Path(pdf_path)
        if not path.exists():
            return {"status": "failed", "error": f"File not found: {path}"}

        doc_id = str(metadata.get("document_id") or path.stem)
        pages = extract_pages_from_pdf(path)
        if not pages:
            return {"status": "failed", "error": f"No extractable text in PDF: {path}", "doc_id": doc_id}

        overall_text = " ".join(t for _, t in pages)
        c_hash = self.chunker.compute_content_hash(overall_text)

        if not force_reindex and self.vector_store.has_document(collection_name, doc_id, c_hash):
            return {"status": "skipped_unchanged", "doc_id": doc_id, "content_hash": c_hash}

        meta = dict(metadata)
        meta["document_id"] = doc_id
        meta["content_hash"] = c_hash
        meta["source_url"] = meta.get("source_url") or str(path)

        chunks = self.chunker.chunk_paged_document(pages, meta)
        if not chunks:
            return {"status": "failed", "error": "Chunking produced no valid chunks", "doc_id": doc_id}

        added = self.vector_store.add_chunks(collection_name, chunks, replace_existing_doc=True)
        return {
            "status": "indexed",
            "doc_id": doc_id,
            "chunks_added": added,
            "page_count": len(pages),
            "collection": collection_name,
            "content_hash": c_hash,
        }

    def index_news_article(
        self,
        article: Dict[str, Any],
        force_reindex: bool = False,
    ) -> Dict[str, Any]:
        """Index a financial news article into news_and_commentary collection."""
        url = article.get("link") or article.get("url") or ""
        title = article.get("title") or "News Article"
        doc_id = f"news_{hash(url or title)}"
        text = article.get("summary") or article.get("content") or title
        if not text or len(text.strip()) < 25:
            return {"status": "skipped", "reason": "empty_text"}

        c_hash = self.chunker.compute_content_hash(text)
        if not force_reindex and self.vector_store.has_document(COLLECTION_NEWS, doc_id, c_hash):
            return {"status": "skipped_unchanged", "doc_id": doc_id}

        metadata = {
            "document_id": doc_id,
            "company_name": article.get("company_name", ""),
            "symbol": (article.get("symbol") or "").upper(),
            "document_type": "news",
            "title": title,
            "source_url": url,
            "publication_date": article.get("published", ""),
            "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
            "content_hash": c_hash,
            "section_heading": article.get("source", "News"),
        }

        chunks = self.chunker.chunk_document(text, metadata, page_number=1)
        added = self.vector_store.add_chunks(COLLECTION_NEWS, chunks)
        return {"status": "indexed", "doc_id": doc_id, "chunks_added": added}

    def index_session_finding(
        self,
        finding: Dict[str, Any],
        session_id: str,
        company_symbol: str,
    ) -> Dict[str, Any]:
        """Index a persistent qualitative finding into the research_findings collection."""
        claim = finding.get("claim") or finding.get("finding_text") or ""
        if not claim or len(claim.strip()) < 15:
            return {"status": "skipped", "reason": "empty_claim"}

        f_id = str(finding.get("finding_id") or f"find_{hash(claim)}_{session_id}")
        rec = self.registry.get_by_symbol(company_symbol)
        comp_name = rec.company_name if rec else company_symbol

        metadata = {
            "document_id": f_id,
            "company_name": comp_name,
            "symbol": company_symbol.upper(),
            "document_type": "session_finding",
            "title": f"Finding: {company_symbol} ({finding.get('topic', 'General')})",
            "source_url": f"session://{session_id}",
            "publication_date": finding.get("created_at", ""),
            "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
            "content_hash": self.chunker.compute_content_hash(claim),
            "section_heading": f"Session Finding ({session_id[:8]})",
        }

        chunks = self.chunker.chunk_document(claim, metadata, page_number=1)
        added = self.vector_store.add_chunks(COLLECTION_FINDINGS, chunks)
        return {"status": "indexed", "doc_id": f_id, "chunks_added": added}

    def ingest_all_for_company(
        self,
        symbol: str,
        limit_disclosures: int = 50,
    ) -> Dict[str, Any]:
        """Index all known stored disclosures and reports for a given symbol."""
        clean_sym = symbol.strip().upper()
        disclosures = database.list_nse_disclosures(company_symbol=clean_sym, limit=limit_disclosures)

        results = {
            "symbol": clean_sym,
            "disclosures_processed": 0,
            "disclosures_indexed": 0,
            "disclosures_skipped": 0,
            "chunks_added": 0,
            "errors": [],
        }

        for disc in disclosures:
            results["disclosures_processed"] += 1
            try:
                res = self.index_nse_disclosure(disc)
                if res.get("status") == "indexed":
                    results["disclosures_indexed"] += 1
                    results["chunks_added"] += res.get("chunks_added", 0)
                else:
                    results["disclosures_skipped"] += 1
            except Exception as exc:
                results["errors"].append(f"{disc.get('disclosure_id')}: {exc}")

        return results

    def ingest_recent_news(self, limit: int = 30) -> Dict[str, Any]:
        """Index recent financial news articles."""
        articles = self.news_service.get_recent_articles(limit=limit)
        results = {"articles_processed": len(articles), "articles_indexed": 0, "chunks_added": 0}
        for art in articles:
            try:
                res = self.index_news_article(art)
                if res.get("status") == "indexed":
                    results["articles_indexed"] += 1
                    results["chunks_added"] += res.get("chunks_added", 0)
            except Exception as exc:
                logger.warning("Failed indexing news article: %s", exc)
        return results


_GLOBAL_INGESTION_SERVICE: Optional[RAGIngestionService] = None


def get_ingestion_service() -> RAGIngestionService:
    """Singleton factory for RAGIngestionService."""
    global _GLOBAL_INGESTION_SERVICE
    if _GLOBAL_INGESTION_SERVICE is None:
        _GLOBAL_INGESTION_SERVICE = RAGIngestionService()
    return _GLOBAL_INGESTION_SERVICE
