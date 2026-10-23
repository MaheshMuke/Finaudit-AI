"""Hybrid retriever combining semantic ChromaDB search with exact keyword matching.

Features:
1. Multi-collection querying (disclosures, reports, news, findings).
2. Metadata filtering by ticker, document type, and date range.
3. Keyword / exact-term boosting for financial metrics, tickers, and announcement numbers.
4. Prompt injection protection: wraps retrieved passages as untrusted external evidence.
5. Explicit separation of retrieval relevance score from factual confidence.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from finaudit.rag.vector_store import (
    ALL_COLLECTIONS,
    COLLECTION_DISCLOSURES,
    COLLECTION_REPORTS,
    COLLECTION_NEWS,
    COLLECTION_FINDINGS,
    FinAuditVectorStore,
    get_vector_store,
)

logger = logging.getLogger(__name__)

INJECTION_PATTERNS = [
    r"(?i)ignore\s+(?:all\s+)?previous\s+instructions",
    r"(?i)system\s*:\s*",
    r"(?i)developer\s*:\s*",
    r"(?i)assistant\s*:\s*",
    r"(?i)you\s+are\s+now\s+in\s+developer\s+mode",
    r"(?i)<\|im_start\|>",
    r"(?i)<\|im_end\|>",
]


def sanitize_evidence_text(raw_text: str) -> str:
    """Sanitize retrieved text against prompt injection patterns.
    Neutralizes commands that attempt to override the system prompt.
    """
    if not raw_text:
        return ""
    cleaned = raw_text
    for pattern in INJECTION_PATTERNS:
        cleaned = re.sub(pattern, "[FILTERED_INSTRUCTION]", cleaned)
    return cleaned.strip()


@dataclass
class RetrievedPassage:
    chunk_id: str
    document_id: str
    text: str
    company_name: str
    symbol: str
    document_type: str
    title: str
    source_url: str
    publication_date: str
    page_number: int
    section_heading: str
    semantic_similarity: float
    keyword_score: float
    hybrid_score: float
    collection: str

    @property
    def untrusted_text(self) -> str:
        """Format chunk as securely tagged, untrusted external evidence."""
        header = (
            f"SOURCE DOCUMENT: {self.title} | COMPANY: {self.company_name} ({self.symbol}) | "
            f"TYPE: {self.document_type} | PAGE: {self.page_number} | "
            f"DATE: {self.publication_date} | URL: {self.source_url}"
        )
        safe_body = sanitize_evidence_text(self.text)
        return (
            f"<untrusted_retrieved_evidence>\n"
            f"[{header}]\n"
            f"{safe_body}\n"
            f"</untrusted_retrieved_evidence>"
        )

    def to_citation(self) -> Dict[str, Any]:
        """Convert passage to user-facing citation record."""
        return {
            "source": self.title or self.document_type,
            "kind": self.document_type,
            "link": self.source_url,
            "url": self.source_url,
            "title": self.title,
            "symbol": self.symbol,
            "page_number": self.page_number,
            "section_heading": self.section_heading,
            "publication_date": self.publication_date,
            "retrieval_method": "hybrid_rag",
            "relevance_score": round(self.hybrid_score, 4),
            "semantic_similarity": round(self.semantic_similarity, 4),
            "excerpt": sanitize_evidence_text(self.text[:300]) + ("..." if len(self.text) > 300 else ""),
            "content_retrieved": True,
        }


class HybridRetriever:
    """Combines dense semantic vector retrieval with sparse term matching."""

    def __init__(self, vector_store: Optional[FinAuditVectorStore] = None):
        self.vector_store = vector_store or get_vector_store()

    def compute_keyword_score(self, text: str, query: str, symbol: Optional[str] = None) -> float:
        """Compute keyword relevance score in [0.0, 1.0].
        Boosts exact ticker matches, percentage/numerical metrics, and financial terms.
        """
        text_lower = (text or "").lower()
        query_lower = (query or "").lower()

        score = 0.0
        # 1. Ticker exact match
        if symbol and symbol.lower() in text_lower:
            score += 0.35

        # 2. Number and metric matches
        numbers_in_query = re.findall(r"\b\d+(?:\.\d+)?%?\b", query_lower)
        if numbers_in_query:
            matched_numbers = sum(1 for n in numbers_in_query if n in text_lower)
            score += 0.35 * (matched_numbers / len(numbers_in_query))

        # 3. Informative query words match (excluding stop words)
        words = [w for w in re.findall(r"[A-Za-z0-9\-]+", query_lower) if len(w) >= 3]
        if words:
            matched_words = sum(1 for w in words if w in text_lower)
            score += 0.30 * (matched_words / len(words))

        return min(1.0, score)

    def search(
        self,
        query: str,
        symbol: Optional[str] = None,
        document_types: Optional[List[str]] = None,
        collections: Optional[List[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 5,
        alpha: float = 0.65,
    ) -> List[RetrievedPassage]:
        """Perform hybrid retrieval across specified collections.
        
        Args:
            query: Natural language query or financial search phrase.
            symbol: Target company stock ticker (e.g. 'TCS').
            document_types: Filter to specific document types ('annual_report', 'nse_announcement', etc.).
            collections: List of collections to search (defaults to all).
            date_from: Lower bound publication date (YYYY-MM-DD or YYYY).
            date_to: Upper bound publication date (YYYY-MM-DD or YYYY).
            limit: Maximum passages to return.
            alpha: Weight for semantic similarity (1 - alpha is keyword weight).
        """
        clean_q = (query or "").strip()
        if not clean_q:
            return []

        clean_sym = symbol.strip().upper() if symbol else None
        target_cols = collections or ALL_COLLECTIONS

        # Build ChromaDB metadata filter
        where_filter: Optional[Dict[str, Any]] = None
        filter_conditions: List[Dict[str, Any]] = []

        if clean_sym:
            filter_conditions.append({"symbol": clean_sym})

        if document_types:
            if len(document_types) == 1:
                filter_conditions.append({"document_type": document_types[0]})
            else:
                filter_conditions.append({"$or": [{"document_type": dt} for dt in document_types]})

        if date_from:
            filter_conditions.append({"publication_date": {"$gte": str(date_from)}})

        if date_to:
            filter_conditions.append({"publication_date": {"$lte": str(date_to)}})

        if len(filter_conditions) == 1:
            where_filter = filter_conditions[0]
        elif len(filter_conditions) > 1:
            where_filter = {"$and": filter_conditions}

        # Query all requested collections
        candidates: List[RetrievedPassage] = []
        seen_texts: Set[str] = set()

        for col_name in target_cols:
            raw_hits = self.vector_store.query(
                collection_name=col_name,
                query_text=clean_q,
                n_results=limit * 2,
                where_filter=where_filter,
            )

            for hit in raw_hits:
                text = hit.get("text", "")
                # Normalize text to detect exact duplicate passages
                text_signature = " ".join(text[:200].split()).lower()
                if text_signature in seen_texts:
                    continue
                seen_texts.add(text_signature)

                meta = hit.get("metadata", {})
                sem_sim = float(hit.get("similarity", 0.5))
                kw_score = self.compute_keyword_score(text, clean_q, clean_sym)
                hybrid_score = (alpha * sem_sim) + ((1.0 - alpha) * kw_score)

                candidates.append(
                    RetrievedPassage(
                        chunk_id=str(hit.get("chunk_id", "")),
                        document_id=str(meta.get("document_id", "")),
                        text=text,
                        company_name=str(meta.get("company_name", "")),
                        symbol=str(meta.get("symbol", clean_sym or "")),
                        document_type=str(meta.get("document_type", "document")),
                        title=str(meta.get("title", "")),
                        source_url=str(meta.get("source_url", "")),
                        publication_date=str(meta.get("publication_date", "")),
                        page_number=int(meta.get("page_number", 1)),
                        section_heading=str(meta.get("section_heading", "General")),
                        semantic_similarity=sem_sim,
                        keyword_score=kw_score,
                        hybrid_score=hybrid_score,
                        collection=col_name,
                    )
                )

        # Rank candidates by hybrid score descending
        candidates.sort(key=lambda p: p.hybrid_score, reverse=True)
        return candidates[:limit]


_GLOBAL_RETRIEVER: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    """Singleton factory for HybridRetriever."""
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        _GLOBAL_RETRIEVER = HybridRetriever()
    return _GLOBAL_RETRIEVER
