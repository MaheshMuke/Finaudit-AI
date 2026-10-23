"""Document chunking and metadata preservation for FinAudit AI RAG.

Extracts semantically coherent passages while preserving strict traceability:
document ID, page number, section heading, publication date, and content hash.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    company_name: str
    symbol: str
    document_type: str
    title: str
    source_url: str
    publication_date: str
    page_number: int
    section_heading: str
    content_hash: str
    retrieval_timestamp: str

    def to_metadata(self) -> Dict[str, Any]:
        """Convert chunk attributes to a ChromaDB-compatible metadata dictionary.
        ChromaDB metadata values must be str, int, float, or bool.
        """
        return {
            "document_id": str(self.document_id),
            "chunk_index": int(self.chunk_index),
            "company_name": str(self.company_name or ""),
            "symbol": str(self.symbol or "").upper(),
            "document_type": str(self.document_type or "general"),
            "title": str(self.title or "")[:200],
            "source_url": str(self.source_url or "")[:500],
            "publication_date": str(self.publication_date or ""),
            "page_number": int(self.page_number or 1),
            "section_heading": str(self.section_heading or "General")[:150],
            "content_hash": str(self.content_hash or ""),
            "retrieval_timestamp": str(self.retrieval_timestamp or ""),
        }


KNOWN_SECTION_PATTERNS = [
    (r"(?i)\bmanagement\s+discussion\s+(?:and\s+analysis|analysis)\b", "Management Discussion & Analysis"),
    (r"(?i)\brisk\s+(?:factors|management|governance)\b", "Risk Factors"),
    (r"(?i)\bfinancial\s+results\b", "Financial Results"),
    (r"(?i)\boperating\s+performance\b", "Operating Performance"),
    (r"(?i)\bsegment\s+(?:reporting|performance|revenue)\b", "Segment Reporting"),
    (r"(?i)\bbusiness\s+overview\b", "Business Overview"),
    (r"(?i)\bdividend\s+(?:recommendation|declaration|payment)\b", "Dividend Notice"),
    (r"(?i)\bcorporate\s+governance\b", "Corporate Governance"),
    (r"(?i)\bauditor(?:'s)?\s+report\b", "Auditor's Report"),
    (r"(?i)\bnotes\s+to\s+financial\s+statements\b", "Notes to Statements"),
    (r"(?i)\boutlook\s+(?:and\s+guidance|statement)?\b", "Guidance & Outlook"),
]


class DocumentChunker:
    """Splits documents into semantically meaningful chunks with provenance."""

    def __init__(
        self,
        target_chunk_chars: int = 700,
        overlap_chars: int = 120,
        min_chunk_chars: int = 35,
    ):
        self.target_chunk_chars = target_chunk_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_chars = min_chunk_chars

    @staticmethod
    def compute_content_hash(text: str) -> str:
        """Compute SHA-256 hash of normalized document text."""
        normalized = " ".join((text or "").split()).strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def detect_section_heading(text: str, default: str = "General") -> str:
        """Detect prominent financial report section headings."""
        first_lines = text[:400]
        for pattern, heading in KNOWN_SECTION_PATTERNS:
            if re.search(pattern, first_lines):
                return heading
        return default

    def clean_text(self, text: str) -> str:
        """Strip control characters, excessive whitespace, and repair linebreaks."""
        if not text:
            return ""
        # Remove null bytes and non-printable characters
        cleaned = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
        # Collapse multiple spaces but preserve paragraph breaks
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def split_into_paragraphs_or_sentences(self, text: str) -> List[str]:
        """Split text by paragraphs, or sentences if paragraphs are oversized."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        units: List[str] = []
        for p in paragraphs:
            if len(p) <= self.target_chunk_chars:
                units.append(p)
            else:
                # Split large paragraphs by sentence boundary
                sentences = re.split(r"(?<=[.!?])\s+", p)
                current = ""
                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue
                    if len(current) + len(s) + 1 <= self.target_chunk_chars:
                        current = f"{current} {s}".strip() if current else s
                    else:
                        if current:
                            units.append(current)
                        current = s
                if current:
                    units.append(current)
        return units

    def chunk_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        page_number: int = 1,
    ) -> List[DocumentChunk]:
        """Chunk a single text string into traceable DocumentChunk objects."""
        cleaned = self.clean_text(text)
        if len(cleaned) < self.min_chunk_chars:
            return []

        doc_id = str(metadata.get("document_id") or metadata.get("title") or "doc")
        c_hash = metadata.get("content_hash") or self.compute_content_hash(cleaned)
        retrieved_at = metadata.get("retrieval_timestamp") or datetime.now(timezone.utc).isoformat()
        company_name = metadata.get("company_name", "")
        symbol = metadata.get("symbol", "").upper()
        doc_type = metadata.get("document_type", "general")
        title = metadata.get("title", "Untitled Document")
        source_url = metadata.get("source_url", "")
        pub_date = metadata.get("publication_date", "")
        section = metadata.get("section_heading") or self.detect_section_heading(cleaned)

        units = self.split_into_paragraphs_or_sentences(cleaned)
        if not units:
            return []

        chunks: List[DocumentChunk] = []
        current_chunk_text = ""
        chunk_idx = 0

        for unit in units:
            if len(current_chunk_text) + len(unit) + 1 <= self.target_chunk_chars:
                current_chunk_text = f"{current_chunk_text}\n\n{unit}".strip() if current_chunk_text else unit
            else:
                if current_chunk_text and len(current_chunk_text) >= self.min_chunk_chars:
                    chunk_id = f"{c_hash[:12]}_p{page_number}_c{chunk_idx}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            document_id=doc_id,
                            chunk_index=chunk_idx,
                            text=current_chunk_text,
                            company_name=company_name,
                            symbol=symbol,
                            document_type=doc_type,
                            title=title,
                            source_url=source_url,
                            publication_date=pub_date,
                            page_number=page_number,
                            section_heading=self.detect_section_heading(current_chunk_text, section),
                            content_hash=c_hash,
                            retrieval_timestamp=retrieved_at,
                        )
                    )
                    chunk_idx += 1

                    # Retain overlap from end of current chunk
                    if self.overlap_chars > 0 and len(current_chunk_text) > self.overlap_chars:
                        overlap = current_chunk_text[-self.overlap_chars :]
                        # Try to start overlap at word boundary
                        space_idx = overlap.find(" ")
                        if space_idx != -1:
                            overlap = overlap[space_idx + 1 :]
                        current_chunk_text = f"{overlap}\n\n{unit}".strip()
                    else:
                        current_chunk_text = unit
                else:
                    current_chunk_text = unit

        # Flush trailing chunk
        if current_chunk_text and len(current_chunk_text) >= self.min_chunk_chars:
            chunk_id = f"{c_hash[:12]}_p{page_number}_c{chunk_idx}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    chunk_index=chunk_idx,
                    text=current_chunk_text,
                    company_name=company_name,
                    symbol=symbol,
                    document_type=doc_type,
                    title=title,
                    source_url=source_url,
                    publication_date=pub_date,
                    page_number=page_number,
                    section_heading=self.detect_section_heading(current_chunk_text, section),
                    content_hash=c_hash,
                    retrieval_timestamp=retrieved_at,
                )
            )

        return chunks

    def chunk_paged_document(
        self,
        pages: List[Tuple[int, str]],
        metadata: Dict[str, Any],
    ) -> List[DocumentChunk]:
        """Chunk a multi-page document (e.g. from PDF layout extraction), preserving page numbers."""
        all_chunks: List[DocumentChunk] = []
        overall_text = " ".join(text for _, text in pages)
        c_hash = metadata.get("content_hash") or self.compute_content_hash(overall_text)
        meta_with_hash = dict(metadata)
        meta_with_hash["content_hash"] = c_hash

        for page_num, page_text in pages:
            if not page_text or len(page_text.strip()) < self.min_chunk_chars:
                continue
            page_chunks = self.chunk_document(
                text=page_text,
                metadata=meta_with_hash,
                page_number=page_num,
            )
            all_chunks.extend(page_chunks)

        return all_chunks
