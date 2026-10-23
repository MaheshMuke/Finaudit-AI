"""Stable evidence records shared by calculations and research responses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any, Iterable


def _stable_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"ev_{sha256(encoded.encode('utf-8')).hexdigest()[:20]}"


@dataclass(frozen=True)
class EvidenceRecord:
    company: str
    source: str
    source_type: str
    metric: str | None = None
    value: Any = None
    unit: str | None = None
    currency: str | None = None
    reporting_period: str | None = None
    duration: str | None = None
    statement_basis: str | None = None
    reporting_basis: str | None = None
    source_url: str | None = None
    source_document_id: str | None = None
    page: int | None = None
    section_heading: str | None = None
    excerpt: str | None = None
    retrieved_at: str | None = None
    quality_warnings: tuple[str, ...] = field(default_factory=tuple)
    period_start: str | None = None
    period_end: str | None = None
    period_type: str | None = None
    publication_date: str | None = None
    extraction_method: str | None = None
    original_value: Any = None
    original_unit: str | None = None
    limitations: tuple[str, ...] = field(default_factory=tuple)
    verification_status: str = "unverified"
    claim_type: str = "sourced_fact"
    relevance_score: float | None = None
    content_hash: str | None = None
    is_contradictory: bool = False
    contradiction_details: str | None = None

    @property
    def evidence_id(self) -> str:
        return _stable_id(
            {
                "company": self.company,
                "source": self.source,
                "source_type": self.source_type,
                "metric": self.metric,
                "value": self.value,
                "unit": self.unit,
                "currency": self.currency,
                "reporting_period": self.reporting_period,
                "period_start": self.period_start,
                "period_end": self.period_end,
                "period_type": self.period_type,
                "statement_basis": self.statement_basis,
                "reporting_basis": self.reporting_basis,
                "source_url": self.source_url,
                "source_document_id": self.source_document_id,
                "page": self.page,
                "section_heading": self.section_heading,
                "content_hash": self.content_hash,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["quality_warnings"] = list(self.quality_warnings)
        result["limitations"] = list(self.limitations)
        result["evidence_id"] = self.evidence_id
        return result


@dataclass(frozen=True)
class CalculatedEvidence:
    company: str
    metric: str
    value: float | None
    formula: str
    inputs: tuple[EvidenceRecord, ...]
    unit: str | None
    currency: str | None
    reporting_period: str | None
    statement_basis: str | None
    status: str
    reason: str | None = None
    claim_type: str = "calculated_metric"
    verification_status: str = "verified"

    @property
    def evidence_id(self) -> str:
        return _stable_id(
            {
                "company": self.company,
                "metric": self.metric,
                "value": self.value,
                "formula": self.formula,
                "input_ids": [item.evidence_id for item in self.inputs],
                "reporting_period": self.reporting_period,
                "statement_basis": self.statement_basis,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["inputs"] = [item.to_dict() for item in self.inputs]
        result["evidence_id"] = self.evidence_id
        return result


@dataclass
class SourceActivityItem:
    """Represents real live activity for an individual research source or tool."""
    activity_id: str
    session_id: str
    request_id: str
    source_name: str
    source_category: str  # "official_filings", "financial_statements", "market_data", "news_web", "vector_rag", "peer_intelligence", "calculations", "session_memory"
    task: str  # e.g., "Fetching NSE announcements and corporate disclosures"
    status: str  # "pending", "in_progress", "retrieved", "partially_retrieved", "extraction_failed", "unavailable", "verification_failed"
    started_at: str
    completed_at: str | None = None
    duration_ms: int = 0
    summary: str = ""
    error_details: str | None = None
    limitations: list[str] = field(default_factory=list)
    freshness: str = "fresh"  # "fresh", "cached", "stale", "incomplete", "conflicting"
    url: str | None = None
    document_title: str | None = None
    document_type: str | None = None
    publication_date: str | None = None
    event_date: str | None = None
    reporting_period: str | None = None
    page_number: int | None = None
    section_heading: str | None = None
    content_hash: str | None = None
    extracted_data: dict[str, Any] = field(default_factory=dict)
    excerpt: str | None = None
    evidence_ids: list[str] = field(default_factory=list)

    @property
    def category(self) -> str:
        return self.source_category

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.source_category
        d["freshness_status"] = self.freshness
        d["research_task"] = self.task
        return d


@dataclass
class ResearchActivitySummary:
    """Concise audit summary of research activity at query conclusion."""
    session_id: str
    request_id: str
    sources_retrieved: list[str]
    sources_partial_or_unavailable: list[dict[str, Any]]
    documents_extracted_count: int
    passages_extracted_count: int
    evidence_records_used_count: int
    data_gaps: list[str] = field(default_factory=list)
    conflicts_detected: list[str] = field(default_factory=list)
    freshness_limitations: list[str] = field(default_factory=list)

    @property
    def total_sources_attempted(self) -> int:
        return len(self.sources_retrieved) + len(self.sources_partial_or_unavailable)

    @property
    def total_sources_retrieved(self) -> int:
        return len(self.sources_retrieved)

    @property
    def total_sources_unavailable(self) -> int:
        return len(self.sources_partial_or_unavailable)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_sources_attempted"] = self.total_sources_attempted
        d["total_sources_retrieved"] = self.total_sources_retrieved
        d["total_sources_unavailable"] = self.total_sources_unavailable
        d["total_passages_extracted"] = self.passages_extracted_count
        d["total_evidence_used"] = self.evidence_records_used_count
        return d


def evidence_from_metric(
    company: str,
    metric: str,
    value: Any,
    *,
    unit: str | None,
    currency: str | None,
    reporting_period: str | None,
    statement_basis: str | None,
    source: str,
    reporting_basis: str | None = None,
    source_type: str = "structured_data",
    source_url: str | None = None,
    retrieved_at: str | None = None,
    period_start: str | None = None,
    period_end: str | None = None,
    period_type: str | None = None,
    publication_date: str | None = None,
    extraction_method: str | None = None,
    original_value: Any = None,
    original_unit: str | None = None,
    limitations: tuple[str, ...] = (),
    section_heading: str | None = None,
    verification_status: str = "verified",
    claim_type: str = "sourced_fact",
    content_hash: str | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        company=company,
        source=source,
        source_type=source_type,
        metric=metric,
        value=value,
        unit=unit,
        currency=currency,
        reporting_period=reporting_period,
        statement_basis=statement_basis,
        reporting_basis=reporting_basis,
        source_url=source_url,
        retrieved_at=retrieved_at,
        period_start=period_start,
        period_end=period_end,
        period_type=period_type,
        publication_date=publication_date,
        extraction_method=extraction_method,
        original_value=original_value,
        original_unit=original_unit,
        limitations=limitations,
        section_heading=section_heading,
        verification_status=verification_status,
        claim_type=claim_type,
        content_hash=content_hash,
    )


def evidence_from_vector_passage(
    company: str,
    title: str,
    excerpt: str,
    *,
    source_url: str | None = None,
    page_number: int | None = None,
    section_heading: str | None = None,
    document_type: str = "nse_disclosure",
    publication_date: str | None = None,
    retrieval_timestamp: str | None = None,
    content_hash: str | None = None,
    relevance_score: float | None = None,
    source_document_id: str | None = None,
    claim_type: str = "sourced_fact",
) -> EvidenceRecord:
    """Build an EvidenceRecord from a vector-retrieved chunk preserving provenance."""
    return EvidenceRecord(
        company=company,
        source=title,
        source_type="vector_retrieved_passage",
        source_url=source_url,
        source_document_id=source_document_id or title,
        page=page_number,
        section_heading=section_heading,
        excerpt=excerpt,
        retrieved_at=retrieval_timestamp,
        publication_date=publication_date,
        extraction_method="ChromaDB Hybrid Vector Search",
        relevance_score=relevance_score,
        content_hash=content_hash,
        verification_status="verified",
        claim_type=claim_type,
    )


def evidence_ids(records: Iterable[EvidenceRecord | CalculatedEvidence]) -> list[str]:
    return [record.evidence_id for record in records]
