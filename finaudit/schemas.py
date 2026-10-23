from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Company(BaseModel):
    company_id: str
    name: str
    exchange: str = "NSE"
    symbol: str | None = None
    sector: str | None = None
    metadata_source: str = "Curated starter catalog; verify before reliance"


class FinancialReport(BaseModel):
    report_id: str
    company_id: str
    financial_year: str | None = None
    statement_basis: Literal["consolidated", "standalone"]
    source_url: str | None = None
    local_path: str
    checksum: str
    retrieval_timestamp: datetime
    extraction_status: str
    report_title: str | None = None
    publication_date: str | None = None
    source_page: str | None = None
    source_kind: str = "manual_upload"
    download_status: str = "manual_upload"
    verification_results: dict = Field(default_factory=dict)


class FinancialMetric(BaseModel):
    company_id: str
    financial_year: str
    statement_basis: str
    metric_key: str
    original_label: str
    value: float | None
    unit: str
    currency: str
    source_report_id: str
    pdf_page_number: int
    source_excerpt: str
    extraction_method: str = "pdfplumber text row matching"
    extraction_confidence: float = Field(ge=0, le=1)
    verification_status: str = "unverified"


class CalculatedMetric(BaseModel):
    metric_key: str
    company_id: str
    financial_year: str
    value: float | None
    formula: str
    input_metric_keys: list[str]
    validation_status: str


class Finding(BaseModel):
    finding_id: str
    title: str
    severity: Literal["low", "medium", "high"]
    period: str
    explanation: str
    supporting_metrics: list[str]
    evidence_references: list[str]
    counterevidence: str
    confidence: float = Field(ge=0, le=1)
    follow_up_questions: list[str]


class AnalysisRequest(BaseModel):
    company_id: str
    report_ids: list[str] = Field(min_length=1)
    competitor_report_ids: list[str] = Field(default_factory=list)
    financial_years: list[str] | None = None
    analysis_mode: Literal[
        "single-company", "competitor-comparison", "trend-analysis", "risk-review"
    ] = "single-company"


class ReportDiscoveryRequest(BaseModel):
    company_id: str
    financial_year: str
    statement_basis: Literal["consolidated", "standalone"] = "consolidated"


class AnalysisRun(BaseModel):
    run_id: str
    selected_company: str
    selected_competitors: list[str]
    selected_years: list[str]
    analysis_mode: str
    workflow_status: str
    results: dict
    creation_timestamp: datetime
