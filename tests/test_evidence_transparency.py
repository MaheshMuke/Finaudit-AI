"""
Tests for Live Source Transparency and Evidence Visualization System.
Verifies:
- Extended EvidenceRecord and CalculatedEvidence models
- Vector passage provenance tracing and retrieval signal demarcation
- Real-time SourceActivityItem and ResearchActivitySummary emission
- Thread-safe ResearchEventHub concurrent request and session isolation
- Orchestrator tool activity mapping and error handling
- Conflicting sources and unsupported claims handling
- Persistent storage of activity and evidence in SQLite session memory
- SSE streaming endpoint (/sessions/{session_id}/messages/stream)
"""

import json
import time
import pytest
from fastapi.testclient import TestClient

from finaudit.evidence import (
    EvidenceRecord,
    CalculatedEvidence,
    SourceActivityItem,
    ResearchActivitySummary,
    evidence_from_vector_passage,
)
from finaudit.rag.hybrid_retriever import RetrievedPassage
from finaudit.activity_stream import ResearchEventHub, event_hub
from finaudit.agent_orchestrator import AgentOrchestrator
from finaudit.api import app


def test_extended_evidence_record_fields():
    """Verify EvidenceRecord has new traceability fields."""
    rec = EvidenceRecord(
        company="TCS",
        source="screener_income_statement",
        source_type="financial_data",
        claim_type="sourced_fact",
        metric="Sales",
        value=240893.0,
        unit="INR crore",
        reporting_period="FY24",
        statement_basis="Consolidated",
        source_url="https://www.screener.in/company/TCS/",
        page=1,
        section_heading="Financial Statements",
        content_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        retrieved_at="2026-10-23T10:00:00",
        verification_status="verified",
        relevance_score=0.92,
        excerpt="Annual Revenue for FY24 stood at 240,893 crore.",
    )

    d = rec.to_dict()
    assert d["evidence_id"].startswith("ev_")
    assert d["claim_type"] == "sourced_fact"
    assert d["section_heading"] == "Financial Statements"
    assert d["content_hash"].startswith("e3b0c4")
    assert d["verification_status"] == "verified"
    assert d["relevance_score"] == 0.92
    assert d["is_contradictory"] is False


def test_calculated_evidence_defaults():
    """Verify CalculatedEvidence auto-populates calculated_metric claim_type."""
    calc = CalculatedEvidence(
        company="TCS",
        metric="EBITDA Margin",
        value=25.0,
        formula="EBITDA / Revenue * 100",
        inputs=(),
        unit="%",
        currency="INR",
        reporting_period="FY24",
        statement_basis="Consolidated",
        status="ok",
    )

    d = calc.to_dict()
    assert d["claim_type"] == "calculated_metric"
    assert d["verification_status"] == "verified"
    assert d["formula"] == "EBITDA / Revenue * 100"
    assert d["value"] == 25.0


def test_vector_passage_provenance_and_retrieval_signal():
    """Verify vector passage conversion preserves document metadata and tags retrieval signal."""
    passage = RetrievedPassage(
        chunk_id="chunk_tcs_q3_fy25_p12",
        document_id="doc_tcs_q3_fy25",
        text="The Board approved an interim dividend of Rs 9 per equity share and special dividend of Rs 18.",
        company_name="Tata Consultancy Services",
        symbol="TCS",
        document_type="nse_disclosure",
        title="TCS Q3 FY25 Earnings Release & Outcome",
        source_url="https://www.nseindia.com/disclosures/tcs_q3_fy25.pdf",
        publication_date="2025-01-09",
        page_number=12,
        section_heading="Corporate Actions & Dividends",
        semantic_similarity=0.88,
        keyword_score=0.85,
        hybrid_score=0.88,
        collection="financial_disclosures",
    )

    ev = evidence_from_vector_passage(
        company=passage.symbol,
        title=passage.title,
        excerpt=passage.text,
        source_url=passage.source_url,
        page_number=passage.page_number,
        section_heading=passage.section_heading,
        content_hash="a1b2c3d4e5f678901234567890abcdef1234567890abcdef1234567890abcdef",
        relevance_score=passage.hybrid_score,
        claim_type="interpretation",
    )
    assert ev.evidence_id.startswith("ev_")
    assert ev.source == "TCS Q3 FY25 Earnings Release & Outcome"
    assert ev.source_type == "vector_retrieved_passage"
    assert ev.claim_type == "interpretation"
    assert ev.relevance_score == 0.88
    assert ev.page == 12
    assert ev.section_heading == "Corporate Actions & Dividends"
    assert ev.source_url == "https://www.nseindia.com/disclosures/tcs_q3_fy25.pdf"
    assert "dividend" in (ev.excerpt or "").lower()


def test_source_activity_item_serialization():
    """Verify SourceActivityItem captures genuine statuses, durations, and limitations."""
    item = SourceActivityItem(
        activity_id="act_screener_tcs",
        session_id="sess_test",
        request_id="req_test",
        source_name="Screener.in Financial Tables",
        source_category="financial_statements",
        task="Extracting quarterly P&L and 10-year audited balance sheet for TCS",
        status="retrieved",
        started_at="2026-10-23T10:00:00Z",
        completed_at="2026-10-23T10:00:01Z",
        duration_ms=124,
        summary="Extracted 12 quarterly revenue rows and operating profit margins.",
        freshness="fresh",
        url="https://www.screener.in/company/TCS/consolidated/",
        document_title="TCS Consolidated Fundamentals",
        document_type="financial_statement",
        reporting_period="Q2FY25",
        content_hash="fe12345678abcdef",
        extracted_data={"Sales": 64259, "OPM": "26%"},
        evidence_ids=["ev_tcs_q2_sales"],
    )

    d = item.to_dict()
    assert d["activity_id"] == "act_screener_tcs"
    assert d["status"] == "retrieved"
    assert d["duration_ms"] == 124
    assert d["freshness"] == "fresh"
    assert d["category"] == "financial_statements"
    assert d["extracted_data"]["Sales"] == 64259
    assert "ev_tcs_q2_sales" in d["evidence_ids"]


def test_research_activity_summary():
    """Verify ResearchActivitySummary correctly audits sources and evidence counts."""
    summary = ResearchActivitySummary(
        session_id="sess_test",
        request_id="req_test",
        sources_retrieved=["Screener.in", "NSE Disclosures", "ChromaDB"],
        sources_partial_or_unavailable=[{"source": "Yahoo Finance RSS", "reason": "network timeout"}],
        documents_extracted_count=8,
        passages_extracted_count=12,
        evidence_records_used_count=6,
        data_gaps=["Commodity correlation unavailable for IT consulting"],
        conflicts_detected=["Differing OPM definitions between standalone and consolidated"],
        freshness_limitations=["FY23 annual report is historical (stale)"],
    )

    d = summary.to_dict()
    assert len(d["sources_retrieved"]) == 3
    assert len(d["sources_partial_or_unavailable"]) == 1
    assert d["evidence_records_used_count"] == 6
    assert len(d["conflicts_detected"]) == 1


def test_concurrent_request_isolation_in_event_hub():
    """Verify that ResearchEventHub cleanly isolates events between concurrent requests."""
    hub = ResearchEventHub()

    # Session 1 with request A
    q_1 = hub.register_stream("sess_alpha", "req_aaa")
    # Session 2 with request B
    q_2 = hub.register_stream("sess_beta", "req_bbb")
    # Same session 1 with concurrent request C
    q_3 = hub.register_stream("sess_alpha", "req_ccc")

    # Emit event for req_aaa
    hub.emit("activity_start", "sess_alpha", "req_aaa", {"activity_id": "act_1"})

    # Emit event for req_bbb
    hub.emit("activity_start", "sess_beta", "req_bbb", {"activity_id": "act_2"})

    # Check q_1 received act_1
    ev_1 = q_1.get_nowait()
    assert ev_1["type"] == "activity_start"
    assert ev_1["data"]["activity_id"] == "act_1"
    assert q_1.empty()

    # Check q_2 received act_2
    ev_2 = q_2.get_nowait()
    assert ev_2["type"] == "activity_start"
    assert ev_2["data"]["activity_id"] == "act_2"
    assert q_2.empty()

    # Check q_3 received nothing (isolated request on same session)
    assert q_3.empty()

    # Clean up listeners
    hub.close_stream("sess_alpha", "req_aaa")
    hub.close_stream("sess_beta", "req_bbb")
    hub.close_stream("sess_alpha", "req_ccc")


def test_orchestrator_tool_activity_mapping():
    """Verify AgentOrchestrator maps tools to typed SourceActivityItems with correct status and metadata."""
    orch = AgentOrchestrator()
    started_iso = "2026-10-23T10:00:00Z"

    # 1. Successful company profile extraction
    act_profile, evs_profile = orch._process_tool_activity(
        tool_name="get_company_profile",
        result={
            "symbol": "INFY",
            "company_name": "Infosys Ltd",
            "industry": "IT Consulting & Software",
            "market_cap": 650000.0,
            "current_price": 1580.0,
            "source_url": "https://www.screener.in/company/INFY/",
            "cached": True,
        },
        duration_ms=45,
        started_iso=started_iso,
        symbol="INFY",
        industry="IT Consulting & Software",
        session_id="test_sess",
        request_id="test_req",
        activity_id="act_profile_1",
    )
    assert act_profile.category == "market_data"
    assert act_profile.status == "retrieved"
    assert len(evs_profile) >= 1
    assert any(e.metric == "current_price" for e in evs_profile)

    # 2. Failed extraction / tool error
    act_err, evs_err = orch._process_tool_activity(
        tool_name="discover_official_annual_reports",
        result={"error": "NSE disclosure portal returned 404 Not Found for ticker"},
        duration_ms=320,
        started_iso=started_iso,
        symbol="NONEXISTENT_XYZ",
        industry="",
        session_id="test_sess",
        request_id="test_req",
        activity_id="act_err_1",
    )
    assert act_err.status == "extraction_failed"
    assert "404" in (act_err.error_details or "")
    assert len(evs_err) == 0

    # 3. Deterministic calculation tool
    act_calc, evs_calc = orch._process_tool_activity(
        tool_name="calculate_change",
        result={
            "metric_name": "Operating Profit",
            "absolute_change": 20.0,
            "percentage_change": 20.0,
            "formula": "((current - previous) / previous) * 100",
            "statement": "Operating Profit increased by 20.00%",
        },
        duration_ms=2,
        started_iso=started_iso,
        symbol="INFY",
        industry="IT Consulting & Software",
        session_id="test_sess",
        request_id="test_req",
        activity_id="act_calc_1",
    )
    assert act_calc.category == "calculations"
    assert act_calc.status == "retrieved"
    assert len(evs_calc) == 1
    assert evs_calc[0].source_type == "deterministic_calculation"


def test_investigate_company_full_transparency():
    """Verify investigate_company generates structured source activity, evidence records, and summary."""
    orch = AgentOrchestrator()
    res = orch.investigate_company(
        query="TCS",
        question="What was TCS revenue growth and operating margin in the latest period?",
        request_id="req_test_full_transparency",
    )

    assert "research_activity" in res
    assert len(res["research_activity"]) > 0

    # Every item must have genuine source information
    for act in res["research_activity"]:
        assert "activity_id" in act
        assert "source_name" in act
        assert "source_category" in act
        assert "status" in act

    # Check research summary
    assert "research_summary" in res
    summary = res["research_summary"]
    assert "sources_retrieved" in summary
    assert summary["documents_extracted_count"] >= 0

    # Check evidence records
    assert "evidence_records" in res or "evidence" in res


def test_contradictory_evidence_preservation():
    """Verify that contradictory evidence maintains both viewpoints and flags contradiction details."""
    rec1 = EvidenceRecord(
        company="TCS",
        source="screener",
        source_type="financial_data",
        metric="Operating Margin",
        value=26.0,
        unit="%",
        reporting_period="Q2FY25",
        statement_basis="Consolidated",
    )
    rec2 = EvidenceRecord(
        company="TCS",
        source="nse_filing",
        source_type="official_filing",
        metric="Operating Margin",
        value=24.5,
        unit="%",
        reporting_period="Q2FY25",
        statement_basis="Standalone",
        is_contradictory=True,
        contradiction_details="Consolidated operating margin is 26.0% while Standalone operating margin is 24.5% due to international subsidiary margins.",
    )

    assert rec2.is_contradictory is True
    assert "Standalone" in (rec2.contradiction_details or "")
    assert rec2.to_dict()["is_contradictory"] is True


def test_api_session_streaming_and_persistence():
    """Verify the /sessions/{session_id}/messages/stream endpoint yields SSE events and persists to SQLite."""
    client = TestClient(app)

    # 1. Create research session
    sess_res = client.post("/sessions", json={"target_company": "TCS", "title": "TCS Live Source Audit"})
    assert sess_res.status_code == 200
    session_id = sess_res.json()["session_id"]

    # 2. Test streaming endpoint
    stream_res = client.post(
        f"/sessions/{session_id}/messages/stream",
        json={"content": "Explain TCS quarterly revenue growth and operating margin"},
    )
    assert stream_res.status_code == 200
    assert "text/event-stream" in stream_res.headers.get("content-type", "")

    raw_sse_text = stream_res.text
    # Verify events are delivered in SSE data envelopes
    assert "connected" in raw_sse_text
    assert "activity_start" in raw_sse_text or "evidence_linked" in raw_sse_text or "done" in raw_sse_text

    # 3. Verify SQLite persistence by retrieving session (polling until worker thread completes)
    messages = []
    for _ in range(30):
        reload_res = client.get(f"/sessions/{session_id}")
        if reload_res.status_code == 200:
            data = reload_res.json()
            messages = data.get("messages", [])
            if len(messages) >= 2:
                break
        time.sleep(0.5)

    assert len(messages) >= 2  # user and assistant messages

    assistant_msg = messages[-1]
    assert assistant_msg["role"] == "assistant"
    metadata = assistant_msg.get("metadata", {})

    # Research activity and summary MUST be preserved across reloads
    assert "research_activity" in metadata
    assert len(metadata["research_activity"]) > 0
    assert "research_summary" in metadata
    assert "evidence_records" in metadata
