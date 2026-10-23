import pytest
from datetime import datetime
from fastapi.testclient import TestClient

from finaudit.api import app
from finaudit.company_registry import CompanyRegistry
from finaudit.competitors import CompetitorService
from finaudit.database import (
    create_research_session,
    get_research_session,
    list_research_sessions,
    update_session_peers,
    delete_research_session,
    save_session_message,
    get_session_messages,
    save_session_finding,
    get_session_findings,
    save_nse_disclosure,
    list_nse_disclosures,
)
from finaudit.macro_service import MacroMarketIntelligenceService
from finaudit.memory_service import MemoryService
from finaudit.research_sources import (
    OfficialQuarterlyDisclosureAdapter,
    classify_disclosure_event,
    extract_disclosure_material_facts,
)


@pytest.fixture
def test_client():
    return TestClient(app)


def test_nse_disclosure_classification_and_material_facts():
    title = (
        "Financial Results for the quarter ended September 30, 2026 and interim dividend "
        "of Rs 10 per share. The approval is subject to regulatory approvals and ongoing litigation."
    )
    event_type = classify_disclosure_event(title)
    assert "financial_results" in event_type.lower() or "dividend" in event_type.lower()

    facts = extract_disclosure_material_facts(title)
    assert len(facts["monetary_values"]) > 0
    assert any("Rs 10" in v or "10" in v for v in facts["monetary_values"])
    assert len(facts["conditions_and_caveats"]) > 0
    assert any("regulatory" in c.lower() for c in facts["conditions_and_caveats"])


def test_multi_company_nse_disclosure_retrieval(monkeypatch):
    """Priority 1: Multi-company announcement retrieval with date range and coverage tracking."""
    adapter = OfficialQuarterlyDisclosureAdapter()

    def fake_fetch(company):
        if company.symbol == "TCS":
            return (
                {
                    "data": [
                        {
                            "desc": "Financial Results for the quarter ended June 30, 2026 (Consolidated)",
                            "attchmntFile": "https://nsearchives.nseindia.com/corporate/tcs_q1_results.pdf",
                            "an_dt": "15-Jul-2026",
                        },
                        {
                            "desc": "Intimation of Board Meeting for Dividend",
                            "attchmntFile": "https://nsearchives.nseindia.com/corporate/tcs_dividend.pdf",
                            "an_dt": "01-Jul-2026",
                        },
                    ]
                },
                "https://www.nseindia.com/api/corporate-announcements",
                [],
            )
        elif company.symbol == "INFY":
            return (
                {
                    "data": [
                        {
                            "desc": "Financial Results for the quarter ended June 30, 2026",
                            "attchmntFile": "https://nsearchives.nseindia.com/corporate/infy_q1.pdf",
                            "an_dt": "18-Jul-2026",
                        }
                    ]
                },
                "https://www.nseindia.com/api/corporate-announcements",
                [],
            )
        else:
            return None, "https://www.nseindia.com/api/corporate-announcements", ["404 Not Found"]

    monkeypatch.setattr(adapter, "_fetch_nse_announcements", fake_fetch)

    res = adapter.discover_multi(
        companies=["TCS", "INFY", "NONEXISTENT_XYZ"],
        from_date="2026-06-01",
        to_date="2026-08-01",
        extract_content=False,
    )

    assert res["status"] in {"completed", "partial"}
    assert res["total_documents"] == 3
    assert res["coverage"]["TCS"]["status"] == "success"
    assert res["coverage"]["INFY"]["status"] == "success"
    assert res["coverage"]["NONEXISTENT_XYZ"]["status"] in {"failed", "unresolved", "not_found", "unavailable"}
    assert len(res["failures"]) >= 1


def test_inaccessible_or_untrusted_attachment_graceful_handling():
    """Priority 1: Graceful failure when attachment is inaccessible or untrusted; never fabricates content."""
    adapter = OfficialQuarterlyDisclosureAdapter()
    reg = CompanyRegistry()
    tcs = reg.get_by_symbol("TCS")

    # Untrusted domain
    untrusted_doc = {
        "url": "https://untrusted-phishing-domain.com/fake_tcs_report.pdf",
        "title": "Fake report",
    }
    res = adapter.download_and_extract_attachment(untrusted_doc, tcs)
    assert res["status"] == "failed"
    assert "Untrusted" in res["error"]
    assert res["extracted_text"] == ""

    # Inaccessible link
    inaccessible_doc = {
        "url": "https://nsearchives.nseindia.com/corporate/non_existent_file_404.pdf",
        "title": "Missing filing",
    }
    res2 = adapter.download_and_extract_attachment(inaccessible_doc, tcs)
    assert res2["status"] == "failed"
    assert res2["extracted_text"] == ""


def test_screener_peer_intelligence_and_core_vs_secondary_ranking():
    """Priority 2: Screener peers, comparability validation, core vs secondary ranking, custom peer set."""
    reg = CompanyRegistry()
    tcs = reg.get_by_symbol("TCS")
    cs = CompetitorService()

    # Default ranking
    intel = cs.get_peer_intelligence(tcs, max_peers=5)
    assert intel["target"]["symbol"] == "TCS"
    assert len(intel["core_peers"]) > 0
    assert len(intel["secondary_peers"]) > 0
    assert intel["total_peers"] <= 5

    # Check that core peers have rationales
    first_core = intel["core_peers"][0]
    assert first_core["tier"] == "core"
    assert "Core Peer" in first_core["selection_reason"]
    assert first_core["comparability_status"] == "verified_industry_match"

    # User-defined custom peer set
    custom_intel = cs.get_peer_intelligence(tcs, custom_peers=["WIPRO", "HCLTECH"])
    assert custom_intel["selection_method"] == "user_custom"
    symbols = [p["symbol"] for p in custom_intel["all_peers"]]
    assert "WIPRO" in symbols
    assert "HCLTECH" in symbols


def test_macro_intelligence_and_peer_relative_performance():
    """Priority 3: Macroeconomic indicators, peer relative returns, and staleness tracking."""
    macro = MacroMarketIntelligenceService()
    snapshot = macro.get_macro_snapshot()

    assert "nifty_50" in snapshot
    assert "usd_inr" in snapshot
    assert "us_10y_yield" in snapshot
    assert "rbi_repo_rate" in snapshot
    assert snapshot["rbi_repo_rate"]["rate_pct"] == 6.50
    assert snapshot["freshness_status"] in {"Live", "Fallback Benchmark"}

    # Relative performance vs peers
    perf = macro.get_peer_relative_performance("TCS", ["INFY", "WIPRO"])
    assert perf["target"]["symbol"] == "TCS"
    assert "market_benchmark" in perf
    assert "comparison_table" in perf
    assert len(perf["comparison_table"]) >= 1


def test_persistent_conversational_memory_across_sessions():
    """Priority 4: Persistent research sessions, message trajectory, structured findings in DB."""
    mem = MemoryService()

    # Create session
    sess = mem.create_session(
        title="Test TCS Audit Session",
        target_company="TCS",
        session_peers=["INFY", "WIPRO"],
    )
    session_id = sess["session_id"]
    assert session_id is not None
    assert sess["target_company"] == "TCS"

    # Save user and assistant messages
    mem.save_message(session_id, "user", "What is the net profit margin?")
    mem.save_message(
        session_id,
        "assistant",
        "TCS reported a net margin of 19.3%.",
        citations=[{"document_id": "NSE_Q1_TCS", "url": "https://nsearchives.nseindia.com/tcs.pdf"}],
    )

    # Retrieve messages
    msgs = mem.get_messages(session_id)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["citations"][0]["document_id"] == "NSE_Q1_TCS"

    # Save structured findings
    mem.save_findings(
        session_id=session_id,
        company_symbol="TCS",
        findings=["Net margin expanded to 19.3%", "BFSI sector contract renewal of $1.2B"],
        reporting_period="2026-06-30",
    )
    findings = mem.get_findings(session_id)
    assert len(findings) == 2
    assert any("19.3%" in f["finding_text"] for f in findings)

    # Clean up
    delete_research_session(session_id)
    assert get_research_session(session_id) is None


def test_contextual_follow_up_reference_resolution():
    """Priority 4 & 6: Resolves follow-up references ('the second company', 'what changed since our last analysis?')."""
    mem = MemoryService()
    sess = mem.create_session("Test Reference Session", "TCS", ["INFY", "HCLTECH", "WIPRO"])
    session_id = sess["session_id"]

    try:
        # 1. Ordinal reference: "the second company"
        res1 = mem.resolve_contextual_references(
            session_id,
            "What were the operating margins of the second company?",
        )
        assert res1["reference_type"] == "ordinal_peer"
        assert res1["resolved_target"] == "HCLTECH"
        assert "HCLTECH" in res1["resolved_query"]

        # 2. Ordinal reference: "the 1st peer"
        res2 = mem.resolve_contextual_references(
            session_id,
            "Compare valuation for the 1st peer",
        )
        assert res2["reference_type"] == "ordinal_peer"
        assert res2["resolved_target"] == "INFY"

        # 3. Delta query: "what changed since our last analysis?"
        res3 = mem.resolve_contextual_references(
            session_id,
            "What changed since our last analysis?",
        )
        assert res3["is_delta_query"] is True
        assert res3["reference_type"] == "delta_query"
        assert res3["resolved_target"] == "TCS"
    finally:
        delete_research_session(session_id)


def test_conversational_api_workflow(test_client):
    """Priority 4 & 5: REST API endpoints for persistent sessions, peer updates, and research messages."""
    # 1. Create Session
    create_resp = test_client.post(
        "/sessions",
        json={"target_company": "TCS", "title": "API Test Session", "session_peers": ["INFY", "WIPRO"]},
    )
    assert create_resp.status_code == 200
    sess_data = create_resp.json()
    session_id = sess_data["session_id"]
    assert sess_data["target_company"] == "TCS"

    # 2. List Sessions
    list_resp = test_client.get("/sessions")
    assert list_resp.status_code == 200
    all_sessions = list_resp.json()
    assert any(s["session_id"] == session_id for s in all_sessions)

    # 3. Update Peers
    put_resp = test_client.put(
        f"/sessions/{session_id}/peers",
        json={"peers": ["INFY", "HCLTECH", "TECHM"]},
    )
    assert put_resp.status_code == 200
    assert put_resp.json()["session_peers"] == ["INFY", "HCLTECH", "TECHM"]

    # 4. Post Message (Turn 1: Conceptual or financial question)
    msg_resp = test_client.post(
        f"/sessions/{session_id}/messages",
        json={"content": "What is operating profit margin and how is it calculated?"},
    )
    assert msg_resp.status_code == 200
    res_data = msg_resp.json()
    assert "conversational_answer" in res_data
    assert len(res_data["conversational_answer"]) > 10

    # 5. Fetch Session details with history
    detail_resp = test_client.get(f"/sessions/{session_id}")
    assert detail_resp.status_code == 200
    detail_data = detail_resp.json()
    assert len(detail_data["messages"]) >= 2  # user + assistant

    # 6. Macro snapshot endpoint
    macro_resp = test_client.get("/macro/snapshot")
    assert macro_resp.status_code == 200
    assert "nifty_50" in macro_resp.json()

    # 7. Clean up
    del_resp = test_client.delete(f"/sessions/{session_id}")
    assert del_resp.status_code == 200
