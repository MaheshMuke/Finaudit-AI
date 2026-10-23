"""Tests verifying the conversational financial reasoning agent behavior in FinAudit AI.

Covers:
1. Configured LLM is genuinely invoked for answer synthesis.
2. LLM invocation and fallback are reported truthfully.
3. The answer is synthesized from evidence rather than a fixed report template.
4. Missing causal evidence triggers further research or an honest limitation.
5. General conceptual questions are answered directly without unnecessary tool calls.
6. Numerical claims remain consistent with deterministic calculations.
7. Unsupported causal claims are not presented as established facts.
8. Live Gemini provider smoke test (run if credentials available).
"""

import os
from typing import Any, Dict
import pytest

from finaudit.agent_orchestrator import (
    AgentOrchestrator,
    ResearchDecisionController,
    DecisionAction,
    classify_question_intent,
    format_currency_unit,
)
from finaudit.llm_gateway import LLMGateway
def _make_sample_metrics(symbol: str = "NTPC"):
    periods = ["2025-06-30", "2026-03-31", "2026-06-30"]
    period_values = {
        "2025-06-30": {"revenue": 1200.0, "net_income": 110.0},
        "2026-03-31": {"revenue": 1400.0, "net_income": 140.0},
        "2026-06-30": {"revenue": 1500.0, "net_income": 150.0},
    }
    facts = []
    for period in periods:
        label = f"{'Mar' if period[5:7] == '03' else 'Jun'} {period[:4]}"
        val = period_values[period]
        facts.append({
            "symbol": symbol,
            "provider": "Screener",
            "statement": "quarterly_results",
            "metric_key": "revenue",
            "original_metric_label": "Sales",
            "period": label,
            "value": val["revenue"],
            "raw_value": str(val["revenue"] * 1e7),
            "metric_unit": "INR_crore",
            "currency": "INR",
            "reporting_basis": "consolidated",
            "source_url": f"https://www.screener.in/company/{symbol}/consolidated/",
            "retrieved_at": "2026-10-22T00:00:00+00:00",
            "data_quality_warnings": [],
        })
        facts.append({
            "symbol": symbol,
            "provider": "Screener",
            "statement": "quarterly_results",
            "metric_key": "net_income",
            "original_metric_label": "Net Profit",
            "period": label,
            "value": val["net_income"],
            "raw_value": str(val["net_income"] * 1e7),
            "metric_unit": "INR_crore",
            "currency": "INR",
            "reporting_basis": "consolidated",
            "source_url": f"https://www.screener.in/company/{symbol}/consolidated/",
            "retrieved_at": "2026-10-22T00:00:00+00:00",
            "data_quality_warnings": [],
        })
    return {
        "symbol": symbol,
        "statement_basis": "quarterly",
        "reporting_basis": "consolidated",
        "period": "Jun 2026",
        "reporting_period": "Jun 2026",
        "previous_period": "Mar 2026",
        "previous_reporting_period": "Mar 2026",
        "revenue": "1,500.00 Cr",
        "previous_revenue": "1,400.00 Cr",
        "revenue_growth": "+7.1%",
        "revenue_growth_numeric": 7.14,
        "net_income": "150.00 Cr",
        "previous_net_income": "140.00 Cr",
        "net_income_growth": "+7.1%",
        "net_income_growth_numeric": 7.14,
        "operating_margin": "20.0%",
        "source": "Screener.in",
        "financial_source_url": f"https://www.screener.in/company/{symbol}/consolidated/",
        "financial_facts": facts,
        "quarterly_series": {
            "periods": ["Jun 2025", "Mar 2026", "Jun 2026"],
            "revenue": [1200.0 * 1e7, 1400.0 * 1e7, 1500.0 * 1e7],
            "net_income": [110.0 * 1e7, 140.0 * 1e7, 150.0 * 1e7],
        },
    }


class MockConversationalLLM:
    """Mock LLM that returns conversational financial reasoning."""

    def __init__(self, answer_override: str = None, should_fail: bool = False):
        self.invoked = False
        self.call_count = 0
        self.last_prompt = ""
        self.should_fail = should_fail
        self.primary_provider = "gemini"
        self.gemini_model = "gemini-3.5-flash-lite"
        self.last_call_metadata = {}
        self.answer_override = answer_override or (
            "NTPC's profitability improved in the latest quarter ending Jun 2026, "
            "as consolidated revenue expanded 7.1% QoQ to 1,500 INR crore and net income rose "
            "to 150 INR crore. Operating leverage appears to be the primary verified contributor, "
            "though fuel cost inflation remains an unverified risk."
        )

    def check_health(self):
        return {
            "active_provider": self.primary_provider,
            "gemini": {"status": "ok", "model": self.gemini_model},
            "ollama": {"status": "unavailable"},
        }

    def generate_json(self, prompt: str, system_instruction: str = "", **kwargs) -> Dict[str, Any]:
        self.invoked = True
        self.call_count += 1
        self.last_prompt = prompt
        self.last_call_metadata = {
            "provider": self.primary_provider,
            "model": self.gemini_model,
            "fallback_used": False,
        }
        if self.should_fail:
            raise RuntimeError("LLM inference service timeout")

        # Check if this is the conversational synthesis prompt
        required_keys = kwargs.get("required_keys") or set()
        if "conversational_answer" in required_keys or "conversational" in prompt.lower():
            return {
                "conversational_answer": self.answer_override,
                "summary": "Consolidated earnings improved with 7.1% revenue growth.",
                "findings": ["Revenue rose 7.1% QoQ to 1,500 INR crore.", "Net income reached 150 INR crore."],
                "risks": ["Fuel cost volatility is an unverified hypothesis."],
                "supported_explanations": ["Operating expansion drove higher net income."],
                "limitations": ["Plant load factor details remain unverified."],
                "next_questions": ["What was the average tariff realization?"],
            }

        # Fallback interpretation JSON
        return {
            "summary": "Financial performance assessment",
            "findings": ["Metrics collected from verified filings."],
            "risks": ["General industry risks."],
            "next_questions": ["Peer comparisons."],
            "limitations": ["Data limited to reported quarters."],
        }


def _setup_mock_tools(orchestrator: AgentOrchestrator, metrics: Dict[str, Any], news: list = None):
    orchestrator.tools.get_company_profile = lambda s: {
        "symbol": s,
        "company_name": f"{s} Limited",
        "industry": "Power",
    }
    orchestrator.tools.get_financial_metrics = lambda s: metrics
    orchestrator.tools.search_news_feeds = lambda kw, limit=5, **kwargs: news or []
    orchestrator.tools.discover_official_quarterly_disclosures = lambda s, p, b: {
        "status": "not_found",
        "documents": [],
    }


def test_llm_genuinely_invoked_for_conversational_synthesis():
    """Verify configured LLM is genuinely called and synthesized into the answer."""
    mock_llm = MockConversationalLLM()
    orchestrator = AgentOrchestrator(llm=mock_llm)
    metrics = _make_sample_metrics("NTPC")
    _setup_mock_tools(orchestrator, metrics)

    result = orchestrator.investigate_company(
        "NTPC",
        question="Why did NTPC's profitability change in the recent quarter?",
    )

    assert result["success"] is True
    # Verify LLM was actually invoked
    assert mock_llm.invoked is True
    assert mock_llm.call_count >= 1
    # Verify telemetry records truthful invocation
    assert result["llm"]["llm_invoked"] is True
    assert result["llm"]["response_used_llm"] is True
    assert result["llm"]["request_succeeded"] is True
    assert result["llm"]["status"] in ("ok", "available")
    # Verify answer contains the conversational synthesis
    assert "conversational_answer" in result
    assert "NTPC's profitability improved" in result["conversational_answer"]
    assert result["research_report"]["conversational_answer"] == result["conversational_answer"]


def test_truthful_reporting_on_llm_failure():
    """Verify that when the LLM fails, fallback is recorded truthfully."""
    failing_llm = MockConversationalLLM(should_fail=True)
    orchestrator = AgentOrchestrator(llm=failing_llm)
    metrics = _make_sample_metrics("NTPC")
    _setup_mock_tools(orchestrator, metrics)

    result = orchestrator.investigate_company(
        "NTPC",
        question="Why did NTPC's profitability change?",
    )

    assert result["success"] is True
    # Telemetry must be truthful
    assert result["llm"]["llm_invoked"] is True
    assert result["llm"]["response_used_llm"] is False
    assert result["llm"]["request_succeeded"] is False
    assert result["llm"]["fallback_used"] is True
    assert "LLM inference service timeout" in (result["llm"]["llm_fallback_reason"] or "")
    # Fallback answer should still be provided gracefully without crashing
    assert result["conversational_answer"] is not None


def test_unconfigured_llm_reported_truthfully():
    """Verify that when no LLM is configured, it is never claimed that LLM was invoked."""
    orchestrator = AgentOrchestrator()
    orchestrator.llm = None
    metrics = _make_sample_metrics("NTPC")
    _setup_mock_tools(orchestrator, metrics)

    result = orchestrator.investigate_company("NTPC", question="Why did profitability change?")
    assert result["success"] is True
    assert result["llm"]["llm_invoked"] is False
    assert result["llm"]["response_used_llm"] is False
    assert result["llm"]["status"] in ("unconfigured", "not_used")


def test_conceptual_question_direct_reasoning_without_tools():
    """Verify that general conceptual questions bypass unnecessary financial tool calls."""
    mock_llm = MockConversationalLLM(
        answer_override=(
            "Operating margin measures profitability from core business activities before interest and taxes, "
            "whereas net margin reflects final bottom-line profitability after accounting for all expenses, "
            "including debt servicing costs and taxes."
        )
    )
    orchestrator = AgentOrchestrator(llm=mock_llm)

    result = orchestrator.investigate_company(
        "GENERAL",
        question="What is the difference between operating margin and net margin?",
    )

    assert result["success"] is True
    # Tools should NOT be executed for a general conceptual question
    assert result["budget"]["tool_calls"] == 0
    assert result["budget"]["steps_used"] == 0
    # LLM must be invoked and response provided
    assert mock_llm.invoked is True
    assert "operating margin measures profitability" in result["conversational_answer"].lower()
    assert result["investigation_status"] in ("verified", "complete")


def test_missing_causal_evidence_triggers_further_research_or_limitation():
    """Verify that missing causal explanation triggers follow-up investigation or honest limitation."""
    mock_llm = MockConversationalLLM(
        answer_override=(
            "While NTPC's revenue grew 7.1%, specific operational drivers such as coal linkage or tariff revisions "
            "cannot be verified from the current filings."
        )
    )
    orchestrator = AgentOrchestrator(llm=mock_llm)
    metrics = _make_sample_metrics("NTPC")
    news_searched = []

    def mock_news(query, limit=5, **kwargs):
        news_searched.append(query)
        return [{"title": "General NTPC market update", "source": "Reuters", "link": "https://example.com/1", "published": "2026-06-30"}]

    _setup_mock_tools(orchestrator, metrics, news=[{"title": "General NTPC update", "source": "News", "link": "https://example.com", "published": "2026-06-30"}])
    orchestrator.tools.search_news_feeds = mock_news

    result = orchestrator.investigate_company(
        "NTPC",
        question="Why did NTPC's profitability increase?",
    )

    assert result["success"] is True
    # Controller should have investigated news context to find causal drivers
    assert len(news_searched) >= 1
    # Report must note uncertainties/limitations honestly
    report = result["research_report"]
    assert "uncertainties" in report
    assert len(report["uncertainties"]) > 0


def test_contradictory_numerical_claim_generates_validation_warning():
    """Verify that if the LLM makes an ungrounded or contradictory directional claim, validation notes it."""
    # LLM hallucinates that revenue declined when evidence shows it grew 7.1%
    hallucinating_llm = MockConversationalLLM(
        answer_override="NTPC's profitability declined significantly as sales fell sharply and revenue declined."
    )
    orchestrator = AgentOrchestrator(llm=hallucinating_llm)
    metrics = _make_sample_metrics("NTPC")
    metrics["revenue_growth_numeric"] = 7.1
    metrics["net_income_growth_numeric"] = 7.1
    _setup_mock_tools(orchestrator, metrics)

    result = orchestrator.investigate_company("NTPC", question="Why did profitability change?")

    assert result["success"] is True
    # The validation pipeline checks the conversational answer against directional growth
    validation_notes = result["validation_notes"]
    assert any("conflicts" in note.lower() or "direction" in note.lower() for note in validation_notes)


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY environment variable not available for live smoke test",
)
def test_live_gemini_smoke_test():
    """Live smoke test verifying genuine invocation of configured Google Gemini provider."""
    gateway = LLMGateway()
    res = gateway.generate_json(
        prompt="Explain why operating cash flow can diverge from net income in a capital goods company. Be concise.",
        system_instruction="Return JSON with keys 'central_finding' (string) and 'key_drivers' (list of strings).",
        required_keys={"central_finding", "key_drivers"},
    )
    assert isinstance(res, dict)
    assert "central_finding" in res
    assert "key_drivers" in res
    assert len(res["key_drivers"]) >= 1
    print("\n[LIVE GEMINI SMOKE TEST PASSED] Gemini response:", res["central_finding"][:100])


def test_intent_classification_all_categories():
    """Verify that user questions are classified into appropriate distinct research intents."""
    # Descriptive
    assert classify_question_intent("What was BEL's quarterly revenue?", "BEL") == "descriptive"
    assert classify_question_intent("How much net profit did BEL report?", "BEL") == "descriptive"
    
    # Causal
    assert classify_question_intent("Why did BEL's operating profit surge in June 2026?", "BEL") == "causal"
    assert classify_question_intent("Why did margins contract?", "BEL") == "causal"
    
    # Comparative
    assert classify_question_intent("Why is BEL performing better than HAL?", "BEL vs HAL") == "comparative"
    assert classify_question_intent("Compare NTPC and Power Grid returns", "NTPC") == "comparative"
    
    # Diagnostic
    assert classify_question_intent("What is driving the change in operating margin?", "BEL") == "diagnostic"
    assert classify_question_intent("What is driving the expansion in debt?", "BEL") == "diagnostic"
    
    # Predictive / scenario
    assert classify_question_intent("What might happen if raw material prices rise by 15%?", "BEL") == "predictive"
    assert classify_question_intent("What would happen to EBITDA under an inflation scenario?", "BEL") == "predictive"
    
    # Conceptual
    assert classify_question_intent("Explain what working capital cycle means.", "GENERAL") == "conceptual"
    assert classify_question_intent("What is EBITDA?", "GENERAL") == "conceptual"


def test_numeric_calculations_alone_do_not_satisfy_causal_question():
    """Verify that numeric change calculations alone do not mark a causal investigation complete."""
    orchestrator = AgentOrchestrator()
    metrics = _make_sample_metrics("BEL")
    plan = orchestrator.build_investigation_plan("Why did BEL's net profit increase?", "BEL")
    assert plan["intent"] == "causal"

    # Evaluation with numeric change and growth calculated, but without verified drivers
    eval_res = orchestrator.evaluate_evidence(
        metrics,
        profitability_analysis={"revenue_growth_qoq": 7.14, "verified_cause": False, "has_drivers": False},
        causal_question=True,
        plan=plan,
        question="Why did BEL's net profit increase?",
    )
    # Must NOT be marked complete merely because percentage changes or metrics exist
    assert eval_res["sufficient_for_report"] is False
    assert any("causal" in gap.lower() or "driver" in gap.lower() for gap in eval_res["evidence_gaps"])


def test_causal_question_triggers_hypothesis_driven_followup():
    """Verify that a causal question triggers follow-up investigation into driver hypotheses."""
    mock_llm = MockConversationalLLM()
    orchestrator = AgentOrchestrator(llm=mock_llm)
    metrics = _make_sample_metrics("BEL")
    _setup_mock_tools(orchestrator, metrics)
    calls = []

    def mock_variance(s):
        calls.append("variance_breakdown")
        return {
            "symbol": s,
            "revenue_change": 100.0,
            "expense_changes": {"raw_materials": 40.0, "employee_cost": 20.0},
            "operating_bridge": [{"driver": "raw_materials", "impact": -40.0}],
        }

    orchestrator.tools.variance_breakdown = mock_variance

    result = orchestrator.investigate_company("BEL", question="Why did BEL's operating profit surge?")
    assert result["success"] is True
    # The decision controller must have invoked hypothesis-driven follow-up tools beyond initial metrics
    follow_up_tools = {"discover_official_quarterly_disclosures", "search_news_feeds", "variance_breakdown"}
    assert any(tool in result["tools_used"] for tool in follow_up_tools)


def test_relevant_official_disclosures_strengthen_hypothesis():
    """Verify that relevant official disclosures substantiate and update candidate explanations."""
    mock_llm = MockConversationalLLM()
    orchestrator = AgentOrchestrator(llm=mock_llm)
    metrics = _make_sample_metrics("BEL")
    _setup_mock_tools(orchestrator, metrics)

    orchestrator.tools.discover_official_quarterly_disclosures = lambda s, p, b: {
        "status": "found",
        "documents": [
            {
                "title": "Investor Presentation Q1 FY27 - Financial Results and Radar Export Deliveries",
                "source": "NSE",
                "url": "https://nsearchives.nseindia.com/corporate/BEL_Q1.pdf",
            }
        ],
    }

    result = orchestrator.investigate_company("BEL", question="Why did BEL's operating profit surge?")
    assert result["success"] is True
    # Official disclosure should be captured in supported explanations or disclosures
    supported = result["research_report"].get("supported_explanations") or []
    assert any("Radar Export Deliveries" in item or "Investor Presentation" in item for item in supported)


def test_unrelated_news_rejected_by_entity_filter():
    """Verify that news not matching the target company entity is strictly rejected."""
    orchestrator = AgentOrchestrator()
    raw_articles = [
        {"title": "Union Bank Q1 net profit jumps 15%", "source_name": "Reuters", "published_at": "2026-06-30", "summary": "Union Bank of India credit growth.", "link": "https://example.com/ub"},
        {"title": "SK Hynix signs high-bandwidth memory agreement", "source_name": "Bloomberg", "published_at": "2026-06-30", "summary": "SK Hynix mass production.", "link": "https://example.com/sk"},
        {"title": "Mindspace REIT leases 1msf office space in Mumbai", "source_name": "Mint", "published_at": "2026-06-30", "summary": "Mindspace Business Parks REIT lease expansion.", "link": "https://example.com/reit"},
        {"title": "BEL secures Rs 1,150 crore defense radar orders", "source_name": "Economic Times", "published_at": "2026-06-30", "summary": "Bharat Electronics Limited (BEL) fresh defense contracts.", "link": "https://example.com/bel"},
    ]
    orchestrator.tools.news_service.search_news = lambda kw, limit=20: raw_articles

    filtered = orchestrator.tools.search_news_feeds(
        keywords=["BEL", "defense radar"],
        limit=5,
        required_entities=["BEL", "Bharat Electronics"],
    )

    assert len(filtered) == 1
    assert filtered[0]["title"] == "BEL secures Rs 1,150 crore defense radar orders"
    titles = [a["title"] for a in filtered]
    assert not any("Union Bank" in t for t in titles)
    assert not any("SK Hynix" in t for t in titles)
    assert not any("Mindspace" in t for t in titles)


def test_annual_and_quarterly_data_remain_distinguishable_and_clean_units():
    """Verify currency units are formatted cleanly without 'INR INR crore' duplication and periods are preserved."""
    assert format_currency_unit("INR", "INR crore") == "INR crore"
    assert format_currency_unit("INR", "crore") == "INR crore"
    assert format_currency_unit("USD", "million") == "USD million"
    assert format_currency_unit("INR", "Cr") == "INR Cr"

    mock_llm = MockConversationalLLM()
    orchestrator = AgentOrchestrator(llm=mock_llm)
    metrics = _make_sample_metrics("BEL")
    _setup_mock_tools(orchestrator, metrics)

    result = orchestrator.investigate_company("BEL", question="What was BEL's quarterly revenue?")
    # Answer and report must NOT contain duplicated "INR INR crore"
    assert "INR INR crore" not in result["conversational_answer"]
    assert "INR INR crore" not in str(result["research_report"])

    # Financial facts keep quarterly statement and period distinct
    facts = result["financial_facts"]
    for fact in facts:
        assert fact["statement"] == "quarterly_results"
        assert fact["period"] in ("Jun 2025", "Mar 2026", "Jun 2026")


def test_conversational_answer_avoids_fixed_template_headers():
    """Verify that both LLM answers and fallback answers do not use rigid template headings."""
    failing_llm = MockConversationalLLM(should_fail=True)
    orchestrator = AgentOrchestrator(llm=failing_llm)
    metrics = _make_sample_metrics("BEL")
    _setup_mock_tools(orchestrator, metrics)

    result = orchestrator.investigate_company("BEL", question="Why did BEL's operating profit surge?")
    ans = result["conversational_answer"]

    banned_headers = [
        "## What the evidence shows",
        "## Risks and alternative explanations",
        "## What remains uncertain",
        "## Methodology & evidence",
    ]
    for header in banned_headers:
        assert header not in ans
        assert header.lower() not in ans.lower()


def test_missing_evidence_acknowledged_honestly():
    """Verify that missing financial evidence (e.g. FCF, segment disclosures) is acknowledged rather than fabricated."""
    mock_llm = MockConversationalLLM(
        answer_override=(
            "BEL's net income grew 7.1% QoQ, but specific segment breakdowns and export disclosures "
            "are not available in the current quarterly filing. Operating margin expansion cannot be fully "
            "attributed to specific programs without formal management commentary."
        )
    )
    orchestrator = AgentOrchestrator(llm=mock_llm)
    metrics = _make_sample_metrics("BEL")
    metrics["free_cash_flow"] = None
    metrics["free_cash_flow_numeric"] = None
    _setup_mock_tools(orchestrator, metrics)

    result = orchestrator.investigate_company("BEL", question="Why did BEL's free cash flow change?")
    assert result["success"] is True
    assert result["research_report"]["free_cash_flow"] is None or result["research_report"]["free_cash_flow"] == "N/A"
    assert len(result["research_report"]["uncertainties"]) > 0 or len(result["research_report"]["limitations"]) > 0


def test_loop_stops_safely_within_budget():
    """Verify that research loop respects step and tool-call budget limits without infinite loops."""
    mock_llm = MockConversationalLLM()
    orchestrator = AgentOrchestrator(llm=mock_llm)
    orchestrator.max_steps = 3
    orchestrator.max_tool_calls = 4
    metrics = _make_sample_metrics("BEL")
    _setup_mock_tools(orchestrator, metrics)

    result = orchestrator.investigate_company("BEL", question="Why did BEL's margin expand?")
    assert result["success"] is True
    assert result["budget"]["steps_used"] <= 3
    assert result["budget"]["tool_calls"] <= 4
    assert result["budget"]["elapsed_ms"] is not None


def test_extract_issue_keywords():
    """Verify issue keywords are extracted while filtering stop words and company entities."""
    from finaudit.web_research import extract_issue_keywords
    kw = extract_issue_keywords(
        "Why did Bajaj Auto's profit fall despite revenue growth?",
        company_name="Bajaj Auto Limited",
        symbol="BAJAJ-AUTO",
    )
    assert "profit" in [k.lower() for k in kw]
    assert "fall" in [k.lower() for k in kw]
    assert "revenue" in [k.lower() for k in kw]
    assert "growth" in [k.lower() for k in kw]
    assert "bajaj" not in [k.lower() for k in kw]
    assert "why" not in [k.lower() for k in kw]


def test_web_research_service_html_and_table_extraction(monkeypatch):
    """Verify WebResearchService extracts clean text and readable tables from HTML."""
    from finaudit.web_research import WebResearchService

    sample_html = """
    <html>
        <head><title>Bajaj Auto Q2 Financial Results Announcement</title></head>
        <body>
            <header><nav>Home | About</nav></header>
            <main>
                <article>
                    <h1>Bajaj Auto Q2 Profit Falls 31%</h1>
                    <p>Bajaj Auto reported a 31% decline in net profit due to higher input costs and a one-off tax provision of Rs 211 crore.</p>
                    <table>
                        <tr><th>Metric</th><th>Q2 FY25</th><th>Q2 FY24</th></tr>
                        <tr><td>Revenue</td><td>13,127 Cr</td><td>10,777 Cr</td></tr>
                        <tr><td>Net Profit</td><td>1,385 Cr</td><td>2,020 Cr</td></tr>
                    </table>
                    <p>Domestic two-wheeler sales remained robust with revenue expanding 21.8% year-on-year.</p>
                </article>
            </main>
            <footer>Copyright 2026</footer>
        </body>
    </html>
    """

    class MockResponse:
        status_code = 200
        headers = {"Content-Type": "text/html", "Last-Modified": "Wed, 22 Oct 2026 00:00:00 GMT"}
        content = sample_html.encode("utf-8")

    service = WebResearchService()
    monkeypatch.setattr(service.session, "get", lambda url, **kw: MockResponse())

    doc = service.fetch_and_extract("https://example.com/bajaj-q2-results")
    assert doc is not None
    assert doc["title"] == "Bajaj Auto Q2 Financial Results Announcement"
    assert doc["content_type"] == "html"
    assert "higher input costs" in doc["text"]
    assert "Table:" in doc["text"]
    assert "13,127 Cr" in doc["text"]

    excerpt = service.extract_relevant_excerpts(doc["text"], keywords=["profit", "fall", "tax", "costs"])
    assert "higher input costs" in excerpt or "one-off tax provision" in excerpt


def test_web_research_service_pdf_extraction(monkeypatch):
    """Verify WebResearchService parses PDF streams into clean text."""
    import fitz
    from finaudit.web_research import WebResearchService

    pdf_doc = fitz.open()
    page = pdf_doc.new_page()
    page.insert_text((50, 50), "Bajaj Auto Investor Presentation Q2. Net profit fell 31% due to deferred tax provision.")
    pdf_bytes = pdf_doc.write()

    class MockPDFResponse:
        status_code = 200
        headers = {"Content-Type": "application/pdf"}
        content = pdf_bytes

    service = WebResearchService()
    monkeypatch.setattr(service.session, "get", lambda url, **kw: MockPDFResponse())

    doc = service.fetch_and_extract("https://example.com/bajaj-presentation.pdf")
    assert doc is not None
    assert doc["content_type"] == "pdf"
    assert "deferred tax provision" in doc["text"]


def test_causal_question_retrieves_and_cites_document_evidence(monkeypatch):
    """Verify that a causal question inspects retrieved documents and passes citations into the LLM prompt."""
    mock_llm = MockConversationalLLM(
        answer_override=(
            "Bajaj Auto's profit declined despite 21.8% revenue growth primarily due to a one-off "
            "deferred tax provision of Rs 211 crore and raw material cost inflation, as documented in its "
            "quarterly earnings disclosure [Source: https://example.com/bajaj-filing]."
        )
    )
    orchestrator = AgentOrchestrator(llm=mock_llm)
    metrics = _make_sample_metrics("BAJAJ-AUTO")
    # Simulate profit falling while revenue grew
    metrics["revenue_growth"] = "+21.8%"
    metrics["net_income_growth"] = "-31.4%"
    # Provide mock news with a document link
    sample_news = [
        {
            "title": "Bajaj Auto Q2 profit drops 31% on one-off tax impact",
            "source_name": "Financial Express",
            "published_at": "2026-10-18T10:00:00Z",
            "summary": "Bajaj Auto reported lower quarterly profit despite record revenue.",
            "link": "https://example.com/bajaj-filing",
        }
    ]
    _setup_mock_tools(orchestrator, metrics, news=sample_news)

    # Mock fetch_document to return explanatory document text
    orchestrator.tools.fetch_document = lambda url, title=None: {
        "url": url,
        "title": title or "Bajaj Auto Q2 Result Disclosure",
        "content_type": "html",
        "text": (
            "Bajaj Auto Limited reported Q2 revenue growth of 21.8%, driven by domestic motorcycle demand. "
            "However, net profit fell 31.4% due to higher input costs and a one-off deferred tax charge of Rs 211 crore."
        ),
        "publication_date": "2026-10-18",
        "length": 250,
        "status": "success",
    }

    result = orchestrator.investigate_company(
        "BAJAJ-AUTO",
        question="Why did Bajaj Auto's profit fall despite revenue growth?",
    )

    assert result["success"] is True
    # Verify retrieved_evidence is captured
    assert "retrieved_evidence" in result
    assert len(result["retrieved_evidence"]) > 0
    first_ev = result["retrieved_evidence"][0]
    assert "https://example.com/bajaj-filing" == first_ev["url"]
    assert "tax" in first_ev["excerpt"].lower() or "costs" in first_ev["excerpt"].lower()

    # Verify sources list includes reviewed web evidence
    sources = result["sources"]
    assert any(s.get("link") == "https://example.com/bajaj-filing" and s.get("content_retrieved") for s in sources)

    # Verify prompt received the evidence section
    assert "RETRIEVED WEB & DOCUMENT EVIDENCE (WITH CITATIONS):" in mock_llm.last_prompt
    assert "https://example.com/bajaj-filing" in mock_llm.last_prompt

    # Verify conversational answer mentions the cited explanation
    ans = result["conversational_answer"]
    assert "one-off" in ans.lower() or "tax" in ans.lower() or "input" in ans.lower()


def test_blocked_source_gracefully_falls_back():
    """Verify that when a source is blocked (e.g. HTTP 403 or unavailable), the agent handles it gracefully."""
    mock_llm = MockConversationalLLM()
    orchestrator = AgentOrchestrator(llm=mock_llm)
    metrics = _make_sample_metrics("NTPC")
    _setup_mock_tools(orchestrator, metrics)

    # Simulate fetch_document failing for blocked URLs
    orchestrator.tools.fetch_document = lambda url, title=None: None

    result = orchestrator.investigate_company(
        "NTPC",
        question="Why did NTPC's fuel costs rise?",
    )
    # The investigation must still complete successfully and not crash
    assert result["success"] is True
    assert result["conversational_answer"] is not None
    assert len(result["conversational_answer"]) > 20


def test_honest_answer_when_no_source_explains_the_cause():
    """Verify that when no retrieved source explains the operational cause, the agent states it honestly."""
    failing_llm = MockConversationalLLM(should_fail=True)
    orchestrator = AgentOrchestrator(llm=failing_llm)
    metrics = _make_sample_metrics("NTPC")
    _setup_mock_tools(orchestrator, metrics)
    # No news and no document explanations
    orchestrator.tools.search_news_feeds = lambda kw, limit=5, **kwargs: []

    result = orchestrator.investigate_company(
        "NTPC",
        question="Why did NTPC's profit change?",
    )
    assert result["success"] is True
    ans = result["conversational_answer"]
    # Fallback answer explicitly notes that disclosures do not establish a confirmed causal explanation
    assert "causal" in ans.lower() or "unverified" in ans.lower() or "not establish" in ans.lower()


def test_sbin_research_empty_cache_fallback_and_deduplication():
    """Regression test for SBIN Ask Anything failure:
    1. An empty local news cache must not prevent web retrieval (falls back to live web/RSS).
    2. Repeated Screener / aggregator links must collapse into exactly one citation.
    3. Attempted providers and status are exposed truthfully in search diagnostics.
    4. When no operational cause is established, the answer explains what was investigated,
       what was found or unavailable, and what remains unknown without returning an empty 1-sentence response.
    """
    from finaudit.web_research import WebResearchService
    from finaudit.agent_orchestrator import deduplicate_sources

    # 1. Verify WebResearchService search_web does not stop at empty local cache
    web_research = WebResearchService()
    # Mock news_service with empty cache
    class EmptyNewsService:
        def search_news(self, *args, **kwargs):
            return []
    web_research.news_service = EmptyNewsService()

    # Search web for SBIN query with empty local cache
    articles = web_research.search_web("SBIN State Bank of India Q1 results", limit=3)
    assert isinstance(articles, list)
    # Search diagnostics must record providers attempted
    diag = getattr(web_research, "last_search_diagnostics", {})
    assert "providers_attempted" in diag
    assert len(diag["providers_attempted"]) > 0

    # 2. Verify source deduplication collapses repeated Screener URLs
    duplicate_sources = [
        {"source": "Screener", "link": "https://www.screener.in/company/SBIN/consolidated/", "metric": "revenue"},
        {"source": "Screener", "link": "https://www.screener.in/company/SBIN/consolidated/", "metric": "net_income"},
        {"source": "Screener", "link": "https://www.screener.in/company/SBIN/consolidated/", "metric": "operating_profit"},
        {"source": "NSE", "link": "https://nsearchives.nseindia.com/corporate/SBIN_07082026135619_Presentation.pdf", "title": "Investor Presentation", "content_retrieved": True},
    ]
    deduped = deduplicate_sources(duplicate_sources)
    screener_entries = [s for s in deduped if "screener.in" in (s.get("link") or "")]
    assert len(screener_entries) == 1, f"Expected 1 collapsed Screener citation, got {len(screener_entries)}"
    assert any("Presentation.pdf" in (s.get("link") or "") for s in deduped)

    # 3. Verify orchestrator answer when no cause is established
    failing_llm = MockConversationalLLM(should_fail=True)
    orchestrator = AgentOrchestrator(llm=failing_llm)
    metrics = _make_sample_metrics("SBIN")
    _setup_mock_tools(orchestrator, metrics)
    orchestrator.tools.search_news_feeds = lambda kw, limit=5, **kwargs: []

    result = orchestrator.investigate_company(
        "SBIN",
        question="Why did SBIN's profitability change?",
    )
    assert result["success"] is True

    # Citations in report must be deduplicated
    research_report = result.get("research_report", {})
    sources = research_report.get("sources", [])
    screener_citations = [s for s in sources if "screener.in" in (s.get("link") or "")]
    assert len(screener_citations) <= 1, f"Expected <= 1 Screener citation, got {len(screener_citations)}"

    # Search diagnostics must be present
    assert "search_diagnostics" in result or "search_diagnostics" in research_report

    # Final answer must not be the old 1-sentence refusal
    old_broken_answer = "The available evidence does not establish what caused the profitability change."
    answer = result.get("conversational_answer", "")
    assert answer != old_broken_answer, "Must not return the hardcoded 1-sentence answer"
    assert len(answer) > 100, f"Answer should be an informative summary, got: {answer}"
    assert "SBIN" in answer or "State Bank of India" in answer or "Limited" in answer
    assert "unverified" in answer.lower() or "not retrieved" in answer.lower() or "disclosures" in answer.lower()



