import time

import pytest
import pandas as pd
from finaudit.company_registry import CompanyRegistry
from finaudit.news_service import NewsService
from finaudit.commodity_service import CommodityService
from finaudit.scenario_engine import ScenarioEngine
from finaudit.watchlist_service import WatchlistService
from finaudit.sector_intelligence import SectorIntelligence
from finaudit.agent_orchestrator import AgentOrchestrator
from finaudit.competitors import CompetitorService
from finaudit.company_registry import CompanyRecord


def test_sector_intelligence_frameworks():
    banking = SectorIntelligence.get_framework("Financial Services")
    assert "ROE" in banking["key_metrics"]
    assert "Gross Margin" in banking["not_applicable"]

    it = SectorIntelligence.get_framework("Information Technology")
    assert "Revenue Growth (YoY)" in it["key_metrics"]


def test_scenario_sensitivity_engine():
    res = ScenarioEngine.run_sensitivity(
        base_revenue=10000.0,
        base_net_income=1000.0,
        base_operating_income=1500.0,
        base_total_debt=2000.0,
        industry="Automobile and Auto Components",
        input_cost_shock_pct=10.0,
        revenue_shock_pct=-5.0,
    )
    assert res["status"] == "success"
    scenarios = res["scenarios"]
    assert len(scenarios) == 3
    bear = next(s for s in scenarios if "Bear" in s["name"])
    assert bear["revenue"] == 9500.0  # -5%
    assert bear["net_income"] < 1000.0  # lower due to shocks
    assert res["assumptions"]["status"] == "illustrative_assumptions_not_forecasts"


@pytest.mark.parametrize(
    ("missing", "expected_input"),
    [
        ({"base_operating_income": None}, "operating income"),
        ({"base_total_debt": None}, "total debt"),
    ],
)
def test_scenario_engine_does_not_impute_missing_financial_inputs(missing, expected_input):
    inputs = {
        "base_revenue": 10000.0,
        "base_net_income": 1000.0,
        "base_operating_income": 1500.0,
        "base_total_debt": 2000.0,
        "industry": "Industrials",
    }
    inputs.update(missing)

    result = ScenarioEngine.run_sensitivity(**inputs)

    assert result["status"] == "insufficient_data"
    assert result["missing_inputs"] == [expected_input]


@pytest.mark.parametrize(
    ("input_override", "expected_status"),
    [
        ({"base_revenue": float("inf")}, "insufficient_data"),
        ({"pass_through_rate": 1.2}, "invalid_assumptions"),
        ({"revenue_shock_pct": -100.0}, "invalid_assumptions"),
    ],
)
def test_scenario_engine_rejects_nonfinite_and_invalid_assumptions(
    input_override, expected_status
):
    inputs = {
        "base_revenue": 10000.0,
        "base_net_income": 1000.0,
        "base_operating_income": 1500.0,
        "base_total_debt": 2000.0,
        "industry": "Industrials",
    }
    inputs.update(input_override)

    assert ScenarioEngine.run_sensitivity(**inputs)["status"] == expected_status


def test_agent_question_classification_selects_relevant_evidence():
    plan = AgentOrchestrator.classify_question(
        "How could a 15% increase in crude oil prices affect this company?"
    )
    assert plan["commodities"] is True
    assert plan["scenario"] is True
    assert plan["news"] is False

    compare_plan = AgentOrchestrator.classify_question("Compare this company with its peers")
    assert compare_plan["peers"] is True
    assert compare_plan["commodities"] is False

    stock_plan = AgentOrchestrator.classify_question(
        "Why did the share price grow more than its competitor?"
    )
    assert stock_plan["peers"] is True
    assert stock_plan["share_price_comparison"] is True


def test_investigation_plan_is_structured_and_selects_relevant_tools():
    plan = AgentOrchestrator.build_investigation_plan(
        "Why has NTPC's profitability changed compared with peers?"
    )

    assert plan["objective"].startswith("Why has NTPC")
    assert "same-basis peer metrics" in plan["required_evidence"]
    assert "deterministic calculations" in plan["required_evidence"]
    assert "get_peer_comparison" in plan["candidate_tools"]
    assert "calc_change" in plan["candidate_tools"]
    assert plan["completion_criteria"]


def _quarterly_profitability_metrics():
    period_values = {
        "2025-03-31": {"revenue": 1000.0, "net_income": 100.0},
        "2025-06-30": {"revenue": 1200.0, "net_income": 110.0},
        "2026-03-31": {"revenue": 1400.0, "net_income": 140.0},
        "2026-06-30": {"revenue": 1500.0, "net_income": 150.0},
    }
    facts = []
    for period, values in period_values.items():
        month = {"03": "Mar", "06": "Jun"}[period[5:7]]
        label = f"{month} {period[:4]}"
        for metric, value in values.items():
            facts.append({
                "symbol": "NTPC",
                "provider": "screener",
                "statement": "quarterly_results",
                "metric_key": metric,
                "original_metric_label": "Sales" if metric == "revenue" else "Net Profit",
                "period": label,
                "period_type": "quarterly",
                "value": value,
                "raw_value": str(value),
                "metric_unit": "INR_crore",
                "currency": "INR",
                "reporting_basis": "consolidated",
                "source_url": "https://www.screener.in/company/NTPC/consolidated/",
                "retrieved_at": "2026-10-22T00:00:00+00:00",
                "data_quality_warnings": [],
            })
    return {
        "symbol": "NTPC",
        "reporting_basis": ["consolidated"],
        "statement_basis": "quarterly",
        "currency": "INR",
        "unit": "INR crore",
        "source": "Screener",
        "original_source": "Screener",
        "financial_source_url": "https://www.screener.in/company/NTPC/consolidated/",
        "financial_retrieved_at": "2026-10-22T00:00:00+00:00",
        "quarterly_series": {
            "periods": list(period_values),
            "revenue": [value["revenue"] * 1e7 for value in period_values.values()],
            "net_income": [value["net_income"] * 1e7 for value in period_values.values()],
        },
        "financial_facts": facts,
    }


def _stub_official_quarterly_disclosures(orchestrator):
    orchestrator.tools.discover_official_quarterly_disclosures = (
        lambda symbol, period, reporting_basis: {
            "status": "not_found",
            "reason": "No test disclosure candidates.",
            "documents": [],
            "errors": [],
            "limitations": ["Mocked official-source adapter."],
        }
    )


def test_profitability_analysis_aligns_qoq_and_same_quarter_yoy_and_margin():
    analysis = AgentOrchestrator.build_profitability_analysis(
        _quarterly_profitability_metrics()
    )

    qoq = analysis["comparisons"]["qoq"]
    yoy = analysis["comparisons"]["yoy"]
    assert analysis["periods"] == {
        "latest_quarter": "2026-06-30",
        "previous_quarter": "2026-03-31",
        "year_ago_quarter": "2025-06-30",
    }
    assert qoq["revenue_change_pct"] == pytest.approx((1500 - 1400) / 1400 * 100)
    assert qoq["net_income_change_pct"] == pytest.approx((150 - 140) / 140 * 100)
    assert qoq["net_margin_current_pct"] == pytest.approx(10.0)
    assert qoq["net_margin_change_percentage_points"] == pytest.approx(0.0)
    assert yoy["revenue_change_pct"] == pytest.approx(25.0)
    assert yoy["net_income_change_pct"] == pytest.approx((150 - 110) / 110 * 100)
    assert analysis["basis_compatibility"] == "comparable"


def test_profitability_analysis_rejects_mixed_basis_and_zero_denominators():
    metrics = _quarterly_profitability_metrics()
    metrics["reporting_basis"] = ["consolidated", "standalone"]
    mixed = AgentOrchestrator.build_profitability_analysis(metrics)
    assert mixed["basis_compatibility"] == "unknown_or_mixed"
    assert mixed["comparisons"]["qoq"]["available"] is False

    metrics = _quarterly_profitability_metrics()
    metrics["financial_facts"][0]["reporting_basis"] = "standalone"
    source_basis_mismatch = AgentOrchestrator.build_profitability_analysis(metrics)
    assert source_basis_mismatch["basis_compatibility"] == "unknown_or_mixed"
    assert source_basis_mismatch["comparisons"]["qoq"]["available"] is False

    metrics = _quarterly_profitability_metrics()
    for fact in metrics["financial_facts"]:
        if fact["period"] == "Mar 2026":
            fact["value"] = 0
    zero_denominator = AgentOrchestrator.build_profitability_analysis(metrics)
    assert zero_denominator["comparisons"]["qoq"]["revenue_change_pct"] is None
    assert zero_denominator["comparisons"]["qoq"]["net_margin_comparison_pct"] is None
    assert zero_denominator["comparisons"]["qoq"]["net_margin_change_percentage_points"] is None
    assert "zero" in zero_denominator["comparisons"]["qoq"]["revenue_change_reason"]

    metrics = _quarterly_profitability_metrics()
    metrics["quarterly_series"]["periods"].append("TTM")
    metrics["quarterly_series"]["revenue"].append(9999 * 1e7)
    metrics["quarterly_series"]["net_income"].append(999 * 1e7)
    ttm_ignored = AgentOrchestrator.build_profitability_analysis(metrics)
    assert ttm_ignored["periods"]["latest_quarter"] == "2026-06-30"


@pytest.mark.parametrize(
    ("mutate", "expected_reason"),
    [
        (
            lambda facts: facts.append({
                **next(
                    fact for fact in facts
                    if fact["period"] == "Jun 2026" and fact["metric_key"] == "revenue"
                ),
                "provider": "other-provider",
                "value": 9999.0,
            }),
            "Conflicting provider facts",
        ),
        (
            lambda facts: facts.__setitem__(
                next(
                    index for index, fact in enumerate(facts)
                    if fact["period"] == "Jun 2026" and fact["metric_key"] == "revenue"
                ),
                {
                    **next(
                        fact for fact in facts
                        if fact["period"] == "Jun 2026" and fact["metric_key"] == "revenue"
                    ),
                    "metric_unit": "percent",
                },
            ),
            "incompatible units",
        ),
    ],
)
def test_profitability_comparison_rejects_conflicting_or_invalid_source_facts(
    mutate, expected_reason
):
    metrics = _quarterly_profitability_metrics()
    mutate(metrics["financial_facts"])

    result = AgentOrchestrator.build_profitability_analysis(metrics)

    assert result["comparisons"]["qoq"]["available"] is False
    assert expected_reason in result["comparisons"]["qoq"]["reason"]


def test_profitability_claims_link_to_facts_and_causal_evidence_stays_unverified():
    metrics = _quarterly_profitability_metrics()
    analysis = AgentOrchestrator.build_profitability_analysis(
        metrics,
        news=[{
            "title": "Dated market context",
            "summary": "A news summary without direct causal proof.",
            "source": "Example News",
            "published": "2026-07-01",
            "link": "https://example.com/news",
        }],
    )
    report = AgentOrchestrator._build_profitability_report(
        {"company_name": "NTPC Limited", "symbol": "NTPC"},
        metrics,
        analysis,
        [],
    )
    assert report["supported_explanations"] == []
    assert analysis["context_hypotheses"][0]["status"] == "hypothetical"
    assert analysis["unresolved_cause"]
    assert report["claims"]
    assert any(
        source["link"] == "https://www.screener.in/company/NTPC/consolidated/"
        for source in report["sources"]
    )
    assert all(claim["supporting_evidence_ids"] for claim in report["claims"])
    assert all(
        evidence_id in {fact["evidence_id"] for fact in analysis["evidence"]}
        for claim in report["claims"]
        for evidence_id in claim["supporting_evidence_ids"]
    )


def test_profitability_bridge_is_only_returned_when_source_lines_reconcile():
    from finaudit.verification import verify_investigation_output

    metrics = _quarterly_profitability_metrics()
    operating_profit = [200.0, 220.0, 300.0, 320.0]
    other_income = [10.0, 12.0, 15.0, 20.0]
    interest = [20.0, 22.0, 25.0, 26.0]
    depreciation = [30.0, 35.0, 40.0, 44.0]
    pbt = [
        op + other - finance - depr
        for op, other, finance, depr in zip(
            operating_profit, other_income, interest, depreciation
        )
    ]
    metrics["quarterly_series"]["net_income"] = [value * 0.75 * 1e7 for value in pbt]
    periods = metrics["quarterly_series"]["periods"]
    for index, period in enumerate(periods):
        label = f"{'Mar' if period[5:7] == '03' else 'Jun'} {period[:4]}"
        for metric, value, unit in (
            ("operating_profit", operating_profit[index], "INR_crore"),
            ("other_income", other_income[index], "INR_crore"),
            ("interest_expense", interest[index], "INR_crore"),
            ("depreciation", depreciation[index], "INR_crore"),
            ("profit_before_tax", pbt[index], "INR_crore"),
            ("tax_rate_percent", 25.0, "percent"),
            ("net_income", pbt[index] * 0.75, "INR_crore"),
        ):
            if metric == "net_income":
                existing_fact = next(
                    fact for fact in metrics["financial_facts"]
                    if fact["metric_key"] == "net_income" and fact["period"] == label
                )
                existing_fact["value"] = value
                existing_fact["raw_value"] = str(value)
                continue
            metrics["financial_facts"].append({
                "symbol": "NTPC",
                "provider": "screener",
                "statement": "quarterly_results",
                "metric_key": metric,
                "original_metric_label": metric,
                "period": label,
                "value": value,
                "raw_value": str(value),
                "metric_unit": unit,
                "currency": "INR" if unit == "INR_crore" else None,
                "reporting_basis": "consolidated",
                "source_url": "https://www.screener.in/company/NTPC/consolidated/",
                "retrieved_at": "2026-10-22T00:00:00+00:00",
                "data_quality_warnings": [],
            })

    analysis = AgentOrchestrator.build_profitability_analysis(metrics)
    assert analysis["profit_bridges"]["qoq"]["status"] == "reconciled"
    assert analysis["profit_bridges"]["qoq"]["net_income_change"] == pytest.approx(15.0)
    assert analysis["profit_bridges"]["qoq"]["contributions"]["tax_expense"] == pytest.approx(-5.0)
    assert analysis["profit_bridges"]["qoq"]["supporting_evidence_ids"]

    report = AgentOrchestrator._build_profitability_report(
        {"company_name": "NTPC Limited", "symbol": "NTPC"},
        metrics,
        analysis,
        [],
    )
    warnings = verify_investigation_output({
        "metrics": {},
        "evidence": analysis["evidence"],
        "evidence_references": [item["evidence_id"] for item in analysis["evidence"]],
        "research_report": report,
    })
    assert warnings == []

    conflicting_metrics = {
        **metrics,
        "financial_facts": [dict(fact) for fact in metrics["financial_facts"]],
    }
    conflicting_fact = next(
        fact for fact in conflicting_metrics["financial_facts"]
        if fact["metric_key"] == "profit_before_tax" and fact["period"] == "Jun 2026"
    )
    conflicting_fact["value"] += 50
    conflicting = AgentOrchestrator.build_profitability_analysis(conflicting_metrics)
    assert conflicting["profit_bridges"]["qoq"]["status"] == "not_reconciled"


def test_profitability_investigation_returns_partial_evidence_without_llm_or_news(monkeypatch):
    orchestrator = AgentOrchestrator()
    _stub_official_quarterly_disclosures(orchestrator)
    metrics = _quarterly_profitability_metrics()
    orchestrator.tools.get_company_profile = lambda query: {
        "symbol": "NTPC",
        "company_name": "NTPC Limited",
        "industry": "Power",
    }
    orchestrator.tools.get_financial_metrics = lambda symbol: metrics
    orchestrator.tools.search_news_feeds = lambda keywords, limit=5: []

    class NoLLM:
        def generate_json(self, prompt, system_instruction):
            raise AssertionError("Causal profitability reports must use deterministic evidence.")

    orchestrator.llm = NoLLM()
    result = orchestrator.investigate_company(
        "NTPC",
        question="Why has NTPC profitability changed?",
    )

    assert result["investigation_status"] == "partial_evidence"
    assert result["stop_reason"] == "evidence_insufficient"
    assert result["llm"]["status"] == "not_used"
    assert result["research_report"]["what_changed"]
    assert result["research_report"]["supported_explanations"] == []
    assert result["research_report"]["uncertainties"]
    assert any(step["tool"] == "search_news_feeds" and step["status"] == "empty" for step in result["research_trace_detail"])
    assert result["verification"]["status"] == "passed"


def test_profitability_investigation_handles_news_retrieval_failure(monkeypatch):
    orchestrator = AgentOrchestrator()
    _stub_official_quarterly_disclosures(orchestrator)
    orchestrator.tools.get_company_profile = lambda query: {
        "symbol": "NTPC",
        "company_name": "NTPC Limited",
        "industry": "Power",
    }
    orchestrator.tools.get_financial_metrics = lambda symbol: _quarterly_profitability_metrics()

    def unavailable_news(keywords, limit=5):
        raise RuntimeError("news provider unavailable")

    orchestrator.tools.search_news_feeds = unavailable_news

    class NoLLM:
        def generate_json(self, prompt, system_instruction):
            raise AssertionError("Causal profitability reports must use deterministic evidence.")

    orchestrator.llm = NoLLM()
    result = orchestrator.investigate_company(
        "NTPC",
        question="Why has NTPC profitability changed?",
    )
    news_step = next(
        step for step in result["research_trace_detail"]
        if step["tool"] == "search_news_feeds"
    )
    assert news_step["status"] == "failed"
    assert result["news"] == []
    assert result["research_report"]["supported_explanations"] == []
    assert any("do not establish a cause" in item for item in result["research_report"]["uncertainties"])


def test_profitability_investigation_plan_discloses_official_disclosure_gap():
    plan = AgentOrchestrator.build_investigation_plan(
        "Why has NTPC profitability changed?", "NTPC"
    )
    assert plan["reporting_periods"] == [
        "latest quarter", "previous quarter", "same quarter in the previous year"
    ]
    assert "search_news_feeds" in plan["candidate_tools"]
    assert any("does not retrieve or review" in gap for gap in plan["capability_gaps"])
    assert "discover_official_quarterly_disclosures" in plan["candidate_tools"]
    assert any("income-statement line-item movements" in item for item in plan["required_evidence"])


def test_evidence_evaluation_reports_gaps_and_period_conflicts():
    evaluation = AgentOrchestrator.evaluate_evidence(
        {
            "reporting_period": "2026-03-31",
            "previous_reporting_period": "2026-03-31",
            "revenue_numeric": 100.0,
            "retrieval_metadata": {"source_warning": "provider metadata unavailable"},
        },
        comparison={"status": "not_comparable"},
        news=[],
    )

    assert evaluation["sufficient_for_report"] is True
    assert "current and previous reporting periods are identical" in evaluation["conflicts"]
    assert "peer evidence is incomplete or not comparable" in evaluation["gaps"]
    assert "provider metadata unavailable" in evaluation["gaps"]


def test_investigation_budget_and_trace_record_tool_outcomes(monkeypatch):
    orchestrator = AgentOrchestrator(max_steps=1, max_tool_calls=1)
    orchestrator.tools.get_company_profile = lambda query: {
        "symbol": "TEST",
        "company_name": "Test Limited",
        "industry": "General",
    }

    result = orchestrator.investigate_company("TEST", question="What are its financials?")

    assert result["stop_reason"] == "budget_exhausted"
    assert result["budget"]["steps_used"] == 1
    assert result["research_trace_detail"][0]["status"] == "success"
    assert any(
        step["status"] == "skipped_budget"
        for step in result["research_trace_detail"]
    )


def test_investigation_time_budget_stops_follow_up_calls_and_llm(monkeypatch):
    orchestrator = AgentOrchestrator(max_execution_seconds=0.01)

    def slow_profile(query):
        time.sleep(0.02)
        return {"symbol": "TEST", "company_name": "Test Limited", "industry": "General"}

    orchestrator.tools.get_company_profile = slow_profile
    result = orchestrator.investigate_company("TEST", question="What are its financials?")

    assert result["stop_reason"] == "time_budget_exhausted"
    assert result["llm"]["status"] == "skipped_budget"
    assert result["budget"]["elapsed_ms"] >= result["budget"]["max_elapsed_ms"]
    assert any(step["status"] == "skipped_time_budget" for step in result["research_trace_detail"])


def test_investigation_trace_skips_equivalent_tool_calls(monkeypatch):
    orchestrator = AgentOrchestrator()
    profile_calls = []

    def profile(query):
        profile_calls.append(query)
        return {"symbol": "TEST", "company_name": "Test Limited", "industry": "General"}

    orchestrator.tools.get_company_profile = profile
    result = orchestrator.investigate_company("TEST", question="What are its financials?")

    assert profile_calls == ["TEST"]
    assert not any(
        step["status"] == "skipped_duplicate"
        for step in result["research_trace_detail"]
    )


def test_profit_comparison_intent_is_explicit():
    plan = AgentOrchestrator.classify_question(
        "Why is Reliance's profit less than its competitors?"
    )
    assert plan["peers"] is True
    assert plan["profit_comparison"] is True
    assert plan["news"] is False


def test_named_cross_company_mentions_are_resolved_independently():
    registry = CompanyRegistry()
    companies = registry.resolve_mentions(
        "Why has Hindustan Copper grown more than Hindalco Industries?"
    )
    assert [company.symbol for company in companies] == ["HINDCOPPER", "HINDALCO"]


def test_explicit_comparison_summary_uses_deterministic_growth_values():
    summary = AgentOrchestrator._build_explicit_comparison_summary(
        "Why has Hindustan Copper grown more than Hindalco Industries?",
        {
            "status": "comparable",
            "comparison_table": [
                {
                    "symbol": "HINDCOPPER",
                    "company_name": "Hindustan Copper Limited",
                    "period": "2025-03-31",
                    "previous_period": "2024-03-31",
                    "current_revenue": 200.0,
                    "previous_revenue": 100.0,
                    "revenue_growth": 100.0,
                    "source": "fixture",
                    "unit": "INR crore",
                    "currency": "INR",
                    "statement_basis": "annual",
                    "data_quality": "comparable",
                },
                {
                    "symbol": "HINDALCO",
                    "company_name": "Hindalco Industries Limited",
                    "period": "2025-03-31",
                    "previous_period": "2024-03-31",
                    "current_revenue": 150.0,
                    "previous_revenue": 100.0,
                    "revenue_growth": 50.0,
                    "source": "fixture",
                    "unit": "INR crore",
                    "currency": "INR",
                    "statement_basis": "annual",
                    "data_quality": "comparable",
                },
            ],
            "limitations": [],
        },
    )
    assert summary["metric"] == "revenue_growth"
    assert summary["comparison_table"][0]["growth"] == 100.0
    assert summary["comparison_table"][0]["statement_basis"] == "annual"
    assert summary["comparison_table"][0]["currency"] == "INR"


def test_causal_investigation_selects_deterministic_driver_tools(monkeypatch):
    orchestrator = AgentOrchestrator()
    _stub_official_quarterly_disclosures(orchestrator)

    class FakeLLM:
        def generate_json(self, prompt, system_instruction):
            return {
                "summary": "The measured change is available; the cause is not verified.",
                "findings": [],
                "risks": [],
                "next_questions": [],
                "limitations": [],
            }

    monkeypatch.setattr(orchestrator, "llm", FakeLLM())
    result = orchestrator.investigate_company(
        "HINDCOPPER",
        question="Why did Hindustan Copper profit change?",
    )
    assert "search_news_feeds" in result["investigation_plan"]["selected_tools"]
    assert result["profitability_analysis"] is not None
    assert result["research_trace"]


def test_adaptive_retrieval_broadens_news_once_when_causal_evidence_is_missing(monkeypatch):
    orchestrator = AgentOrchestrator()
    _stub_official_quarterly_disclosures(orchestrator)
    news_calls = []
    orchestrator.tools.get_company_profile = lambda query: {
        "symbol": "NTPC",
        "company_name": "NTPC Limited",
        "industry": "Power",
    }
    orchestrator.tools.get_financial_metrics = lambda symbol: _quarterly_profitability_metrics()

    def search_news(keywords, limit=5):
        news_calls.append((keywords, limit))
        return [] if len(news_calls) == 1 else [{
            "title": "NTPC quarterly results",
            "source": "Test feed",
            "published": "2026-07-01",
            "summary": "Test context only.",
            "link": "https://example.test/ntpc-results",
        }]

    orchestrator.tools.search_news_feeds = search_news
    result = orchestrator.investigate_company(
        "NTPC", question="Why has NTPC profitability changed?"
    )

    assert len(news_calls) == 2
    assert news_calls[0] != news_calls[1]
    assert any(
        decision["decision"] == "search_news_feeds"
        and "No dated context" in decision["reason"]
        for decision in result["research_decisions"]
    )
    assert result["news"][0]["link"] == "https://example.test/ntpc-results"
    assert result["stop_reason"] == "evidence_insufficient"


def test_planner_selects_scenario_and_primary_document_tools():
    scenario_plan = AgentOrchestrator.build_investigation_plan(
        "What are NTPC's business risks and downside scenario?"
    )

    assert scenario_plan["question_type"]["business_risks"] is True
    assert scenario_plan["question_type"]["scenario"] is True
    assert "run_scenario_analysis" in scenario_plan["candidate_tools"]
    assert "discover_official_annual_reports" in scenario_plan["candidate_tools"]


def test_news_tool_deduplicates_exact_titles_and_preserves_publication_provenance():
    orchestrator = AgentOrchestrator()
    orchestrator.tools.news_service.search_news = lambda keywords, limit: [
        {
            "title": "NTPC announces quarterly results",
            "source_name": "Publisher A",
            "published_at": "2026-07-01",
            "summary": "Reported result excerpt.",
            "link": "https://publisher-a.test/ntpc",
        },
        {
            "title": "NTPC announces quarterly results",
            "source_name": "Publisher B",
            "published_at": "2026-07-02",
            "summary": "Syndicated duplicate.",
            "link": "https://publisher-b.test/ntpc",
        },
    ]

    articles = orchestrator.tools.search_news_feeds(["NTPC"], limit=5)

    assert len(articles) == 1
    assert articles[0]["publication_date"] == "2026-07-01"
    assert articles[0]["event_date"] is None
    assert articles[0]["excerpt"] == "Reported result excerpt."


def test_risk_and_scenario_plan_executes_bounded_tools_and_labels_document_leads(monkeypatch):
    orchestrator = AgentOrchestrator()
    orchestrator.tools.get_company_profile = lambda query: {
        "symbol": "NTPC",
        "company_name": "NTPC Limited",
        "industry": "Power",
    }
    orchestrator.tools.get_financial_metrics = lambda symbol: _quarterly_profitability_metrics()
    orchestrator.tools.run_scenario_analysis = lambda symbol, metrics=None: {
        "status": "insufficient_data",
        "message": "Scenario inputs are incomplete.",
        "missing_inputs": ["total debt"],
    }
    orchestrator.tools.discover_official_annual_reports = lambda symbol, period, basis: {
        "status": "candidates_found",
        "documents": [{
            "title": "NTPC annual report candidate",
            "url": "https://ntpc.co.in/investors/report.pdf",
            "retrieved_at": "2026-10-01T00:00:00+00:00",
            "content_retrieved": False,
            "statement_basis": basis,
        }],
        "limitations": ["Annual reports are not substitutes for quarterly disclosures."],
    }

    class FakeLLM:
        def generate_json(self, prompt, system_instruction):
            return {
                "summary": "Scenario analysis is unavailable because an input is missing.",
                "findings": [],
                "risks": [],
                "next_questions": [],
                "limitations": ["Annual report link is a candidate; contents were not reviewed."],
            }

    monkeypatch.setattr(orchestrator, "llm", FakeLLM())
    result = orchestrator.investigate_company(
        "NTPC", question="What are NTPC's business risks and downside scenario?"
    )

    assert result["scenario_analysis"]["status"] == "insufficient_data"
    assert result["scenario_analysis"]["missing_inputs"] == ["total debt"]
    assert result["official_source_discovery"]["documents"][0]["content_retrieved"] is False
    assert any(
        source["kind"] == "official_annual_report_candidate"
        and source["content_retrieved"] is False
        for source in result["sources"]
    )
    assert "discover_official_annual_reports" in result["investigation_plan"]["selected_tools"]


def test_comparison_summary_distinguishes_absolute_profit_and_margin():
    peer_research = {
        "status": "comparable",
        "target": {
            "symbol": "TARGET",
            "company_name": "Target",
            "net_income": 100.0,
            "net_margin": 5.0,
            "period": "2025-03-31",
            "source": "fixture",
        },
        "peers": [
            {
                "symbol": "PEER",
                "company_name": "Peer",
                "net_income": 200.0,
                "net_margin": 10.0,
                "period": "2025-03-31",
                "source": "fixture",
            }
        ],
        "rejected_peers": [],
    }
    result = AgentOrchestrator._build_comparison_summary(
        "Why is profit lower than competitors?", peer_research
    )
    assert result["metric"] == "net_income"
    target_row = next(row for row in result["peers_ranked"] if row["is_target"])
    assert target_row["value"] == 100.0
    margin = AgentOrchestrator._build_comparison_summary(
        "Why is profit margin lower than competitors?", peer_research
    )
    assert margin["metric"] == "net_margin"


def test_peer_selection_reason_excludes_target():
    target = CompanyRecord(
        symbol="TARGET",
        company_name="Target",
        industry="Aerospace",
        yahoo_ticker="TARGET.NS",
    )
    peer = CompanyRecord(
        symbol="PEER",
        company_name="Peer",
        industry="Aerospace",
        yahoo_ticker="PEER.NS",
    )
    registry = CompanyRegistry.__new__(CompanyRegistry)
    registry._companies = {"TARGET": target, "PEER": peer}
    registry._name_index = {}
    peers = registry.get_peers(target)
    assert [item.symbol for item in peers] == ["PEER"]
    assert target.symbol not in [item.symbol for item in peers]
    assert "Same classified industry" in registry.peer_selection_reason(target, peer)


def test_peer_ranking_prefers_business_similarity_over_broad_industry():
    target = CompanyRecord(
        symbol="TARGET",
        company_name="Target Pharmaceuticals Ltd.",
        industry="Healthcare",
        yahoo_ticker="TARGET.NS",
    )
    pharma = CompanyRecord(
        symbol="PHARMA",
        company_name="Comparable Pharma Ltd.",
        industry="Healthcare",
        yahoo_ticker="PHARMA.NS",
    )
    chemicals = CompanyRecord(
        symbol="CHEM",
        company_name="Acutaas Chemicals Ltd.",
        industry="Healthcare",
        yahoo_ticker="CHEM.NS",
    )
    registry = CompanyRegistry.__new__(CompanyRegistry)
    registry._companies = {"TARGET": target, "PHARMA": pharma, "CHEM": chemicals}
    registry._name_index = {}
    ranked = registry.rank_peers(
        target,
        evidence={
            "TARGET": {"business_summary": "generic pharmaceutical formulations", "revenue": 1000, "has_financial_data": True},
            "PHARMA": {"business_summary": "pharmaceutical generics and APIs", "revenue": 900, "has_financial_data": True},
            "CHEM": {"business_summary": "industrial chemical manufacturing", "revenue": 950, "has_financial_data": True},
        },
        max_peers=2,
    )
    assert [entry["company"].symbol for entry in ranked] == ["PHARMA", "CHEM"]
    assert ranked[0]["confidence"] == "supported"
    assert "Shared business-profile" in ranked[0]["reason"]


def test_peer_ranking_marks_missing_business_metadata_provisional():
    target = CompanyRecord(
        symbol="TARGET",
        company_name="Target Ltd.",
        industry="Healthcare",
        yahoo_ticker="TARGET.NS",
    )
    peer = CompanyRecord(
        symbol="PEER",
        company_name="Peer Ltd.",
        industry="Healthcare",
        yahoo_ticker="PEER.NS",
    )
    registry = CompanyRegistry.__new__(CompanyRegistry)
    registry._companies = {"TARGET": target, "PEER": peer}
    registry._name_index = {}
    ranked = registry.rank_peers(target, evidence={"TARGET": {}, "PEER": {}}, max_peers=1)
    assert ranked[0]["confidence"] == "provisional"
    assert "metadata unavailable" in ranked[0]["reason"]


def test_comparison_rejects_mismatched_reporting_periods_and_bases():
    target = CompanyRecord(
        symbol="TARGET",
        company_name="Target Ltd.",
        industry="Metals",
        yahoo_ticker="TARGET.NS",
    )
    peer = CompanyRecord(
        symbol="PEER",
        company_name="Peer Ltd.",
        industry="Metals",
        yahoo_ticker="PEER.NS",
    )

    def statements(period):
        return {
            "income_statement_annual": pd.DataFrame([
                {"item": "Total Revenue", "period_end": period, "value": 1000.0},
                {"item": "Net Income", "period_end": period, "value": 100.0},
            ])
        }

    class FakeManager:
        def get_company_data(self, company):
            return statements("2025-03-31" if company.symbol == "TARGET" else "2024-03-31")

        @staticmethod
        def fetch_market_data(company):
            return {}

    service = CompetitorService.__new__(CompetitorService)
    service.data_manager = FakeManager()
    result = service.compare_companies([target, peer])
    assert result["status"] == "partial"
    assert all(
        row["data_quality"] == "incompatible_reporting_period"
        for row in result["comparison_table"]
    )


def test_reliance_uses_curated_integrated_energy_peers():
    registry = CompanyRegistry()
    target = registry.get_by_symbol("RELIANCE")
    peers = registry.get_peers(target, max_peers=4)
    symbols = [peer.symbol for peer in peers]
    assert "RELIANCE" not in symbols
    assert symbols == ["ONGC", "BPCL", "IOC", "GAIL"]
    assert "Curated listed integrated-energy peer" in registry.peer_selection_reason(
        target, peers[0]
    )


def test_watchlist_operations(tmp_path):
    db_file = tmp_path / "test_wl.sqlite3"
    ws = WatchlistService(db_path=db_file)

    ws.add_item("TCS", "Tata Consultancy Services Ltd.")
    items = ws.get_items("default")
    assert len(items) == 1
    assert items[0]["symbol"] == "TCS"

    ws.remove_item("TCS", "default")
    assert len(ws.get_items("default")) == 0


def test_news_service_deduplication(tmp_path, monkeypatch):
    import feedparser

    db_file = tmp_path / "test_news.sqlite3"
    ns = NewsService(db_path=db_file)

    class MockEntry:
        link = "https://example.com/test-article-1"
        title = "RBI Keeps Rates Unchanged"
        summary = "Monetary policy committee decision."
        published = "2026-10-20"

    class MockFeed:
        entries = [MockEntry(), MockEntry()]  # Duplicate entries

    monkeypatch.setattr(feedparser, "parse", lambda *args, **kwargs: MockFeed())
    count = ns.fetch_latest_feeds()
    # Deduplication ensures unique link
    assert count == 1
    articles = ns.search_news(["RBI"])
    assert len(articles) == 1
    assert articles[0]["title"] == "RBI Keeps Rates Unchanged"


def test_commodity_service_indicators(monkeypatch):
    cs = CommodityService()
    # Mock yfinance ticker history
    class MockTicker:
        def history(self, *args, **kwargs):
            return pd.DataFrame({"Close": [80.0, 82.0, 85.0]})

    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", lambda *args, **kwargs: MockTicker())

    indicators = cs.get_indicators(force_refresh=True)
    assert len(indicators) > 0
    crude = next(i for i in indicators if i["name"] == "Brent Crude")
    assert crude["last_price"] == 85.0
    assert crude["change_pct"] is not None
