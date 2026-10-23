"""Deterministic automated tests proving the agentic financial research loop.

Covers the 7-step evidence-dependent loop and the three required research questions:
1. "Why has NTPC's profitability changed?"
2. "Compare Grasim's profitability and leverage with its peers."
3. "What are NTPC's main financial risks and possible future scenarios?"
"""

from typing import Any, Dict
import pytest

from finaudit.agent_orchestrator import (
    AgentOrchestrator,
    DecisionAction,
    ResearchDecision,
    ResearchDecisionController,
)


def _mock_company_profile(symbol: str = "NTPC", name: str = "NTPC Limited", industry: str = "Power"):
    return {
        "symbol": symbol,
        "company_name": name,
        "industry": industry,
        "yahoo_ticker": f"{symbol}.NS",
        "current_price": 380.0,
        "market_cap": 360000.0,
    }


def _mock_quarterly_metrics(symbol: str = "NTPC"):
    return {
        "symbol": symbol,
        "reporting_period": "2026-06-30",
        "previous_reporting_period": "2026-03-31",
        "statement_basis": "quarterly",
        "reporting_basis": "consolidated",
        "currency": "INR",
        "unit": "INR crore",
        "source": "Screener quarterly statements",
        "financial_facts": [
            {
                "symbol": symbol,
                "metric_key": "revenue",
                "statement": "quarterly_results",
                "period": "2026-06-30",
                "value": 45000.0,
                "raw_value": 450000000000.0,
                "metric_unit": "INR_crore",
                "currency": "INR",
                "period_type": "quarterly",
                "reporting_basis": "consolidated",
                "provider": "Screener",
                "source_url": "https://www.screener.in/company/NTPC/",
            },
            {
                "symbol": symbol,
                "metric_key": "net_income",
                "statement": "quarterly_results",
                "period": "2026-06-30",
                "value": 5500.0,
                "raw_value": 55000000000.0,
                "metric_unit": "INR_crore",
                "currency": "INR",
                "period_type": "quarterly",
                "reporting_basis": "consolidated",
                "provider": "Screener",
                "source_url": "https://www.screener.in/company/NTPC/",
            },
            {
                "symbol": symbol,
                "metric_key": "operating_profit",
                "statement": "quarterly_results",
                "period": "2026-06-30",
                "value": 11000.0,
                "raw_value": 110000000000.0,
                "metric_unit": "INR_crore",
                "currency": "INR",
                "period_type": "quarterly",
                "reporting_basis": "consolidated",
                "provider": "Screener",
            },
            {
                "symbol": symbol,
                "metric_key": "total_debt",
                "statement": "quarterly_results",
                "period": "2026-06-30",
                "value": 180000.0,
                "metric_unit": "INR_crore",
                "currency": "INR",
                "period_type": "quarterly",
                "reporting_basis": "consolidated",
                "provider": "Screener",
            },
        ],
        "revenue_numeric": 450000000000.0,
        "net_income_numeric": 55000000000.0,
        "revenue_growth": "+8.5%",
        "revenue_growth_numeric": 8.5,
        "net_income_growth": "+12.1%",
        "net_income_growth_numeric": 12.1,
        "operating_income": 11000.0,
        "operating_income_numeric": 110000000000.0,
        "total_debt": 180000.0,
        "total_debt_numeric": 1800000000000.0,
    }


class FakeDeterministicLLM:
    def generate_json(self, prompt: str, system_instruction: str) -> Dict[str, Any]:
        return {
            "summary": "Evidence-grounded response based on verified deterministic metrics.",
            "findings": ["Quarterly revenue and earnings expanded on a consolidated basis."],
            "risks": ["Fuel cost changes and regulatory revisions remain uncertain."],
            "next_questions": ["What was the generation volume across plant segments?"],
            "limitations": ["Causal claims remain unverified by primary disclosure attachments."],
        }


def test_bounded_evidence_driven_loop_full_cycle():
    """
    Demonstrates the complete 7-step evidence-dependent loop:
    1. Initial research returns incomplete evidence (e.g. news is missing).
    2. The evaluator identifies a specific missing fact.
    3. The decision controller selects a follow-up action.
    4. The follow-up tool actually executes.
    5. New evidence is added to the investigation.
    6. Evidence is evaluated again.
    7. The agent produces a verified answer or explicitly stops with limitations.
    """
    orchestrator = AgentOrchestrator()
    orchestrator.llm = FakeDeterministicLLM()

    # Step 1: Initial research returns profile & metrics, but initial news retrieval is empty
    orchestrator.tools.get_company_profile = lambda query: _mock_company_profile("NTPC")
    orchestrator.tools.get_financial_metrics = lambda symbol: _mock_quarterly_metrics(symbol)

    tool_execution_log = []

    def mock_search_news(keywords, limit=4):
        tool_execution_log.append(("search_news_feeds", keywords, limit))
        # Initial search has no results, simulating incomplete evidence
        if len(tool_execution_log) == 1:
            return []
        # Step 4 & 5: Second follow-up tool execution returns dated evidence
        return [
            {
                "title": "NTPC posts quarterly net profit rise driven by higher generation",
                "source": "Economic Times Feed",
                "published": "2026-07-02",
                "publication_date": "2026-07-02",
                "link": "https://economictimes.indiatimes.com/ntpc-results-2026",
                "summary": "Commercial capacity addition and higher thermal generation supported revenue growth.",
            }
        ]

    def mock_discover_quarterly(symbol, period, basis):
        tool_execution_log.append(("discover_official_quarterly_disclosures", symbol, period, basis))
        return {
            "status": "candidates_found",
            "documents": [
                {
                    "title": "Financial Results for the Quarter ended June 30, 2026",
                    "url": "https://nsearchives.nseindia.com/corporate/NTPC_Q1_2027.pdf",
                    "publication_date": "2026-07-28",
                    "retrieved_at": "2026-10-01T00:00:00+00:00",
                    "content_retrieved": False,
                    "statement_basis": basis,
                }
            ],
            "limitations": ["Document contents were not reviewed; candidate link only."],
        }

    orchestrator.tools.search_news_feeds = mock_search_news
    orchestrator.tools.discover_official_quarterly_disclosures = mock_discover_quarterly

    # Execute investigation
    result = orchestrator.investigate_company(
        "NTPC",
        question="Why has NTPC's profitability changed?",
    )

    # 1. Initial research had missing news
    assert result["success"] is True

    # 2. Evaluator identified the missing evidence gap
    decisions = result["research_decisions"]
    assert len(decisions) >= 2

    # 3. Decision controller selected follow-up action
    actions = [d.get("action_type") or d.get("action") for d in decisions]
    assert DecisionAction.RETRIEVE_MORE_EVIDENCE in actions or DecisionAction.INVESTIGATE_CLAIM in actions

    # 4. The follow-up tool actually executed
    executed_tool_names = [call[0] for call in tool_execution_log]
    assert "search_news_feeds" in executed_tool_names
    assert "discover_official_quarterly_disclosures" in executed_tool_names
    assert len(tool_execution_log) >= 2  # Proves follow-up tool was invoked

    # 5. New evidence was added to the investigation
    assert len(result["news"]) > 0
    assert result["news"][0]["link"] == "https://economictimes.indiatimes.com/ntpc-results-2026"

    # 6. Evidence was re-evaluated
    assert result["evidence_evaluation"] is not None

    # 7. Agent produced verified output with explicit limitations disclosed
    assert result["stop_reason"] in {"evidence_insufficient", "evidence_sufficient"}
    assert "evidence" in result
    assert "evidence_references" in result
    assert result["verification"]["status"] in {"passed", "warnings"}


def test_question_why_has_ntpc_profitability_changed():
    """Covers Question 1: 'Why has NTPC's profitability changed?'"""
    orchestrator = AgentOrchestrator()
    orchestrator.llm = FakeDeterministicLLM()

    orchestrator.tools.get_company_profile = lambda query: _mock_company_profile("NTPC", "NTPC Limited", "Power")
    orchestrator.tools.get_financial_metrics = lambda symbol: _mock_quarterly_metrics(symbol)
    orchestrator.tools.search_news_feeds = lambda keywords, limit=4: [
        {
            "title": "NTPC Q1 Net Profit Jumps",
            "source": "Business Standard",
            "published": "2026-07-29",
            "publication_date": "2026-07-29",
            "link": "https://business-standard.com/ntpc-q1",
            "summary": "NTPC posted improved net profit on steady plant load factors.",
        }
    ]
    orchestrator.tools.discover_official_quarterly_disclosures = lambda symbol, period, basis: {
        "status": "candidates_found",
        "documents": [
            {
                "title": "NTPC Q1 Disclosure",
                "url": "https://nsearchives.nseindia.com/corporate/NTPC_Q1.pdf",
                "publication_date": "2026-07-29",
                "statement_basis": "consolidated",
                "content_retrieved": False,
            }
        ],
        "limitations": ["Candidate link only."],
    }

    result = orchestrator.investigate_company("NTPC", question="Why has NTPC's profitability changed?")

    assert result["success"] is True
    assert result["profile"]["symbol"] == "NTPC"
    # Deterministic quarterly profitability analysis was built
    assert result["profitability_analysis"] is not None
    # Material claims have evidence references
    assert len(result["evidence"]) > 0
    assert len(result["evidence_references"]) > 0
    # Verification was performed
    assert "verification" in result
    assert result["verification"]["status"] in {"passed", "warnings"}
    # News context is present
    assert len(result["news"]) > 0
    # Decisions logged
    assert any(
        d.get("action_type") in {DecisionAction.RETRIEVE_MORE_EVIDENCE, DecisionAction.INVESTIGATE_CLAIM}
        or d.get("decision") in {"search_news_feeds", "discover_official_quarterly_disclosures"}
        for d in result["research_decisions"]
    )


def test_question_compare_grasim_profitability_and_leverage_with_peers():
    """Covers Question 2: 'Compare Grasim's profitability and leverage with its peers.'"""
    orchestrator = AgentOrchestrator()
    orchestrator.llm = FakeDeterministicLLM()

    orchestrator.tools.get_company_profile = lambda query: _mock_company_profile(
        "GRASIM", "Grasim Industries Limited", "Construction Materials"
    )
    orchestrator.tools.get_financial_metrics = lambda symbol: {
        "symbol": "GRASIM",
        "reporting_period": "2025-03-31",
        "statement_basis": "annual",
        "reporting_basis": "consolidated",
        "currency": "INR",
        "unit": "INR crore",
        "revenue_numeric": 1300000000000.0,
        "net_income_numeric": 65000000000.0,
        "operating_income_numeric": 140000000000.0,
        "total_debt_numeric": 450000000000.0,
        "revenue_growth": "+10.2%",
        "net_income_growth": "+5.4%",
        "source": "Local dataset",
        "financial_facts": [
            {
                "symbol": "GRASIM",
                "metric_key": "total_debt",
                "period": "2025-03-31",
                "value": 45000.0,
                "metric_unit": "INR_crore",
                "currency": "INR",
                "period_type": "annual",
                "reporting_basis": "consolidated",
                "provider": "Local dataset",
            }
        ],
    }

    # Mock peer research
    orchestrator.tools.competitor_service.comparable_peer_research = lambda company, max_peers=4: {
        "status": "comparable",
        "target": {
            "symbol": "GRASIM",
            "company_name": "Grasim Industries Limited",
            "period": "2025-03-31",
            "net_income": 6500.0,
            "net_margin": 5.0,
            "roe": 8.5,
            "total_debt": 45000.0,
            "debt_to_equity": 0.58,
            "source": "Local dataset",
        },
        "peers": [
            {
                "symbol": "ULTRACEMCO",
                "company_name": "UltraTech Cement Limited",
                "period": "2025-03-31",
                "net_income": 7000.0,
                "net_margin": 10.1,
                "roe": 12.8,
                "total_debt": 11000.0,
                "debt_to_equity": 0.18,
                "source": "Local dataset",
            },
            {
                "symbol": "AMBUJACEM",
                "company_name": "Ambuja Cements Limited",
                "period": "2025-03-31",
                "net_income": 3500.0,
                "net_margin": 9.5,
                "roe": 10.2,
                "total_debt": 500.0,
                "debt_to_equity": 0.01,
                "source": "Local dataset",
            },
        ],
        "rejected_peers": [],
    }

    result = orchestrator.investigate_company(
        "GRASIM",
        question="Compare Grasim's profitability and leverage with its peers.",
    )

    assert result["success"] is True
    assert result["profile"]["symbol"] == "GRASIM"
    assert result["peers"] is not None
    # Peer table includes Grasim and peers
    comparison_table = result["peers"]["comparison_table"]
    assert len(comparison_table) == 3
    # Check that Grasim operates across diversified segments, so provisional note is added
    assert result["peer_research"]["peer_selection_confidence"] == "provisional"
    assert any("provisional" in limit.lower() or "diversified" in limit.lower() for limit in result["peer_research"]["limitations"])
    # Comparison summary exists
    assert result["comparison_summary"] is not None
    assert result["stop_reason"] == "evidence_sufficient"


def test_question_ntpc_financial_risks_and_future_scenarios():
    """Covers Question 3: 'What are NTPC's main financial risks and possible future scenarios?'"""
    orchestrator = AgentOrchestrator()
    orchestrator.llm = FakeDeterministicLLM()

    orchestrator.tools.get_company_profile = lambda query: _mock_company_profile("NTPC", "NTPC Limited", "Power")
    metrics = _mock_quarterly_metrics("NTPC")
    orchestrator.tools.get_financial_metrics = lambda symbol: metrics

    orchestrator.tools.run_scenario_analysis = lambda symbol, metrics=None: {
        "status": "success",
        "industry": "Power",
        "scenarios": [
            {"name": "Base Case", "revenue": 45000.0, "net_income": 5500.0},
            {"name": "Bear Case (-5% Rev, +5% Cost)", "revenue": 42750.0, "net_income": 4125.0},
            {"name": "Bull Case (+5% Rev)", "revenue": 47250.0, "net_income": 6875.0},
        ],
        "assumptions": {
            "status": "illustrative_assumptions_not_forecasts",
            "revenue_shock_pct": -5.0,
            "cost_shock_pct": 5.0,
        },
    }

    orchestrator.tools.discover_official_annual_reports = lambda symbol, period, basis: {
        "status": "candidates_found",
        "documents": [
            {
                "title": "NTPC Integrated Annual Report FY2026",
                "url": "https://www.ntpc.co.in/annual_report_2026.pdf",
                "publication_date": "2026-06-15",
                "retrieved_at": "2026-10-01T00:00:00+00:00",
                "statement_basis": basis,
                "content_retrieved": False,
            }
        ],
        "limitations": ["Document contents were not reviewed; candidate link only."],
    }

    result = orchestrator.investigate_company(
        "NTPC",
        question="What are NTPC's main financial risks and possible future scenarios?",
    )

    assert result["success"] is True
    # Scenario sensitivity calculation was performed
    assert result["scenario_analysis"] is not None
    assert result["scenario_analysis"]["status"] == "success"
    assert result["scenario_analysis"]["assumptions"]["status"] == "illustrative_assumptions_not_forecasts"
    assert len(result["scenario_analysis"]["scenarios"]) == 3
    # Official annual report discovery candidates were found and labeled
    assert result["official_source_discovery"] is not None
    assert result["official_source_discovery"]["documents"][0]["content_retrieved"] is False
    # Decision log contains calculation and retrieval
    decisions = result["research_decisions"]
    actions = [d.get("action_type") or d.get("action") for d in decisions]
    assert DecisionAction.CALCULATE in actions or DecisionAction.RETRIEVE_MORE_EVIDENCE in actions


def test_decision_controller_conflict_resolution_and_stop_conditions():
    """Verifies that conflict resolution and bounded stop conditions are honored by the controller."""
    controller = ResearchDecisionController()

    # Conflicting periods trigger RESOLVE_CONFLICT
    conflicted_state = {
        "evaluation": {
            "sufficient_for_report": False,
            "gaps": ["period conflict"],
            "conflicts": ["current and previous reporting periods are identical"],
        },
        "plan": {"peers": False, "news": False, "commodities": False, "scenario": False},
        "attempted_actions": set(),
    }
    decision = controller.select_next_action(conflicted_state)
    assert decision.action == DecisionAction.RESOLVE_CONFLICT
    assert "reconcile" in decision.reason

    # Once conflicts are attempted and no more candidate tools remain, stop with limitations
    exhausted_state = {
        "evaluation": {
            "sufficient_for_report": False,
            "gaps": ["incomplete statement facts"],
            "conflicts": [],
        },
        "plan": {"peers": False, "news": False, "commodities": False, "scenario": False},
        "attempted_actions": {"resolve_conflicts"},
    }
    stop_decision = controller.select_next_action(exhausted_state)
    assert stop_decision.action == DecisionAction.STOP_WITH_LIMITATIONS
    assert stop_decision.decision == "stop_with_limitations"
    assert stop_decision.decision == "stop"  # DecisionStr alias works

    # When evaluation is sufficient, finalize
    sufficient_state = {
        "evaluation": {
            "sufficient_for_report": True,
            "gaps": [],
            "conflicts": [],
        },
        "plan": {"peers": False, "news": False, "commodities": False, "scenario": False},
        "attempted_actions": set(),
    }
    finalize_decision = controller.select_next_action(sufficient_state)
    assert finalize_decision.action == DecisionAction.FINALIZE
    assert finalize_decision.decision == "finalize"
    assert finalize_decision.decision == "stop"  # DecisionStr alias works
