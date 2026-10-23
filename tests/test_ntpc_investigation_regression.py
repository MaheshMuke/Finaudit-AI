"""Regression tests for FinAudit Ask Anything execution issues demonstrated by the NTPC trace.

Tests coverage:
1. LLM configured but not invoked is distinguishable from an LLM-generated answer.
2. Successful LLM invocation is recorded accurately.
3. Missing causal evidence triggers follow-up research or an honest stop.
4. Failed optional annual-report discovery does not automatically invalidate unrelated supported findings.
5. The controller cannot mark a profitability-causation question complete when required driver evidence has not been investigated.
6. Financial units and claim provenance are validated.
"""

from typing import Any, Dict
import pytest

from finaudit.agent_orchestrator import (
    AgentOrchestrator,
    DecisionAction,
    ResearchDecisionController,
)
from finaudit.verification import verify_investigation_output
from finaudit.evidence import EvidenceRecord, evidence_from_metric


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
        "reporting_basis": ["consolidated"],
        "statement_basis": "quarterly",
        "reporting_period": "2026-06-30",
        "previous_reporting_period": "2026-03-31",
        "currency": "INR",
        "unit": "INR crore",
        "source": "Screener",
        "original_source": "Screener",
        "financial_source_url": f"https://www.screener.in/company/{symbol}/consolidated/",
        "financial_retrieved_at": "2026-10-22T00:00:00+00:00",
        "revenue_numeric": 1500.0 * 1e7,
        "net_income_numeric": 150.0 * 1e7,
        "revenue_growth": "+7.1%",
        "net_income_growth": "+7.1%",
        "quarterly_series": {
            "periods": periods,
            "revenue": [val["revenue"] * 1e7 for val in period_values.values()],
            "net_income": [val["net_income"] * 1e7 for val in period_values.values()],
        },
        "financial_facts": facts,
    }


class MockLLMGateway:
    def __init__(self, primary_provider="gemini", gemini_model="gemini-1.5-flash"):
        self.primary_provider = primary_provider
        self.gemini_model = gemini_model
        self.last_call_metadata = {}
        self.call_count = 0

    def check_health(self):
        return {
            "active_provider": self.primary_provider,
            "gemini": {"status": "ok", "model": self.gemini_model},
            "ollama": {"status": "unavailable"},
        }

    def generate_json(self, prompt: str, system_instruction: str = "") -> Dict[str, Any]:
        self.call_count += 1
        self.last_call_metadata = {
            "provider": self.primary_provider,
            "model": self.gemini_model,
            "call_index": self.call_count,
            "fallback_used": False,
        }
        return {
            "summary": "NTPC demonstrated steady revenue growth driven by capacity additions.",
            "findings": ["Revenue expanded 7.1% QoQ.", "Net margin remained stable at 10.0%."],
            "risks": ["Fuel cost variability remains a key driver."],
            "next_questions": ["What was the plant load factor during the quarter?"],
            "limitations": ["Candidate disclosures were not reviewed in full."],
        }


def test_llm_configured_not_invoked_distinguishable_from_llm_generated_answer():
    """Requirement 1 & 4: Distinguish configured-not-invoked from an LLM-generated answer."""
    orchestrator = AgentOrchestrator()
    mock_llm = MockLLMGateway(primary_provider="gemini", gemini_model="gemini-1.5-flash")
    orchestrator.llm = mock_llm

    metrics = _make_sample_metrics("NTPC")
    orchestrator.tools.get_company_profile = lambda s: {"symbol": "NTPC", "company_name": "NTPC Limited", "industry": "Power"}
    orchestrator.tools.get_financial_metrics = lambda s: metrics
    orchestrator.tools.search_news_feeds = lambda kw, limit=5: []
    orchestrator.tools.discover_official_quarterly_disclosures = lambda s, p, b: {"status": "not_found", "documents": []}

    # 1. When LLM is configured, it is genuinely invoked for causal profitability questions
    result_profitability = orchestrator.investigate_company("NTPC", question="Why has NTPC's profitability changed?")

    assert result_profitability["llm_invoked"] is True
    assert result_profitability["response_used_llm"] is True
    assert result_profitability["provider_configured"] is True
    assert result_profitability["provider_available"] is True
    assert result_profitability["llm_provider"] == "gemini"
    assert result_profitability["llm_model"] == "gemini-1.5-flash"
    assert result_profitability["llm_stage"] == "interpretation"
    assert result_profitability["llm_request_status"] == "succeeded"
    assert result_profitability["llm_fallback_reason"] is None
    assert result_profitability["request_succeeded"] is True
    assert mock_llm.call_count >= 1

    # 2. When LLM is unconfigured, deterministic fallback is used and recorded truthfully
    orchestrator.llm = None
    result_unconfigured = orchestrator.investigate_company("NTPC", question="Why has NTPC's profitability changed?")
    assert result_unconfigured["llm_invoked"] is False
    assert result_unconfigured["response_used_llm"] is False
    assert result_unconfigured["request_succeeded"] is False
    assert result_unconfigured["llm_fallback_reason"] == "no_llm_configured"

    # They are strictly distinguishable
    assert result_profitability["llm_invoked"] != result_unconfigured["llm_invoked"]
    assert result_profitability["response_used_llm"] != result_unconfigured["response_used_llm"]


def test_successful_llm_invocation_recorded_accurately():
    """Requirement 1 & 4: Successful LLM invocation records all required trace fields."""
    orchestrator = AgentOrchestrator()
    mock_llm = MockLLMGateway(primary_provider="gemini", gemini_model="gemini-1.5-flash")
    orchestrator.llm = mock_llm

    metrics = _make_sample_metrics("NTPC")
    orchestrator.tools.get_company_profile = lambda s: {"symbol": "NTPC", "company_name": "NTPC Limited", "industry": "Power"}
    orchestrator.tools.get_financial_metrics = lambda s: metrics

    result = orchestrator.investigate_company("NTPC", question="Summarize corporate governance and board oversight.")

    assert result["llm_invoked"] is True
    assert result["llm_provider"] == "gemini"
    assert result["llm_model"] == "gemini-1.5-flash"
    assert result["llm_stage"] == "interpretation"
    assert result["llm_request_status"] == "succeeded"
    assert result["llm_fallback_reason"] is None
    assert result["provider_configured"] is True
    assert result["provider_available"] is True
    assert result["request_succeeded"] is True
    assert result["response_used_llm"] is True
    assert "interpretation" in result
    assert result["interpretation"]["summary"]


def test_missing_causal_evidence_triggers_followup_research_or_honest_stop():
    """Requirement 2 & 4: Missing causal evidence triggers follow-up or honest stop with limitations."""
    orchestrator = AgentOrchestrator()
    metrics = _make_sample_metrics("NTPC")
    orchestrator.tools.get_company_profile = lambda s: {"symbol": "NTPC", "company_name": "NTPC Limited", "industry": "Power"}
    orchestrator.tools.get_financial_metrics = lambda s: metrics
    orchestrator.tools.search_news_feeds = lambda kw, limit=5: []
    orchestrator.tools.discover_official_quarterly_disclosures = lambda s, p, b: {
        "status": "not_found",
        "reason": "No causal disclosures found on exchange feed.",
        "documents": [],
    }

    result = orchestrator.investigate_company("NTPC", question="Why has NTPC's profitability changed?")

    # Because news and official disclosures were empty, causal evidence was missing
    assert result["investigation_status"] == "partial_evidence"
    assert result["stop_reason"] == "evidence_insufficient"
    assert result["evidence_evaluation"]["sufficient_for_report"] is False
    assert any("causal" in gap for gap in result["evidence_evaluation"]["gaps"])

    # Limitations explicitly state unverified causal drivers
    report = result["research_report"]
    assert report["supported_explanations"] == []
    assert len(report["uncertainties"]) > 0


def test_failed_optional_annual_report_discovery_does_not_invalidate_supported_findings():
    """Requirement 2, 3 & 4: Failed optional primary discovery does not invalidate unrelated supported findings."""
    metrics = _make_sample_metrics("NTPC")

    # When official discovery fails:
    official_discovery = {
        "status": "unavailable",
        "reason": "Official server timeout; optional discovery failed.",
        "documents": [],
    }

    # Evaluate evidence with optional official report discovery failing
    evaluation = AgentOrchestrator.evaluate_evidence(
        metrics,
        official_source_discovery=official_discovery,
        plan={"peers": False, "news": False, "scenario": False, "business_risks": False},
        question="What are NTPC's reported quarterly revenue and profit?",
    )

    # Material claims from valid financial facts remain supported
    assert evaluation["material_claims_supported"] is True
    assert evaluation["sufficient_for_report"] is True


def test_controller_cannot_mark_profitability_causation_complete_without_driver_evidence():
    """Requirement 2 & 4: Controller cannot mark profitability-causation question complete without driver evidence."""
    controller = ResearchDecisionController()

    state_without_drivers = {
        "causal_question": True,
        "profitability_question": True,
        "evaluation": {
            "sufficient_for_report": False,
            "gaps": ["no causal explanation is established by the retrieved disclosures"],
            "conflicts": [],
        },
        "profitability_analysis": {
            "supported_explanations": [],  # No confirmed causal driver
        },
        "plan": {"peers": False, "news": False, "commodities": False, "scenario": False},
        "attempted_actions": {"expanded_news_search", "quarterly_disclosures"},
        "news": [],
    }

    decision = controller.select_next_action(state_without_drivers)

    # Controller must NOT finalize
    assert decision.action != DecisionAction.FINALIZE
    assert decision.decision != "finalize"
    assert decision.action == DecisionAction.STOP_WITH_LIMITATIONS
    assert "stopping with explicit limitations" in decision.reason or "No distinct relevant retrieval" in decision.reason


def test_financial_units_and_claim_provenance_are_validated():
    """Requirement 3 & 4: Financial units and claim provenance are validated."""
    # 1. Test evidence_from_metric enforces valid metric_unit and normalization
    evidence = evidence_from_metric(
        "NTPC",
        "revenue",
        1500.0,
        unit="INR crore",
        currency="INR",
        reporting_period="2026-06-30",
        source="Screener",
        statement_basis="quarterly",
        reporting_basis="consolidated",
        extraction_method="normalized provider fact",
        period_end="2026-06-30",
    )
    assert evidence.unit == "INR crore"
    assert evidence.value == 1500.0
    assert evidence.reporting_basis == "consolidated"

    # 2. Test verify_investigation_output detects ungrounded or mismatched claims
    ev_id = evidence.evidence_id
    valid_report = {
        "metrics": {"reporting_period": "2026-06-30", "currency": "INR"},
        "evidence": [evidence.to_dict()],
        "evidence_references": [ev_id],
        "research_report": {
            "claims": [
                {
                    "claim": "Revenue was 1500.0 INR crore",
                    "metric": "revenue",
                    "value": 1500.0,
                    "unit": "INR crore",
                    "period": "2026-06-30",
                    "basis": "consolidated",
                    "supporting_evidence_ids": [ev_id],
                }
            ],
            "sources": [{"kind": "financial_data", "source": "Screener", "link": "https://screener.in"}],
        },
    }
    warnings = verify_investigation_output(valid_report)
    assert len(warnings) == 0

    # 3. Test that a claim with missing evidence id produces verification warnings
    unsupported_report = {
        "metrics": {"reporting_period": "2026-06-30"},
        "evidence": [],
        "evidence_references": [],
        "research_report": {
            "claims": [
                {
                    "claim": "Net profit doubled without provenance",
                    "metric": "net_income",
                    "value": 9999.0,
                    "supporting_evidence_ids": ["missing_evidence_id"],
                }
            ],
            "sources": [],
        },
    }
    unsupported_warnings = verify_investigation_output(unsupported_report)
    assert len(unsupported_warnings) > 0
    assert any("Evidence identifier does not resolve: missing_evidence_id" in w for w in unsupported_warnings)
