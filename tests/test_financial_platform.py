import pytest
import pandas as pd
from finaudit.company_registry import CompanyRegistry, CompanyRecord
from finaudit.financial_calc import FinancialCalculator, safe_div, growth_pct, format_currency_inr
from finaudit.data_service import FinancialDataManager
from finaudit.ollama_client import OllamaProvider, parse_ollama_json
from finaudit.llm_gateway import LLMGenerationError, LLMGateway
from finaudit.research_tools import DeterministicResearchTools


def test_company_registry_resolution():
    registry = CompanyRegistry()

    # Exact symbol
    rel = registry.resolve("RELIANCE")
    assert rel is not None
    assert rel.symbol == "RELIANCE"
    assert rel.yahoo_ticker == "RELIANCE.NS"

    # Full name
    tcs = registry.resolve("Tata Consultancy Services Ltd.")
    assert tcs is not None
    assert tcs.symbol == "TCS"

    # Alias / common name
    infy = registry.resolve("infosys")
    assert infy is not None
    assert infy.symbol == "INFY"

    # Tata Motors resolution
    tm = registry.resolve("Tata Motors")
    assert tm is not None
    assert tm.symbol in ["TMPV", "TMCV"]


def test_safe_div_and_growth_calculations():
    # Normal division
    assert safe_div(100, 50) == 2.0
    # Zero division
    assert safe_div(100, 0) is None
    # None handling
    assert safe_div(None, 50) is None
    assert safe_div(100, None) is None

    # Growth calculation
    assert growth_pct(120, 100) == 20.0
    assert growth_pct(80, 100) == -20.0
    assert growth_pct(100, 0) is None
    assert growth_pct(None, 100) is None
    assert growth_pct(-50, -100) == 50.0
    assert growth_pct(50, -100) == 150.0
    assert safe_div(100, float("inf")) is None
    assert growth_pct(float("inf"), 100) is None


def test_format_currency_inr():
    assert format_currency_inr(None) == "Not available"
    assert "Cr" in format_currency_inr(100_000_000)  # 10 Cr
    assert "₹" in format_currency_inr(5000)
    assert format_currency_inr(float("inf")) == "Not available"


def test_legacy_cache_does_not_claim_an_unverified_provider(tmp_path, monkeypatch):
    import finaudit.data_service as data_service

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    pd.DataFrame([
        {
            "symbol": "TEST",
            "company": "Test Limited",
            "statement": "income_statement_annual",
            "item": "Total Revenue",
            "period_end": "2025-03-31",
            "value": 1000.0,
        }
    ]).to_csv(cache_dir / "TEST.csv", index=False)
    monkeypatch.setattr(data_service, "CACHE_DIR", cache_dir)

    class UnavailableScreener:
        def get_cached_facts(self, symbol, fresh_only=False):
            return []

        def get_or_refresh(self, symbol):
            raise RuntimeError("offline")

    manager = FinancialDataManager()
    manager.screener = UnavailableScreener()
    company = CompanyRecord(
        symbol="TEST",
        company_name="Test Limited",
        industry="General",
        yahoo_ticker="TEST.NS",
    )

    statements = manager.get_company_data(company)

    assert statements["_metadata"]["source"] == "Unknown original provider"
    assert statements["_metadata"]["original_source"] is None
    assert statements["_metadata"]["cache_layer"] == "local_normalized_cache"
    assert "provider or source URL" in statements["_metadata"]["source_warning"]
    calculated = FinancialCalculator.calculate_statement_metrics(statements)
    assert calculated["metric_metadata"]["original_source"] is None
    assert calculated["metric_metadata"]["cache_layer"] == "local_normalized_cache"


def test_statement_metrics_calculation():
    # Create mock long table format statements
    income_data = pd.DataFrame([
        {"symbol": "TEST", "item": "Total Revenue", "period_end": "2024-03-31", "value": 1000.0},
        {"symbol": "TEST", "item": "Total Revenue", "period_end": "2025-03-31", "value": 1200.0},
        {"symbol": "TEST", "item": "Net Income", "period_end": "2024-03-31", "value": 100.0},
        {"symbol": "TEST", "item": "Net Income", "period_end": "2025-03-31", "value": 150.0},
        {"symbol": "TEST", "item": "Operating Income", "period_end": "2025-03-31", "value": 200.0},
    ])
    balance_data = pd.DataFrame([
        {"symbol": "TEST", "item": "Total Assets", "period_end": "2025-03-31", "value": 5000.0},
        {"symbol": "TEST", "item": "Stockholders Equity", "period_end": "2025-03-31", "value": 2500.0},
        {"symbol": "TEST", "item": "Total Debt", "period_end": "2025-03-31", "value": 500.0},
    ])
    cashflow_data = pd.DataFrame([
        {"symbol": "TEST", "item": "Operating Cash Flow", "period_end": "2025-03-31", "value": 180.0},
        {"symbol": "TEST", "item": "Free Cash Flow", "period_end": "2025-03-31", "value": 120.0},
    ])

    statements = {
        "income_statement_annual": income_data,
        "balance_sheet_annual": balance_data,
        "cash_flow_annual": cashflow_data,
    }

    metrics = FinancialCalculator.calculate_statement_metrics(statements)
    assert metrics["primary_basis"] == "annual"
    ratios = metrics["latest_ratios"]
    assert ratios["revenue"] == 1200.0
    assert ratios["revenue_growth"] == 20.0
    assert ratios["net_income"] == 150.0
    assert ratios["net_income_growth"] == 50.0
    assert ratios["operating_margin"] == (200.0 / 1200.0) * 100
    assert ratios["debt_to_equity"] == 0.2  # 500 / 2500
    assert ratios["roe"] == (150.0 / 2500.0) * 100
    assert ratios["statement_basis"] == "annual"
    assert ratios["reporting_period"] == "2025-03-31"
    assert ratios["previous_period"] == "2024-03-31"
    assert ratios["metric_provenance"]["net_margin"]["reporting_period"] == "2025-03-31"
    assert ratios["metric_provenance"]["roe"]["formula"] == "net_income / total_equity * 100"
    assert ratios["metric_provenance"]["roe"]["inputs"] == {
        "net_income": 150.0,
        "total_equity": 2500.0,
    }
    assert ratios["metric_provenance"]["debt_to_equity"]["inputs"] == {
        "total_debt": 500.0,
        "total_equity": 2500.0,
    }


def test_quarterly_metrics_are_labelled_separately_from_annual_metrics():
    quarterly = pd.DataFrame([
        {"item": "Total Revenue", "period_end": "2026-06-30", "value": 800.0},
        {"item": "Total Revenue", "period_end": "2026-03-31", "value": 700.0},
        {"item": "Net Income", "period_end": "2026-06-30", "value": 80.0},
        {"item": "Net Income", "period_end": "2026-03-31", "value": 70.0},
    ])
    annual = pd.DataFrame([
        {"item": "Total Revenue", "period_end": "2026-03-31", "value": 3200.0},
        {"item": "Total Revenue", "period_end": "2025-03-31", "value": 3000.0},
        {"item": "Net Income", "period_end": "2026-03-31", "value": 400.0},
        {"item": "Net Income", "period_end": "2025-03-31", "value": 350.0},
    ])
    result = FinancialCalculator.calculate_statement_metrics({
        "income_statement_quarterly": quarterly,
        "income_statement_annual": annual,
    })
    assert result["metric_metadata"]["statement_basis"] == "quarterly"
    assert result["latest_ratios"]["reporting_period"] == "2026-06-30"
    assert result["annual_series"]["periods"][-1] == "2026-03-31"
    assert result["latest_ratios"]["metric_provenance"]["revenue"]["statement_basis"] == "quarterly"


def test_statement_metrics_ignore_non_date_periods_without_mixing_ttm():
    annual = pd.DataFrame([
        {"item": "Total Revenue", "period_end": "2024-03-31", "value": 1000.0},
        {"item": "Total Revenue", "period_end": "2025-03-31", "value": 1200.0},
        {"item": "Total Revenue", "period_end": "TTM", "period_type": "ttm", "value": 1300.0},
        {"item": "Net Income", "period_end": "2024-03-31", "value": 100.0},
        {"item": "Net Income", "period_end": "2025-03-31", "value": 150.0},
        {"item": "Net Income", "period_end": "TTM", "period_type": "ttm", "value": 160.0},
    ])

    result = FinancialCalculator.calculate_statement_metrics({
        "income_statement_annual": annual,
    }, prefer_annual=True)

    assert result["annual_series"]["periods"] == ["2024-03-31", "2025-03-31"]
    assert result["latest_ratios"]["reporting_period"] == "2025-03-31"
    assert result["latest_ratios"]["revenue"] == 1200.0


def test_statement_metrics_ignore_malformed_and_missing_periods():
    annual = pd.DataFrame([
        {"item": "Total Revenue", "period_end": "2024-03-31", "value": 1000.0},
        {"item": "Total Revenue", "period_end": "not-a-date", "value": 900.0},
        {"item": "Total Revenue", "period_end": None, "value": 800.0},
        {"item": "Net Income", "period_end": "2024-03-31", "value": 100.0},
        {"item": "Net Income", "period_end": "not-a-date", "value": 90.0},
    ])

    result = FinancialCalculator.calculate_statement_metrics({
        "income_statement_annual": annual,
    })

    assert result["annual_series"]["periods"] == ["2024-03-31"]
    assert result["latest_ratios"]["reporting_period"] == "2024-03-31"


def test_screener_equity_denominator_includes_capital_and_reserves():
    statements = {
        "income_statement_annual": pd.DataFrame([
            {"item": "Net Income", "period_end": "2025-03-31", "value": 200.0},
        ]),
        "balance_sheet_annual": pd.DataFrame([
            {"item": "Equity Capital", "period_end": "2025-03-31", "value": 100.0},
            {"item": "reserves", "period_end": "2025-03-31", "value": 900.0},
        ]),
        "_metadata": {
            "source": "Screener",
            "original_source": "Screener",
            "reporting_basis": ["consolidated"],
        },
    }

    result = FinancialCalculator.calculate_statement_metrics(statements, prefer_annual=True)

    assert result["latest_ratios"]["total_equity"] == 1000.0
    assert result["latest_ratios"]["roe"] == 20.0
    assert result["metric_metadata"]["reporting_basis"] == ["consolidated"]


def test_invalid_roe_denominator_is_unavailable():
    statements = {
        "income_statement_annual": pd.DataFrame([
            {"item": "Net Income", "period_end": "2025-03-31", "value": 200.0},
        ]),
        "balance_sheet_annual": pd.DataFrame([
            {"item": "Stockholders Equity", "period_end": "2025-03-31", "value": 0.0},
        ]),
    }

    result = FinancialCalculator.calculate_statement_metrics(statements, prefer_annual=True)

    assert result["latest_ratios"]["roe"] is None
    assert result["latest_ratios"]["metric_status"]["roe"]["status"] == "unavailable"


def test_current_ratio_uses_reported_current_assets_and_liabilities_only():
    statements = {
        "income_statement_annual": pd.DataFrame([
            {"item": "Net Income", "period_end": "2025-03-31", "value": 20.0},
        ]),
        "balance_sheet_annual": pd.DataFrame([
            {"item": "Total Current Assets", "period_end": "2025-03-31", "value": 120.0},
            {"item": "Total Current Liabilities", "period_end": "2025-03-31", "value": 60.0},
        ]),
    }

    ratios = FinancialCalculator.calculate_statement_metrics(
        statements, prefer_annual=True
    )["latest_ratios"]

    assert ratios["current_ratio"] == 2.0
    assert ratios["metric_provenance"]["current_ratio"]["inputs"] == {
        "total_current_assets": 120.0,
        "current_liabilities": 60.0,
    }

    statements["balance_sheet_annual"].loc[1, "value"] = 0.0
    invalid = FinancialCalculator.calculate_statement_metrics(
        statements, prefer_annual=True
    )["latest_ratios"]
    assert invalid["current_ratio"] is None
    assert "zero" in invalid["metric_status"]["current_ratio"]["reason"]


def test_deterministic_research_tools_return_source_and_consistent_units():
    income = pd.DataFrame([
        {"symbol": "TEST", "item": "Total Revenue", "period_end": "2024-03-31", "value": 100_000_000.0},
        {"symbol": "TEST", "item": "Total Revenue", "period_end": "2025-03-31", "value": 150_000_000.0},
        {"symbol": "TEST", "item": "Net Income", "period_end": "2024-03-31", "value": 10_000_000.0},
        {"symbol": "TEST", "item": "Net Income", "period_end": "2025-03-31", "value": 25_000_000.0},
    ])

    class FakeManager:
        def get_company_data(self, company):
            return {
                "income_statement_annual": income,
                "balance_sheet_annual": pd.DataFrame(),
                "cash_flow_annual": pd.DataFrame(),
            }

    company = CompanyRecord(
        symbol="TEST",
        company_name="Test Limited",
        industry="Metals",
        yahoo_ticker="TEST.NS",
    )
    tools = DeterministicResearchTools(FakeManager())
    change = tools.calc_change(company, "net_income")
    assert change["growth_pct"] == 150.0
    assert change["unit"] == "INR crore"
    assert change["previous_value"] == 1.0
    assert change["current_value"] == 2.5
    assert change["source"]

    breakdown = tools.variance_breakdown(company)
    assert breakdown["status"] == "success"
    assert breakdown["drivers"][0]["unit"] == "INR crore"


def test_ollama_mocked_analysis(monkeypatch):
    import requests

    class MockOllamaResponse:
        status_code = 200

        def json(self):
            return {
                "message": {
                    "content": '{"business_model": {"summary": "Leading conglomerate"}, "share_price_drivers": {"positive_catalysts": ["Strong 5G growth"]}}'
                }
            }

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: MockOllamaResponse())

    provider = OllamaProvider(base_url="http://localhost:11434", model="qwen3:8b")
    result = provider.analyze_company({"company": {"name": "Test Co"}})
    assert "business_model" in result
    assert result["business_model"]["summary"] == "Leading conglomerate"
    assert "share_price_drivers" in result


def test_ollama_parser_accepts_fenced_json_with_trailing_text():
    parsed = parse_ollama_json('```json\n{"summary": "Grounded"}\n```\nDone.')
    assert parsed == {"summary": "Grounded"}


def test_ollama_parser_accepts_control_characters_in_model_strings():
    parsed = parse_ollama_json('{"summary": "Grounded\nwith detail"}')
    assert parsed["summary"] == "Grounded\nwith detail"


def test_llm_gateway_classifies_timeout_without_retry(monkeypatch):
    import requests

    monkeypatch.setenv("LLM_PRIMARY", "ollama")
    calls = []

    def timeout(*args, **kwargs):
        calls.append(kwargs["timeout"])
        raise requests.exceptions.Timeout("slow model")

    monkeypatch.setattr(requests, "post", timeout)
    gateway = LLMGateway()

    with pytest.raises(LLMGenerationError) as exc_info:
        gateway.generate_json("short prompt", "return JSON")

    assert exc_info.value.kind == "timeout"
    assert len(calls) == 1


def test_llm_gateway_accepts_fenced_structured_response(monkeypatch):
    import requests

    monkeypatch.setenv("LLM_PRIMARY", "ollama")
    class MockResponse:
        status_code = 200
        text = ""

        def json(self):
            return {
                "message": {
                    "content": (
                        "```json\n"
                        '{"summary":"Grounded","findings":[],"risks":[],'
                        '"next_questions":[],"limitations":[]}\n'
                        "```\n"
                    )
                }
            }

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: MockResponse())
    result = LLMGateway().generate_json("prompt", "return JSON")
    assert result["summary"] == "Grounded"


def test_llm_gateway_rejects_wrong_structured_schema(monkeypatch):
    import requests

    monkeypatch.setenv("LLM_PRIMARY", "ollama")
    class MockResponse:
        status_code = 200
        text = ""

        def json(self):
            return {"message": {"content": '{"summary":"Only one field"}'}}

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: MockResponse())
    with pytest.raises(LLMGenerationError) as exc_info:
        LLMGateway().generate_json("prompt", "return JSON")
    assert exc_info.value.kind == "malformed_response"


def test_llm_gateway_uses_gemini_json_api(monkeypatch):
    import requests

    monkeypatch.setenv("LLM_PRIMARY", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    class MockResponse:
        status_code = 200
        text = ""

        def json(self):
            return {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": (
                                '{"summary":"Grounded","findings":[],"risks":[],'
                                '"next_questions":[],"limitations":[]}'
                            )
                        }]
                    }
                }]
            }

    calls = []
    monkeypatch.setattr(
        requests,
        "post",
        lambda url, **kwargs: (calls.append((url, kwargs)) or MockResponse()),
    )
    result = LLMGateway().generate_json("prompt", "return JSON")
    assert result["summary"] == "Grounded"
    assert calls[0][0].endswith("/gemini-2.5-flash-lite:generateContent")
    assert calls[0][1]["params"]["key"] == "test-gemini-key"
    assert calls[0][1]["json"]["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_model_env_quotes_are_removed(monkeypatch):
    from finaudit.llm import GeminiProvider

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", ' "gemini-2.5-flash-lite" ')
    assert GeminiProvider().model_name == "gemini-2.5-flash-lite"


def test_agent_interpretation_flags_unsupported_growth_claims():
    from finaudit.agent_orchestrator import AgentOrchestrator

    warnings = AgentOrchestrator._validate_interpretation(
        {"summary": "Revenue increased materially.", "findings": []},
        {"revenue_growth": None, "net_income_growth": None},
    )

    assert warnings
    assert "comparable revenue periods" in warnings[0]
