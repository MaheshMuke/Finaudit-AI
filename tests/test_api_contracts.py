from fastapi.testclient import TestClient

from finaudit import api


def test_unknown_watchlist_symbol_is_rejected():
    with TestClient(api.app) as client:
        response = client.post("/watchlist", json={"symbol": "NOT_A_REAL_NSE_SYMBOL"})

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_analysis_contract_exposes_current_price_and_metadata(monkeypatch):
    monkeypatch.setattr(
        api.engine,
        "run_full_analysis",
        lambda symbol: {
            "success": True,
            "company": {
                "symbol": symbol,
                "company_name": "Example Ltd.",
                "industry": "Information Technology",
                "currency": "INR",
                "source": "registry",
            },
            "market_data": {
                "current_price": 123.4,
                "source": "Yahoo Finance via yfinance",
                "retrieved_at": "2026-01-01T00:00:00+00:00",
                "status": "online",
            },
            "calculations": {
                "latest_ratios": {},
                "metric_metadata": {
                    "currency": "INR",
                    "display_unit": "INR crore",
                    "reporting_period": "2025-03-31",
                    "statement_basis": "annual",
                },
            },
            "competitors": {
                "peers": [],
                "comparison_table": [],
                "rankings": {},
            },
            "ai_analysis": {},
            "statements": {},
        },
    )

    with TestClient(api.app) as client:
        response = client.get("/analyses/EXAMPLE")

    assert response.status_code == 200
    payload = response.json()
    assert payload["market_data"]["current_price"] == 123.4
    assert payload["data_metadata"]["financial"]["display_unit"] == "INR crore"


def test_portfolio_impact_contract_is_exposed(monkeypatch):
    monkeypatch.setattr(
        api.impact_service,
        "analyze_impact",
        lambda question, symbols: {
            "success": True,
            "analysis": {
                "short_answer": "Mixed exposure.",
                "transmission_mechanism": "Input costs affect margins.",
                "company_impacts": [],
                "indicators_to_monitor": ["Operating margin"],
            },
            "news_evidence": [],
            "financial_evidence": [],
        },
    )
    monkeypatch.setattr(
        api.watchlist_service,
        "get_items",
        lambda list_name="default": [{"symbol": "TCS"}],
    )

    with TestClient(api.app) as client:
        response = client.post("/portfolio-impact", json={"question": "Assess crude oil risk"})

    assert response.status_code == 200
    assert response.json()["analysis"]["short_answer"] == "Mixed exposure."
