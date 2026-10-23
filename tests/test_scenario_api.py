from fastapi.testclient import TestClient

from finaudit.api import app


def test_simulate_scenario_applies_borrowing_cost_shock():
    with TestClient(app) as client:
        response = client.post(
            "/scenarios/simulate",
            json={
                "industry": "Automobile and Auto Components",
                "base_revenue": 10000.0,
                "base_net_income": 1000.0,
                "base_operating_income": 1500.0,
                "base_total_debt": 2000.0,
                "input_cost_shock_pct": 10.0,
                "revenue_shock_pct": -5.0,
                "borrowing_cost_shock_bps": 250.0,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["assumptions"]["rate_hike_bps"] == 250.0

    bear_case = next(scenario for scenario in payload["scenarios"] if "Bear" in scenario["name"])
    assert bear_case["net_income"] == 633.75
