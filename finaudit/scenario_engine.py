from typing import Any, Dict, List, Optional
import math
from finaudit.financial_calc import safe_div


class ScenarioEngine:
    """
    Deterministic sensitivity analysis engine calculating impact of:
    - Input cost / commodity price shocks
    - Currency movements (USD/INR)
    - Revenue slowdown / demand shocks
    - Borrowing cost / interest rate spikes
    """

    @classmethod
    def run_sensitivity(
        cls,
        base_revenue: Optional[float],
        base_net_income: Optional[float],
        base_operating_income: Optional[float],
        base_total_debt: Optional[float],
        industry: str,
        revenue_shock_pct: float = -5.0,
        input_cost_shock_pct: float = 5.0,
        rate_hike_bps: float = 50.0,
        pass_through_rate: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Compute Bull, Base, and Bear cases with transparent assumptions.
        """
        missing_inputs = [
            name
            for name, value in (
                ("revenue", base_revenue),
                ("net income", base_net_income),
                ("operating income", base_operating_income),
                ("total debt", base_total_debt),
            )
            if value is None
        ]
        if missing_inputs:
            return {
                "status": "insufficient_data",
                "message": "Scenario inputs are incomplete; missing values are not imputed.",
                "missing_inputs": missing_inputs,
            }
        invalid_inputs = []
        for name, value in (
            ("revenue", base_revenue),
            ("net income", base_net_income),
            ("operating income", base_operating_income),
            ("total debt", base_total_debt),
            ("revenue shock", revenue_shock_pct),
            ("input cost shock", input_cost_shock_pct),
            ("rate hike", rate_hike_bps),
            ("pass-through rate", pass_through_rate),
        ):
            try:
                is_finite = math.isfinite(float(value))
            except (TypeError, ValueError):
                is_finite = False
            if not is_finite:
                invalid_inputs.append(name)
        if invalid_inputs:
            return {
                "status": "insufficient_data",
                "message": "Scenario inputs must be finite numeric values.",
                "invalid_inputs": invalid_inputs,
            }
        if pass_through_rate < 0 or pass_through_rate > 1 or revenue_shock_pct <= -100:
            return {
                "status": "invalid_assumptions",
                "message": "Pass-through must be between 0 and 1, and the revenue shock must be greater than -100%.",
            }

        rev = base_revenue
        pat = base_net_income
        op_inc = base_operating_income
        debt = base_total_debt
        if rev <= 0 or debt < 0:
            return {
                "status": "insufficient_data",
                "message": "Revenue must be positive and total debt cannot be negative.",
                "invalid_inputs": ["revenue" if rev <= 0 else "total debt"],
            }

        # Estimate operating expenses
        op_expenses = max(0.0, rev - op_inc)
        # Assumed raw material / input cost fraction of expenses (typical ~40-60%)
        industry_lower = industry.lower()
        if "financial" in industry_lower or "bank" in industry_lower or "insurance" in industry_lower:
            cost_exposed_fraction = 0.0
        elif "service" not in industry_lower and "tech" not in industry_lower:
            cost_exposed_fraction = 0.45
        else:
            cost_exposed_fraction = 0.15
        exposed_input_cost = op_expenses * cost_exposed_fraction

        # Scenarios
        # 1. Base Case (No new shock)
        base_case = {
            "name": "Base Case (Current Trajectory)",
            "revenue": rev,
            "net_income": pat,
            "net_margin": safe_div(pat, rev) * 100,
            "net_income_impact": 0.0,
        }

        # 2. Bear Case (Input inflation + revenue contraction + rate hike)
        rev_bear = rev * (1 + revenue_shock_pct / 100.0)
        # Unabsorbed cost increase = shock * (1 - pass_through)
        unabsorbed_cost = exposed_input_cost * (input_cost_shock_pct / 100.0) * (1.0 - pass_through_rate)
        # Additional debt servicing interest
        interest_spike = debt * (rate_hike_bps / 10000.0)
        pat_bear = pat + (rev_bear - rev) * 0.25 - unabsorbed_cost - interest_spike

        bear_case = {
            "name": "Bear Case (Cost Inflation & Slowdown)",
            "revenue": rev_bear,
            "net_income": pat_bear,
            "net_margin": safe_div(pat_bear, rev_bear) * 100,
            "net_income_impact": pat_bear - pat,
            "impact_pct": safe_div(pat_bear - pat, abs(pat)) * 100,
        }

        # 3. Bull Case (Operating leverage & lower input inflation)
        rev_bull = rev * 1.08
        pat_bull = pat + (rev_bull - rev) * 0.35 + (exposed_input_cost * 0.03)

        bull_case = {
            "name": "Bull Case (Acceleration & Margin Expansion)",
            "revenue": rev_bull,
            "net_income": pat_bull,
            "net_margin": safe_div(pat_bull, rev_bull) * 100,
            "net_income_impact": pat_bull - pat,
            "impact_pct": safe_div(pat_bull - pat, abs(pat)) * 100,
        }

        return {
            "status": "success",
            "assumptions": {
                "status": "illustrative_assumptions_not_forecasts",
                "revenue_shock_pct": revenue_shock_pct,
                "input_cost_shock_pct": input_cost_shock_pct,
                "rate_hike_bps": rate_hike_bps,
                "pass_through_rate": pass_through_rate,
                "cost_exposed_fraction": cost_exposed_fraction,
                "bear_case_revenue_sensitivity": 0.25,
                "bull_case_revenue_growth_pct": 8.0,
                "bull_case_revenue_sensitivity": 0.35,
                "bull_case_cost_saving_fraction": 0.03,
            },
            "scenarios": [base_case, bear_case, bull_case],
        }
