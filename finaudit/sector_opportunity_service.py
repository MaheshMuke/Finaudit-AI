from typing import Any, Dict, List, Optional
import pandas as pd
from finaudit.company_registry import get_company_registry
from finaudit.data_service import get_data_manager
from finaudit.financial_calc import FinancialCalculator


class SectorOpportunityService:
    """
    Computes comparative Sector Opportunity Scores based on:
    - Median revenue growth across sector constituents in Nifty 500
    - Median net profit margins
    - Median Return on Equity (ROE)
    - Available balance-sheet health
    """

    def __init__(self):
        self.registry = get_company_registry()
        self.data_manager = get_data_manager()

    def get_sector_radar(self) -> List[Dict[str, Any]]:
        all_companies = self.registry.list_all()

        # Group companies by industry
        by_industry: Dict[str, List[any]] = {}
        for c in all_companies:
            by_industry.setdefault(c.industry, []).append(c)

        radar_results = []
        for ind, comps in by_industry.items():
            if len(comps) < 4:
                continue

            # Prefer constituents with complete metrics, then use registry order as a
            # stable tie-breaker. This avoids treating arbitrary registry position as
            # a proxy for company size or data quality.
            scored = []
            for comp in comps:
                statements = self.data_manager.get_company_data(comp)
                calc = FinancialCalculator.calculate_statement_metrics(
                    statements, prefer_annual=True
                )
                ratios = calc.get("latest_ratios", {})
                quality = sum(
                    ratios.get(key) is not None
                    for key in ("revenue_growth", "net_margin", "roe")
                )
                scored.append((quality, comp))
            sample = [
                comp for quality, comp in sorted(
                    scored, key=lambda item: (-item[0], item[1].symbol)
                )[:6]
                if quality > 0
            ]
            growth_list = []
            margin_list = []
            roe_list = []
            periods = []

            for comp in sample:
                stmts = self.data_manager.get_company_data(comp)
                calc = FinancialCalculator.calculate_statement_metrics(
                    stmts, prefer_annual=True
                )
                ratios = calc.get("latest_ratios", {})
                if ratios.get("period"):
                    periods.append(ratios["period"])
                if ratios.get("revenue_growth") is not None:
                    growth_list.append(ratios["revenue_growth"])
                if ratios.get("net_margin") is not None:
                    margin_list.append(ratios["net_margin"])
                if ratios.get("roe") is not None:
                    roe_list.append(ratios["roe"])

            if not growth_list:
                continue

            med_growth = float(pd.Series(growth_list).median())
            med_margin = float(pd.Series(margin_list).median()) if margin_list else None
            med_roe = float(pd.Series(roe_list).median()) if roe_list else None

            # Transparent deterministic opportunity score (0 to 100)
            score = (
                min(max(med_growth * 1.5, -20), 40)
                + (min(max(med_margin * 1.2, 0), 30) if med_margin is not None else 0)
                + (min(max(med_roe * 1.0, 0), 30) if med_roe is not None else 0)
            )
            score = round(max(0, min(100, score + 20)), 1)

            radar_results.append({
                "industry": ind,
                "company_count": len(comps),
                "sample_size": len(sample),
                "reporting_periods": sorted(set(periods)),
                "opportunity_score": score,
                "median_revenue_growth": round(med_growth, 2),
                "median_net_margin": round(med_margin, 2) if med_margin is not None else None,
                "median_roe": round(med_roe, 2) if med_roe is not None else None,
                "methodology": "Historical medians from up to six annual-reporting constituents ranked by completeness of revenue growth, margin, and ROE; this score is not a stock-return forecast.",
                "representative_companies": [c.company_name for c in sample[:3]],
            })

        radar_results.sort(key=lambda x: x["opportunity_score"], reverse=True)
        return radar_results
