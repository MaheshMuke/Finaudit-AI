"""Deterministic, source-aware tools used by the research investigator."""

from typing import Any, Dict, List, Optional

import pandas as pd

from finaudit.company_registry import CompanyRecord
from finaudit.data_service import FinancialDataManager
from finaudit.financial_calc import FinancialCalculator, growth_pct


ITEM_ALIASES = {
    "revenue": ["Total Revenue", "Operating Revenue", "Gross Revenue", "Revenue"],
    "net_income": [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income From Continuing And Discontinued Operation",
    ],
}


class DeterministicResearchTools:
    """Financial research tools that never delegate arithmetic to an LLM."""

    def __init__(self, data_manager: Optional[FinancialDataManager] = None):
        self.data_manager = data_manager or FinancialDataManager()

    def _annual_metrics(self, company: CompanyRecord) -> Dict[str, Any]:
        statements = self.data_manager.get_company_data(company)
        financial = FinancialCalculator.calculate_statement_metrics(
            statements,
            "financial" in company.industry.lower() or "bank" in company.industry.lower(),
            prefer_annual=True,
        )
        return financial

    def get_financials(
        self,
        company: CompanyRecord,
        items: Optional[List[str]] = None,
        periods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        calculated = self._annual_metrics(company)
        series = calculated.get("annual_series", {})
        requested = items or ["revenue", "net_income"]
        period_values = series.get("periods", [])
        selected_periods = [
            period for period in period_values if not periods or period in periods
        ]
        rows = []
        for item in requested:
            values = series.get(item, [])
            for period in selected_periods:
                index = period_values.index(period)
                value = values[index] if index < len(values) else None
                rows.append(
                    {
                        "symbol": company.symbol,
                        "company_name": company.company_name,
                        "item": item,
                        "period": period,
                        "value": value / 1e7 if value is not None else None,
                        "reported_value": value,
                        "unit": "INR crore",
                        "currency": calculated.get("metric_metadata", {}).get(
                            "currency", "INR"
                        ),
                        "statement_basis": calculated.get("primary_basis", "annual"),
                        "source": calculated.get("metric_metadata", {}).get(
                            "source", "Local financial dataset"
                        ),
                        "data_quality": "reported" if value is not None else "missing",
                    }
                )
        return {
            "status": "success" if rows else "missing_data",
            "rows": rows,
            "source": calculated.get("metric_metadata", {}).get(
                "source", "Local financial dataset"
            ),
        }

    def calc_change(
        self, company: CompanyRecord, item: str = "net_income"
    ) -> Dict[str, Any]:
        financials = self.get_financials(company, [item])
        rows = [row for row in financials["rows"] if row["value"] is not None]
        if len(rows) < 2:
            return {
                "status": "missing_data",
                "symbol": company.symbol,
                "item": item,
                "source": financials["source"],
            }
        previous, current = rows[-2], rows[-1]
        return {
            "status": "success",
            "symbol": company.symbol,
            "company_name": company.company_name,
            "item": item,
            "previous_period": previous["period"],
            "current_period": current["period"],
            "previous_value": previous["value"],
            "current_value": current["value"],
            "absolute_change": current["value"] - previous["value"],
            "growth_pct": growth_pct(current["value"], previous["value"]),
            "formula": "(current - previous) / abs(previous) * 100",
            "unit": current["unit"],
            "currency": current["currency"],
            "statement_basis": current["statement_basis"],
            "source": current["source"],
        }

    def variance_breakdown(self, company: CompanyRecord) -> Dict[str, Any]:
        calculated = self._annual_metrics(company)
        raw_income = calculated.get("raw_tables", {}).get("annual_income", pd.DataFrame())
        periods = calculated.get("annual_series", {}).get("periods", [])
        if raw_income.empty or len(periods) < 2:
            return {"status": "missing_data", "symbol": company.symbol, "drivers": []}
        previous_period, current_period = periods[-2], periods[-1]
        drivers = []
        for item, values in raw_income.iterrows():
            previous = values.get(previous_period)
            current = values.get(current_period)
            if pd.isna(previous) or pd.isna(current):
                continue
            drivers.append(
                {
                    "item": str(item),
                    "previous_value": float(previous) / 1e7,
                    "current_value": float(current) / 1e7,
                    "absolute_change": float(current - previous) / 1e7,
                    "unit": "INR crore",
                }
            )
        drivers.sort(key=lambda row: abs(row["absolute_change"]), reverse=True)
        return {
            "status": "success" if drivers else "missing_data",
            "symbol": company.symbol,
            "previous_period": previous_period,
            "current_period": current_period,
            "drivers": drivers[:10],
            "source": calculated.get("metric_metadata", {}).get(
                "source", "Local financial dataset"
            ),
        }
