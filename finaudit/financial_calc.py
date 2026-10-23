from typing import Any, Dict, List, Optional
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None:
        return None
    if den == 0 or not np.isfinite(den) or not np.isfinite(num):
        return None
    return float(num / den)


def growth_pct(curr: Optional[float], prev: Optional[float]) -> Optional[float]:
    if curr is None or prev is None:
        return None
    if prev == 0 or not np.isfinite(prev) or not np.isfinite(curr):
        return None
    return float(((curr - prev) / abs(prev)) * 100)


def format_currency_inr(val: Optional[float]) -> str:
    """Format large numbers into Cr (Crores) or Lakhs."""
    if val is None or not np.isfinite(val):
        return "Not available"
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e7:
        return f"{sign}₹{abs_val / 1e7:,.2f} Cr"
    elif abs_val >= 1e5:
        return f"{sign}₹{abs_val / 1e5:,.2f} Lakh"
    return f"{sign}₹{abs_val:,.2f}"


def format_pct(val: Optional[float]) -> str:
    if val is None or not np.isfinite(val):
        return "Not available"
    return f"{val:+.2f}%" if val != 0 else "0.00%"


def format_ratio(val: Optional[float]) -> str:
    if val is None or not np.isfinite(val):
        return "Not available"
    return f"{val:.2f}x"


class FinancialCalculator:
    """
    Deterministic calculation engine for statements loaded from yfinance long-table format.
    Handles income statements, balance sheets, and cash flows.
    """

    @classmethod
    def calculate_statement_metrics(
        cls,
        statements: Dict[str, pd.DataFrame],
        is_bank_or_financial: bool = False,
        prefer_annual: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract multi-period series and latest ratios.
        Supports automatic fallback from quarterly to annual data.
        """
        inc_q = statements.get("income_statement_quarterly", pd.DataFrame())
        inc_a = statements.get("income_statement_annual", pd.DataFrame())
        bs_q = statements.get("balance_sheet_quarterly", pd.DataFrame())
        bs_a = statements.get("balance_sheet_annual", pd.DataFrame())
        cf_q = statements.get("cash_flow_quarterly", pd.DataFrame())
        cf_a = statements.get("cash_flow_annual", pd.DataFrame())

        # Determine primary basis
        has_quarterly = not inc_q.empty and len(inc_q["period_end"].unique()) >= 1
        primary_basis = "quarterly" if has_quarterly else "annual"

        # Pivot long table: periods as columns, items as index
        def pivot_statement(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty or "item" not in df.columns or "period_end" not in df.columns:
                return pd.DataFrame()
            # Sort periods descending
            df_clean = df.dropna(subset=["value"]).copy()
            raw_periods = df_clean["period_end"].astype("string").str.strip()
            parsed_periods = pd.to_datetime(raw_periods, errors="coerce")
            invalid_periods = raw_periods.notna() & parsed_periods.isna()
            if invalid_periods.any():
                logger.warning(
                    "Ignoring non-date financial periods in chronological calculation: %s",
                    sorted(set(raw_periods[invalid_periods].tolist())),
                )
            df_clean = df_clean.loc[parsed_periods.notna()].copy()
            if df_clean.empty:
                return pd.DataFrame()
            df_clean["period_end"] = parsed_periods.loc[df_clean.index].dt.strftime("%Y-%m-%d")
            pivoted = df_clean.pivot_table(
                index="item", columns="period_end", values="value", aggfunc="last"
            )
            # Reorder columns chronologically ascending (older -> newer)
            sorted_cols = sorted(pivoted.columns)
            return pivoted[sorted_cols]

        annual_income = pivot_statement(inc_a)
        quarterly_income = pivot_statement(inc_q)
        annual_balance = pivot_statement(bs_a)
        quarterly_balance = pivot_statement(bs_q)
        annual_cashflow = pivot_statement(cf_a)
        quarterly_cashflow = pivot_statement(cf_q)

        # Work on annual series for multi-year trends
        annual_summary = cls._extract_series(
            annual_income, annual_balance, annual_cashflow, is_bank_or_financial
        )
        quarterly_summary = cls._extract_series(
            quarterly_income, quarterly_balance, quarterly_cashflow, is_bank_or_financial
        )
        annual_summary["period_type"] = "annual"
        quarterly_summary["period_type"] = "quarterly"

        # Primary latest ratios (use latest available period)
        latest_series = (
            annual_summary
            if prefer_annual and annual_summary.get("periods")
            else quarterly_summary
            if has_quarterly and quarterly_summary.get("periods")
            else annual_summary
        )
        latest_ratios = cls._compute_latest_ratios(latest_series, is_bank_or_financial)
        selected_basis = "annual" if latest_series is annual_summary else primary_basis
        latest_ratios["statement_basis"] = selected_basis
        latest_ratios["reporting_period"] = latest_ratios.get("period")
        latest_ratios["previous_period"] = latest_ratios.get("previous_period")
        source_metadata = statements.get("_metadata", {})
        if source_metadata.get("screener_ratio_facts"):
            latest_ratios["reported_screener_ratios"] = source_metadata[
                "screener_ratio_facts"
            ]
        metric_context = {
            "reporting_period": latest_ratios.get("period"),
            "previous_period": latest_ratios.get("previous_period"),
            "statement_basis": selected_basis,
            "reporting_basis": source_metadata.get("reporting_basis"),
            "currency": "INR",
            "raw_value_unit": "INR",
            "display_unit": "INR crore",
            "unit": "INR",
            "source": source_metadata.get("source", "Yahoo Finance via yfinance"),
        }
        formula_provenance = {
            "net_margin": {
                "formula": "net_income / revenue * 100",
                "inputs": {
                    "net_income": latest_ratios.get("net_income"),
                    "revenue": latest_ratios.get("revenue"),
                },
            },
            "roe": {
                "formula": "net_income / total_equity * 100",
                "inputs": {
                    "net_income": latest_ratios.get("net_income"),
                    "total_equity": latest_ratios.get("total_equity"),
                },
            },
            "debt_to_equity": {
                "formula": "total_debt / total_equity",
                "inputs": {
                    "total_debt": latest_ratios.get("total_debt"),
                    "total_equity": latest_ratios.get("total_equity"),
                },
            },
            "current_ratio": {
                "formula": "total_current_assets / current_liabilities",
                "inputs": {
                    "total_current_assets": latest_ratios.get("current_assets"),
                    "current_liabilities": latest_ratios.get("current_liabilities"),
                },
            },
            "free_cash_flow": {
                "formula": "reported free_cash_flow or operating_cash_flow - capital_expenditure",
                "inputs": {
                    "free_cash_flow": latest_ratios.get("free_cash_flow"),
                    "operating_cash_flow": latest_ratios.get("operating_cash_flow"),
                },
            },
        }
        latest_ratios["metric_provenance"] = {
            metric: {**metric_context, **formula_provenance.get(metric, {})}
            for metric in (
                "revenue",
                "net_income",
                "revenue_growth",
                "net_income_growth",
                "net_margin",
                "operating_margin",
                "free_cash_flow",
                "roe",
                "roa",
                "debt_to_equity",
                "current_ratio",
            )
        }

        return {
            "primary_basis": selected_basis,
            "metric_metadata": {
                "source": source_metadata.get("source", "Yahoo Finance via yfinance"),
                "original_source": source_metadata.get("original_source"),
                "cache_layer": source_metadata.get("cache_layer"),
                "source_warning": source_metadata.get("source_warning"),
                "currency": "INR",
                "raw_unit": "INR",
                "display_unit": "INR crore",
                "reporting_period": latest_ratios.get("period"),
                "statement_basis": selected_basis,
                "reporting_basis": source_metadata.get("reporting_basis"),
                "retrieval": source_metadata,
            },
            "has_quarterly": has_quarterly,
            "annual_series": annual_summary,
            "quarterly_series": quarterly_summary,
            "latest_ratios": latest_ratios,
            "raw_tables": {
                "annual_income": annual_income,
                "quarterly_income": quarterly_income,
                "annual_balance": annual_balance,
                "quarterly_balance": quarterly_balance,
                "annual_cashflow": annual_cashflow,
                "quarterly_cashflow": quarterly_cashflow,
            },
        }

    @classmethod
    def _extract_series(
        cls,
        inc: pd.DataFrame,
        bs: pd.DataFrame,
        cf: pd.DataFrame,
        is_financial: bool,
    ) -> Dict[str, Any]:
        """Extract standardized time series from pivoted statement tables."""
        if inc.empty and bs.empty:
            return {"periods": []}

        # Available periods across statements
        all_periods = sorted(list(set(inc.columns).union(set(bs.columns)).union(set(cf.columns))))
        if not all_periods:
            return {"periods": []}

        def get_val(df: pd.DataFrame, candidates: List[str], period: str) -> Optional[float]:
            if df.empty or period not in df.columns:
                return None
            for cand in candidates:
                if cand in df.index:
                    val = df.loc[cand, period]
                    if pd.notna(val):
                        return float(val)
            return None

        revenue_keys = ["Total Revenue", "Operating Revenue", "Gross Revenue", "Revenue"]
        net_income_keys = ["Net Income", "Net Income Common Stockholders", "Net Income From Continuing And Discontinued Operation"]
        operating_income_keys = ["Operating Income", "EBIT", "Operating Profit"]
        gross_profit_keys = ["Gross Profit"]
        depreciation_keys = ["Reconciled Depreciation", "Depreciation And Amortization In Income Statement"]
        total_assets_keys = ["Total Assets"]
        total_liabilities_keys = ["Total Liabilities Net Minority Interest", "Total Liabilities"]
        total_equity_keys = [
            "Stockholders Equity",
            "Total Equity Gross Minority Interest",
            "Common Stock Equity",
        ]
        current_assets_keys = ["Total Current Assets", "Current Assets"]
        current_liabilities_keys = ["Total Current Liabilities", "Current Liabilities"]
        total_debt_keys = ["Total Debt", "Long Term Debt And Capital Lease Obligation"]
        cash_keys = ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]
        operating_cf_keys = ["Operating Cash Flow", "Cash Flowsfromusedin Operating Activities"]
        capex_keys = ["Capital Expenditure"]
        fcf_keys = ["Free Cash Flow"]

        series = {
            "periods": all_periods,
            "period_type": "unknown",
            "revenue": [get_val(inc, revenue_keys, p) for p in all_periods],
            "net_income": [get_val(inc, net_income_keys, p) for p in all_periods],
            "operating_income": [get_val(inc, operating_income_keys, p) for p in all_periods],
            "gross_profit": [get_val(inc, gross_profit_keys, p) for p in all_periods],
            "total_assets": [get_val(bs, total_assets_keys, p) for p in all_periods],
            "total_equity": [get_val(bs, total_equity_keys, p) for p in all_periods],
            "current_assets": [get_val(bs, current_assets_keys, p) for p in all_periods],
            "current_liabilities": [get_val(bs, current_liabilities_keys, p) for p in all_periods],
            "total_debt": [get_val(bs, total_debt_keys, p) for p in all_periods],
            "operating_cash_flow": [get_val(cf, operating_cf_keys, p) for p in all_periods],
            "free_cash_flow": [get_val(cf, fcf_keys, p) for p in all_periods],
        }

        # Screener reports equity capital and reserves separately. They are
        # both required to form total equity; capital alone is not a valid ROE
        # or debt-to-equity denominator.
        for i, p in enumerate(all_periods):
            if series["total_equity"][i] is None:
                equity_capital = get_val(bs, ["Equity Capital"], p)
                reserves = get_val(bs, ["Reserves", "reserves"], p)
                if equity_capital is not None and reserves is not None:
                    series["total_equity"][i] = equity_capital + reserves

        # Calculate free cash flow if missing and components exist (OCF - CapEx)
        for i, p in enumerate(all_periods):
            if series["free_cash_flow"][i] is None:
                ocf = series["operating_cash_flow"][i]
                capex = get_val(cf, capex_keys, p)
                if ocf is not None and capex is not None:
                    # yfinance reports capex as negative
                    series["free_cash_flow"][i] = ocf + capex if capex < 0 else ocf - capex

        return series

    @classmethod
    def _compute_latest_ratios(
        cls, series: Dict[str, Any], is_financial: bool
    ) -> Dict[str, Any]:
        periods = series.get("periods", [])
        if not periods:
            return {}

        idx = -1  # Latest period
        prev_idx = -2 if len(periods) >= 2 else None
        previous_period = periods[prev_idx] if prev_idx is not None else None

        rev = series["revenue"][idx]
        rev_prev = series["revenue"][prev_idx] if prev_idx is not None else None
        ni = series["net_income"][idx]
        ni_prev = series["net_income"][prev_idx] if prev_idx is not None else None
        op_inc = series["operating_income"][idx]
        gp = series["gross_profit"][idx]
        assets = series["total_assets"][idx]
        equity = series["total_equity"][idx]
        current_assets = series.get("current_assets", [None] * len(periods))[idx]
        current_liabilities = series.get("current_liabilities", [None] * len(periods))[idx]
        debt = series["total_debt"][idx]
        ocf = series["operating_cash_flow"][idx]
        fcf = series["free_cash_flow"][idx]

        def status(value: Any, reason: str) -> Dict[str, Any]:
            return {
                "status": "calculated" if value is not None else "unavailable",
                "reason": None if value is not None else reason,
            }

        revenue_margin = safe_div(op_inc, rev)
        net_margin = safe_div(ni, rev)
        debt_equity = safe_div(debt, equity)
        current_ratio = safe_div(current_assets, current_liabilities)
        roe = safe_div(ni, equity)
        roa = safe_div(ni, assets)

        ratios = {
            "period": periods[idx],
            "reporting_period": periods[idx],
            "previous_period": previous_period,
            "revenue": rev,
            "revenue_display": format_currency_inr(rev),
            "revenue_growth": growth_pct(rev, rev_prev),
            "revenue_growth_display": format_pct(growth_pct(rev, rev_prev)),
            "net_income": ni,
            "net_income_display": format_currency_inr(ni),
            "net_income_growth": growth_pct(ni, ni_prev),
            "net_income_growth_display": format_pct(growth_pct(ni, ni_prev)),
            "operating_income": op_inc,
            "operating_income_display": format_currency_inr(op_inc),
            "operating_margin": revenue_margin * 100 if revenue_margin is not None else None,
            "operating_margin_display": format_pct(revenue_margin * 100 if revenue_margin is not None else None),
            "net_margin": net_margin * 100 if net_margin is not None else None,
            "net_margin_display": format_pct(net_margin * 100 if net_margin is not None else None),
            "gross_margin": safe_div(gp, rev) * 100 if safe_div(gp, rev) is not None else None,
            "gross_margin_display": format_pct(safe_div(gp, rev) * 100 if safe_div(gp, rev) is not None else None),
            "total_assets": assets,
            "total_assets_display": format_currency_inr(assets),
            "total_equity": equity,
            "total_equity_display": format_currency_inr(equity),
            "total_debt": debt,
            "total_debt_display": format_currency_inr(debt),
            "debt_to_equity": debt_equity,
            "debt_to_equity_display": format_ratio(debt_equity),
            "current_assets": current_assets,
            "current_assets_display": format_currency_inr(current_assets),
            "current_liabilities": current_liabilities,
            "current_liabilities_display": format_currency_inr(current_liabilities),
            "current_ratio": current_ratio,
            "current_ratio_display": format_ratio(current_ratio),
            "roe": roe * 100 if roe is not None else None,
            "roe_display": format_pct(roe * 100 if roe is not None else None),
            "roa": roa * 100 if roa is not None else None,
            "roa_display": format_pct(roa * 100 if roa is not None else None),
            "operating_cash_flow": ocf,
            "operating_cash_flow_display": format_currency_inr(ocf),
            "free_cash_flow": fcf,
            "free_cash_flow_display": format_currency_inr(fcf),
            "ocf_to_ni": safe_div(ocf, ni),
            "ocf_to_ni_display": format_ratio(safe_div(ocf, ni)),
        }
        ratios["metric_status"] = {
            "revenue": status(rev, "revenue is not reported for the selected period"),
            "net_income": status(ni, "net income is not reported for the selected period"),
            "revenue_growth": status(
                growth_pct(rev, rev_prev),
                "current or previous revenue is missing, or previous revenue is zero",
            ),
            "net_income_growth": status(
                growth_pct(ni, ni_prev),
                "current or previous net income is missing, or previous net income is zero",
            ),
            "operating_margin": status(
                ratios["operating_margin"],
                "operating income or revenue is missing, or revenue is zero",
            ),
            "net_margin": status(
                ratios["net_margin"],
                "net income or revenue is missing, or revenue is zero",
            ),
            "debt_to_equity": status(
                debt_equity,
                "debt or equity is missing, or equity is zero",
            ),
            "current_ratio": status(
                current_ratio,
                "current assets or current liabilities are missing, or current liabilities are zero",
            ),
            "roe": status(roe, "net income or equity is missing, or equity is zero"),
            "roa": status(roa, "net income or assets is missing, or assets is zero"),
            "total_debt": status(debt, "total debt is not reported for the selected period"),
            "total_equity": status(equity, "total equity is not reported for the selected period"),
            "operating_cash_flow": status(
                ocf, "operating cash flow is not reported for the selected period"
            ),
            "ocf_to_ni": status(
                safe_div(ocf, ni),
                "operating cash flow or net income is missing, or net income is zero",
            ),
            "free_cash_flow": status(
                fcf,
                "free cash flow and its operating-cash-flow/capex inputs are unavailable",
            ),
        }

        return ratios
