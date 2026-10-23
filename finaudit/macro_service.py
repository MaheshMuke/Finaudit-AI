"""Macroeconomic, sector benchmark, and relative market intelligence service."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

try:
    import yfinance as yf
except ImportError:
    yf = None

from finaudit.company_registry import CompanyRecord, get_company_registry

logger = logging.getLogger(__name__)

# Standard Indian macro benchmarks
MACRO_BENCHMARKS = {
    "nifty_50": "^NSEI",
    "usd_inr": "USDINR=X",
    "us_10y_yield": "^TNX",
}

# Current baseline monetary policy benchmarks
POLICY_RATES = {
    "rbi_repo_rate_pct": 6.50,
    "rbi_stance": "withdrawal of accommodation / neutral transition",
    "india_cpi_inflation_pct": 4.85,
}


class MacroMarketIntelligenceService:
    """Provides market-wide benchmarking, macro indicators, and relative peer performance."""

    def __init__(self):
        self.registry = get_company_registry()
        self._cache: Dict[str, Any] = {}

    def get_macro_environment(self) -> Dict[str, Any]:
        """Fetch current macro benchmark indicators (Nifty 50, USD/INR, 10Y Yield, RBI Repo Rate)."""
        now = datetime.now(timezone.utc).isoformat()
        macro_data = {
            "retrieved_at": now,
            "status": "online",
            "policy_rates": POLICY_RATES,
            "benchmarks": {
                "nifty_50": {"current": None, "1y_return_pct": None, "symbol": "^NSEI"},
                "usd_inr": {"current": None, "1y_return_pct": None, "symbol": "USDINR=X"},
                "us_10y_yield": {"current": None, "symbol": "^TNX"},
            },
        }

        if yf is None:
            macro_data["status"] = "yfinance_not_available"
            return macro_data

        for key, ticker_symbol in MACRO_BENCHMARKS.items():
            try:
                t = yf.Ticker(ticker_symbol)
                hist = t.history(period="1y")
                if hist is not None and not hist.empty and "Close" in hist.columns:
                    closes = hist["Close"].dropna()
                    if len(closes) > 0:
                        cur_val = round(float(closes.iloc[-1]), 2)
                        start_val = round(float(closes.iloc[0]), 2)
                        ret_pct = round(((cur_val - start_val) / start_val) * 100, 2) if start_val > 0 else None
                        macro_data["benchmarks"][key]["current"] = cur_val
                        macro_data["benchmarks"][key]["1y_return_pct"] = ret_pct
            except Exception as exc:
                logger.debug("Failed fetching macro ticker %s: %s", ticker_symbol, exc)

        macro_data["nifty_50"] = macro_data["benchmarks"]["nifty_50"]
        macro_data["usd_inr"] = macro_data["benchmarks"]["usd_inr"]
        macro_data["us_10y_yield"] = macro_data["benchmarks"]["us_10y_yield"]
        macro_data["rbi_repo_rate"] = {
            "rate_pct": POLICY_RATES["rbi_repo_rate_pct"],
            "stance": POLICY_RATES["rbi_stance"],
            "cpi_inflation_pct": POLICY_RATES["india_cpi_inflation_pct"],
        }
        macro_data["freshness_status"] = "Live" if macro_data.get("status") == "online" else "Fallback Benchmark"
        return macro_data

    def get_macro_snapshot(self) -> Dict[str, Any]:
        """Alias for get_macro_environment."""
        return self.get_macro_environment()

    def compare_relative_performance(
        self,
        target_company: CompanyRecord | str,
        peer_symbols: List[str],
        period: str = "1y",
    ) -> Dict[str, Any]:
        """Compare target company stock performance vs peers and Nifty 50 over a matching period.
        
        Isolates market-wide movement (beta/market return) from company-specific alpha.
        """
        if isinstance(target_company, str):
            rec = self.registry.get_by_symbol(target_company.strip().upper())
            target_company = rec or CompanyRecord(
                symbol=target_company.strip().upper(),
                company_name=target_company.strip().upper(),
                industry="Unknown",
                isin="",
                yahoo_ticker=f"{target_company.strip().upper()}.NS",
            )

        all_symbols = [target_company.symbol] + [s.strip().upper() for s in peer_symbols if s.strip().upper() != target_company.symbol]
        performance_records = []
        nifty_return: Optional[float] = None


        # Fetch Nifty 50 benchmark return
        if yf is not None:
            try:
                nifty_hist = yf.Ticker("^NSEI").history(period=period)
                if nifty_hist is not None and not nifty_hist.empty and "Close" in nifty_hist.columns:
                    closes = nifty_hist["Close"].dropna()
                    if len(closes) >= 2:
                        nifty_return = round(((float(closes.iloc[-1]) - float(closes.iloc[0])) / float(closes.iloc[0])) * 100, 2)
            except Exception as e:
                logger.debug("Nifty 50 return fetch failed: %s", e)

        for sym in all_symbols:
            rec = self.registry.get_by_symbol(sym)
            if not rec:
                continue
            ticker_str = rec.yahoo_ticker
            comp_record: Dict[str, Any] = {
                "symbol": sym,
                "company_name": rec.company_name,
                "is_target": sym == target_company.symbol,
                "period": period,
                "return_pct": None,
                "annualized_volatility_pct": None,
                "max_drawdown_pct": None,
                "alpha_vs_market_pct": None,
                "status": "online",
            }
            if yf is not None:
                try:
                    t = yf.Ticker(ticker_str)
                    hist = t.history(period=period)
                    if hist is not None and not hist.empty and "Close" in hist.columns:
                        closes = hist["Close"].dropna()
                        if len(closes) >= 2:
                            start_p = float(closes.iloc[0])
                            end_p = float(closes.iloc[-1])
                            ret = round(((end_p - start_p) / start_p) * 100, 2)
                            comp_record["return_pct"] = ret
                            if nifty_return is not None:
                                comp_record["alpha_vs_market_pct"] = round(ret - nifty_return, 2)

                            daily_ret = closes.pct_change().dropna()
                            if len(daily_ret) > 5:
                                comp_record["annualized_volatility_pct"] = round(float(daily_ret.std() * np.sqrt(252) * 100), 2)
                            cum_max = closes.cummax()
                            comp_record["max_drawdown_pct"] = round(float(((closes - cum_max) / cum_max * 100).min()), 2)
                except Exception as err:
                    comp_record["status"] = f"error ({err})"
            performance_records.append(comp_record)

        # Sort peers by return
        target_entry = next((r for r in performance_records if r["is_target"]), None)
        target_return = target_entry["return_pct"] if target_entry else None

        return {
            "period": period,
            "market_benchmark": {
                "index": "Nifty 50 (^NSEI)",
                "return_pct": nifty_return,
            },
            "target": target_entry,
            "peers": [r for r in performance_records if not r["is_target"]],
            "comparison_table": performance_records,
            "interpretation": (
                f"{target_company.company_name} ({target_company.symbol}) delivered {target_return}% over {period}, "
                f"{'outperforming' if target_return is not None and nifty_return is not None and target_return > nifty_return else 'underperforming'} "
                f"the Nifty 50 benchmark ({nifty_return}%)."
                if target_return is not None and nifty_return is not None
                else "Comparative price history partially available."
            ),
        }

    get_peer_relative_performance = compare_relative_performance


_macro_service_instance: Optional[MacroMarketIntelligenceService] = None



def get_macro_service() -> MacroMarketIntelligenceService:
    global _macro_service_instance
    if _macro_service_instance is None:
        _macro_service_instance = MacroMarketIntelligenceService()
    return _macro_service_instance
