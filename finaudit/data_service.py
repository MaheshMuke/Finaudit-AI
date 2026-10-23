import logging
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
try:
    import yfinance as yf
except ImportError:
    yf = None

from finaudit.company_registry import CompanyRecord, ROOT_DIR, OUTPUT_DIR, CACHE_DIR
from finaudit.screener_ingestion import ScreenerAdapter, facts_to_statement_frames

logger = logging.getLogger(__name__)

DATASET_FILES = {
    "income_statement_annual": OUTPUT_DIR / "income_statement_annual.csv",
    "income_statement_quarterly": OUTPUT_DIR / "income_statement_quarterly.csv",
    "balance_sheet_annual": OUTPUT_DIR / "balance_sheet_annual.csv",
    "balance_sheet_quarterly": OUTPUT_DIR / "balance_sheet_quarterly.csv",
    "cash_flow_annual": OUTPUT_DIR / "cash_flow_annual.csv",
    "cash_flow_quarterly": OUTPUT_DIR / "cash_flow_quarterly.csv",
}


class FinancialDataManager:
    def __init__(self):
        self._cache_loaded = False
        self._dfs: Dict[str, pd.DataFrame] = {}
        self.screener = ScreenerAdapter()
        self.last_retrieval_metadata: Dict[str, object] = {}

    def get_company_data(self, company: CompanyRecord) -> Dict[str, Any]:
        """
        Retrieve financial statements for a company.
        First tries per-company cached CSV (fastest, pre-merged).
        Falls back to output/*.csv datasets.
        """
        symbol = company.symbol
        cache_file = CACHE_DIR / f"{symbol}.csv"

        result: Dict[str, pd.DataFrame] = {
            "income_statement_annual": pd.DataFrame(),
            "income_statement_quarterly": pd.DataFrame(),
            "balance_sheet_annual": pd.DataFrame(),
            "balance_sheet_quarterly": pd.DataFrame(),
            "cash_flow_annual": pd.DataFrame(),
            "cash_flow_quarterly": pd.DataFrame(),
        }

        screener_facts = self.screener.get_cached_facts(symbol, fresh_only=True)
        if screener_facts:
            self.last_retrieval_metadata = {
                "source": "Screener",
                "original_source": "Screener",
                "provider": "screener",
                "cache_layer": "local_financial_facts",
                "cache_status": "fresh",
                "symbol": symbol,
            }
            frames = facts_to_statement_frames(screener_facts)
            frames["_metadata"] = {
                **frames.get("_metadata", {}),
                **self.last_retrieval_metadata,
            }
            return frames

        stale_screener_facts = self.screener.get_cached_facts(symbol)
        if stale_screener_facts:
            try:
                refreshed = self.screener.get_or_refresh(symbol)
                self.last_retrieval_metadata = {
                    "source": "Screener",
                    "original_source": "Screener",
                    "provider": "screener",
                    "cache_layer": "local_financial_facts",
                    "cache_status": refreshed.get("status"),
                    "symbol": symbol,
                    "batch_id": refreshed.get("batch_id"),
                    "warnings": refreshed.get("findings", []),
                }
                frames = facts_to_statement_frames(refreshed.get("facts", []))
                frames["_metadata"] = {
                    **frames.get("_metadata", {}),
                    **self.last_retrieval_metadata,
                }
                return frames
            except (RuntimeError, ValueError) as exc:
                logger.warning("Screener refresh unavailable for %s; using prior/local data: %s", symbol, exc)
                self.last_retrieval_metadata = {
                    "source": "Screener",
                    "original_source": "Screener",
                    "provider": "screener",
                    "cache_layer": "local_financial_facts",
                    "cache_status": "stale_refresh_failed",
                    "symbol": symbol,
                    "error": str(exc),
                }

        # Prefer an on-demand Screener retrieval before using legacy local files.
        # Legacy CSVs have no provider metadata, so they must not be presented as
        # Screener observations or as having a verified original source.
        try:
            refreshed = self.screener.get_or_refresh(symbol)
            self.last_retrieval_metadata = {
                "source": "Screener",
                "original_source": "Screener",
                "provider": "screener",
                "cache_layer": "local_financial_facts",
                "cache_status": refreshed.get("status"),
                "symbol": symbol,
                "batch_id": refreshed.get("batch_id"),
                "warnings": refreshed.get("findings", []),
            }
            frames = facts_to_statement_frames(refreshed.get("facts", []))
            frames["_metadata"] = {
                **frames.get("_metadata", {}),
                **self.last_retrieval_metadata,
            }
            return frames
        except (RuntimeError, ValueError) as exc:
            logger.warning("Screener retrieval unavailable for %s; using legacy local data: %s", symbol, exc)
            screener_warning = str(exc)

        # 1. Try local cache file
        if cache_file.exists() and cache_file.stat().st_size > 50:
            try:
                df = pd.read_csv(cache_file)
                if not df.empty and "statement" in df.columns:
                    for stmt in result.keys():
                        part = df[df["statement"] == stmt]
                        if not part.empty:
                            result[stmt] = part.drop(columns=["statement"], errors="ignore")
                    if any(not df_stmt.empty for df_stmt in result.values()):
                        self.last_retrieval_metadata = {
                            "source": "Unknown original provider",
                            "original_source": None,
                            "provider": "unknown",
                            "cache_layer": "local_normalized_cache",
                            "cache_status": "fresh_or_untracked",
                            "symbol": symbol,
                            "source_warning": "Legacy cache contains no provider or source URL metadata.",
                        }
                        if screener_warning:
                            self.last_retrieval_metadata["screener_refresh_warning"] = screener_warning
                        result["_metadata"] = self.last_retrieval_metadata
                        return result
            except Exception as e:
                logger.warning("Failed reading cache file %s: %s", cache_file, e)

        # 2. Try output/*.csv files
        for stmt_name, path in DATASET_FILES.items():
            if path.exists():
                try:
                    # Filter for target symbol using chunks or read
                    # To be fast without reading whole 11MB file every time, we can filter
                    df_all = self._get_dataset(stmt_name)
                    if df_all is not None and not df_all.empty:
                        company_slice = df_all[df_all["symbol"] == symbol]
                        if not company_slice.empty:
                            result[stmt_name] = company_slice
                except Exception as e:
                    logger.warning("Failed slicing from %s for %s: %s", stmt_name, symbol, e)

        if any(not df_stmt.empty for df_stmt in result.values()):
            self.last_retrieval_metadata = {
                "source": "Unknown original provider",
                "original_source": None,
                "provider": "unknown",
                "cache_layer": "local_normalized_dataset",
                "cache_status": "fresh_or_untracked",
                "symbol": symbol,
                "source_warning": "Legacy dataset contains no provider or source URL metadata.",
            }
            if screener_warning:
                self.last_retrieval_metadata["screener_refresh_warning"] = screener_warning
            result["_metadata"] = self.last_retrieval_metadata
            return result

        self.last_retrieval_metadata = {
            "source": "Unavailable",
            "original_source": None,
            "provider": "unknown",
            "cache_layer": "none",
            "cache_status": "failed",
            "symbol": symbol,
            "error": screener_warning,
        }
        result["_metadata"] = self.last_retrieval_metadata
        return result

    def _get_dataset(self, key: str) -> Optional[pd.DataFrame]:
        """Lazy load output dataset in memory if requested."""
        if key not in self._dfs:
            path = DATASET_FILES.get(key)
            if path and path.exists():
                self._dfs[key] = pd.read_csv(path)
            else:
                self._dfs[key] = pd.DataFrame()
        return self._dfs[key]

    @staticmethod
    def fetch_market_data(company: CompanyRecord) -> Dict[str, any]:
        """
        On-demand retrieval of share price, 1y historical closes, and valuation metrics via yfinance.
        Never blocks the analysis if offline or failing.
        """
        ticker_sym = company.yahoo_ticker
        market_info = {
            "ticker": ticker_sym,
            "current_price": None,
            "previous_close": None,
            "currency": "INR",
            "market_cap": None,
            "pe_ratio": None,
            "pe_ratio_type": None,
            "pb_ratio": None,
            "dividend_yield": None,
            "beta": None,
            "fifty_two_week_high": None,
            "fifty_two_week_low": None,
            "business_summary": "",
            "history_dates": [],
            "history_prices": [],
            "history_volumes": [],
            "one_year_return": None,
            "annualized_volatility": None,
            "max_drawdown": None,
            "status": "online",
            "source": "Yahoo Finance via yfinance",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }

        if yf is None:
            market_info["status"] = "yfinance_not_installed"
            return market_info

        try:
            ticker = yf.Ticker(ticker_sym)
            # 1. Fetch info
            info = {}
            try:
                info = ticker.info or {}
            except Exception:
                info = {}

            market_info["current_price"] = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
            market_info["previous_close"] = info.get("previousClose")
            market_info["currency"] = info.get("currency", "INR")
            market_info["market_cap"] = info.get("marketCap")
            trailing_pe = info.get("trailingPE")
            forward_pe = info.get("forwardPE")
            if trailing_pe is not None:
                market_info["pe_ratio"] = trailing_pe
                market_info["pe_ratio_type"] = "trailing"
            elif forward_pe is not None:
                market_info["pe_ratio"] = forward_pe
                market_info["pe_ratio_type"] = "forward"
            market_info["pb_ratio"] = info.get("priceToBook")
            market_info["dividend_yield"] = info.get("dividendYield")
            market_info["beta"] = info.get("beta")
            market_info["fifty_two_week_high"] = info.get("fiftyTwoWeekHigh")
            market_info["fifty_two_week_low"] = info.get("fiftyTwoWeekLow")
            market_info["business_summary"] = info.get("longBusinessSummary", "")

            # 2. Fetch 1y price history
            hist = None
            try:
                hist = ticker.history(period="1y")
            except Exception:
                hist = None

            if hist is not None and not hist.empty and "Close" in hist.columns:
                closes = hist["Close"].dropna()
                if len(closes) > 0:
                    market_info["history_dates"] = [d.strftime("%Y-%m-%d") for d in closes.index]
                    market_info["history_prices"] = [round(float(v), 2) for v in closes.values]
                    if "Volume" in hist.columns:
                        market_info["history_volumes"] = [int(v) for v in hist["Volume"].fillna(0).values]

                    # Performance calculations
                    start_price = float(closes.iloc[0])
                    end_price = float(closes.iloc[-1])
                    if start_price > 0:
                        market_info["one_year_return"] = round(((end_price - start_price) / start_price) * 100, 2)

                    # Volatility
                    daily_returns = closes.pct_change().dropna()
                    if len(daily_returns) > 5:
                        vol = daily_returns.std() * np.sqrt(252) * 100
                        market_info["annualized_volatility"] = round(float(vol), 2)

                    # Max Drawdown
                    cum_max = closes.cummax()
                    drawdowns = (closes - cum_max) / cum_max * 100
                    market_info["max_drawdown"] = round(float(drawdowns.min()), 2)

                    if market_info["current_price"] is None:
                        market_info["current_price"] = round(end_price, 2)

        except Exception as exc:
            logger.warning("Failed fetching live market data for %s: %s", ticker_sym, exc)
            market_info["status"] = f"offline ({type(exc).__name__})"

        return market_info


_manager_instance: Optional[FinancialDataManager] = None


def get_data_manager() -> FinancialDataManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = FinancialDataManager()
    return _manager_instance
