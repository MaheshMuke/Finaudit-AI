import logging
from typing import Any, Dict, Optional
from finaudit.company_registry import CompanyRecord, get_company_registry
from finaudit.data_service import get_data_manager
from finaudit.financial_calc import FinancialCalculator
from finaudit.competitors import CompetitorService
from finaudit.ollama_client import OllamaProvider, check_ollama_health
from finaudit.llm import GeminiProvider, configured_provider_name

logger = logging.getLogger(__name__)


class AnalysisEngine:
    """
    Coordinates the complete 1-click analysis pipeline:
    1. Resolve company
    2. Retrieve financial statements from local Nifty 500 dataset
    3. Fetch live market info via yfinance
    4. Compute deterministic financial ratios and growth trends
    5. Compare against genuine sector peers
    6. Generate AI business model, dependencies, catalysts, and risks via Ollama
    """

    def __init__(self):
        self.registry = get_company_registry()
        self.data_manager = get_data_manager()
        self.competitor_service = CompetitorService()
        self.ollama_provider = OllamaProvider()
        self.gemini_provider = GeminiProvider() if configured_provider_name() == "Gemini" else None

    def run_full_analysis(
        self,
        query: str,
        progress_callback=None,
    ) -> Dict[str, Any]:
        def report_progress(msg: str):
            if progress_callback:
                progress_callback(msg)

        # Step 1: Resolve Company
        report_progress(f"Resolving '{query}' against Nifty 500 company registry...")
        company = self.registry.resolve(query)
        if not company:
            matches = self.registry.search(query, limit=5)
            return {
                "success": False,
                "error": f"Could not find company matching '{query}'.",
                "suggestions": [m.dict() for m in matches],
            }

        # Step 2: Retrieve local financial statements
        report_progress(f"Retrieving financial statements for {company.company_name} ({company.symbol})...")
        statements = self.data_manager.get_company_data(company)

        # Step 3: Fetch on-demand market data (prices, 52w, business summary)
        report_progress(f"Fetching market price and valuation metrics for {company.yahoo_ticker}...")
        market_data = self.data_manager.fetch_market_data(company)

        # Step 4: Deterministic calculations
        report_progress("Computing growth rates, margins, leverage, and cash flow ratios...")
        is_financial = (
            "financial" in company.industry.lower()
            or "bank" in company.industry.lower()
            or "insurance" in company.industry.lower()
        )
        calc_results = FinancialCalculator.calculate_statement_metrics(statements, is_financial)

        # Step 5: Peer Comparison
        report_progress(f"Identifying and comparing peers in '{company.industry}' sector...")
        competitor_results = self.competitor_service.compare(company, max_peers=4)

        # Step 6: Configured AI provider
        provider_name = configured_provider_name()
        report_progress(
            f"Generating narrative intelligence, catalysts, and risk factors via {provider_name or 'configured AI provider'}..."
        )
        ollama_status = check_ollama_health() if provider_name != "Gemini" else {
            "available": False,
            "provider": "gemini",
            "configured": True,
            "model": self.gemini_provider.model_name if self.gemini_provider else None,
        }

        ai_analysis = None
        ai_error = None
        llm_metadata = {
            "provider": provider_name,
            "model": (
                self.gemini_provider.model_name
                if provider_name == "Gemini" and self.gemini_provider
                else self.ollama_provider.model
                if provider_name == "Ollama"
                else None
            ),
            "generation_status": "not_started",
            "fallback_used": False,
        }

        if provider_name == "Gemini" and self.gemini_provider:
            try:
                latest_ratios = calc_results.get("latest_ratios", {})
                annual_series = calc_results.get("annual_series", {})
                compact_context = {
                    "company": {
                        "name": company.company_name,
                        "symbol": company.symbol,
                        "industry": company.industry,
                        "sector": company.industry,
                        "business_summary": market_data.get("business_summary") or "",
                        # Asset-management specific metadata
                        "is_asset_manager": any(kw in company.company_name.lower() for kw in
                            ["asset management", "amc", "mutual fund", "fund management"]),
                    },
                    "metrics": latest_ratios,
                    "metric_evidence": {
                        "statement_basis": calc_results.get("metric_metadata", {}).get("statement_basis"),
                        "reporting_period": calc_results.get("metric_metadata", {}).get("reporting_period"),
                        "previous_period": latest_ratios.get("previous_period"),
                        "currency": calc_results.get("metric_metadata", {}).get("currency", "INR"),
                        "unit": calc_results.get("metric_metadata", {}).get("display_unit", "INR crore"),
                        "per_metric": latest_ratios.get("metric_provenance", {}),
                    },
                    "competitors": competitor_results.get("comparison_table", [])[:4],
                    "trends": {
                        "periods": annual_series.get("periods", []),
                        "revenue": annual_series.get("revenue", []),
                        "net_income": annual_series.get("net_income", []),
                    },
                    "stock_performance": {
                        "one_year_return_display": market_data.get("one_year_return"),
                        "range_52w_display": (
                            f"₹{market_data.get('fifty_two_week_low')} - ₹{market_data.get('fifty_two_week_high')}"
                            if market_data.get("fifty_two_week_low")
                            else "N/A"
                        ),
                        "pe_ratio_display": market_data.get("pe_ratio"),
                        "market_cap_display": market_data.get("market_cap"),
                    },
                    "data_limitations": (
                        ["Quarterly data unavailable; used annual statements."]
                        if not calc_results.get("has_quarterly")
                        else []
                    ),
                }
                ai_analysis = self.gemini_provider.analyze_company(compact_context)
                llm_metadata = dict(self.gemini_provider.last_call_metadata)
            except Exception as e:
                logger.warning("Gemini narrative generation failed: %s", e)
                ai_error = f"Gemini narrative analysis failed ({type(e).__name__}: {e}). Displaying numerical metrics and charts."
                llm_metadata = dict(self.gemini_provider.last_call_metadata)
                llm_metadata["generation_status"] = "failed"
        elif ollama_status.get("available") and ollama_status.get("model_installed"):
            try:
                # Prepare compact context for Ollama
                latest_ratios = calc_results.get("latest_ratios", {})
                annual_series = calc_results.get("annual_series", {})
                compact_context = {
                    "company": {
                        "name": company.company_name,
                        "symbol": company.symbol,
                        "industry": company.industry,
                        "sector": company.industry,
                        "business_summary": market_data.get("business_summary") or "",
                    },
                    "metrics": latest_ratios,
                    "metric_evidence": {
                        "statement_basis": calc_results.get("metric_metadata", {}).get("statement_basis"),
                        "reporting_period": calc_results.get("metric_metadata", {}).get("reporting_period"),
                        "previous_period": latest_ratios.get("previous_period"),
                        "currency": calc_results.get("metric_metadata", {}).get("currency", "INR"),
                        "unit": calc_results.get("metric_metadata", {}).get("display_unit", "INR crore"),
                        "per_metric": latest_ratios.get("metric_provenance", {}),
                    },
                    "competitors": competitor_results.get("comparison_table", []),
                    "trends": {
                        "periods": annual_series.get("periods", []),
                        "revenue": annual_series.get("revenue", []),
                        "net_income": annual_series.get("net_income", []),
                    },
                    "stock_performance": {
                        "one_year_return_display": market_data.get("one_year_return"),
                        "range_52w_display": (
                            f"₹{market_data.get('fifty_two_week_low')} - ₹{market_data.get('fifty_two_week_high')}"
                            if market_data.get('fifty_two_week_low')
                            else "N/A"
                        ),
                        "pe_ratio_display": market_data.get("pe_ratio"),
                        "market_cap_display": market_data.get("market_cap"),
                    },
                    "data_limitations": (
                        ["Quarterly data unavailable; used annual statements."]
                        if not calc_results.get("has_quarterly")
                        else []
                    ),
                }

                ai_analysis = self.ollama_provider.analyze_company(compact_context)
                llm_metadata["generation_status"] = "success"
            except Exception as e:
                logger.warning("Ollama narrative generation failed: %s", e)
                ai_error = f"Ollama narrative analysis failed ({type(e).__name__}: {e}). Displaying numerical metrics and charts."
                llm_metadata["generation_status"] = "failed"
                llm_metadata["error_code"] = type(e).__name__.lower()
        elif not ollama_status.get("available"):
            ai_error = ollama_status.get("message", "Ollama is not running. Showing financial data and charts.")
        else:
            ai_error = ollama_status.get(
                "message",
                f"Configured Ollama model '{ollama_status.get('target_model', 'unknown')}' is unavailable.",
            )

        report_progress("Analysis complete!")

        return {
            "success": True,
            "company": company.dict(),
            "market_data": market_data,
            "statements": statements,
            "calculations": calc_results,
            "competitors": competitor_results,
            "ai_analysis": ai_analysis,
            "ai_error": ai_error,
            "ollama_status": ollama_status,
            "llm_provider": provider_name or "none",
            "llm_metadata": llm_metadata,
        }
