from typing import Any, Dict, List, Optional
from finaudit.company_registry import get_company_registry
from finaudit.data_service import get_data_manager
from finaudit.financial_calc import FinancialCalculator
from finaudit.news_service import get_news_service
from finaudit.commodity_service import get_commodity_service
from finaudit.llm_gateway import get_llm_gateway


class PortfolioImpactService:
    """
    Answers natural-language questions about watchlists and portfolios:
    - Analyzes macro shocks (Crude, FX, RBI rates)
    - Maps direct vs. indirect business dependencies
    - Identifies positive vs. negative company exposures
    """

    def __init__(self):
        self.registry = get_company_registry()
        self.data_manager = get_data_manager()
        self.news_service = get_news_service()
        self.commodity_service = get_commodity_service()
        self.llm = get_llm_gateway()

    def analyze_impact(
        self,
        question: str,
        symbols: List[str],
    ) -> Dict[str, Any]:
        if not symbols:
            return {"error": "Please add companies to your watchlist first."}

        # Step 1: Collect exposure profiles for all watchlist companies
        watchlist_profiles = []
        for sym in symbols:
            comp = self.registry.get_by_symbol(sym)
            if comp:
                stmts = self.data_manager.get_company_data(comp)
                calc = FinancialCalculator.calculate_statement_metrics(stmts)
                ratios = calc.get("latest_ratios", {})
                commodities = self.commodity_service.get_sector_commodities(comp.industry)
                watchlist_profiles.append({
                    "symbol": comp.symbol,
                    "name": comp.company_name,
                    "industry": comp.industry,
                    "revenue": ratios.get("revenue_display"),
                    "net_margin": ratios.get("net_margin_display"),
                    "debt_to_equity": ratios.get("debt_to_equity_display"),
                    "relevant_commodities": [c["name"] for c in commodities],
                })

        # Step 2: Fetch related news
        news_items = self.news_service.search_news([question] + [p["industry"] for p in watchlist_profiles][:3], limit=5)

        # Step 3: Prompt Ollama for structured impact ranking
        prompt = f"""
Question: {question}

Watchlist Holdings Context ({len(watchlist_profiles)} companies):
{watchlist_profiles}

Related News Headlines:
{[n['title'] for n in news_items]}

Analyze how the event, commodity, or macroeconomic shift described in the question affects these specific companies.
Return a JSON object with this EXACT structure:
{{
  "short_answer": "...",
  "macro_driver": "...",
  "transmission_mechanism": "...",
  "company_impacts": [
    {{
      "symbol": "...",
      "company_name": "...",
      "impact": "Potentially Positive" | "Potentially Negative" | "Mixed / Neutral",
      "mechanism": "...",
      "materiality": "High" | "Medium" | "Low"
    }}
  ],
  "indicators_to_monitor": ["..."],
  "confidence_and_limitations": "..."
}}
"""
        sys_msg = (
            "You are an expert financial risk strategist. Analyze transmission mechanisms "
            "based strictly on business models and stated financials. Distinguish direct from indirect exposure. "
            "Output ONLY valid JSON."
        )

        try:
            ai_res = self.llm.generate_json(prompt, sys_msg)
            return {
                "success": True,
                "analysis": ai_res,
                "news_evidence": news_items,
                "financial_evidence": watchlist_profiles,
            }
        except Exception as e:
            # Deterministic fallback breakdown
            fallback_impacts = []
            q_lower = question.lower()
            for p in watchlist_profiles:
                ind = p["industry"].lower()
                impact = "Mixed / Neutral"
                mech = "Indirect macroeconomic exposure."
                if "crude" in q_lower or "oil" in q_lower:
                    if "oil" in ind or "gas" in ind:
                        impact = "Potentially Positive"
                        mech = "Upstream realisations expand."
                    elif "paint" in ind or "chemical" in ind or "auto" in ind or "aviation" in ind:
                        impact = "Potentially Negative"
                        mech = "Input cost inflation pressures margins."
                elif "rate" in q_lower or "rbi" in q_lower:
                    if "financial" in ind or "bank" in ind:
                        impact = "Mixed / Neutral"
                        mech = "Net Interest Margin re-pricing vs deposit costs."
                    elif "realty" in ind or "auto" in ind:
                        impact = "Potentially Negative"
                        mech = "Higher EMI borrowing costs dampen consumer demand."

                fallback_impacts.append({
                    "symbol": p["symbol"],
                    "company_name": p["name"],
                    "impact": impact,
                    "mechanism": mech,
                    "materiality": "Medium",
                })

            return {
                "success": True,
                "analysis": {
                    "short_answer": f"Evaluated {len(watchlist_profiles)} companies against query '{question}'.",
                    "macro_driver": question,
                    "transmission_mechanism": "Evaluated via sector input/output exposure rules.",
                    "company_impacts": fallback_impacts,
                    "indicators_to_monitor": ["Quarterly operating margins", "Commodity benchmark prices"],
                    "confidence_and_limitations": f"Generated via rule-based fallback ({e}).",
                },
                "news_evidence": news_items,
                "financial_evidence": watchlist_profiles,
            }
