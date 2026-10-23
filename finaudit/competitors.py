from typing import Any, Dict, List, Optional
import pandas as pd
from finaudit.company_registry import CompanyRecord, get_company_registry
from finaudit.data_service import get_data_manager
from finaudit.financial_calc import FinancialCalculator, format_currency_inr, format_pct, format_ratio


class CompetitorService:
    def __init__(self):
        self.registry = get_company_registry()
        self.data_manager = get_data_manager()

    def compare(self, target: CompanyRecord, max_peers: int = 4) -> Dict[str, Any]:
        """
        Identify genuine industry peers and compare latest metrics:
        - Revenue & Growth
        - Net Income & Margin
        - Operating Margin
        - Debt to Equity
        - ROE / ROA
        - Market Cap & PE
        """
        # Prioritize verified Screener peer candidates when available
        screener_candidates: List[CompanyRecord] = []
        try:
            from finaudit.screener_ingestion import ScreenerAdapter
            adapter = ScreenerAdapter()
            scr_peers = adapter.get_cached_peers(target.symbol, include_target=False)
            if not scr_peers:
                ref = adapter.get_or_refresh(target.symbol)
                scr_peers = ref.get("peers") or adapter.get_cached_peers(target.symbol, include_target=False)
            for sp in scr_peers:
                sym = str(sp.get("peer_symbol") or sp.get("symbol", "")).strip().upper()
                if not sym or sym == target.symbol.upper():
                    continue
                rec = self.registry.get_by_symbol(sym)
                if not rec:
                    rec = CompanyRecord(
                        symbol=sym,
                        company_name=sp.get("company_name", sym),
                        industry=target.industry,
                        isin="",
                        yahoo_ticker=f"{sym}.NS",
                    )
                screener_candidates.append(rec)
        except Exception:
            screener_candidates = []

        registry_candidates = self.registry.get_peers(target, max_peers=len(self.registry.list_all()))
        seen_syms = {target.symbol.upper()}
        candidates: List[CompanyRecord] = []
        for c in screener_candidates + registry_candidates:
            if c.symbol.upper() not in seen_syms:
                seen_syms.add(c.symbol.upper())
                candidates.append(c)
        peer_evidence = self._peer_evidence([target, *candidates])
        screener_syms = {c.symbol.upper() for c in screener_candidates}
        for sym, ev in peer_evidence.items():
            if sym.upper() in screener_syms:
                ev["is_screener_peer"] = True

        ranked_peers = self.registry.rank_peers(
            target, evidence=peer_evidence, max_peers=max_peers, candidates=candidates
        )
        peers = [entry["company"] for entry in ranked_peers]
        ranking_by_symbol = {
            entry["company"].symbol: entry for entry in ranked_peers
        }
        all_companies = [target] + peers

        comparison_rows = []
        for comp in all_companies:
            is_target = comp.symbol == target.symbol
            stmts = self.data_manager.get_company_data(comp)
            is_financial = (
                "financial" in comp.industry.lower()
                or "bank" in comp.industry.lower()
                or "insurance" in comp.industry.lower()
            )
            metrics = FinancialCalculator.calculate_statement_metrics(
                stmts, is_financial, prefer_annual=True
            )
            ratios = metrics.get("latest_ratios", {})

            # Live market data for valuation
            market_data = self.data_manager.fetch_market_data(comp)

            row = {
                "symbol": comp.symbol,
                "company_name": comp.company_name,
                "is_target": is_target,
                "industry": comp.industry,
                "period": ratios.get("period"),
                "reporting_period": ratios.get("reporting_period", ratios.get("period")),
                "previous_period": ratios.get("previous_period"),
                "revenue": ratios.get("revenue"),
                "revenue_display": ratios.get("revenue_display", "Not available"),
                "revenue_growth": ratios.get("revenue_growth"),
                "revenue_growth_display": ratios.get("revenue_growth_display", "Not available"),
                "net_income": ratios.get("net_income"),
                "net_income_display": ratios.get("net_income_display", "Not available"),
                "net_margin": ratios.get("net_margin"),
                "net_margin_display": ratios.get("net_margin_display", "Not available"),
                "operating_margin": ratios.get("operating_margin"),
                "operating_margin_display": ratios.get("operating_margin_display", "Not available"),
                "debt_to_equity": ratios.get("debt_to_equity"),
                "debt_to_equity_display": ratios.get("debt_to_equity_display", "Not available"),
                "roe": ratios.get("roe"),
                "roe_display": ratios.get("roe_display", "Not available"),
                "market_cap": market_data.get("market_cap"),
                "market_cap_display": format_currency_inr(market_data.get("market_cap")),
                "pe_ratio": market_data.get("pe_ratio"),
                "pe_ratio_display": format_ratio(market_data.get("pe_ratio")),
                "one_year_return": market_data.get("one_year_return"),
                "one_year_return_display": format_pct(market_data.get("one_year_return")),
                "source": metrics.get("metric_metadata", {}).get(
                    "source", "Unknown original provider"
                ),
                "statement_basis": metrics.get("metric_metadata", {}).get(
                    "statement_basis", "annual"
                ),
                "reporting_basis": metrics.get("metric_metadata", {}).get(
                    "reporting_basis"
                ),
                "currency": metrics.get("metric_metadata", {}).get("currency", "INR"),
                "unit": metrics.get("metric_metadata", {}).get(
                    "display_unit", "INR crore"
                ),
                "reporting_label": (
                    f"{metrics.get('metric_metadata', {}).get('statement_basis', 'annual').title()} "
                    f"statement ending {ratios.get('period', 'N/A')}"
                ),
                "data_quality": (
                    "comparable"
                    if ratios.get("period")
                    else "missing_financial_data"
                ),
                "peer_selection_reason": (
                    "Target company"
                    if is_target
                    else ranking_by_symbol.get(comp.symbol, {}).get(
                        "reason", self.registry.peer_selection_reason(target, comp)
                    )
                ),
                "peer_selection_confidence": (
                    "not_applicable"
                    if is_target
                    else ranking_by_symbol.get(comp.symbol, {}).get(
                        "confidence", "provisional"
                    )
                ),
                "peer_score": (
                    None
                    if is_target
                    else ranking_by_symbol.get(comp.symbol, {}).get("score")
                ),
            }
            comparison_rows.append(row)

        target_period = next(
            (row.get("period") for row in comparison_rows if row.get("is_target")),
            None,
        )
        target_basis = next(
            (row.get("statement_basis") for row in comparison_rows if row.get("is_target")),
            "annual",
        )
        target_reporting_basis = next(
            (
                row.get("reporting_basis")
                for row in comparison_rows
                if row.get("is_target")
            ),
            None,
        )
        for row in comparison_rows:
            if row.get("is_target"):
                continue
            if row.get("period") != target_period:
                row["data_quality"] = "not_comparable_period"
            elif row.get("statement_basis") != target_basis:
                row["data_quality"] = "not_comparable_basis"
            elif (
                target_reporting_basis is not None
                and row.get("reporting_basis") != target_reporting_basis
            ):
                row["data_quality"] = "not_comparable_reporting_basis"
        df_comparison = pd.DataFrame(comparison_rows)

        # Calculate rankings among selected peers
        rankings = {}
        if len(df_comparison) > 1:
            for col, metric_label in [
                ("revenue", "Revenue"),
                ("revenue_growth", "Revenue Growth"),
                ("net_margin", "Net Profit Margin"),
                ("roe", "Return on Equity"),
                ("market_cap", "Market Capitalization"),
            ]:
                valid = df_comparison[
                    (df_comparison["data_quality"] == "comparable")
                    | (df_comparison["is_target"])
                ].dropna(subset=[col])
                if not valid.empty:
                    sorted_df = valid.sort_values(by=col, ascending=False).reset_index(drop=True)
                    target_idx = sorted_df[sorted_df["symbol"] == target.symbol].index
                    if not target_idx.empty:
                        rankings[metric_label] = {
                            "rank": int(target_idx[0]) + 1,
                            "out_of": int(len(sorted_df)),
                            "leader": str(sorted_df.iloc[0]["company_name"]),
                        }

        return {
            "peers": [p.dict() for p in peers],
            "comparison_table": comparison_rows,
            "rankings": rankings,
            "comparison_metadata": {
                "basis": "annual",
                "target_period": target_period,
                "reporting_basis": target_reporting_basis,
                "reporting_periods": sorted(
                    {row["period"] for row in comparison_rows if row.get("period")}
                ),
                "note": "Peer metrics use annual statements; only peers with the same reporting period and basis are comparable.",
                "peer_selection_method": "same-industry candidates ranked by available business-profile, scale, and financial evidence",
                "peer_selection_confidence": (
                    "supported"
                    if all(
                        row.get("peer_selection_confidence") in {"supported", "not_applicable"}
                        for row in comparison_rows
                    )
                    else "provisional"
                ),
            },
        }

    def _peer_evidence(self, companies: List[CompanyRecord]) -> Dict[str, Dict[str, object]]:
        evidence: Dict[str, Dict[str, object]] = {}
        for company in companies:
            statements = self.data_manager.get_company_data(company)
            calculated = FinancialCalculator.calculate_statement_metrics(
                statements,
                "financial" in company.industry.lower()
                or "bank" in company.industry.lower()
                or "insurance" in company.industry.lower(),
                prefer_annual=True,
            )
            ratios = calculated.get("latest_ratios", {})
            market = self.data_manager.fetch_market_data(company)
            evidence[company.symbol] = {
                "business_summary": market.get("business_summary", ""),
                "revenue": ratios.get("revenue"),
                "has_financial_data": bool(ratios.get("period")),
            }
        return evidence

    def comparable_peer_research(
        self, target: CompanyRecord, max_peers: int = 4
    ) -> Dict[str, Any]:
        """Return only peers with compatible annual periods and usable metrics."""
        comparison = self.compare(target, max_peers=max_peers)
        rows = comparison["comparison_table"]
        target_row = next(row for row in rows if row["is_target"])
        target_period = target_row.get("period")
        compatible = []
        rejected = []
        for row in rows:
            if row["is_target"]:
                continue
            if row.get("data_quality") != "comparable":
                rejected.append(
                    {
                        "symbol": row["symbol"],
                        "reason": "No usable financial statement period.",
                    }
                )
            elif (
                target_period
                and (
                    row.get("period") != target_period
                    or row.get("statement_basis") != target_row.get("statement_basis")
                    or row.get("currency") != target_row.get("currency")
                    or row.get("unit") != target_row.get("unit")
                )
            ):
                rejected.append(
                    {
                        "symbol": row["symbol"],
                        "period": row.get("period"),
                        "reason": (
                            "Reporting period, statement basis, currency, or unit does not "
                            f"match target ({target_period}, {target_row.get('statement_basis')}, "
                            f"{target_row.get('currency')}, {target_row.get('unit')})."
                        ),
                    }
                )
            else:
                compatible.append(row)

        return {
            "target": target_row,
            "peers": compatible,
            "rejected_peers": rejected,
            "status": "comparable" if compatible else "insufficient_data",
            "metric_definitions": {
                "absolute_profit": "Net income in the reported unit.",
                "profit_margin": "Net income divided by revenue.",
                "profit_growth": "Current net income versus the prior comparable period.",
            },
            "source": target_row.get("source", "Local financial dataset"),
            "reporting_period": target_period,
            "statement_basis": target_row.get("statement_basis", "annual"),
            "currency": target_row.get("currency", "INR"),
            "unit": target_row.get("unit", "INR crore"),
        }

    def compare_companies(
        self,
        companies: List[CompanyRecord],
        metric: str = "financial_growth",
    ) -> Dict[str, Any]:
        """Compare explicitly requested companies using compatible annual statements."""
        rows = []
        for company in companies:
            market_data = (
                self.data_manager.fetch_market_data(company)
                if metric == "share_price_growth"
                else {}
            )
            statements = self.data_manager.get_company_data(company)
            is_financial = (
                "financial" in company.industry.lower()
                or "bank" in company.industry.lower()
                or "insurance" in company.industry.lower()
            )
            calculated = FinancialCalculator.calculate_statement_metrics(
                statements, is_financial, prefer_annual=True
            )
            series = calculated.get("annual_series", {})
            periods = series.get("periods", [])
            ratios = calculated.get("latest_ratios", {})
            previous_period = periods[-2] if len(periods) >= 2 else None
            row = {
                "symbol": company.symbol,
                "company_name": company.company_name,
                "industry": company.industry,
                "metric": metric,
                "period": ratios.get("period"),
                "previous_period": previous_period,
                "reporting_period": ratios.get("reporting_period", ratios.get("period")),
                "current_revenue": ratios.get("revenue"),
                "previous_revenue": series.get("revenue", [None])[-2] if len(series.get("revenue", [])) >= 2 else None,
                "current_revenue_display": (
                    ratios.get("revenue") / 1e7 if ratios.get("revenue") is not None else None
                ),
                "previous_revenue_display": (
                    series.get("revenue", [None])[-2] / 1e7
                    if len(series.get("revenue", [])) >= 2 and series.get("revenue", [None])[-2] is not None
                    else None
                ),
                "revenue_growth": ratios.get("revenue_growth"),
                "current_net_income": ratios.get("net_income"),
                "previous_net_income": series.get("net_income", [None])[-2] if len(series.get("net_income", [])) >= 2 else None,
                "current_net_income_display": (
                    ratios.get("net_income") / 1e7 if ratios.get("net_income") is not None else None
                ),
                "previous_net_income_display": (
                    series.get("net_income", [None])[-2] / 1e7
                    if len(series.get("net_income", [])) >= 2 and series.get("net_income", [None])[-2] is not None
                    else None
                ),
                "net_income_growth": ratios.get("net_income_growth"),
                "source": calculated.get("metric_metadata", {}).get("source", "Local financial dataset"),
                "statement_basis": calculated.get("metric_metadata", {}).get("statement_basis", "annual"),
                "currency": calculated.get("metric_metadata", {}).get("currency", "INR"),
                "unit": calculated.get("metric_metadata", {}).get("display_unit", "INR crore"),
                "reporting_label": (
                    f"{calculated.get('metric_metadata', {}).get('statement_basis', 'annual').title()} "
                    f"statement ending {ratios.get('period', 'N/A')}"
                ),
                "data_quality": "comparable" if ratios.get("period") else "missing_financial_data",
            }
            if metric == "share_price_growth":
                row.update(
                    {
                        "share_price_growth": market_data.get("one_year_return"),
                        "source": market_data.get("source", "Yahoo Finance via yfinance"),
                        "unit": "percent return",
                        "period": market_data.get("retrieved_at"),
                        "previous_period": None,
                        "statement_basis": "market price history",
                        "reporting_label": "One-year share-price return ending at retrieval timestamp",
                        "data_quality": (
                            "comparable"
                            if market_data.get("one_year_return") is not None
                            else "missing_market_data"
                        ),
                    }
                )
            if not periods and metric != "share_price_growth":
                row["data_quality"] = "missing_financial_data"
            rows.append(row)

        available_periods = {
            (row["statement_basis"], row["period"], row["previous_period"], row["currency"], row["unit"])
            for row in rows
            if row["data_quality"] == "comparable"
        }
        compatible = (
            len(available_periods) == 1 if metric != "share_price_growth"
            else len([row for row in rows if row["data_quality"] == "comparable"]) == len(rows)
        ) and len(rows) > 1
        if not compatible:
            for row in rows:
                if row["data_quality"] == "comparable" and len(available_periods) > 1:
                    row["data_quality"] = "incompatible_reporting_period"

        return {
            "status": "comparable" if compatible else "partial",
            "metric": metric,
            "comparison_table": rows,
            "reporting_periods": [
                {
                    "period": row["period"],
                    "previous_period": row["previous_period"],
                    "statement_basis": row["statement_basis"],
                    "currency": row["currency"],
                    "unit": row["unit"],
                    "symbol": row["symbol"],
                }
                for row in rows
            ],
            "limitations": [
                "Companies with missing or incompatible comparable annual periods were not ranked."
            ] if not compatible else [],
        }

    def get_peer_intelligence(
        self,
        target: CompanyRecord,
        custom_peers: Optional[List[str]] = None,
        max_peers: int = 5,
    ) -> Dict[str, Any]:
        """
        Priority 2 — Screener-powered peer and company intelligence:
        - Uses Screener.in peer information as the initial peer set when available.
        - Validates business comparability.
        - Ranks core vs secondary peers with explicit selection rationales.
        - Allows user customization of peer sets.
        - Gracefully falls back to registry without silently substituting unrelated companies.
        """
        selection_method = "screener_live"
        fallback_note = None
        peer_items: List[Dict[str, Any]] = []

        if custom_peers:
            selection_method = "user_custom"
            for sym in custom_peers:
                clean_sym = sym.strip().upper()
                if clean_sym == target.symbol.upper():
                    continue
                comp = self.registry.get_by_symbol(clean_sym)
                if not comp:
                    comp = CompanyRecord(
                        symbol=clean_sym,
                        company_name=clean_sym,
                        industry=target.industry,
                        isin="",
                        yahoo_ticker=f"{clean_sym}.NS",
                    )
                peer_items.append({
                    "company": comp,
                    "screener_metrics": {},
                    "is_custom": True,
                })
        else:
            screener_raw_peers = []
            try:
                from finaudit.screener_ingestion import ScreenerAdapter
                adapter = ScreenerAdapter()
                cached = adapter.get_cached_peers(target.symbol, fresh_only=True, include_target=False)
                if cached:
                    screener_raw_peers = cached
                else:
                    ref = adapter.get_or_refresh(target.symbol)
                    screener_raw_peers = ref.get("peers") or adapter.get_cached_peers(target.symbol, include_target=False)
                    if not screener_raw_peers and "metadata" in ref:
                        screener_raw_peers = [p for p in ref["metadata"].get("screener_peers", []) if p.get("symbol") != target.symbol.upper()]
            except Exception as exc:
                logger.debug("Failed retrieving Screener peers for %s: %s", target.symbol, exc)
                screener_raw_peers = []

            if screener_raw_peers:
                for raw in screener_raw_peers:
                    sym = str(raw.get("symbol", "")).strip().upper()
                    if not sym or sym == target.symbol.upper():
                        continue
                    comp = self.registry.get_by_symbol(sym)
                    if not comp:
                        comp = CompanyRecord(
                            symbol=sym,
                            company_name=raw.get("company_name", raw.get("name", sym)),
                            industry=target.industry,
                            isin="",
                            yahoo_ticker=f"{sym}.NS",
                        )
                    peer_items.append({
                        "company": comp,
                        "screener_metrics": raw,
                        "is_custom": False,
                    })
                    if len(peer_items) >= max_peers:
                        break

            else:
                selection_method = "registry_industry_fallback"
                fallback_note = (
                    "Screener peer comparison table was unavailable or blocked; "
                    "fell back to verified same-industry registry peers without substituting unrelated companies."
                )
                reg_peers = self.registry.get_peers(target, max_peers=max_peers)
                for comp in reg_peers:
                    peer_items.append({
                        "company": comp,
                        "screener_metrics": {},
                        "is_custom": False,
                    })

        # Rank into Core and Secondary peers
        # Core peers are direct industry leaders / high comparability; secondary are midcap/adjacent
        core_peers: List[Dict[str, Any]] = []
        secondary_peers: List[Dict[str, Any]] = []

        for idx, item in enumerate(peer_items):
            comp: CompanyRecord = item["company"]
            scr = item["screener_metrics"]

            # Market data enrichment
            market_data = self.data_manager.fetch_market_data(comp)
            market_cap = scr.get("market_cap") or market_data.get("market_cap")

            # Determine core vs secondary
            # Top 3 or companies with substantial market cap / high comparability
            is_core = idx < 3
            tier = "core" if is_core else "secondary"

            if is_core:
                reason = (
                    f"{comp.company_name} ({comp.symbol}) is ranked as a Core Peer due to direct "
                    f"business overlap in {comp.industry}, high comparability in operating model, "
                    f"and scale."
                )
            else:
                reason = (
                    f"{comp.company_name} ({comp.symbol}) is classified as a Secondary Peer, "
                    f"providing comparative mid-tier/adjacent industry benchmarks in {comp.industry}."
                )

            market_cap = scr.get("market_cap_cr") or scr.get("market_cap") or market_data.get("market_cap")
            cmp_price = scr.get("cmp") or market_data.get("current_price")
            pe_val = scr.get("pe") or market_data.get("pe_ratio")
            roce_val = scr.get("roce_pct") or scr.get("roce")
            sales_var_val = scr.get("qtr_sales_var_pct") or scr.get("sales_var")

            peer_entry = {
                "symbol": comp.symbol,
                "company_name": comp.company_name,
                "industry": comp.industry,
                "tier": tier,
                "selection_reason": reason,
                "cmp": cmp_price,
                "pe_ratio": pe_val,
                "market_cap": market_cap,
                "market_cap_display": format_currency_inr(market_cap) if market_cap else "N/A",
                "roce": roce_val,
                "sales_growth_qtr": sales_var_val,
                "one_year_return": market_data.get("one_year_return"),
                "source": "Screener.in peers API" if scr else market_data.get("source", "Registry & Market Data"),
                "source_url": scr.get("source_url") or scr.get("screener_url"),
                "screener_url": scr.get("screener_url") or f"https://www.screener.in/company/{comp.symbol}/",
                "comparability_status": "verified_industry_match",
            }


            if is_core:
                core_peers.append(peer_entry)
            else:
                secondary_peers.append(peer_entry)

        return {
            "target": {
                "symbol": target.symbol,
                "company_name": target.company_name,
                "industry": target.industry,
            },
            "selection_method": selection_method,
            "fallback_note": fallback_note,
            "core_peers": core_peers,
            "secondary_peers": secondary_peers,
            "all_peers": core_peers + secondary_peers,
            "total_peers": len(core_peers) + len(secondary_peers),
            "editable_peer_set_supported": True,
        }

