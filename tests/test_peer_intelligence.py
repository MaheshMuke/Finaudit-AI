import pytest
import pandas as pd
from finaudit.company_registry import CompanyRecord, CompanyRegistry, CURATED_PEER_GROUPS
from finaudit.financial_calc import FinancialCalculator, growth_pct, safe_div, format_currency_inr

def _make_company(symbol, name, industry):
    return CompanyRecord(symbol=symbol, company_name=name, industry=industry, yahoo_ticker=f"{symbol}.NS")

def _income(symbol, periods, revenues, net_incomes):
    rows = []
    for p, r, ni in zip(periods, revenues, net_incomes):
        rows.append({"symbol": symbol, "item": "Total Revenue", "period_end": p, "value": r})
        rows.append({"symbol": symbol, "item": "Net Income", "period_end": p, "value": ni})
    return pd.DataFrame(rows)

class TestFinancialSubtypeInference:
    def test_amc_from_company_name(self):
        r = CompanyRegistry.__new__(CompanyRegistry)
        assert r.infer_financial_subtype(_make_company("NAM-INDIA", "Nippon Life India Asset Management Ltd.", "Financial Services")) == "asset_manager"

    def test_hdfc_amc_detected(self):
        r = CompanyRegistry.__new__(CompanyRegistry)
        assert r.infer_financial_subtype(_make_company("HDFCAMC", "HDFC Asset Management Company Ltd.", "Financial Services")) == "asset_manager"

    def test_uti_amc_detected(self):
        r = CompanyRegistry.__new__(CompanyRegistry)
        assert r.infer_financial_subtype(_make_company("UTIAMC", "UTI Asset Management Company Ltd.", "Financial Services")) == "asset_manager"

    def test_bank_detected(self):
        r = CompanyRegistry.__new__(CompanyRegistry)
        assert r.infer_financial_subtype(_make_company("SBIN", "State Bank of India", "Financial Services")) == "bank"

    def test_insurance_detected(self):
        r = CompanyRegistry.__new__(CompanyRegistry)
        assert r.infer_financial_subtype(_make_company("LICI", "Life Insurance Corporation of India", "Financial Services")) == "insurance"

    def test_nbfc_detected(self):
        r = CompanyRegistry.__new__(CompanyRegistry)
        assert r.infer_financial_subtype(_make_company("BAJFINANCE", "Bajaj Finance Ltd.", "Financial Services")) == "nbfc"

    def test_exchange_detected(self):
        r = CompanyRegistry.__new__(CompanyRegistry)
        assert r.infer_financial_subtype(_make_company("BSE", "BSE Ltd.", "Financial Services")) == "stock_exchange"

    def test_wealth_manager_detected(self):
        r = CompanyRegistry.__new__(CompanyRegistry)
        assert r.infer_financial_subtype(_make_company("NUVAMA", "Nuvama Wealth Management Ltd.", "Financial Services")) == "wealth_manager"

    def test_non_financial_returns_other(self):
        r = CompanyRegistry.__new__(CompanyRegistry)
        assert r.infer_financial_subtype(_make_company("TCS", "Tata Consultancy Services Ltd.", "IT")) == "other_financial"

    def test_summary_enriches_detection(self):
        r = CompanyRegistry.__new__(CompanyRegistry)
        generic = _make_company("X", "Generic Financial Ltd.", "Financial Services")
        assert r.infer_financial_subtype(generic) == "other_financial"
        assert r.infer_financial_subtype(generic, business_summary="manages mutual fund assets") == "asset_manager"

class TestPeerSelectionQuality:
    @pytest.fixture(scope="class")
    def registry(self):
        return CompanyRegistry()

    def test_nam_india_curated_peers_are_amc(self, registry):
        curated = CURATED_PEER_GROUPS.get("NAM-INDIA", ())
        assert len(curated) >= 3
        for symbol in curated:
            company = registry.get_by_symbol(symbol)
            assert company is not None, f"{symbol} not found"
            assert registry.infer_financial_subtype(company) == "asset_manager", f"{symbol} not AMC"

    def test_amc_peers_exclude_banks_in_top4(self, registry):
        nam = registry.get_by_symbol("NAM-INDIA")
        if nam is None:
            pytest.skip("NAM-INDIA not in registry")
        top4 = registry.rank_peers(nam, evidence={}, max_peers=10)[:4]
        subtypes = [registry.infer_financial_subtype(r["company"]) for r in top4]
        bank_peers = [(r["company"].symbol, t) for r, t in zip(top4, subtypes) if t == "bank"]
        assert not bank_peers, f"Banks in top-4 AMC peers: {bank_peers}"

    def test_amc_score_above_bank_by_margin(self, registry):
        nam = registry.get_by_symbol("NAM-INDIA")
        if nam is None:
            pytest.skip("NAM-INDIA not in registry")
        ranked = registry.rank_peers(nam, evidence={}, max_peers=100)
        scores = {r["company"].symbol: float(r["score"]) for r in ranked}
        amc_sym = next((s for s in ("HDFCAMC", "ABSLAMC", "ICICIAMC", "UTIAMC") if s in scores), None)
        bank_sym = next((s for s in ("SBIN", "HDFCBANK", "ICICIBANK", "AXISBANK") if s in scores), None)
        if amc_sym and bank_sym:
            assert scores[amc_sym] > scores[bank_sym] + 30, f"AMC {amc_sym}={scores[amc_sym]:.1f} bank {bank_sym}={scores[bank_sym]:.1f}"

    def test_pharma_peers_not_banks(self, registry):
        sun = registry.get_by_symbol("SUNPHARMA")
        if sun is None:
            pytest.skip("SUNPHARMA not in registry")
        ranked = registry.rank_peers(sun, evidence={}, max_peers=4)
        for entry in ranked:
            ind = entry["company"].industry.lower()
            assert any(k in ind for k in ("healthcare", "pharma", "chemical")), f"Non-pharma: {entry['company'].symbol} ({ind})"

    def test_manufacturing_peers_not_banks(self, registry):
        maruti = registry.get_by_symbol("MARUTI")
        if maruti is None:
            pytest.skip("MARUTI not in registry")
        for entry in registry.rank_peers(maruti, evidence={}, max_peers=4):
            assert "bank" not in entry["company"].company_name.lower(), f"Bank as Maruti peer: {entry['company'].symbol}"

    def test_bank_vs_amc_receives_penalty(self, registry):
        nam = registry.get_by_symbol("NAM-INDIA")
        if nam is None:
            pytest.skip("NAM-INDIA not in registry")
        ranked = registry.rank_peers(nam, evidence={}, max_peers=100)
        sbin_rank = next((r for r in ranked if r["company"].symbol == "SBIN"), None)
        if sbin_rank:
            assert float(sbin_rank["score"]) < 55, f"SBIN should score <55 vs AMC NAM-INDIA, got {sbin_rank['score']}"

class TestFinancialPeriodsAndRatios:
    def test_annual_period_preserved(self):
        stmts = {"income_statement_annual": _income("T", ["2024-03-31", "2025-03-31"], [1e9, 1.2e9], [1e8, 1.5e8]), "balance_sheet_annual": pd.DataFrame(), "cash_flow_annual": pd.DataFrame()}
        result = FinancialCalculator.calculate_statement_metrics(stmts, prefer_annual=True)
        assert result["latest_ratios"]["period"] == "2025-03-31"
        assert result["latest_ratios"]["previous_period"] == "2024-03-31"
        assert result["metric_metadata"]["statement_basis"] == "annual"

    def test_quarterly_not_confused_with_annual(self):
        q = _income("Q", ["2025-12-31", "2026-03-31"], [3e8, 3.2e8], [3e7, 3.5e7])
        a = _income("Q", ["2024-03-31", "2025-03-31"], [1e9, 9e8], [1e8, 8.5e7])
        result = FinancialCalculator.calculate_statement_metrics({"income_statement_quarterly": q, "income_statement_annual": a, "balance_sheet_annual": pd.DataFrame(), "cash_flow_annual": pd.DataFrame()})
        assert result["metric_metadata"]["statement_basis"] == "quarterly"
        assert result["latest_ratios"]["period"] == "2026-03-31"
        assert "2025-03-31" in result["annual_series"]["periods"]

    def test_net_margin_correct(self):
        stmts = {"income_statement_annual": _income("M", ["2025-03-31"], [4e9], [8e8]), "balance_sheet_annual": pd.DataFrame(), "cash_flow_annual": pd.DataFrame()}
        ratios = FinancialCalculator.calculate_statement_metrics(stmts, prefer_annual=True)["latest_ratios"]
        assert ratios["net_margin"] == pytest.approx(20.0, abs=0.01)
        assert ratios["revenue"] == 4e9

    def test_revenue_growth_correct(self):
        stmts = {"income_statement_annual": _income("G", ["2024-03-31", "2025-03-31"], [1e3, 1.25e3], [1e2, 1.2e2]), "balance_sheet_annual": pd.DataFrame(), "cash_flow_annual": pd.DataFrame()}
        assert FinancialCalculator.calculate_statement_metrics(stmts, prefer_annual=True)["latest_ratios"]["revenue_growth"] == pytest.approx(25.0, abs=0.01)

    def test_display_unit_no_duplicate_currency(self):
        stmts = {"income_statement_annual": _income("U", ["2025-03-31"], [1e9], [1e8]), "balance_sheet_annual": pd.DataFrame(), "cash_flow_annual": pd.DataFrame()}
        du = FinancialCalculator.calculate_statement_metrics(stmts)["metric_metadata"]["display_unit"]
        assert du == "INR crore", f"Got: {du!r}"
        assert du.count("INR") == 1, f"Duplicated INR in: {du!r}"

    def test_growth_pct_negative_base(self):
        assert growth_pct(-50, -100) == pytest.approx(50.0, abs=0.01)
        assert growth_pct(50, -100) == pytest.approx(150.0, abs=0.01)
        assert growth_pct(0, 0) is None
        assert growth_pct(None, 100) is None

    def test_safe_div_zero(self):
        assert safe_div(100, 0) is None
        assert safe_div(0, 0) is None
        assert safe_div(None, 5) is None

    def test_format_currency_no_inr_label(self):
        result = format_currency_inr(1_000_000_000)
        assert "Cr" in result
        assert "INR" not in result

class TestPeerSelectionReason:
    def test_nam_india_reason_mentions_amc(self):
        registry = CompanyRegistry()
        nam = registry.get_by_symbol("NAM-INDIA")
        hdfc = registry.get_by_symbol("HDFCAMC")
        if not nam or not hdfc:
            pytest.skip("company not in registry")
        reason = registry.peer_selection_reason(nam, hdfc)
        assert any(kw in reason.lower() for kw in ("asset-management", "amc", "mutual fund")), f"Got: {reason!r}"

    def test_reliance_reason_mentions_energy(self):
        registry = CompanyRegistry()
        rel = registry.get_by_symbol("RELIANCE")
        ongc = registry.get_by_symbol("ONGC")
        if not rel or not ongc:
            pytest.skip("company not in registry")
        reason = registry.peer_selection_reason(rel, ongc)
        assert any(kw in reason.lower() for kw in ("energy", "reliance")), f"Got: {reason!r}"


class TestCompetitorServiceScreenerIntegration:
    def test_compare_prioritizes_screener_peers(self, monkeypatch):
        from finaudit.competitors import CompetitorService
        from finaudit.screener_ingestion import ScreenerAdapter

        mock_peers = [
            {
                "symbol": "INFY",
                "peer_symbol": "INFY",
                "company_name": "Infosys",
                "screener_url": "https://www.screener.in/company/INFY/consolidated/",
                "source_url": "https://www.screener.in/api/company/6599230/peers/",
                "cmp": 1160.0,
                "pe": 15.0,
                "market_cap_cr": 470000.0,
                "roce_pct": 39.9,
                "is_target": False,
            },
            {
                "symbol": "HCLTECH",
                "peer_symbol": "HCLTECH",
                "company_name": "HCL Technologies",
                "screener_url": "https://www.screener.in/company/HCLTECH/consolidated/",
                "source_url": "https://www.screener.in/api/company/6599230/peers/",
                "cmp": 1425.0,
                "pe": 20.4,
                "market_cap_cr": 387000.0,
                "roce_pct": 30.4,
                "is_target": False,
            },
        ]

        monkeypatch.setattr(
            ScreenerAdapter,
            "get_cached_peers",
            lambda self, symbol, **kwargs: mock_peers,
        )

        service = CompetitorService()
        tcs = service.registry.get_by_symbol("TCS")
        assert tcs is not None

        comparison = service.compare(tcs, max_peers=3)
        assert "comparison_table" in comparison
        table_symbols = [r["symbol"] for r in comparison["comparison_table"]]
        assert "TCS" in table_symbols
        assert "INFY" in table_symbols
        assert "HCLTECH" in table_symbols

        intel = service.get_peer_intelligence(tcs, max_peers=3)
        assert intel["selection_method"] == "screener_live"
        assert len(intel["all_peers"]) >= 2
        p0 = intel["all_peers"][0]
        assert p0["symbol"] == "INFY"
        assert p0["screener_url"] == "https://www.screener.in/company/INFY/consolidated/"
        assert p0["cmp"] == 1160.0

