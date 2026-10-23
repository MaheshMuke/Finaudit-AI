from pathlib import Path

import pandas as pd
import pytest

from finaudit import database
from finaudit.screener_ingestion import (
    ScreenerAdapter,
    _period_to_date,
    facts_to_statement_frames,
    normalize_screener_rows,
    parse_screener_number,
)


class FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0
        self.headers = {}

    def get(self, url, timeout):
        self.calls += 1
        return FakeResponse(self.text)


TABLES = """
<table>
<tr><th></th><th>Mar 2025</th><th>Mar 2024</th></tr>
<tr><td>Sales +</td><td>100,000</td><td>90,000</td></tr>
<tr><td>Net Profit</td><td>18,000</td><td>15,000</td></tr>
</table>
<table>
<tr><th></th><th>Mar 2025</th></tr>
<tr><td>Total Assets</td><td>250,000</td></tr>
<tr><td>Borrowings</td><td>-</td></tr>
</table>
<table>
<tr><th></th><th>Mar 2025</th></tr>
<tr><td>ROE %</td><td>18.2%</td></tr>
</table>
"""


@pytest.fixture
def isolated_database(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(database, "UPLOAD_DIR", tmp_path / "data" / "uploads")
    monkeypatch.setattr(database, "REPORTS_DIR", tmp_path / "data" / "reports")
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "data" / "test.sqlite3")
    database.initialize()


def test_numeric_parser_preserves_missing_values():
    assert parse_screener_number("1,23,456") == 123456
    assert parse_screener_number("(2,450)") == -2450
    assert parse_screener_number("18.2%") == 18.2
    assert parse_screener_number("-") is None


def test_normalizer_preserves_raw_values_and_metadata():
    facts, findings = normalize_screener_rows(
        [
            {
                "statement": "quarterly_results",
                "metric": "Sales +",
                "period": "Jun 2025",
                "value": "100,000",
                "raw_value": "100,000",
                "period_type": "quarterly",
                "source_url": "https://www.screener.in/company/TCS/consolidated/",
            },
            {
                "statement": "quarterly_results",
                "metric": "Net Profit",
                "period": "Jun 2025",
                "value": "-",
                "raw_value": "-",
                "period_type": "quarterly",
            },
        ],
        symbol="TCS",
        company_name="Tata Consultancy Services Ltd.",
        source_url="https://www.screener.in/company/TCS/consolidated/",
        retrieved_at="2026-10-22T00:00:00+00:00",
    )

    assert not findings
    assert facts[0].metric_key == "revenue"
    assert facts[0].original_metric_label == "Sales +"
    assert facts[0].value == 100000
    assert facts[0].raw_value == "100,000"
    assert facts[1].value is None


def test_normalizer_preserves_ttm_period_label_and_type():
    facts, findings = normalize_screener_rows(
        [{
            "statement": "annual_profit_loss",
            "metric": "Sales",
            "period": "TTM",
            "period_type": "ttm",
            "value": "100,000",
        }],
        symbol="TCS",
        company_name="Tata Consultancy Services Ltd.",
        source_url="https://www.screener.in/company/TCS/consolidated/",
        retrieved_at="2026-10-22T00:00:00+00:00",
    )

    assert not findings
    frames = facts_to_statement_frames([facts[0].to_dict()])
    row = frames["income_statement_annual"].iloc[0]
    assert row["period"] == "TTM"
    assert pd.isna(row["period_end"])
    assert row["period_type"] == "ttm"


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        ("Mar 2026", "2026-03-31"),
        ("Jun 2026", "2026-06-30"),
        ("Sep 2024", "2024-09-30"),
        ("Dec 2025", "2025-12-31"),
    ],
)
def test_period_to_date_uses_reporting_month_end(period, expected):
    assert _period_to_date(period) == expected


def test_period_to_date_does_not_turn_ttm_into_a_date():
    assert _period_to_date("TTM") is None


def test_adapter_extracts_sections_and_period_types():
    adapter = ScreenerAdapter(session=FakeSession(TABLES), timeout=1)
    facts, findings, metadata = adapter.fetch("TCS")

    assert not findings
    assert {fact.statement for fact in facts} == {
        "annual_profit_loss",
        "balance_sheet",
        "ratios",
    }
    assert {fact.period_type for fact in facts} == {"annual", "annual_snapshot"}
    assert metadata["sections_missing"] == [
        "cash_flow",
        "quarterly_results",
    ]


def test_refresh_is_idempotent_and_cache_hit_skips_fetch(isolated_database):
    session = FakeSession(TABLES)
    adapter = ScreenerAdapter(session=session, ttl_seconds=3600, timeout=1)

    first = adapter.refresh("TCS")
    second = adapter.refresh("TCS")
    assert first["facts"] == second["facts"]

    with database.connect() as db:
        facts_count = db.execute("SELECT COUNT(*) FROM financial_facts").fetchone()[0]
        observations_count = db.execute(
            "SELECT COUNT(*) FROM financial_fact_observations"
        ).fetchone()[0]
    assert facts_count == first["facts"]
    assert observations_count == first["facts"] * 2

    session.calls = 0
    cached = adapter.get_or_refresh("TCS")
    assert cached["status"] == "cache_hit"
    assert session.calls == 0


def test_failed_refresh_preserves_previous_valid_facts(isolated_database):
    adapter = ScreenerAdapter(session=FakeSession(TABLES), ttl_seconds=3600, timeout=1)
    first = adapter.refresh("TCS")

    class FailingSession(FakeSession):
        def get(self, url, timeout):
            raise RuntimeError("source unavailable")

    adapter.session = FailingSession("")
    with pytest.raises(RuntimeError):
        adapter.refresh("TCS")

    with database.connect() as db:
        row = db.execute(
            "SELECT COUNT(*) FROM financial_facts WHERE symbol='TCS'"
        ).fetchone()
        failed = db.execute(
            "SELECT COUNT(*) FROM financial_ingestion_batches WHERE status='failed'"
        ).fetchone()
    assert row[0] == first["facts"]
    assert failed[0] == 1


PEERS_HTML = """
<div data-page-results>
  <table class="data-table text-nowrap striped mark-visited highlight-on-hover">
    <tbody>
      <tr>
        <th>S.No.</th>
        <th>Company</th>
        <th>CMP Rs.</th>
        <th>P/E</th>
        <th>Mar Cap Rs.Cr.</th>
        <th>Div Yld %</th>
        <th>NP Qtr Rs.Cr.</th>
        <th>Qtr Profit Var %</th>
        <th>Sales Qtr Rs.Cr.</th>
        <th>Qtr Sales Var %</th>
        <th>ROCE %</th>
      </tr>
      <tr>
        <td>1.</td>
        <td><a href="/company/TCS/consolidated/">TCS</a></td>
        <td>2550.00</td>
        <td>16.90</td>
        <td>924000.00</td>
        <td>2.45</td>
        <td>13934.00</td>
        <td>7.41</td>
        <td>73188.00</td>
        <td>11.23</td>
        <td>63.03</td>
      </tr>
      <tr>
        <td>2.</td>
        <td><a href="/company/INFY/consolidated/">Infosys</a></td>
        <td>1160.00</td>
        <td>15.03</td>
        <td>470000.00</td>
        <td>4.09</td>
        <td>7775.00</td>
        <td>12.25</td>
        <td>48211.00</td>
        <td>14.03</td>
        <td>39.95</td>
      </tr>
      <tr>
        <td>3.</td>
        <td><a href="/company/HCLTECH/consolidated/">HCL Technologies</a></td>
        <td>1425.00</td>
        <td>20.45</td>
        <td>387000.00</td>
        <td>3.70</td>
        <td>5132.00</td>
        <td>21.18</td>
        <td>35620.00</td>
        <td>11.51</td>
        <td>30.40</td>
      </tr>
      <tr>
        <td></td>
        <td>Median: 73 Co.</td>
        <td>200.00</td>
        <td>18.00</td>
        <td>700.00</td>
        <td>0.30</td>
        <td>10.00</td>
        <td>9.00</td>
        <td>85.00</td>
        <td>15.00</td>
        <td>22.00</td>
      </tr>
    </tbody>
  </table>
</div>
"""

MAIN_HTML_WITH_WAREHOUSE = """
<div data-warehouse-id="6599230">
""" + TABLES + """
</div>
"""


class PeerMockSession:
    def __init__(self, main_text: str, peer_text: str):
        self.main_text = main_text
        self.peer_text = peer_text
        self.calls = 0
        self.headers = {}

    def get(self, url, timeout):
        self.calls += 1
        if "peers" in str(url):
            return FakeResponse(self.peer_text)
        return FakeResponse(self.main_text)


def test_parse_peer_table_html_extracts_all_fields_and_filters_median():
    adapter = ScreenerAdapter(session=FakeSession(""), timeout=1)
    peers = adapter.parse_peer_table_html(
        PEERS_HTML,
        target_symbol="TCS",
        source_url="https://www.screener.in/api/company/6599230/peers/",
    )

    assert len(peers) == 3  # Median row filtered out
    tcs = peers[0]
    assert tcs["symbol"] == "TCS"
    assert tcs["peer_symbol"] == "TCS"
    assert tcs["company_name"] == "TCS"
    assert tcs["is_target"] is True
    assert tcs["rank"] == 1
    assert tcs["cmp"] == 2550.0
    assert tcs["pe"] == 16.9
    assert tcs["market_cap_cr"] == 924000.0
    assert tcs["div_yield_pct"] == 2.45
    assert tcs["qtr_profit_cr"] == 13934.0
    assert tcs["qtr_profit_var_pct"] == 7.41
    assert tcs["qtr_sales_cr"] == 73188.0
    assert tcs["qtr_sales_var_pct"] == 11.23
    assert tcs["roce_pct"] == 63.03
    assert tcs["screener_url"] == "https://www.screener.in/company/TCS/consolidated/"
    assert tcs["source_url"] == "https://www.screener.in/api/company/6599230/peers/"

    infy = peers[1]
    assert infy["symbol"] == "INFY"
    assert infy["company_name"] == "Infosys"
    assert infy["is_target"] is False
    assert infy["rank"] == 2
    assert infy["cmp"] == 1160.0
    assert infy["screener_url"] == "https://www.screener.in/company/INFY/consolidated/"


def test_parse_peer_table_html_handles_missing_or_malformed_gracefully():
    adapter = ScreenerAdapter(session=FakeSession(""), timeout=1)
    assert adapter.parse_peer_table_html("") == []
    assert adapter.parse_peer_table_html("<div>No tables here</div>") == []
    assert adapter.parse_peer_table_html("<table><tr><th>Bad</th></tr></table>") == []


def test_fetch_peers_handles_missing_warehouse_or_network_error_gracefully():
    adapter = ScreenerAdapter(session=FakeSession(""), timeout=1)
    # Empty warehouse ID
    assert adapter.fetch_peers("") == []

    # Network error
    class ErrSession(FakeSession):
        def get(self, url, timeout):
            raise RuntimeError("Peer endpoint timeout")

    err_adapter = ScreenerAdapter(session=ErrSession(""), timeout=1)
    assert err_adapter.fetch_peers("12345", target_symbol="TCS") == []


def test_refresh_persists_peers_in_sqlite_and_caches_them(isolated_database):
    session = PeerMockSession(MAIN_HTML_WITH_WAREHOUSE, PEERS_HTML)
    adapter = ScreenerAdapter(session=session, ttl_seconds=3600, timeout=1)

    result = adapter.refresh("TCS")
    assert result["status"] in ("success", "partial")
    assert result["peers"] == 3

    # Check persistence in sqlite
    cached_all = adapter.get_cached_peers("TCS")
    assert len(cached_all) == 3
    assert [p["symbol"] for p in cached_all] == ["TCS", "INFY", "HCLTECH"]

    # Exclude target
    cached_peers = adapter.get_cached_peers("TCS", include_target=False)
    assert len(cached_peers) == 2
    assert [p["symbol"] for p in cached_peers] == ["INFY", "HCLTECH"]
    assert cached_peers[0]["screener_url"] == "https://www.screener.in/company/INFY/consolidated/"

    # Test cache hit on get_or_refresh
    session.calls = 0
    hit = adapter.get_or_refresh("TCS")
    assert hit["status"] == "cache_hit"
    assert len(hit["peers"]) == 3
    assert session.calls == 0


def test_comparative_research_uses_screener_peers(isolated_database):
    from finaudit.competitors import CompetitorService
    from finaudit.company_registry import get_company_registry

    session = PeerMockSession(MAIN_HTML_WITH_WAREHOUSE, PEERS_HTML)
    adapter = ScreenerAdapter(session=session, ttl_seconds=3600, timeout=1)
    adapter.refresh("TCS")

    registry = get_company_registry()
    tcs = registry.get_by_symbol("TCS")
    assert tcs is not None

    service = CompetitorService()
    intel = service.get_peer_intelligence(tcs, max_peers=3)
    assert intel["selection_method"] == "screener_live"
    assert intel["total_peers"] >= 2
    symbols = [p["symbol"] for p in intel["all_peers"]]
    assert "INFY" in symbols
    assert "HCLTECH" in symbols
    # Verify screener_url is exposed
    infy_entry = next(p for p in intel["all_peers"] if p["symbol"] == "INFY")
    assert "screener.in" in infy_entry["screener_url"]
    assert infy_entry["cmp"] == 1160.0

