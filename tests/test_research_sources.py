from finaudit.company_registry import CompanyRecord
from finaudit.research_sources import (
    OfficialAnnualReportAdapter,
    OfficialQuarterlyDisclosureAdapter,
    financial_year_for_period,
)


def company():
    return CompanyRecord(
        symbol="NTPC",
        company_name="NTPC Limited",
        industry="Power",
        yahoo_ticker="NTPC.NS",
    )


def test_financial_year_uses_indian_april_to_march_year():
    assert financial_year_for_period("2026-03-31") == "FY2026"
    assert financial_year_for_period("2026-06-30") == "FY2027"
    assert financial_year_for_period("TTM") is None


def test_official_adapter_returns_candidates_without_claiming_content_review():
    adapter = OfficialAnnualReportAdapter(
        discover_candidates_fn=lambda company, year, basis: (
            [{
                "title": f"{company.name} annual report {year}",
                "source_kind": "nse_corporate_disclosure",
                "source_page": "https://www.nseindia.com/filings",
                "pdf_url": "https://ntpc.co.in/investors/report.pdf",
                "publication_date": "2026-05-01",
                "financial_year": year,
                "statement_basis": basis,
            }],
            [],
        )
    )

    result = adapter.discover(company(), "FY2026", "consolidated")

    assert result["status"] == "candidates_found"
    assert result["documents"][0]["url"].endswith("report.pdf")
    assert result["documents"][0]["content_retrieved"] is False
    assert any("not substitutes for quarterly" in item for item in result["limitations"])


def test_official_adapter_reports_basis_and_provider_failures():
    adapter = OfficialAnnualReportAdapter(
        discover_candidates_fn=lambda *_: (_ for _ in ()).throw(TimeoutError("timeout"))
    )

    mixed_basis = adapter.discover(company(), "FY2026", "unknown")
    timed_out = adapter.discover(company(), "FY2026", "standalone")

    assert mixed_basis["status"] == "unavailable"
    assert "single consolidated or standalone" in mixed_basis["reason"]
    assert timed_out["status"] == "unavailable"
    assert "TimeoutError" in timed_out["reason"]


def test_quarterly_disclosure_adapter_returns_only_trusted_relevant_candidates():
    adapter = OfficialQuarterlyDisclosureAdapter(
        fetch_announcements_fn=lambda company: (
            {
                "data": [
                    {
                        "desc": "Financial Results for the quarter ended June 30, 2026 (Consolidated)",
                        "attchmntFile": "https://nsearchives.nseindia.com/corporate/ntpc_results.pdf",
                        "an_dt": "15-Aug-2026",
                    },
                    {
                        "desc": "Financial Results for the quarter ended June 30, 2026 (Standalone)",
                        "attchmntFile": "https://nsearchives.nseindia.com/corporate/ntpc_standalone.pdf",
                        "an_dt": "15-Aug-2026",
                    },
                    {
                        "desc": "Financial Results quarterly",
                        "attchmntFile": "https://untrusted.test/results.pdf",
                        "an_dt": "15-Aug-2026",
                    },
                    {
                        "desc": "Annual General Meeting notice",
                        "attchmntFile": "https://nsearchives.nseindia.com/corporate/agm.pdf",
                    },
                ]
            },
            "https://www.nseindia.com/api/corporate-announcements",
            [],
        )
    )

    result = adapter.discover(company(), "2026-06-30", "consolidated")

    assert result["status"] == "candidates_found"
    assert len(result["documents"]) == 1
    assert result["documents"][0]["statement_basis"] == "consolidated"
    assert result["documents"][0]["publication_date"] == "2026-08-15"
    assert result["documents"][0]["content_retrieved"] is False
    assert any("not evidence for a causal explanation" in limitation for limitation in result["limitations"])


def test_quarterly_disclosure_adapter_reports_unavailable_without_inventing_sources():
    adapter = OfficialQuarterlyDisclosureAdapter(
        fetch_announcements_fn=lambda company: (
            None,
            "https://www.nseindia.com/api/corporate-announcements",
            ["NSE announcements returned HTTP 403."],
        )
    )

    result = adapter.discover(company(), "2026-06-30", "consolidated")

    assert result["status"] == "unavailable"
    assert result["documents"] == []
    assert result["errors"] == ["NSE announcements returned HTTP 403."]
