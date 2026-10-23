import json
from pathlib import Path
import uuid

import pytest

from finaudit import database
from finaudit.calculations import calculate_metrics, find_anomalies
from finaudit.extraction import _unit_scale, extract_pdf, parse_number
from finaudit.reporting import build_report
from finaudit.service import (
    create_analysis,
    execute_analysis,
    get_analysis,
    upload_report,
    verify_result,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PDF = ROOT / "finaudit_test" / "infosys_fy2025_26.pdf"


@pytest.fixture
def isolated_database(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(database, "UPLOAD_DIR", tmp_path / "data" / "uploads")
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "data" / "test.sqlite3")
    database.initialize()
    return tmp_path


def test_parse_number_supports_indian_commas_and_missing_values():
    assert parse_number("1,23,456.75") == 123456.75
    assert parse_number("(2,450)") == -2450
    assert parse_number("-") is None
    assert parse_number("—") is None
    assert parse_number("0") == 0


def test_units_require_currency_evidence_for_comparison():
    assert _unit_scale("(In ` crore)") == ("INR crore", 1.0, "INR")
    assert _unit_scale("(USD in millions)") == ("USD crore", 0.1, "USD")
    assert _unit_scale("(amounts in millions)") == (
        "reported units (unverified)",
        1.0,
        "unknown",
    )


def test_extracts_sample_values_with_dynamic_page_provenance():
    extracted = extract_pdf(SAMPLE_PDF, "consolidated")
    current_revenue = next(
        row for row in extracted["metrics"]
        if row["metric_key"] == "revenue" and row["financial_year"] == "FY2026"
    )
    assert extracted["financial_years"] == ["FY2026", "FY2025"]
    assert current_revenue["value"] == 178650
    assert current_revenue["pdf_page_number"] == 288
    assert "Revenue from operations" in current_revenue["source_excerpt"]


def test_calculations_handle_missing_values_and_zero_denominators():
    rows = [
        {"financial_year": "FY2026", "metric_key": "revenue", "value": 100.0},
        {"financial_year": "FY2026", "metric_key": "profit_after_tax", "value": 0.0},
        {"financial_year": "FY2025", "metric_key": "revenue", "value": 0.0},
        {"financial_year": "FY2025", "metric_key": "profit_after_tax", "value": 25.0},
    ]
    calculations = calculate_metrics(rows, "TEST")
    current = {row["metric_key"]: row for row in calculations if row["financial_year"] == "FY2026"}
    assert current["net_profit_margin_percent"]["value"] == 0
    assert current["profit_growth_percent"]["value"] == -100
    assert current["revenue_growth_percent"]["value"] is None
    assert current["operating_cash_flow_to_profit"]["value"] is None
    assert current["current_ratio"]["value"] is None


def test_anomaly_rules_use_explicit_thresholds():
    metrics = [
        {"financial_year": "FY2026", "metric_key": "revenue", "value": 140, "pdf_page_number": 5},
        {"financial_year": "FY2025", "metric_key": "revenue", "value": 100, "pdf_page_number": 5},
        {"financial_year": "FY2026", "metric_key": "profit_after_tax", "value": 20, "pdf_page_number": 6},
        {"financial_year": "FY2025", "metric_key": "profit_after_tax", "value": 10, "pdf_page_number": 6},
        {"financial_year": "FY2026", "metric_key": "operating_cash_flow", "value": 50, "pdf_page_number": 8},
        {"financial_year": "FY2025", "metric_key": "operating_cash_flow", "value": 100, "pdf_page_number": 8},
    ]
    calculations = calculate_metrics(metrics, "TEST")
    findings = find_anomalies(metrics, calculations, "TEST")
    assert any("Material revenue movement" in item["title"] for item in findings)
    assert any("weaker operating cash flow" in item["title"] for item in findings)


def test_sample_upload_analysis_persists_and_reconciles(isolated_database):
    content = SAMPLE_PDF.read_bytes()
    report = upload_report(
        "INFY",
        "Infosys Limited",
        content,
        "consolidated",
        "https://www.infosys.com/investors/",
    )
    assert report["extraction_status"] == "extracted"
    assert report["extracted_metrics"] >= 15
    run_id = create_analysis("INFY", [report["report_id"]], [], "single-company")
    execute_analysis(run_id, [report["report_id"]], [])
    result = get_analysis(run_id)
    assert result["workflow_status"] == "completed"
    company = result["results"]["companies"][0]
    assert not verify_result(result["results"])
    html_report = build_report(result)
    assert "PDF page 288" in html_report
    assert "Not available" in html_report
    assert "SHA-256" in html_report
    reconciliation = next(
        row for row in company["calculated_metrics"]
        if row["metric_key"] == "net_profit_margin_percent"
        and row["financial_year"] == "FY2026"
    )
    assert reconciliation["value"] == pytest.approx(16.5, abs=0.01)

    competitor_report_id = str(uuid.uuid4())
    with database.connect() as db:
        db.execute(
            """INSERT INTO reports
               (report_id, company_id, financial_year, statement_basis, source_url,
                local_path, checksum, retrieval_timestamp, extraction_status)
               SELECT ?, 'TCS', financial_year, statement_basis, source_url,
                      local_path, checksum, retrieval_timestamp, extraction_status
               FROM reports WHERE report_id=?""",
            (competitor_report_id, report["report_id"]),
        )
        db.execute(
            """INSERT INTO financial_metrics
               (report_id, company_id, financial_year, statement_basis, metric_key,
                original_label, value, unit, currency, pdf_page_number, source_excerpt,
                extraction_method, extraction_confidence, verification_status)
               SELECT ?, 'TCS', financial_year, statement_basis, metric_key,
                      original_label, value, unit, currency, pdf_page_number,
                      source_excerpt, extraction_method, extraction_confidence,
                      verification_status
               FROM financial_metrics WHERE report_id=?""",
            (competitor_report_id, report["report_id"]),
        )
    comparison_id = create_analysis(
        "INFY",
        [report["report_id"]],
        [competitor_report_id],
        "competitor-comparison",
    )
    execute_analysis(comparison_id, [report["report_id"]], [competitor_report_id])
    comparison = get_analysis(comparison_id)
    assert comparison["workflow_status"] == "completed"
    assert any(
        row["metric_key"] == "revenue"
        and row["financial_year"] == "FY2026"
        and row["absolute_difference"] == 0
        for row in comparison["results"]["comparison_rows"]
    )
    selected_year_run = create_analysis(
        "INFY", [report["report_id"]], financial_years=["FY2026"]
    )
    execute_analysis(selected_year_run, [report["report_id"]], [], ["FY2026"])
    selected_year_result = get_analysis(selected_year_run)
    assert selected_year_result["selected_years"] == ["FY2026"]
    selected_calculations = selected_year_result["results"]["companies"][0][
        "calculated_metrics"
    ]
    assert {row["financial_year"] for row in selected_calculations} == {"FY2026"}
    assert any(
        row["financial_year"] == "FY2025"
        for row in selected_year_result["results"]["companies"][0]["metrics"]
    )


def test_upload_rejects_non_pdf_and_basis_mismatch(isolated_database):
    with pytest.raises(ValueError, match="valid PDF"):
        upload_report("BAD", "Bad File", b"not pdf", "consolidated")
    standalone = extract_pdf(SAMPLE_PDF, "standalone")
    assert standalone["statement_basis"] == "standalone"
    assert any(row["metric_key"] == "revenue" for row in standalone["metrics"])


def test_sample_output_includes_page_cited_findings_and_json(isolated_database):
    report = upload_report(
        "INFY", "Infosys Limited", SAMPLE_PDF.read_bytes(), "consolidated"
    )
    run_id = create_analysis("INFY", [report["report_id"]])
    execute_analysis(run_id, [report["report_id"]], [])
    analysis = get_analysis(run_id)
    assert analysis["results"]["data_quality_warnings"]
    assert analysis["results"]["verification_issues"] == []
    assert json.loads(json.dumps(analysis["results"]))["companies"][0]["metrics"]
