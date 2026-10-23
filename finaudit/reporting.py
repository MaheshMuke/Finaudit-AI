import html
import json
from datetime import datetime, timezone

from finaudit.extraction import METRICS


def _number(value, suffix=""):
    if value is None:
        return "Not available"
    return f"{value:,.2f}{suffix}"


def build_report(analysis: dict) -> str:
    result = analysis["results"]
    company_ids = [item["company_id"] for item in result.get("companies", [])]
    years = analysis.get("selected_years", [])
    output = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>FinAudit AI analysis</title><style>",
        "body{font:16px/1.55 Segoe UI,Arial,sans-serif;max-width:1050px;margin:40px auto;padding:0 20px;color:#17212b}",
        "h1,h2{color:#12334b}table{border-collapse:collapse;width:100%;margin:16px 0 28px}",
        "th,td{border:1px solid #d8e0e6;padding:8px;text-align:left;vertical-align:top}",
        "th{background:#edf3f7}.warning{background:#fff7df;padding:12px;border-left:4px solid #d99a00}",
        ".finding{border:1px solid #d8e0e6;padding:12px;margin:12px 0}",
        "</style></head><body>",
        "<h1>FinAudit AI — Financial analysis</h1>",
        f"<p><b>Company:</b> {html.escape(', '.join(company_ids))}<br>",
        f"<b>Reporting years:</b> {html.escape(', '.join(years) or 'Not identified')}<br>",
        f"<b>Mode:</b> {html.escape(analysis['analysis_mode'])}<br>",
        f"<b>Generated:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>",
        "<h2>Executive summary</h2>",
        "<p>This report presents extracted financial-statement values, deterministic calculations, "
        "and rules-based indicators. Each factual value retains its source report page.</p>",
    ]
    for company in result.get("companies", []):
        revenue_growth = next(
            (
                item for item in company["calculated_metrics"]
                if item["metric_key"] == "revenue_growth_percent"
                and item["value"] is not None
            ),
            None,
        )
        if revenue_growth:
            revenue_row = next(
                (
                    item for item in company["metrics"]
                    if item["metric_key"] == "revenue"
                    and item["financial_year"] == revenue_growth["financial_year"]
                ),
                None,
            )
            if revenue_row:
                output.append(
                    "<p><b>{}</b>: revenue was {} {} in {}, a calculated year-over-year "
                    "change of {:.2f}% (source: PDF page {}). This is a historical "
                    "calculation, not a forecast.</p>".format(
                        html.escape(company["company_id"]),
                        _number(revenue_row["value"]),
                        html.escape(revenue_row["unit"]),
                        html.escape(revenue_growth["financial_year"]),
                        revenue_growth["value"],
                        revenue_row["pdf_page_number"],
                    )
                )
    output.extend(
        [
        "<h2>Data coverage and trends</h2>",
        ]
    )
    for company in result.get("companies", []):
        name = html.escape(company["company_id"])
        output.append(        f"<h3>{name} — {html.escape(company.get('report_title') or 'Annual report')} "
        f"({html.escape(company['statement_basis'])})</h3>")
        source_label = html.escape(
            company.get("source_url") or "User-uploaded report; no URL recorded"
        )
        output.append(
            f"<p>Report ID: {html.escape(company['report_id'])}; "
            f"SHA-256: {html.escape(company['checksum'])}; "
            f"source: {source_label}; "
            f"source page: {html.escape(company.get('source_page') or 'Not recorded')}; "
            f"publication date: {html.escape(company.get('publication_date') or 'Not available')}</p>"
        )
        output.append(
            "<table><thead><tr><th>Year</th><th>Metric</th><th>Value</th>"
            "<th>Source</th><th>Confidence</th></tr></thead><tbody>"
        )
        for metric in company["metrics"]:
            output.append(
                "<tr><td>{}</td><td>{}</td><td>{} {}</td><td>PDF page {}</td>"
                "<td>{:.0%} — unverified</td></tr>".format(
                    html.escape(metric["financial_year"]),
                    html.escape(metric["metric_key"].replace("_", " ").title()),
                    _number(metric["value"]),
                    html.escape(metric["unit"]),
                    metric["pdf_page_number"],
                    metric["extraction_confidence"],
                )
            )
        available = {
            (metric["financial_year"], metric["metric_key"])
            for metric in company["metrics"]
        }
        for year in years:
            for metric_key in METRICS:
                if (year, metric_key) not in available:
                    output.append(
                        "<tr><td>{}</td><td>{}</td><td>Not available</td>"
                        "<td>No extracted row</td><td>Not calculated</td></tr>".format(
                            html.escape(year),
                            html.escape(metric_key.replace("_", " ").title()),
                        )
                    )
        for calculated in company["calculated_metrics"]:
            output.append(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>Calculated</td>"
                "<td>{}; formula: {}</td></tr>".format(
                    html.escape(calculated["financial_year"]),
                    html.escape(calculated["metric_key"].replace("_", " ").title()),
                    _number(calculated["value"], "%" if "percent" in calculated["metric_key"] else ""),
                    html.escape(calculated["validation_status"]),
                    html.escape(calculated["formula"]),
                )
            )
        output.append("</tbody></table>")
    output.append("<h2>Risk and anomaly indicators</h2>")
    findings = [
        finding for company in result.get("companies", [])
        for finding in company.get("findings", [])
    ]
    if findings:
        for finding in findings:
            output.append(
                "<div class='finding'><b>{}</b> — {}<p>{}</p><p>Evidence: {}</p>"
                "<p>Limitations: {}</p></div>".format(
                    html.escape(finding["title"]),
                    html.escape(finding["severity"].upper()),
                    html.escape(finding["explanation"]),
                    html.escape(", ".join(finding["evidence_references"])),
                    html.escape(finding["counterevidence"]),
                )
            )
    else:
        output.append("<p>No configured rule crossed its review threshold. This is not evidence that no risks exist.</p>")
    comparison_rows = result.get("comparison_rows", [])
    if comparison_rows:
        output.append(
            "<h2>Competitor comparison</h2><table><thead><tr><th>Year</th>"
            "<th>Metric</th><th>Selected company</th><th>Competitor</th>"
            "<th>Absolute difference</th><th>Difference (%)</th></tr></thead><tbody>"
        )
        for row in comparison_rows:
            output.append(
                "<tr><td>{}</td><td>{}</td><td>{}: {} {}</td>"
                "<td>{}: {} {}</td><td>{} {}</td><td>{}</td></tr>".format(
                    html.escape(row["financial_year"]),
                    html.escape(row["metric_key"].replace("_", " ").title()),
                    html.escape(row["selected_company"]),
                    _number(row["selected_value"]),
                    html.escape(row["unit"]),
                    html.escape(row["competitor_company"]),
                    _number(row["competitor_value"]),
                    html.escape(row["unit"]),
                    _number(row["absolute_difference"]),
                    html.escape(row["unit"]),
                    _number(row["percentage_difference"], "%"),
                )
            )
        output.append("</tbody></table>")

    interpretation = result.get("llm_interpretation")
    if interpretation:
        output.extend(
            [
                "<h2>AI interpretation (hypotheses)</h2><p>",
                html.escape(interpretation).replace("\n", "<br>"),
                "</p>",
            ]
        )
    if result.get("comparison_warnings"):
        output.append("<h2>Comparability warnings</h2><ul>")
        output.extend(
            f"<li>{html.escape(warning)}</li>"
            for warning in result["comparison_warnings"]
        )
        output.append("</ul>")
    output.append("<h2>Data quality and limitations</h2><ul>")
    output.extend(
        f"<li>{html.escape(warning)}</li>"
        for warning in result.get("data_quality_warnings", [])
    )
    output.extend(
        [
            "<li>Automated extraction may misread unusual table layouts; verify source lines and page references.</li>",
            "<li>Ratios use only available values. Missing or zero denominators are shown as unavailable.</li>",
            "<li>Comparisons require consistent reporting years, statement basis, currencies, units, and metric definitions.</li>",
            "<li>Indicators are prompts for investigation, not proof of wrongdoing or a certified statutory audit.</li>",
            "</ul><h2>Methodology</h2><p>PDF statement headings and labeled rows are identified dynamically. "
            "Financial ratios and anomaly thresholds are calculated in Python; LLM output, if configured, "
            "is interpretive only and is not authoritative for arithmetic or verification.</p>",
            "</body></html>",
        ]
    )
    return "".join(output)


def structured_export(analysis: dict) -> str:
    return json.dumps(analysis, indent=2, ensure_ascii=False)
