import json
import hashlib
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from finaudit import database
from finaudit.calculations import calculate_metrics, find_anomalies
from finaudit.extraction import extract_pdf
from finaudit.report_sources import (
    discover_candidates,
    download_and_verify,
)
from finaudit.schemas import Company

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update_step(run_id: str, step: str, status: str | None = None) -> None:
    with database.connect() as db:
        if status:
            db.execute(
                """UPDATE analysis_runs SET current_step=?, workflow_status=?,
                   updated_at=? WHERE run_id=?""",
                (step, status, _now(), run_id),
            )
        else:
            db.execute(
                "UPDATE analysis_runs SET current_step=?, updated_at=? WHERE run_id=?",
                (step, _now(), run_id),
            )


def upload_report(
    company_id: str,
    company_name: str,
    content: bytes,
    statement_basis: str,
    source_url: str | None = None,
    source_page_url: str | None = None,
    storage_path: str | Path | None = None,
    report_title: str | None = None,
    publication_date: str | None = None,
    source_kind: str = "manual_upload",
    verification_results: dict | None = None,
    financial_year_override: str | None = None,
) -> dict:
    database.initialize()
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Report exceeds the 25 MB upload limit.")
    if not content.startswith(b"%PDF-"):
        raise ValueError("Upload must be a valid PDF document.")
    normalized_id = re.sub(r"[^A-Za-z0-9_-]", "", company_id.upper())
    if not normalized_id or len(normalized_id) > 30:
        raise ValueError("Company identifier must be 1-30 letters or digits.")
    safe_name = re.sub(r"\s+", " ", company_name).strip()
    if not safe_name or len(safe_name) > 160:
        raise ValueError("Company name must be between 1 and 160 characters.")
    company = database.get_company(normalized_id)
    if not company:
        company = Company(
            company_id=normalized_id,
            name=safe_name,
            symbol=normalized_id,
            metadata_source="User-provided during report upload",
        )
        database.upsert_company(company)

    checksum = hashlib.sha256(content).hexdigest()
    with database.connect() as db:
        duplicate = db.execute(
            """SELECT * FROM reports WHERE company_id=? AND checksum=?
               AND statement_basis=?""",
            (normalized_id, checksum, statement_basis),
        ).fetchone()
    if duplicate and not verification_results:
        return dict(duplicate)

    local_path = Path(storage_path) if storage_path else database.UPLOAD_DIR / f"{checksum}.pdf"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not local_path.exists():
        local_path.write_bytes(content)
    extraction = extract_pdf(local_path, statement_basis)
    report_id = str(uuid.uuid4())
    now = _now()
    status = "extracted" if extraction["metrics"] else "needs_review"
    with database.connect() as db:
        duplicate = db.execute(
            """SELECT report_id FROM reports WHERE company_id=? AND checksum=?
               AND statement_basis=?""",
            (normalized_id, checksum, statement_basis),
        ).fetchone()
        if duplicate:
            report_id = duplicate["report_id"]
            db.execute(
                """UPDATE reports SET financial_year=?, source_url=COALESCE(?, source_url),
                   local_path=?, report_title=COALESCE(?, report_title),
                   publication_date=COALESCE(?, publication_date),
                   source_page=COALESCE(?, source_page), source_kind=?,
                   download_status=?, verification_json=?,
                   retrieval_timestamp=?
                   WHERE report_id=?""",
                (
                    financial_year_override or extraction["financial_years"][0],
                    source_url,
                    str(local_path),
                    report_title,
                    publication_date,
                    source_page_url,
                    source_kind,
                    "verified" if verification_results else "manual_upload",
                    json.dumps(verification_results or {}),
                    now,
                    report_id,
                ),
            )
            # Existing extracted metric rows remain linked to this same checksum.
            duplicate_report = dict(
                db.execute(
                    "SELECT * FROM reports WHERE report_id=?", (report_id,)
                ).fetchone()
            )
            return duplicate_report
        report_id = str(uuid.uuid4())
        report_year = financial_year_override or extraction["financial_years"][0]
        db.execute(
            """INSERT INTO reports
               (report_id, company_id, financial_year, statement_basis, source_url,
                local_path, checksum, retrieval_timestamp, extraction_status,
                report_title, publication_date, source_page, source_kind, download_status,
                verification_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                report_id,
                normalized_id,
                report_year,
                statement_basis,
                source_url,
                str(local_path),
                checksum,
                now,
                status,
                report_title,
                publication_date,
                source_page_url,
                source_kind,
                "verified" if verification_results else "manual_upload",
                json.dumps(verification_results or {}),
            ),
        )
        db.executemany(
            """INSERT INTO financial_metrics
               (report_id, company_id, financial_year, statement_basis, metric_key,
                original_label, value, unit, currency, pdf_page_number, source_excerpt,
                extraction_method, extraction_confidence, verification_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    report_id,
                    normalized_id,
                    row["financial_year"],
                    statement_basis,
                    row["metric_key"],
                    row["original_label"],
                    row["value"],
                    row["unit"],
                    row["currency"],
                    row["pdf_page_number"],
                    row["source_excerpt"],
                    row["extraction_method"],
                    row["extraction_confidence"],
                    row["verification_status"],
                )
                for row in extraction["metrics"]
            ],
        )
    return {
        "report_id": report_id,
        "company_id": normalized_id,
        "financial_year": report_year,
        "statement_basis": statement_basis,
        "source_url": source_url,
        "local_path": str(local_path),
        "checksum": checksum,
        "retrieval_timestamp": now,
        "extraction_status": status,
        "extracted_metrics": len(extraction["metrics"]),
        "financial_years": extraction["financial_years"],
        "warnings": extraction["warnings"],
        "report_title": report_title,
        "publication_date": publication_date,
        "source_page": source_page_url,
        "source_kind": source_kind,
        "download_status": "verified" if verification_results else "manual_upload",
        "verification_results": verification_results or {},
    }


def create_report_discovery(
    company_id: str, financial_year: str, statement_basis: str
) -> str:
    database.initialize()
    company = database.get_company(company_id.upper())
    if not company:
        raise ValueError(f"Unknown company identifier: {company_id}")
    if statement_basis not in {"consolidated", "standalone"}:
        raise ValueError("Statement basis must be consolidated or standalone.")
    from finaudit.report_sources import _normalize_fy

    normalized_year = _normalize_fy(financial_year)
    discovery_id = str(uuid.uuid4())
    database.save_discovery(
        discovery_id,
        company.company_id,
        normalized_year,
        statement_basis,
        "queued",
        [],
        [],
        _now(),
    )
    return discovery_id


def execute_report_discovery(discovery_id: str, search_provider=None) -> None:
    discovery = database.get_discovery(discovery_id)
    if not discovery:
        raise ValueError(f"Unknown discovery ID: {discovery_id}")
    database.update_discovery(discovery_id, "searching", [], [])
    company = database.get_company(discovery["company_id"])
    if not company:
        database.update_discovery(
            discovery_id, "failed", [], ["Selected company is no longer available."]
        )
        return
    try:
        candidates, errors = discover_candidates(
            company,
            discovery["financial_year"],
            discovery["statement_basis"],
            search_provider=search_provider,
        )
        for candidate in candidates:
            existing = database.get_verified_report(
                company.company_id,
                discovery["financial_year"],
                discovery["statement_basis"],
            )
            if existing:
                candidate["download_status"] = "verified_reuse_available"
                candidate["report_id"] = existing["report_id"]
                candidate["local_path"] = existing["local_path"]
                candidate["verification_results"] = json.loads(
                    existing["verification_json"] or "{}"
                )
        database.update_discovery(
            discovery_id,
            "completed" if candidates else "not_found",
            candidates,
            errors,
        )
    except Exception as exc:
        logger.exception("Report discovery %s failed", discovery_id)
        database.update_discovery(
            discovery_id,
            "failed",
            [],
            [f"Report discovery failed ({type(exc).__name__}): {exc}"],
        )


def download_discovered_report(discovery_id: str, candidate_id: str) -> dict:
    discovery = database.get_discovery(discovery_id)
    if not discovery:
        raise ValueError(f"Unknown discovery ID: {discovery_id}")
    candidates = discovery["candidates"]
    candidate = next(
        (item for item in candidates if item["candidate_id"] == candidate_id),
        None,
    )
    if not candidate:
        raise ValueError(f"Unknown report candidate: {candidate_id}")
    if candidate.get("download_status") == "downloading":
        raise ValueError("This report candidate is already downloading.")
    company = database.get_company(discovery["company_id"])
    if not company:
        raise ValueError("Selected company is no longer available.")

    candidate["download_status"] = "downloading"
    candidate["error"] = None
    database.update_discovery(discovery_id, "downloading", candidates, discovery["errors"])
    target_year = discovery["financial_year"]
    basis = discovery["statement_basis"]
    try:
        reusable = database.get_verified_report(company.company_id, target_year, basis)
        if reusable:
            candidate["download_status"] = "reused"
            candidate["report_id"] = reusable["report_id"]
            candidate["local_path"] = reusable["local_path"]
            candidate["verification_results"] = json.loads(
                reusable["verification_json"] or "{}"
            )
            database.update_discovery(discovery_id, "completed", candidates, discovery["errors"])
            return reusable

        verified = download_and_verify(
            candidate,
            company,
            target_year,
            basis,
        )
        if verified["reused_report"]:
            report = verified["reused_report"]
        else:
            try:
                report = upload_report(
                    company.company_id,
                    company.name,
                    Path(verified["local_path"]).read_bytes(),
                    basis,
                    source_url=verified["final_url"],
                    source_page_url=candidate.get("source_page"),
                    storage_path=verified["local_path"],
                    report_title=candidate["title"],
                    publication_date=verified.get("publication_date"),
                    source_kind=candidate["source_kind"],
                    verification_results=verified["verification_results"],
                    financial_year_override=target_year,
                )
            except Exception:
                Path(verified["local_path"]).unlink(missing_ok=True)
                raise
        candidate["download_status"] = "verified"
        candidate["report_id"] = report["report_id"]
        candidate["local_path"] = report["local_path"]
        candidate["verification_results"] = verified["verification_results"]
        candidate["checksum"] = report["checksum"]
        candidate["downloaded_at"] = _now()
        database.update_discovery(discovery_id, "completed", candidates, discovery["errors"])
        return report
    except Exception as exc:
        candidate["download_status"] = "failed"
        candidate["error"] = str(exc)
        database.update_discovery(discovery_id, "ready", candidates, discovery["errors"])
        logger.warning(
            "Report download failed for discovery %s candidate %s (%s)",
            discovery_id,
            candidate_id,
            type(exc).__name__,
        )
        raise


def get_report_discovery(discovery_id: str) -> dict | None:
    return database.get_discovery(discovery_id)


def _load_report_data(report_id: str) -> tuple[dict, list[dict]]:
    report = database.get_report(report_id)
    if not report:
        raise ValueError(f"Unknown report ID: {report_id}")
    with database.connect() as db:
        rows = db.execute(
            """SELECT * FROM financial_metrics WHERE report_id=?
               ORDER BY financial_year DESC, metric_key""",
            (report_id,),
        ).fetchall()
    return report, [dict(row) for row in rows]


def create_analysis(
    company_id: str,
    report_ids: list[str],
    competitor_report_ids: list[str] | None = None,
    analysis_mode: str = "single-company",
    financial_years: list[str] | None = None,
) -> str:
    database.initialize()
    if not report_ids:
        raise ValueError("Select at least one uploaded report.")
    competitor_report_ids = competitor_report_ids or []
    if analysis_mode == "competitor-comparison" and not competitor_report_ids:
        raise ValueError(
            "Competitor comparison requires at least one competitor report."
        )
    if financial_years and (
        len(set(financial_years)) != len(financial_years)
        or any(not re.fullmatch(r"FY20\d{2}", year) for year in financial_years)
    ):
        raise ValueError("Financial years must be unique values such as FY2026.")
    run_id = str(uuid.uuid4())
    now = _now()
    with database.connect() as db:
        db.execute(
            """INSERT INTO analysis_runs
               (run_id, selected_company, selected_competitors, selected_years,
                analysis_mode, workflow_status, current_step, results,
                creation_timestamp, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                company_id,
                json.dumps(competitor_report_ids),
                json.dumps(financial_years or []),
                analysis_mode,
                "queued",
                "Queued",
                "{}",
                now,
                now,
            ),
        )
    return run_id


def execute_analysis(
    run_id: str,
    report_ids: list[str],
    competitor_report_ids: list[str],
    financial_years: list[str] | None = None,
) -> None:
    database.initialize()
    with database.connect() as db:
        run = db.execute(
            "SELECT * FROM analysis_runs WHERE run_id=?", (run_id,)
        ).fetchone()
    if not run:
        raise ValueError(f"Unknown analysis ID: {run_id}")
    try:
        _update_step(run_id, "Loading extracted metrics", "running")
        report_data = [_load_report_data(report_id) for report_id in report_ids]
        selected_company = run["selected_company"]
        if any(report["company_id"] != selected_company for report, _ in report_data):
            raise ValueError("Selected company reports do not match the analysis company.")
        if len({report["statement_basis"] for report, _ in report_data}) != 1:
            raise ValueError("Reports with different statement bases cannot be combined.")

        requested_years = set(financial_years or [])
        all_company_results = []
        all_calculations = []
        all_findings = []
        selected_years = set()
        _update_step(run_id, "Calculating verified ratios")
        primary_rows = {}
        for report, rows in report_data:
            for row in rows:
                row["source_report_id"] = row["report_id"]
                primary_rows.setdefault(
                    (row["financial_year"], row["metric_key"]), row
                )
        available_years = {year for year, _ in primary_rows}
        if requested_years and not requested_years.issubset(available_years):
            missing = ", ".join(sorted(requested_years - available_years))
            raise ValueError(f"Selected financial year(s) are not available: {missing}.")
        selected_years.update(requested_years or available_years)
        calculation_years = requested_years or available_years
        context_years = calculation_years | {
            f"FY{int(year[2:]) - 1}" for year in calculation_years
        }
        primary_rows = [
            row for row in primary_rows.values()
            if row["financial_year"] in context_years
        ]
        if len({row["unit"] for row in primary_rows}) > 1:
            raise ValueError(
                "Reports with different units cannot be combined; upload comparable reports."
            )
        if primary_rows:
            report = report_data[0][0]
            calculations = [
                row for row in calculate_metrics(primary_rows, report["company_id"])
                if row["financial_year"] in calculation_years
            ]
            findings = find_anomalies(primary_rows, calculations, report["company_id"])
            all_company_results.append(
                {
                    "company_id": report["company_id"],
                    "report_id": report["report_id"],
                    "statement_basis": report["statement_basis"],
                    "source_url": report["source_url"],
                    "source_page": report.get("source_page"),
                    "report_title": report.get("report_title"),
                    "publication_date": report.get("publication_date"),
                    "source_kind": report.get("source_kind"),
                    "checksum": report["checksum"],
                    "retrieval_timestamp": report["retrieval_timestamp"],
                    "metrics": primary_rows,
                    "calculated_metrics": calculations,
                    "findings": findings,
                }
            )
            all_calculations.extend(calculations)
            all_findings.extend(findings)

        comparison_warnings = []
        primary_units = {
            (row["financial_year"], row["metric_key"]): (
                row["unit"],
                row["currency"],
            )
            for row in primary_rows
        }
        for report_id in competitor_report_ids:
            competitor_report, competitor_rows = _load_report_data(report_id)
            if competitor_report["company_id"] == selected_company:
                raise ValueError("A competitor report must belong to a different company.")
            if competitor_report["statement_basis"] != report_data[0][0]["statement_basis"]:
                raise ValueError("Competitor statement basis must match the selected company.")
            competitor_calculations = calculate_metrics(
                competitor_rows, competitor_report["company_id"]
            )
            competitor_rows = [
                row for row in competitor_rows
                if row["financial_year"] in context_years
            ]
            competitor_calculations = [
                row for row in competitor_calculations
                if row["financial_year"] in calculation_years
            ]
            for row in competitor_rows:
                row["source_report_id"] = row["report_id"]
                primary_unit, primary_currency = primary_units.get(
                    (row["financial_year"], row["metric_key"]), (None, None)
                )
                if primary_unit and (
                    row["unit"] != primary_unit
                    or row["currency"] != primary_currency
                    or row["currency"] == "unknown"
                ):
                    comparison_warnings.append(
                        f"{row['metric_key']} for {competitor_report['company_id']} "
                        f"{row['financial_year']} has incompatible or unverified units/currency."
                    )
            all_company_results.append(
                {
                    "company_id": competitor_report["company_id"],
                    "report_id": competitor_report["report_id"],
                    "statement_basis": competitor_report["statement_basis"],
                    "source_url": competitor_report["source_url"],
                    "source_page": competitor_report.get("source_page"),
                    "report_title": competitor_report.get("report_title"),
                    "publication_date": competitor_report.get("publication_date"),
                    "source_kind": competitor_report.get("source_kind"),
                    "checksum": competitor_report["checksum"],
                    "retrieval_timestamp": competitor_report["retrieval_timestamp"],
                    "metrics": competitor_rows,
                    "calculated_metrics": competitor_calculations,
                    "findings": [],
                }
            )
            all_calculations.extend(competitor_calculations)

        _update_step(run_id, "Checking consistency and evidence references")
        comparison_rows = []
        if competitor_report_ids:
            for company_result in all_company_results:
                units = {row["unit"] for row in company_result["metrics"]}
                if len(units) != 1 or "reported units (unverified)" in units:
                    comparison_warnings.append(
                        f"{company_result['company_id']} has multiple or unknown units; confirm comparability."
                    )
            selected_metrics = {
                (row["financial_year"], row["metric_key"]): row
                for row in all_company_results[0]["metrics"]
            }
            for competitor in all_company_results[1:]:
                for metric in competitor["metrics"]:
                    if metric["financial_year"] not in calculation_years:
                        continue
                    selected = selected_metrics.get(
                        (metric["financial_year"], metric["metric_key"])
                    )
                    if not selected or selected["value"] is None or metric["value"] is None:
                        continue
                    if (
                        selected["unit"] != metric["unit"]
                        or selected["currency"] != metric["currency"]
                        or selected["currency"] == "unknown"
                    ):
                        continue
                    difference = metric["value"] - selected["value"]
                    comparison_rows.append(
                        {
                            "financial_year": metric["financial_year"],
                            "metric_key": metric["metric_key"],
                            "selected_company": all_company_results[0]["company_id"],
                            "selected_value": selected["value"],
                            "competitor_company": competitor["company_id"],
                            "competitor_value": metric["value"],
                            "absolute_difference": round(difference, 4),
                            "percentage_difference": (
                                round(difference / abs(selected["value"]) * 100, 4)
                                if selected["value"] != 0
                                else None
                            ),
                            "unit": metric["unit"],
                        }
                    )
            year_sets = {
                result["company_id"]: {
                    metric["financial_year"] for metric in result["metrics"]
                }
                for result in all_company_results
            }
            common_years = set.intersection(*year_sets.values()) if year_sets else set()
            if not common_years:
                comparison_warnings.append(
                    "No overlapping reporting years; side-by-side comparisons are not available."
                )
            elif not comparison_rows and not any(
                "incompatible or unverified" in warning
                for warning in comparison_warnings
            ):
                comparison_warnings.append(
                    "No matching supported metric values were available for comparison."
                )

        result = {
            "workflow": [
                "Company and report resolution",
                "Report extraction and provenance",
                "Deterministic metric calculations",
                "Anomaly rule review",
                "Numerical and citation checks",
                "Evidence-backed report generation",
            ],
            "companies": all_company_results,
            "comparison_rows": comparison_rows,
            "comparison_warnings": comparison_warnings,
            "data_quality_warnings": [
                "Extracted values are unverified until checked against the cited PDF page.",
                "This analytical tool is not a statutory audit and cannot determine fraud.",
            ],
            "llm_status": "Not configured; deterministic analysis is complete.",
        }
        from finaudit.llm import configured_provider_name

        if configured_provider_name():
            _update_step(run_id, "Generating a bounded evidence-based interpretation")
            try:
                from finaudit.llm import GeminiProvider, OpenAIProvider, configured_provider_name

                provider_class = (
                    GeminiProvider
                    if configured_provider_name() == "Gemini"
                    else OpenAIProvider
                )
                result["llm_interpretation"] = provider_class().interpret(
                    [row for company in all_company_results for row in company["metrics"]],
                    all_findings,
                )
                result["llm_status"] = "Interpretation generated; numerical results remain deterministic."
            except Exception as exc:
                logger.error(
                    "LLM interpretation failed for run %s (%s)",
                    run_id,
                    type(exc).__name__,
                )
                result["llm_status"] = (
                    f"Interpretation failed ({type(exc).__name__}); "
                    "deterministic results are still available."
                )
                result["data_quality_warnings"].append(
                    "Optional AI interpretation failed; review server logs. Deterministic calculations were retained."
                )
        verification_issues = verify_result(result)
        result["verification_issues"] = verification_issues
        if verification_issues:
            result["data_quality_warnings"].extend(verification_issues)
        with database.connect() as db:
            db.execute("DELETE FROM calculated_metrics WHERE run_id=?", (run_id,))
            db.execute("DELETE FROM findings WHERE run_id=?", (run_id,))
            db.executemany(
                """INSERT INTO calculated_metrics
                   (run_id, company_id, financial_year, metric_key, value,
                    formula, input_metric_keys, validation_status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        run_id,
                        row["company_id"],
                        row["financial_year"],
                        row["metric_key"],
                        row["value"],
                        row["formula"],
                        json.dumps(row["input_metric_keys"]),
                        row["validation_status"],
                    )
                    for row in all_calculations
                ],
            )
            db.executemany(
                """INSERT INTO findings
                   (finding_id, run_id, title, severity, period, explanation,
                    supporting_metrics, evidence_references, counterevidence,
                    confidence, follow_up_questions)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        item["finding_id"],
                        run_id,
                        item["title"],
                        item["severity"],
                        item["period"],
                        item["explanation"],
                        json.dumps(item["supporting_metrics"]),
                        json.dumps(item["evidence_references"]),
                        item["counterevidence"],
                        item["confidence"],
                        json.dumps(item["follow_up_questions"]),
                    )
                    for item in all_findings
                ],
            )
            db.execute(
                """UPDATE analysis_runs SET workflow_status='completed',
                   current_step='Complete', selected_years=?, results=?, error=NULL,
                   updated_at=? WHERE run_id=?""",
                (
                    json.dumps(sorted(selected_years, reverse=True)),
                    json.dumps(result),
                    _now(),
                    run_id,
                ),
            )
    except Exception as exc:
        logger.error(
            "Analysis %s failed (%s): %s",
            run_id,
            type(exc).__name__,
            exc,
        )
        with database.connect() as db:
            db.execute(
                """UPDATE analysis_runs SET workflow_status='failed',
                   current_step='Failed', error=?, updated_at=? WHERE run_id=?""",
                (str(exc), _now(), run_id),
            )


def get_analysis(run_id: str) -> dict | None:
    database.initialize()
    with database.connect() as db:
        row = db.execute(
            "SELECT * FROM analysis_runs WHERE run_id=?", (run_id,)
        ).fetchone()
    if not row:
        return None
    result = dict(row)
    result["selected_competitors"] = json.loads(result["selected_competitors"])
    result["selected_years"] = json.loads(result["selected_years"])
    result["results"] = json.loads(result["results"] or "{}")
    return result


def get_findings(run_id: str) -> list[dict]:
    database.initialize()
    with database.connect() as db:
        rows = db.execute(
            "SELECT * FROM findings WHERE run_id=? ORDER BY severity DESC",
            (run_id,),
        ).fetchall()
    return [
        {
            **dict(row),
            "supporting_metrics": json.loads(row["supporting_metrics"]),
            "evidence_references": json.loads(row["evidence_references"]),
            "follow_up_questions": json.loads(row["follow_up_questions"]),
        }
        for row in rows
    ]


def list_analyses(limit: int = 20) -> list[dict]:
    database.initialize()
    with database.connect() as db:
        rows = db.execute(
            """SELECT run_id, selected_company, selected_competitors,
                      selected_years, analysis_mode, workflow_status,
                      current_step, creation_timestamp
               FROM analysis_runs ORDER BY creation_timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [
        {
            **dict(row),
            "selected_competitors": json.loads(row["selected_competitors"]),
            "selected_years": json.loads(row["selected_years"]),
        }
        for row in rows
    ]


def available_years(report_ids: list[str]) -> list[str]:
    if not report_ids:
        return []
    placeholders = ",".join("?" for _ in report_ids)
    with database.connect() as db:
        rows = db.execute(
            f"""SELECT DISTINCT financial_year FROM financial_metrics
                WHERE report_id IN ({placeholders})
                ORDER BY financial_year DESC""",
            report_ids,
        ).fetchall()
    return [row["financial_year"] for row in rows]


def verify_result(result: dict) -> list[str]:
    issues = []
    for company in result.get("companies", []):
        rows = {
            (item["financial_year"], item["metric_key"]): item
            for item in company["metrics"]
        }
        for calculation in company["calculated_metrics"]:
            value = calculation["value"]
            if value is None:
                continue
            if calculation["metric_key"] == "net_profit_margin_percent":
                profit = rows.get((calculation["financial_year"], "profit_after_tax"), {}).get("value")
                revenue = rows.get((calculation["financial_year"], "revenue"), {}).get("value")
                if profit is None or revenue in (None, 0) or abs(value - profit / revenue * 100) > 0.01:
                    issues.append(f"Profit margin verification failed for {company['company_id']} {calculation['financial_year']}.")
            if calculation["metric_key"] in {"revenue_growth_percent", "profit_growth_percent"}:
                source_metric = (
                    "revenue" if calculation["metric_key"] == "revenue_growth_percent"
                    else "profit_after_tax"
                )
                year = calculation["financial_year"]
                prior_year = f"FY{int(year[2:]) - 1}"
                current = rows.get((year, source_metric), {}).get("value")
                previous = rows.get((prior_year, source_metric), {}).get("value")
                expected = (
                    (current - previous) / abs(previous) * 100
                    if current is not None and previous not in (None, 0)
                    else None
                )
                if expected is None or abs(value - expected) > 0.01:
                    issues.append(f"{source_metric} growth verification failed for {company['company_id']} {year}.")
            for metric in company["metrics"]:
                if metric["metric_key"] not in calculation["input_metric_keys"]:
                    continue
                if not metric["source_excerpt"] or metric["pdf_page_number"] < 1:
                    issues.append(f"Missing citation for {metric['metric_key']} on {company['company_id']}.")
        years = sorted(
            {year for year, _ in rows},
            key=lambda year: int(year[2:]),
            reverse=True,
        )
        if years:
            year = years[0]
            operating = rows.get((year, "operating_cash_flow"), {}).get("value")
            investing = rows.get((year, "investing_cash_flow"), {}).get("value")
            financing = rows.get((year, "financing_cash_flow"), {}).get("value")
            fx = rows.get((year, "fx_effect"), {}).get("value")
            reported_change = rows.get((year, "net_change_cash"), {}).get("value")
            if all(item is not None for item in (operating, investing, financing)):
                calculated_change = operating + investing + financing
                if reported_change is not None and abs(calculated_change - reported_change) >= 0.01:
                    issues.append(
                        f"Cash-flow activity reconciliation failed for {company['company_id']} {year}: "
                        f"difference {calculated_change - reported_change:.2f}."
                    )
            opening = rows.get((year, "opening_cash"), {}).get("value")
            closing = rows.get((year, "cash_and_cash_equivalents"), {}).get("value")
            if all(item is not None for item in (opening, closing, reported_change, fx)):
                cash_movement = reported_change + fx
                cash_difference = cash_movement - (closing - opening)
                if abs(cash_difference) >= 0.01:
                    issues.append(
                        f"Cash balance reconciliation failed for {company['company_id']} {year}: "
                        f"difference {cash_difference:.2f}."
                    )
            elif all(item is not None for item in (operating, investing, financing, fx, opening, closing)):
                cash_movement = operating + investing + financing + fx
                cash_difference = cash_movement - (closing - opening)
                if abs(cash_difference) >= 0.01:
                    issues.append(
                        f"Cash balance reconciliation failed for {company['company_id']} {year}: "
                        f"difference {cash_difference:.2f}."
                    )
    return issues
