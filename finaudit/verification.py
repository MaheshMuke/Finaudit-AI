"""Independent checks for deterministic financial research output."""

from __future__ import annotations

import math
from typing import Any


def _numbers_match(actual: Any, expected: float, tolerance: float = 1e-6) -> bool:
    return isinstance(actual, (int, float)) and math.isfinite(float(actual)) and math.isclose(
        float(actual), expected, rel_tol=tolerance, abs_tol=tolerance
    )


def verify_calculated_metric(
    value: Any,
    formula: str,
    inputs: dict[str, Any],
) -> tuple[bool, str | None]:
    """Verify the supported formula families used by the deterministic engine."""
    if value is None:
        return False, "calculation is unavailable"
    if formula == "profit_after_tax / revenue * 100":
        numerator, denominator = inputs.get("profit_after_tax"), inputs.get("revenue")
        expected = None if denominator in (None, 0) or numerator is None else numerator / denominator * 100
    elif formula == "operating_cash_flow / profit_after_tax":
        numerator, denominator = inputs.get("operating_cash_flow"), inputs.get("profit_after_tax")
        expected = None if denominator in (None, 0) or numerator is None else numerator / denominator
    elif formula == "total_current_assets / current_liabilities":
        numerator, denominator = inputs.get("total_current_assets"), inputs.get("current_liabilities")
        expected = None if denominator in (None, 0) or numerator is None else numerator / denominator
    else:
        return False, f"unsupported formula: {formula}"
    if expected is None:
        return False, "required input is missing or has a zero denominator"
    return (_numbers_match(value, float(expected)), None if _numbers_match(value, float(expected)) else "value does not match formula inputs")


def verify_evidence_references(
    references: list[str], evidence: list[dict[str, Any]]
) -> list[str]:
    available = {item.get("evidence_id") for item in evidence}
    return [
        f"Evidence identifier does not resolve: {reference}"
        for reference in references
        if reference not in available
    ]


def verify_investigation_output(report: dict[str, Any]) -> list[str]:
    """Check invariants independently of the LLM narrative."""
    warnings: list[str] = []
    metrics = report.get("metrics") or {}
    if metrics.get("revenue_growth_numeric") is None and metrics.get("revenue_growth") not in (None, "N/A"):
        warnings.append("Revenue growth has a display value but no numeric value.")
    if metrics.get("net_income_growth_numeric") is None and metrics.get("net_income_growth") not in (None, "N/A"):
        warnings.append("Net-income growth has a display value but no numeric value.")
    if metrics.get("reporting_period") == metrics.get("previous_reporting_period") and metrics.get("reporting_period"):
        warnings.append("Current and comparison reporting periods are identical.")
    evidence = report.get("evidence") or []
    warnings.extend(verify_evidence_references(report.get("evidence_references") or [], evidence))
    research_report = report.get("research_report") or {}
    evidence_by_id = {
        item.get("evidence_id"): item
        for item in evidence
        if item.get("evidence_id")
    }
    for claim in research_report.get("claims") or []:
        claim_id = claim.get("claim_id", "unknown")
        references = claim.get("supporting_evidence_ids") or []
        warnings.extend(
            f"Claim {claim_id}: {warning}"
            for warning in verify_evidence_references(references, evidence)
        )
        if claim.get("status") == "inferred" and not references:
            warnings.append(f"Claim {claim_id} has no supporting financial evidence.")
            continue
        if claim.get("status") == "verified" and claim.get("source_verification_status") == "unverified":
            warnings.append(f"Claim {claim_id} is marked verified but its source is unverified.")
        if claim.get("claim_type") in {"causal_explanation", "causal_claim"}:
            causal_evidence = [
                evidence_by_id[reference]
                for reference in references
                if reference in evidence_by_id
            ]
            if claim.get("status") == "verified" and not any(
                item.get("source_url")
                and (item.get("excerpt") or item.get("source_document_id"))
                and (item.get("publication_date") or item.get("reporting_period"))
                for item in causal_evidence
            ):
                warnings.append(
                    f"Claim {claim_id} lacks dated primary-source evidence for its causal explanation."
                )
            if claim.get("status") not in {"verified", "inferred", "hypothetical", "unsupported"}:
                warnings.append(f"Claim {claim_id} has an unknown causal evidence status.")
        if claim.get("claim_type") == "profit_bridge":
            facts = {
                (item.get("reporting_period"), item.get("metric")): item.get("value")
                for item in (
                    evidence_by_id[reference]
                    for reference in references
                    if reference in evidence_by_id
                )
            }
            previous_period = claim.get("comparison_period")
            current_period = claim.get("current_period")
            required = (
                "operating_profit",
                "other_income",
                "interest_expense",
                "depreciation",
                "profit_before_tax",
                "tax_rate_percent",
                "net_income",
            )
            try:
                prior = {metric: facts[(previous_period, metric)] for metric in required}
                current = {metric: facts[(current_period, metric)] for metric in required}
                contributions = claim.get("contributions") or {}

                def close(left: Any, right: Any) -> bool:
                    return (
                        isinstance(left, (int, float))
                        and isinstance(right, (int, float))
                        and math.isclose(float(left), float(right), rel_tol=0.02, abs_tol=1.0)
                    )

                reconciles = all(
                    close(
                        row["profit_before_tax"],
                        row["operating_profit"] + row["other_income"]
                        - row["interest_expense"] - row["depreciation"],
                    )
                    and close(
                        row["net_income"],
                        row["profit_before_tax"]
                        * (1 - row["tax_rate_percent"] / 100),
                    )
                    for row in (prior, current)
                )
                expected_contributions = {
                    "operating_profit": current["operating_profit"] - prior["operating_profit"],
                    "other_income": current["other_income"] - prior["other_income"],
                    "interest_expense": -(current["interest_expense"] - prior["interest_expense"]),
                    "depreciation": -(current["depreciation"] - prior["depreciation"]),
                    "tax_expense": -(
                        current["profit_before_tax"] * current["tax_rate_percent"] / 100
                        - prior["profit_before_tax"] * prior["tax_rate_percent"] / 100
                    ),
                }
                reconciles = reconciles and all(
                    close(contributions.get(name), expected)
                    for name, expected in expected_contributions.items()
                )
                reconciles = reconciles and close(
                    claim.get("value"),
                    current["net_income"] - prior["net_income"],
                ) and close(
                    sum(contributions.values()),
                    current["net_income"] - prior["net_income"],
                )
            except (KeyError, TypeError, ZeroDivisionError):
                reconciles = False
            if not reconciles:
                warnings.append(f"Claim {claim_id} does not reconcile to its cited line items.")
            continue
        if claim.get("claim_type") != "deterministic_calculation":
            continue
        inputs = [
            evidence_by_id[reference]
            for reference in references
            if reference in evidence_by_id
        ]
        bases = {
            str(item.get("reporting_basis") or "").strip().lower()
            for item in inputs
            if item.get("reporting_basis")
        }
        statement_bases = {
            str(item.get("statement_basis") or "").strip().lower()
            for item in inputs
            if item.get("statement_basis")
        }
        if len(bases) > 1 or len(statement_bases) > 1:
            warnings.append(f"Claim {claim_id} mixes incompatible reporting bases.")
            continue
        claim_periods = {
            item.get("reporting_period")
            for item in inputs
            if item.get("reporting_period")
        }
        if claim_periods - {claim.get("current_period"), claim.get("comparison_period")}:
            warnings.append(f"Claim {claim_id} cites a period outside its stated comparison.")
            continue
        facts = {
            (item.get("reporting_period"), item.get("metric")): item.get("value")
            for item in inputs
        }
        current_period = claim.get("current_period")
        comparison_period = claim.get("comparison_period")
        metric = str(claim_id).rsplit("-", 1)[-1]
        try:
            if metric == "revenue":
                current = facts[(current_period, "revenue")]
                previous = facts[(comparison_period, "revenue")]
                expected = None if previous in (None, 0) else (current - previous) / abs(previous) * 100
            elif metric == "net_income":
                current = facts[(current_period, "net_income")]
                previous = facts[(comparison_period, "net_income")]
                expected = None if previous in (None, 0) else (current - previous) / abs(previous) * 100
            elif metric == "net_margin":
                current_revenue = facts[(current_period, "revenue")]
                current_income = facts[(current_period, "net_income")]
                previous_revenue = facts[(comparison_period, "revenue")]
                previous_income = facts[(comparison_period, "net_income")]
                if current_revenue in (None, 0) or previous_revenue in (None, 0):
                    expected = None
                else:
                    expected = (
                        current_income / current_revenue
                        - previous_income / previous_revenue
                    ) * 100
            else:
                expected = None
        except (KeyError, TypeError, ZeroDivisionError):
            expected = None
        if expected is None or not _numbers_match(claim.get("value"), float(expected)):
            warnings.append(f"Claim {claim_id} does not reconcile to its cited financial facts.")
    return warnings
