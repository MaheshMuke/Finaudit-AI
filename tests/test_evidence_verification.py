from finaudit.evidence import evidence_from_metric
from finaudit.verification import (
    verify_calculated_metric,
    verify_evidence_references,
    verify_investigation_output,
)


def test_evidence_identifier_is_stable_and_serializable():
    record = evidence_from_metric(
        "ABC",
        "revenue",
        100.0,
        unit="INR crore",
        currency="INR",
        reporting_period="2025-03-31",
        statement_basis="annual",
        source="local statements",
    )

    assert record.evidence_id == record.evidence_id
    assert record.to_dict()["evidence_id"].startswith("ev_")
    assert record.to_dict()["reporting_period"] == "2025-03-31"


def test_evidence_preserves_period_and_extraction_provenance():
    record = evidence_from_metric(
        "ABC",
        "revenue",
        100.0,
        unit="INR crore",
        currency="INR",
        reporting_period="2025-03-31",
        statement_basis="quarterly",
        reporting_basis="consolidated",
        source="Screener",
        source_url="https://www.screener.in/company/ABC/",
        period_end="2025-03-31",
        period_type="quarterly",
        extraction_method="normalized statement table",
        original_value="₹100 crore",
        original_unit="INR_crore",
    )

    serialized = record.to_dict()
    assert serialized["period_end"] == "2025-03-31"
    assert serialized["period_type"] == "quarterly"
    assert serialized["original_value"] == "₹100 crore"
    assert serialized["source_url"].startswith("https://")


def test_verifier_flags_mixed_basis_calculations_and_unsubstantiated_causal_claims():
    evidence = [
        {
            "evidence_id": "rev-current",
            "metric": "revenue",
            "value": 200,
            "unit": "INR crore",
            "currency": "INR",
            "reporting_period": "2025-03-31",
            "statement_basis": "annual",
            "reporting_basis": "consolidated",
            "source_url": "https://source.test/report",
        },
        {
            "evidence_id": "rev-prior",
            "metric": "revenue",
            "value": 100,
            "unit": "INR crore",
            "currency": "INR",
            "reporting_period": "2024-03-31",
            "statement_basis": "annual",
            "reporting_basis": "standalone",
            "source_url": "https://source.test/report",
        },
    ]
    report = {
        "metrics": {},
        "evidence": evidence,
        "evidence_references": ["rev-current", "rev-prior"],
        "research_report": {
            "claims": [
                {
                    "claim_id": "revenue-growth",
                    "claim_type": "deterministic_calculation",
                    "status": "inferred",
                    "value": 100,
                    "current_period": "2025-03-31",
                    "comparison_period": "2024-03-31",
                    "supporting_evidence_ids": ["rev-current", "rev-prior"],
                },
                {
                    "claim_id": "cause",
                    "claim_type": "causal_explanation",
                    "status": "verified",
                    "source_verification_status": "verified",
                    "supporting_evidence_ids": ["rev-current"],
                },
            ],
        },
    }

    warnings = verify_investigation_output(report)

    assert any("mixes incompatible reporting bases" in warning for warning in warnings)
    assert any("lacks dated primary-source evidence" in warning for warning in warnings)


def test_formula_verification_rejects_inconsistent_value():
    valid, reason = verify_calculated_metric(
        25.0,
        "profit_after_tax / revenue * 100",
        {"profit_after_tax": 25.0, "revenue": 100.0},
    )
    assert valid is True
    assert reason is None

    valid, reason = verify_calculated_metric(
        30.0,
        "profit_after_tax / revenue * 100",
        {"profit_after_tax": 25.0, "revenue": 100.0},
    )
    assert valid is False
    assert reason == "value does not match formula inputs"


def test_missing_growth_is_explicitly_unavailable():
    valid, reason = verify_calculated_metric(
        None,
        "profit_after_tax / revenue * 100",
        {"profit_after_tax": 10.0, "revenue": 0.0},
    )
    assert valid is False
    assert reason == "calculation is unavailable"


def test_evidence_references_must_resolve():
    assert verify_evidence_references(
        ["ev_present", "ev_missing"], [{"evidence_id": "ev_present"}]
    ) == ["Evidence identifier does not resolve: ev_missing"]
