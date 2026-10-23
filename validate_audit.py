
import json
import re
from pathlib import Path

import pdfplumber

PDF_PATH = Path("finaudit_test/infosys_fy2025_26.pdf")
JSON_PATH = Path("finaudit_output/financial_audit.json")

NUMBER_PATTERN = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?|[–—-]")


def parse_number(token):
    token = token.strip()

    if token in {"–", "—", "-"}:
        return None

    negative = token.startswith("(") and token.endswith(")")
    value = float(token.replace(",", "").replace("(", "").replace(")", ""))

    return -value if negative else value


def extract_line_value(page_text, label):
    matches = [
        line for line in page_text.splitlines()
        if label.lower() in line.lower()
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one line for {label!r}; found {len(matches)}"
        )

    tokens = NUMBER_PATTERN.findall(matches[0])

    if len(tokens) < 2:
        raise ValueError(f"Could not parse values from: {matches[0]}")

    return {
        "current": parse_number(tokens[-2]),
        "previous": parse_number(tokens[-1]),
        "source_line": matches[0],
    }


def main():
    audit = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    metrics = audit["metrics"]

    def current(name):
        item = metrics[name]
        if item["status"] != "EXTRACTED":
            raise ValueError(f"{name} needs manual review")
        return item["FY2026"]

    # Independently extract two more figures from PDF page 296.
    with pdfplumber.open(PDF_PATH) as pdf:
        cash_flow_page = pdf.pages[295].extract_text() or ""

    fx = extract_line_value(
        cash_flow_page,
        "Effect of exchange rate changes on cash and cash equivalents"
    )["current"]

    opening_cash = extract_line_value(
        cash_flow_page,
        "Cash and cash equivalents at the beginning of the period"
    )["current"]

    operating = current("operating_cash_flow")
    investing = current("investing_cash_flow")
    financing = current("financing_cash_flow")
    closing_cash = current("closing_cash")

    calculated_change = operating + investing + financing + fx
    reported_change = closing_cash - opening_cash

    difference = round(calculated_change - reported_change, 2)

    checks = {
        "cash_flow_reconciliation": {
            "operating_cash_flow": operating,
            "investing_cash_flow": investing,
            "financing_cash_flow": financing,
            "fx_effect": fx,
            "calculated_change": calculated_change,
            "reported_change": reported_change,
            "difference": difference,
            "status": (
                "PASS" if abs(difference) < 0.01 else "REVIEW_REQUIRED"
            ),
        }
    }

    # Independently recalculate the profit margin.
    revenue = current("revenue")
    profit = current("profit_after_tax")
    margin = profit / revenue * 100

    checks["profit_margin"] = {
        "calculated_percent": round(margin, 2),
        "reported_by_previous_script": audit["calculated_metrics"][
            "profit_margin_FY2026_percent"
        ],
        "status": (
            "PASS"
            if abs(
                margin
                - audit["calculated_metrics"]["profit_margin_FY2026_percent"]
            ) < 0.01
            else "REVIEW_REQUIRED"
        ),
    }

    result_path = Path("finaudit_output/validation_results.json")
    result_path.write_text(
        json.dumps(checks, indent=2),
        encoding="utf-8",
    )

    print("\n========== INDEPENDENT VALIDATION ==========")
    for name, result in checks.items():
        print(f"\n{name}: {result['status']}")
        print(json.dumps(result, indent=2))

    print("\nSaved:", result_path.resolve())


if __name__ == "__main__":
    main()
