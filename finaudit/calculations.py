from collections.abc import Iterable


def calculate_metrics(metrics: Iterable[dict], company_id: str) -> list[dict]:
    metrics = list(metrics)
    indexed = {
        (row["financial_year"], row["metric_key"]): row for row in metrics
    }
    years = sorted({row["financial_year"] for row in metrics}, reverse=True)
    results = []

    def add(year: str, key: str, value, formula: str, inputs: list[str], status: str):
        results.append(
            {
                "metric_key": key,
                "company_id": company_id,
                "financial_year": year,
                "value": round(value, 4) if value is not None else None,
                "formula": formula,
                "input_metric_keys": inputs,
                "validation_status": status,
            }
        )

    for year in years:
        prior_year = f"FY{int(year[2:]) - 1}" if year.startswith("FY") else ""

        def value(metric_key: str, selected_year: str = year):
            row = indexed.get((selected_year, metric_key))
            return row.get("value") if row else None

        revenue = value("revenue")
        profit = value("profit_after_tax")
        operating_cash = value("operating_cash_flow")
        add(
            year,
            "net_profit_margin_percent",
            profit / revenue * 100 if profit is not None and revenue not in (None, 0) else None,
            "profit_after_tax / revenue * 100",
            ["profit_after_tax", "revenue"],
            "calculated" if profit is not None and revenue not in (None, 0) else "unavailable",
        )
        add(
            year,
            "operating_cash_flow_to_profit",
            operating_cash / profit if operating_cash is not None and profit not in (None, 0) else None,
            "operating_cash_flow / profit_after_tax",
            ["operating_cash_flow", "profit_after_tax"],
            "calculated" if operating_cash is not None and profit not in (None, 0) else "unavailable",
        )
        current_assets = value("total_current_assets")
        current_liabilities = value("current_liabilities")
        add(
            year,
            "current_ratio",
            current_assets / current_liabilities if current_assets is not None and current_liabilities not in (None, 0) else None,
            "total_current_assets / current_liabilities",
            ["total_current_assets", "current_liabilities"],
            "calculated" if current_assets is not None and current_liabilities not in (None, 0) else "unavailable",
        )

        for source_metric, growth_key, label in (
            ("revenue", "revenue_growth_percent", "revenue"),
            ("profit_after_tax", "profit_growth_percent", "profit"),
        ):
            current = value(source_metric)
            prior = value(source_metric, prior_year)
            growth = (
                (current - prior) / abs(prior) * 100
                if current is not None and prior not in (None, 0)
                else None
            )
            add(
                year,
                growth_key,
                growth,
                f"({label}_current_year - {label}_prior_year) / abs({label}_prior_year) * 100",
                [source_metric],
                "calculated" if growth is not None else "unavailable",
            )
    return results


def find_anomalies(metrics: list[dict], calculations: list[dict], company: str) -> list[dict]:
    indexed = {(row["financial_year"], row["metric_key"]): row for row in metrics}
    calculated = {(row["financial_year"], row["metric_key"]): row for row in calculations}
    years = sorted({row["financial_year"] for row in metrics}, reverse=True)
    if len(years) < 2:
        return []
    current, previous = years[:2]
    findings = []

    def add(title, severity, explanation, metric_keys, refs, confidence, follow_up):
        findings.append(
            {
                "finding_id": f"{company}-{title.lower().replace(' ', '-')}",
                "title": title,
                "severity": severity,
                "period": f"{previous} to {current}",
                "explanation": explanation,
                "supporting_metrics": metric_keys,
                "evidence_references": refs,
                "counterevidence": "A two-year change alone does not establish a cause; review accounting policies, acquisitions, and report disclosures.",
                "confidence": confidence,
                "follow_up_questions": [follow_up],
            }
        )

    revenue_growth = calculated.get((current, "revenue_growth_percent"), {}).get("value")
    profit_growth = calculated.get((current, "profit_growth_percent"), {}).get("value")
    operating_current = indexed.get((current, "operating_cash_flow"), {})
    operating_prior = indexed.get((previous, "operating_cash_flow"), {})
    if revenue_growth is not None and abs(revenue_growth) >= 25:
        add(
            "Material revenue movement",
            "medium",
            f"Revenue changed by {revenue_growth:.1f}% year over year, crossing the transparent 25% review threshold.",
            ["revenue", "revenue_growth_percent"],
            [f"PDF page {indexed[(current, 'revenue')]['pdf_page_number']}"],
            0.82,
            "What segment, volume, pricing, or acquisition factors explain the movement?",
        )
    if (
        profit_growth is not None
        and profit_growth > 0
        and operating_current.get("value") is not None
        and operating_prior.get("value") not in (None, 0)
    ):
        cash_change = (
            (operating_current["value"] - operating_prior["value"])
            / abs(operating_prior["value"])
            * 100
        )
        if cash_change < -20:
            add(
                "Profit growth with weaker operating cash flow",
                "medium",
                f"Profit increased {profit_growth:.1f}% while operating cash flow declined {abs(cash_change):.1f}% year over year.",
                ["profit_after_tax", "profit_growth_percent", "operating_cash_flow"],
                [
                    f"PDF page {indexed[(current, 'profit_after_tax')]['pdf_page_number']}",
                    f"PDF page {operating_current['pdf_page_number']}",
                ],
                0.88,
                "Are working-capital movements, tax payments, or timing differences disclosed?",
            )
    for finding in findings:
        finding["title"] = f"{company}: {finding['title']}"
    return findings
