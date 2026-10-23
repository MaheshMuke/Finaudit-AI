"""On-demand Screener retrieval, normalization, validation, and SQLite caching."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from io import StringIO
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Iterable

import pandas as pd
import requests

from finaudit import database
from finaudit.company_registry import CompanyRecord, get_company_registry

logger = logging.getLogger(__name__)

SCREENER_BASE_URL = "https://www.screener.in/company"
SUPPORTED_SECTIONS = (
    "quarterly_results",
    "annual_profit_loss",
    "balance_sheet",
    "cash_flow",
    "ratios",
)
DEFAULT_TTL_SECONDS = 24 * 60 * 60
DEFAULT_TIMEOUT = 20
MAX_RETRIES = 2

SECTION_LABELS = {
    "quarterly_results": {
        "sales", "revenue", "expenses", "operating profit", "net profit",
        "profit before tax", "opm %", "eps in rs",
    },
    "annual_profit_loss": {
        "sales", "revenue", "expenses", "operating profit", "net profit",
        "profit before tax", "opm %", "eps in rs",
    },
    "balance_sheet": {
        "equity capital", "reserves", "borrowings", "borrowing",
        "total assets", "total liabilities", "fixed assets", "investments",
    },
    "cash_flow": {
        "cash from operating activity", "cash from investing activity",
        "cash from financing activity", "free cash flow", "net cash flow",
    },
    "ratios": {
        "roe %", "roce %", "debtor days", "inventory days", "days payable",
        "cash conversion cycle", "working capital days",
    },
}

METRIC_KEYS = {
    "sales": "revenue",
    "revenue": "revenue",
    "expenses": "expenses",
    "operating profit": "operating_profit",
    "opm %": "operating_margin_percent",
    "other income": "other_income",
    "interest": "interest_expense",
    "depreciation": "depreciation",
    "profit before tax": "profit_before_tax",
    "tax %": "tax_rate_percent",
    "net profit": "net_income",
    "eps in rs": "eps",
    "dividend payout %": "dividend_payout_percent",
    "financing profit": "financing_profit",
    "financing margin %": "financing_margin_percent",
    "gross npa %": "gross_npa_percent",
    "net npa %": "net_npa_percent",
    "equity capital": "equity_capital",
    "reserves": "reserves",
    "borrowings": "borrowings",
    "borrowing": "borrowings",
    "cwip": "capital_work_in_progress",
    "deposits": "deposits",
    "fixed assets": "fixed_assets",
    "investments": "investments",
    "other assets": "other_assets",
    "other liabilities": "other_liabilities",
    "total assets": "total_assets",
    "total liabilities": "total_liabilities",
    "cash from operating activity": "operating_cash_flow",
    "cash from investing activity": "investing_cash_flow",
    "cash from financing activity": "financing_cash_flow",
    "free cash flow": "free_cash_flow",
    "net cash flow": "net_cash_flow",
    "cfo/ebitda share of operating profit converting into cash: cfo (before tax) / operating profit": "cash_conversion_ratio",
    "roe %": "roe_percent",
    "roce %": "roce_percent",
    "debtor days": "debtor_days",
    "inventory days": "inventory_days",
    "days payable": "days_payable",
    "cash conversion cycle": "cash_conversion_cycle",
    "working capital days": "working_capital_days",
}


@dataclass(frozen=True)
class ScreenerFact:
    symbol: str
    company_name: str
    provider: str
    source_type: str
    statement: str
    metric_key: str
    original_metric_label: str
    period: str
    period_type: str
    value: float | None
    raw_value: str
    metric_unit: str | None
    currency: str | None
    reporting_basis: str
    source_url: str
    retrieved_at: str
    data_quality_status: str
    data_quality_warnings: tuple[str, ...]
    verification_status: str = "unverified"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["data_quality_warnings"] = list(self.data_quality_warnings)
        return result


@dataclass(frozen=True)
class ScreenerPeer:
    target_symbol: str
    peer_symbol: str
    company_name: str
    screener_url: str
    source_url: str
    rank: int | None
    cmp: float | None
    pe: float | None
    market_cap_cr: float | None
    div_yield_pct: float | None
    qtr_profit_cr: float | None
    qtr_profit_var_pct: float | None
    qtr_sales_cr: float | None
    qtr_sales_var_pct: float | None
    roce_pct: float | None
    is_target: bool
    retrieved_at: str
    data_quality_status: str
    data_quality_warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["data_quality_warnings"] = list(self.data_quality_warnings)
        result["symbol"] = self.peer_symbol
        return result


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_label(label: Any) -> str:
    return re.sub(r"\s+", " ", str(label).replace("\xa0", " ")).strip()


def _canonical_label(label: Any) -> str:
    normalized = _normalize_label(label)
    return normalized.rstrip("+").strip().lower()


def parse_screener_number(raw: Any) -> float | None:
    """Parse Screener display values while preserving the original raw value separately."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = _normalize_label(raw)
    if not text or text in {"-", "—", "–", "NA", "N/A"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    cleaned = re.sub(r"[,%₹]", "", text).replace(",", "").replace("(", "").replace(")", "")
    cleaned = cleaned.strip()
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative else value


def _period_type(section: str, period: str) -> str:
    text = _normalize_label(period).lower()
    if "ttm" in text:
        return "ttm"
    if section == "quarterly_results":
        return "quarterly"
    if section == "balance_sheet":
        return "annual_snapshot"
    return "annual"


def _metric_unit(metric: str) -> tuple[str | None, str | None]:
    if metric.endswith("%"):
        return "percent", None
    if metric == "EPS in Rs":
        return "INR_per_share", "INR"
    if metric in {"Debtor Days", "Inventory Days", "Days Payable", "Cash Conversion Cycle", "Working Capital Days"}:
        return "days", None
    return "INR_crore", "INR"


def _section_for_labels(labels: set[str]) -> str | None:
    scores = {
        section: len(labels.intersection(candidates))
        for section, candidates in SECTION_LABELS.items()
    }
    section, score = max(scores.items(), key=lambda item: item[1])
    return section if score else None


def normalize_screener_rows(
    rows: Iterable[dict[str, Any]],
    *,
    symbol: str,
    company_name: str,
    source_url: str,
    retrieved_at: str,
    reporting_basis: str = "consolidated",
) -> tuple[list[ScreenerFact], list[dict[str, Any]]]:
    """Normalize Colab-compatible long rows without assuming all values are present."""
    facts: list[ScreenerFact] = []
    findings: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        statement = _normalize_label(row.get("statement"))
        original_label = _normalize_label(row.get("metric") or row.get("original_metric_label"))
        period = _normalize_label(row.get("period"))
        if not statement or not original_label or not period:
            findings.append({"row": index, "status": "rejected", "reason": "missing statement, metric, or period"})
            continue
        canonical = METRIC_KEYS.get(_canonical_label(original_label), _canonical_label(original_label).replace(" ", "_"))
        raw_value = str(row.get("raw_value") if row.get("raw_value") is not None else row.get("value", ""))
        parsed = parse_screener_number(row.get("value") if row.get("value") is not None else raw_value)
        unit = row.get("metric_unit") or _metric_unit(original_label)[0]
        currency = "INR" if unit in {"INR_crore", "INR_per_share"} else row.get("currency")
        warnings: list[str] = []
        if parsed is None and raw_value.strip() not in {"", "-", "—", "–", "NA", "N/A"}:
            warnings.append("numeric value could not be parsed")
        if not row.get("source_url"):
            warnings.append("source URL is missing")
        facts.append(
            ScreenerFact(
                symbol=symbol,
                company_name=company_name,
                provider="screener",
                source_type="structured_web",
                statement=statement,
                metric_key=canonical,
                original_metric_label=original_label,
                period=period,
                period_type=_normalize_label(row.get("period_type")) or _period_type(statement, period),
                value=parsed,
                raw_value=raw_value,
                metric_unit=unit,
                currency=currency,
                reporting_basis=_normalize_label(row.get("reporting_basis") or reporting_basis),
                source_url=_normalize_label(row.get("source_url") or source_url),
                retrieved_at=_normalize_label(row.get("retrieved_at") or retrieved_at),
                data_quality_status="warning" if warnings else "available",
                data_quality_warnings=tuple(warnings),
                verification_status=_normalize_label(row.get("verification_status") or "unverified"),
            )
        )
    return facts, findings


class ScreenerAdapter:
    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        ttl_seconds: int | None = None,
        timeout: int | None = None,
    ):
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "FinAudit AI research client")
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else int(
            os.getenv("SCREENER_FINANCIAL_TTL_SECONDS", str(DEFAULT_TTL_SECONDS))
        )
        self.timeout = timeout if timeout is not None else int(
            os.getenv("SCREENER_REQUEST_TIMEOUT", str(DEFAULT_TIMEOUT))
        )

    @staticmethod
    def resolve_company(symbol: str) -> CompanyRecord:
        company = get_company_registry().get_by_symbol(symbol.strip().upper())
        if not company:
            raise ValueError(f"Unsupported or unknown company symbol: {symbol}")
        return company

    def source_url(self, symbol: str) -> str:
        company = self.resolve_company(symbol)
        return f"{SCREENER_BASE_URL}/{company.symbol}/consolidated/"

    def fetch(self, symbol: str) -> tuple[list[ScreenerFact], list[dict[str, Any]], dict[str, Any]]:
        company = self.resolve_company(symbol)
        url = self.source_url(company.symbol)
        retrieved_at = _now().isoformat()
        response = self._get(url)
        try:
            tables = pd.read_html(StringIO(response.text))
        except (ValueError, ImportError) as exc:
            raise ValueError(f"Screener page for {company.symbol} contained no readable tables") from exc
        rows: list[dict[str, Any]] = []
        sections_found: set[str] = set()
        for table in tables:
            if table.empty or table.shape[1] < 2:
                continue
            table = table.copy()
            table.columns = [_normalize_label(column) for column in table.columns]
            first_column = table.columns[0]
            labels = {_canonical_label(value) for value in table[first_column].tolist()}
            section = _section_for_labels(labels)
            if not section:
                continue
            if section == "quarterly_results":
                period_columns = [_normalize_label(column).lower() for column in table.columns[1:]]
                if period_columns and all(
                    column.startswith("mar ") or "ttm" in column
                    for column in period_columns
                ):
                    section = "annual_profit_loss"
            sections_found.add(section)
            for _, record in table.iterrows():
                label = _normalize_label(record.iloc[0])
                if not label or _canonical_label(label) not in METRIC_KEYS:
                    continue
                for period_column, value in record.iloc[1:].items():
                    rows.append(
                        {
                            "statement": section,
                            "metric": label,
                            "period": period_column,
                            "period_type": _period_type(section, str(period_column)),
                            "value": value,
                            "raw_value": _normalize_label(value),
                            "source_url": url,
                            "retrieved_at": retrieved_at,
                            "reporting_basis": "consolidated",
                            "metric_unit": _metric_unit(label)[0],
                        }
                    )
        facts, findings = normalize_screener_rows(
            rows,
            symbol=company.symbol,
            company_name=company.company_name,
            source_url=url,
            retrieved_at=retrieved_at,
        )
        warehouse_match = re.search(r'data-warehouse-id=["\']?(\d+)', response.text, re.I)
        warehouse_id = warehouse_match.group(1) if warehouse_match else None
        screener_peers = self.fetch_peers(warehouse_id, target_symbol=company.symbol) if warehouse_id else []
        if not screener_peers:
            screener_peers = self.parse_peer_table_html(
                response.text,
                target_symbol=company.symbol,
                source_url=url,
                retrieved_at=retrieved_at,
            )

        metadata = {
            "symbol": company.symbol,
            "source_url": url,
            "retrieved_at": retrieved_at,
            "sections_found": sorted(sections_found),
            "sections_missing": sorted(set(SUPPORTED_SECTIONS) - sections_found),
            "warehouse_id": warehouse_id,
            "screener_peers": screener_peers,
        }
        return facts, findings, metadata

    def parse_peer_table_html(
        self,
        html_text: str,
        *,
        target_symbol: str = "",
        source_url: str = "",
        retrieved_at: str | None = None,
    ) -> list[dict[str, Any]]:
        """Extract and normalize peer companies from HTML containing Screener peer comparison table."""
        if not html_text or not html_text.strip():
            return []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_text, "html.parser")
            tables = soup.find_all("table")
            target_table = None
            for tbl in tables:
                text = tbl.get_text()
                if "CMP" in text and ("Mar Cap" in text or "P/E" in text or "ROCE" in text or "NP Qtr" in text):
                    target_table = tbl
                    break
            if target_table is None:
                return []

            headers: list[str] = []
            header_row = target_table.find("tr")
            if header_row:
                for th in header_row.find_all(["th", "td"]):
                    headers.append(_normalize_label(th.get_text()).lower())

            col_map: dict[str, int] = {}
            for idx, h in enumerate(headers):
                if any(k in h for k in ("s.no", "sno", "sr")):
                    col_map["rank"] = idx
                elif any(k in h for k in ("name", "company")):
                    col_map["name"] = idx
                elif "cmp" in h:
                    col_map["cmp"] = idx
                elif "p/e" in h or h == "pe":
                    col_map["pe"] = idx
                elif "mar cap" in h or "market cap" in h:
                    col_map["market_cap_cr"] = idx
                elif "div yld" in h or "dividend" in h:
                    col_map["div_yield_pct"] = idx
                elif any(k in h for k in ("profit var", "qtr profit var")):
                    col_map["qtr_profit_var_pct"] = idx
                elif any(k in h for k in ("sales var", "qtr sales var")):
                    col_map["qtr_sales_var_pct"] = idx
                elif any(k in h for k in ("np qtr", "net profit", "np")):
                    col_map["qtr_profit_cr"] = idx
                elif any(k in h for k in ("sales qtr", "sales", "revenue")):
                    col_map["qtr_sales_cr"] = idx
                elif "roce" in h:
                    col_map["roce_pct"] = idx

            retrieved_time = retrieved_at or _now().isoformat()
            peers: list[dict[str, Any]] = []

            for tr in target_table.find_all("tr"):
                tds = tr.find_all("td")
                if not tds:
                    continue
                a = tr.find("a")
                if not a or not a.get("href"):
                    continue
                href = a.get("href", "").strip()
                m = re.search(r"/company/([A-Za-z0-9_\-]+)/", href)
                if not m:
                    continue
                sym = m.group(1).upper()
                name = _normalize_label(a.get_text()) or sym
                if "median" in name.lower() or "median" in tr.get_text().lower():
                    continue

                full_screener_url = f"https://www.screener.in{href}" if href.startswith("/") else href

                def cell_val(key: str) -> Any:
                    if key in col_map and col_map[key] < len(tds):
                        return _normalize_label(tds[col_map[key]].get_text())
                    return None

                rank_val = None
                raw_rank = cell_val("rank")
                if raw_rank:
                    digits = re.sub(r"\D", "", raw_rank)
                    if digits:
                        try:
                            rank_val = int(digits)
                        except ValueError:
                            rank_val = None

                is_target = bool(target_symbol and sym == target_symbol.upper())
                warnings: list[str] = []

                cmp_val = parse_screener_number(cell_val("cmp"))
                pe_val = parse_screener_number(cell_val("pe"))
                mcap_val = parse_screener_number(cell_val("market_cap_cr"))
                div_val = parse_screener_number(cell_val("div_yield_pct"))
                np_val = parse_screener_number(cell_val("qtr_profit_cr"))
                np_var = parse_screener_number(cell_val("qtr_profit_var_pct"))
                sales_val = parse_screener_number(cell_val("qtr_sales_cr"))
                sales_var = parse_screener_number(cell_val("qtr_sales_var_pct"))
                roce_val = parse_screener_number(cell_val("roce_pct"))

                peer_obj = ScreenerPeer(
                    target_symbol=target_symbol.upper() if target_symbol else sym,
                    peer_symbol=sym,
                    company_name=name,
                    screener_url=full_screener_url,
                    source_url=source_url,
                    rank=rank_val,
                    cmp=cmp_val,
                    pe=pe_val,
                    market_cap_cr=mcap_val,
                    div_yield_pct=div_val,
                    qtr_profit_cr=np_val,
                    qtr_profit_var_pct=np_var,
                    qtr_sales_cr=sales_val,
                    qtr_sales_var_pct=sales_var,
                    roce_pct=roce_val,
                    is_target=is_target,
                    retrieved_at=retrieved_time,
                    data_quality_status="available" if not warnings else "warning",
                    data_quality_warnings=tuple(warnings),
                )
                peers.append(peer_obj.to_dict())

            return peers
        except Exception as exc:
            logger.debug("Error parsing peer table HTML: %s", exc)
            return []

    def fetch_peers(self, warehouse_id: str, target_symbol: str = "") -> list[dict[str, Any]]:
        """Fetch and parse live peer comparison table for this company from Screener."""
        if not warehouse_id:
            return []
        url = f"https://www.screener.in/api/company/{warehouse_id}/peers/"
        retrieved_at = _now().isoformat()
        try:
            resp = self._get(url)
            return self.parse_peer_table_html(
                resp.text,
                target_symbol=target_symbol,
                source_url=url,
                retrieved_at=retrieved_at,
            )
        except Exception as exc:
            logger.debug("Failed fetching Screener peer table for warehouse %s: %s", warehouse_id, exc)
            return []

    def _get(self, url: str) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    time.sleep(0.25 * (attempt + 1))
        raise RuntimeError(f"Screener retrieval failed: {last_error}") from last_error

    def get_cached_facts(self, symbol: str, *, fresh_only: bool = False) -> list[dict[str, Any]]:
        database.initialize()
        query = """
            SELECT * FROM financial_facts
            WHERE symbol=? AND provider='screener' AND is_current=1
        """
        params: list[Any] = [symbol.upper()]
        if fresh_only:
            query += " AND cache_expires_at > ?"
            params.append(_now().isoformat())
        query += " ORDER BY statement, period, metric_key"
        with database.connect() as db:
            return [dict(row) for row in db.execute(query, params).fetchall()]

    def refresh(self, symbol: str) -> dict[str, Any]:
        company = self.resolve_company(symbol)
        database.initialize()
        batch_id = str(uuid.uuid4())
        started_at = _now().isoformat()
        with database.connect() as db:
            db.execute(
                """INSERT INTO financial_ingestion_batches
                   (batch_id, symbol, provider, requested_sections, started_at, status)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (batch_id, company.symbol, "screener", json.dumps(SUPPORTED_SECTIONS), started_at, "running"),
            )
        try:
            facts, findings, metadata = self.fetch(company.symbol)
            expires_at = (_now() + timedelta(seconds=self.ttl_seconds)).isoformat()
            inserted = 0
            updated = 0
            with database.connect() as db:
                for section in SUPPORTED_SECTIONS:
                    section_facts = [fact for fact in facts if fact.statement == section]
                    db.execute(
                        """INSERT OR REPLACE INTO financial_retrieval_sections
                           (batch_id, section, status, row_count, retrieved_at)
                           VALUES (?, ?, ?, ?, ?)""",
                        (batch_id, section, "success" if section_facts else "empty", len(section_facts), metadata["retrieved_at"]),
                    )
                for fact in facts:
                    payload = fact.to_dict()
                    existing = db.execute(
                        """SELECT fact_id, value FROM financial_facts
                           WHERE symbol=? AND provider=? AND statement=? AND metric_key=?
                             AND period=? AND period_type=? AND reporting_basis=?""",
                        (fact.symbol, fact.provider, fact.statement, fact.metric_key, fact.period, fact.period_type, fact.reporting_basis),
                    ).fetchone()
                    fact_id = existing["fact_id"] if existing else f"fact_{uuid.uuid4().hex}"
                    change_type = "inserted" if not existing else "unchanged" if existing["value"] == fact.value else "updated"
                    db.execute(
                        """INSERT INTO financial_facts
                           (fact_id, batch_id, symbol, company_name, provider, source_type,
                            statement, metric_key, original_metric_label, period, period_type,
                            value, raw_value, metric_unit, currency, reporting_basis, source_url,
                            retrieved_at, ingested_at, cache_expires_at, data_quality_status,
                            data_quality_warnings, verification_status, is_current)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT(symbol, provider, statement, metric_key, period, period_type, reporting_basis)
                           DO UPDATE SET
                             fact_id=excluded.fact_id, batch_id=excluded.batch_id,
                             company_name=excluded.company_name, source_type=excluded.source_type,
                             original_metric_label=excluded.original_metric_label, value=excluded.value,
                             raw_value=excluded.raw_value, metric_unit=excluded.metric_unit,
                             currency=excluded.currency, source_url=excluded.source_url,
                             retrieved_at=excluded.retrieved_at, ingested_at=excluded.ingested_at,
                             cache_expires_at=excluded.cache_expires_at,
                             data_quality_status=excluded.data_quality_status,
                             data_quality_warnings=excluded.data_quality_warnings,
                             verification_status=excluded.verification_status, is_current=1""",
                        (
                            fact_id, batch_id, fact.symbol, fact.company_name, fact.provider,
                            fact.source_type, fact.statement, fact.metric_key, fact.original_metric_label,
                            fact.period, fact.period_type, fact.value, fact.raw_value, fact.metric_unit,
                            fact.currency, fact.reporting_basis, fact.source_url, fact.retrieved_at,
                            started_at, expires_at, fact.data_quality_status,
                            json.dumps(fact.data_quality_warnings), fact.verification_status, 1,
                        ),
                    )
                    db.execute(
                        """INSERT INTO financial_fact_observations
                           (fact_id, batch_id, value, raw_value, observed_at, source_url, change_type)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (fact_id, batch_id, fact.value, fact.raw_value, fact.retrieved_at, fact.source_url, change_type),
                    )
                    if change_type == "inserted":
                        inserted += 1
                    elif change_type == "updated":
                        updated += 1
                peers = metadata.get("screener_peers", [])
                for peer in peers:
                    db.execute(
                        """INSERT INTO screener_peers
                           (batch_id, target_symbol, peer_symbol, company_name, screener_url, source_url,
                            rank, cmp, pe, market_cap_cr, div_yield_pct, qtr_profit_cr, qtr_profit_var_pct,
                            qtr_sales_cr, qtr_sales_var_pct, roce_pct, is_target, retrieved_at,
                            cache_expires_at, is_current)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                           ON CONFLICT(target_symbol, peer_symbol) DO UPDATE SET
                             batch_id=excluded.batch_id,
                             company_name=excluded.company_name,
                             screener_url=excluded.screener_url,
                             source_url=excluded.source_url,
                             rank=excluded.rank,
                             cmp=excluded.cmp,
                             pe=excluded.pe,
                             market_cap_cr=excluded.market_cap_cr,
                             div_yield_pct=excluded.div_yield_pct,
                             qtr_profit_cr=excluded.qtr_profit_cr,
                             qtr_profit_var_pct=excluded.qtr_profit_var_pct,
                             qtr_sales_cr=excluded.qtr_sales_cr,
                             qtr_sales_var_pct=excluded.qtr_sales_var_pct,
                             roce_pct=excluded.roce_pct,
                             is_target=excluded.is_target,
                             retrieved_at=excluded.retrieved_at,
                             cache_expires_at=excluded.cache_expires_at,
                             is_current=1""",
                        (
                            batch_id,
                            peer.get("target_symbol") or company.symbol,
                            peer.get("peer_symbol") or peer.get("symbol", ""),
                            peer.get("company_name", ""),
                            peer.get("screener_url", f"https://www.screener.in/company/{peer.get('symbol', '')}/consolidated/"),
                            peer.get("source_url", metadata.get("source_url", "")),
                            peer.get("rank"),
                            peer.get("cmp"),
                            peer.get("pe"),
                            peer.get("market_cap_cr"),
                            peer.get("div_yield_pct"),
                            peer.get("qtr_profit_cr"),
                            peer.get("qtr_profit_var_pct"),
                            peer.get("qtr_sales_cr"),
                            peer.get("qtr_sales_var_pct"),
                            peer.get("roce_pct"),
                            1 if peer.get("is_target") else 0,
                            peer.get("retrieved_at", metadata.get("retrieved_at", _now().isoformat())),
                            expires_at,
                        ),
                    )
                status = (
                    "success"
                    if facts and not findings and not metadata["sections_missing"]
                    else "partial"
                    if facts
                    else "failed"
                )
                db.execute(
                    """UPDATE financial_ingestion_batches
                       SET completed_at=?, status=?, total_rows=?, inserted_rows=?,
                           updated_rows=?, rejected_rows=?, error_summary=?
                       WHERE batch_id=?""",
                    (_now().isoformat(), status, len(facts) + len(findings), inserted, updated, len(findings), json.dumps(findings), batch_id),
                )
            return {
                "batch_id": batch_id,
                "symbol": company.symbol,
                "status": status,
                "facts": len(facts),
                "peers": len(peers),
                "findings": findings,
                "metadata": metadata,
            }
        except Exception as exc:
            with database.connect() as db:
                db.execute(
                    """UPDATE financial_ingestion_batches
                       SET completed_at=?, status=?, error_summary=?
                       WHERE batch_id=?""",
                    (_now().isoformat(), "failed", json.dumps([str(exc)]), batch_id),
                )
            raise

    def get_cached_peers(
        self,
        symbol: str,
        *,
        fresh_only: bool = False,
        include_target: bool = True,
    ) -> list[dict[str, Any]]:
        database.initialize()
        query = """
            SELECT * FROM screener_peers
            WHERE target_symbol=? AND is_current=1
        """
        params: list[Any] = [symbol.upper()]
        if fresh_only:
            query += " AND cache_expires_at > ?"
            params.append(_now().isoformat())
        if not include_target:
            query += " AND is_target=0"
        query += " ORDER BY rank ASC, market_cap_cr DESC"
        with database.connect() as db:
            rows = [dict(row) for row in db.execute(query, params).fetchall()]
            for r in rows:
                if "peer_symbol" in r and "symbol" not in r:
                    r["symbol"] = r["peer_symbol"]
                if "is_target" in r:
                    r["is_target"] = bool(r["is_target"])
            return rows

    def get_or_refresh(self, symbol: str) -> dict[str, Any]:
        company = self.resolve_company(symbol)
        fresh = self.get_cached_facts(company.symbol, fresh_only=True)
        fresh_peers = self.get_cached_peers(company.symbol, fresh_only=True)
        if fresh:
            return {
                "status": "cache_hit",
                "symbol": company.symbol,
                "facts": fresh,
                "peers": fresh_peers,
                "refreshed": False,
            }
        result = self.refresh(company.symbol)
        result["facts"] = self.get_cached_facts(company.symbol)
        result["peers"] = self.get_cached_peers(company.symbol)
        result["refreshed"] = True
        return result


def facts_to_statement_frames(facts: Iterable[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """Adapt cached Screener facts to the existing calculator's long-table contract."""
    frames: dict[str, list[dict[str, Any]]] = {
        "income_statement_annual": [],
        "income_statement_quarterly": [],
        "balance_sheet_annual": [],
        "balance_sheet_quarterly": [],
        "cash_flow_annual": [],
    }
    statement_map = {
        "annual_profit_loss": "income_statement_annual",
        "quarterly_results": "income_statement_quarterly",
        "balance_sheet": "balance_sheet_annual",
        "cash_flow": "cash_flow_annual",
    }
    fact_list = list(facts)
    ratio_facts = [
        {
            "metric_key": fact["metric_key"],
            "period": fact["period"],
            "period_type": fact["period_type"],
            "value": fact["value"],
            "unit": fact.get("metric_unit"),
            "source_url": fact.get("source_url"),
        }
        for fact in fact_list
        if fact["statement"] == "ratios"
    ]
    for fact in fact_list:
        target = statement_map.get(fact["statement"])
        if not target:
            continue
        value = fact["value"]
        if value is not None and fact.get("metric_unit") == "INR_crore":
            value = value * 1e7
        frames[target].append(
            {
                "symbol": fact["symbol"],
                "company": fact["company_name"],
                "item": _display_item_for_calculator(fact["metric_key"]),
                "period_end": _period_to_date(fact["period"]),
                "period": fact["period"],
                "period_type": fact["period_type"],
                "reporting_basis": fact["reporting_basis"],
                "source_url": fact["source_url"],
                "value": value,
            }
        )
    result: dict[str, Any] = {key: pd.DataFrame(value) for key, value in frames.items()}
    result["_metadata"] = {
        "screener_ratio_facts": ratio_facts,
        "reporting_basis": sorted(
            {
                fact["reporting_basis"]
                for fact in fact_list
                if fact["statement"] in statement_map
            }
        ),
    }
    return result


def _period_to_date(period: str) -> str | None:
    if period.strip().upper() == "TTM":
        return None
    month_period = re.fullmatch(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(20\d{2})",
        period.strip(),
        flags=re.IGNORECASE,
    )
    if month_period:
        month = pd.to_datetime(
            f"{month_period.group(1)} 1 {month_period.group(2)}",
            format="%b %d %Y",
        )
        return (month + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    parsed = pd.to_datetime(period, errors="coerce")
    if pd.notna(parsed):
        return parsed.strftime("%Y-%m-%d")
    return None


def _display_item_for_calculator(metric_key: str) -> str:
    return {
        "revenue": "Total Revenue",
        "net_income": "Net Income",
        "operating_profit": "Operating Income",
        "operating_cash_flow": "Operating Cash Flow",
        "free_cash_flow": "Free Cash Flow",
        "total_assets": "Total Assets",
        "total_liabilities": "Total Liabilities",
        "borrowings": "Total Debt",
        "equity_capital": "Equity Capital",
    }.get(metric_key, metric_key)
