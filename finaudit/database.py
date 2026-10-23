import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

from finaudit.schemas import Company

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
DATA_DIR = Path(os.getenv("FINAUDIT_DATA_DIR", ROOT / "data")).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
REPORTS_DIR = DATA_DIR / "reports"
DB_PATH = DATA_DIR / "finaudit.sqlite3"

COMPANIES = [
    Company(company_id="INFY", name="Infosys Limited", symbol="INFY", sector="IT Services"),
    Company(company_id="TCS", name="Tata Consultancy Services Limited", symbol="TCS", sector="IT Services"),
    Company(company_id="WIPRO", name="Wipro Limited", symbol="WIPRO", sector="IT Services"),
    Company(company_id="HCLTECH", name="HCL Technologies Limited", symbol="HCLTECH", sector="IT Services"),
    Company(company_id="TECHM", name="Tech Mahindra Limited", symbol="TECHM", sector="IT Services"),
    Company(company_id="RELIANCE", name="Reliance Industries Limited", symbol="RELIANCE", sector="Diversified"),
    Company(company_id="ICICIBANK", name="ICICI Bank Limited", symbol="ICICIBANK", sector="Banking"),
]


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize() -> None:
    with connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS companies (
                company_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                exchange TEXT NOT NULL,
                symbol TEXT,
                sector TEXT,
                metadata_source TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reports (
                report_id TEXT PRIMARY KEY,
                company_id TEXT NOT NULL REFERENCES companies(company_id),
                financial_year TEXT,
                statement_basis TEXT NOT NULL,
                source_url TEXT,
                local_path TEXT NOT NULL,
                checksum TEXT NOT NULL,
                retrieval_timestamp TEXT NOT NULL,
                extraction_status TEXT NOT NULL,
                report_title TEXT,
                publication_date TEXT,
                source_page TEXT,
                source_kind TEXT NOT NULL DEFAULT 'manual_upload',
                download_status TEXT NOT NULL DEFAULT 'manual_upload',
                verification_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_reports_company_year
                ON reports(company_id, financial_year);
            CREATE TABLE IF NOT EXISTS financial_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT NOT NULL REFERENCES reports(report_id),
                company_id TEXT NOT NULL REFERENCES companies(company_id),
                financial_year TEXT NOT NULL,
                statement_basis TEXT NOT NULL,
                metric_key TEXT NOT NULL,
                original_label TEXT NOT NULL,
                value REAL,
                unit TEXT NOT NULL,
                currency TEXT NOT NULL,
                pdf_page_number INTEGER NOT NULL,
                source_excerpt TEXT NOT NULL,
                extraction_method TEXT NOT NULL,
                extraction_confidence REAL NOT NULL,
                verification_status TEXT NOT NULL,
                UNIQUE(report_id, financial_year, metric_key)
            );
            CREATE INDEX IF NOT EXISTS idx_metrics_report
                ON financial_metrics(report_id, financial_year);
            CREATE TABLE IF NOT EXISTS analysis_runs (
                run_id TEXT PRIMARY KEY,
                selected_company TEXT NOT NULL,
                selected_competitors TEXT NOT NULL,
                selected_years TEXT NOT NULL,
                analysis_mode TEXT NOT NULL,
                workflow_status TEXT NOT NULL,
                current_step TEXT,
                results TEXT,
                error TEXT,
                creation_timestamp TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS calculated_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES analysis_runs(run_id),
                company_id TEXT NOT NULL,
                financial_year TEXT NOT NULL,
                metric_key TEXT NOT NULL,
                value REAL,
                formula TEXT NOT NULL,
                input_metric_keys TEXT NOT NULL,
                validation_status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS findings (
                finding_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES analysis_runs(run_id),
                title TEXT NOT NULL,
                severity TEXT NOT NULL,
                period TEXT NOT NULL,
                explanation TEXT NOT NULL,
                supporting_metrics TEXT NOT NULL,
                evidence_references TEXT NOT NULL,
                counterevidence TEXT NOT NULL,
                confidence REAL NOT NULL,
                follow_up_questions TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS report_discoveries (
                discovery_id TEXT PRIMARY KEY,
                company_id TEXT NOT NULL REFERENCES companies(company_id),
                financial_year TEXT NOT NULL,
                statement_basis TEXT NOT NULL,
                status TEXT NOT NULL,
                candidates_json TEXT NOT NULL DEFAULT '[]',
                errors_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS financial_ingestion_batches (
                batch_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                provider TEXT NOT NULL,
                requested_sections TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL,
                total_rows INTEGER NOT NULL DEFAULT 0,
                inserted_rows INTEGER NOT NULL DEFAULT 0,
                updated_rows INTEGER NOT NULL DEFAULT 0,
                rejected_rows INTEGER NOT NULL DEFAULT 0,
                error_summary TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS financial_facts (
                fact_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL REFERENCES financial_ingestion_batches(batch_id),
                symbol TEXT NOT NULL,
                company_name TEXT NOT NULL,
                provider TEXT NOT NULL,
                source_type TEXT NOT NULL,
                statement TEXT NOT NULL,
                metric_key TEXT NOT NULL,
                original_metric_label TEXT NOT NULL,
                period TEXT NOT NULL,
                period_type TEXT NOT NULL,
                value REAL,
                raw_value TEXT,
                metric_unit TEXT,
                currency TEXT,
                reporting_basis TEXT,
                source_url TEXT NOT NULL,
                retrieved_at TEXT NOT NULL,
                ingested_at TEXT NOT NULL,
                cache_expires_at TEXT NOT NULL,
                data_quality_status TEXT NOT NULL,
                data_quality_warnings TEXT NOT NULL DEFAULT '[]',
                verification_status TEXT NOT NULL,
                is_current INTEGER NOT NULL DEFAULT 1,
                UNIQUE(symbol, provider, statement, metric_key, period, period_type, reporting_basis)
            );
            CREATE INDEX IF NOT EXISTS idx_financial_facts_lookup
                ON financial_facts(symbol, statement, period_type, period);
            CREATE INDEX IF NOT EXISTS idx_financial_facts_metric
                ON financial_facts(symbol, metric_key, period);
            CREATE TABLE IF NOT EXISTS financial_fact_observations (
                observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id TEXT NOT NULL REFERENCES financial_facts(fact_id),
                batch_id TEXT NOT NULL REFERENCES financial_ingestion_batches(batch_id),
                value REAL,
                raw_value TEXT,
                observed_at TEXT NOT NULL,
                source_url TEXT NOT NULL,
                change_type TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_financial_observations_fact
                ON financial_fact_observations(fact_id, observed_at);
            CREATE TABLE IF NOT EXISTS financial_retrieval_sections (
                batch_id TEXT NOT NULL REFERENCES financial_ingestion_batches(batch_id),
                section TEXT NOT NULL,
                status TEXT NOT NULL,
                row_count INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                retrieved_at TEXT,
                PRIMARY KEY(batch_id, section)
            );
            CREATE TABLE IF NOT EXISTS research_sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                company_symbol TEXT NOT NULL,
                selected_peers TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_research_sessions_symbol
                ON research_sessions(company_symbol, updated_at);
            CREATE TABLE IF NOT EXISTS session_messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES research_sessions(session_id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                intent TEXT,
                evidence_references TEXT NOT NULL DEFAULT '[]',
                citations TEXT NOT NULL DEFAULT '[]',
                structured_payload TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_session_messages_session
                ON session_messages(session_id, created_at);
            CREATE TABLE IF NOT EXISTS session_findings (
                finding_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES research_sessions(session_id) ON DELETE CASCADE,
                topic TEXT NOT NULL,
                claim TEXT NOT NULL,
                claim_type TEXT NOT NULL,
                supporting_evidence TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_session_findings_session
                ON session_findings(session_id, created_at);
            CREATE TABLE IF NOT EXISTS nse_disclosures (
                disclosure_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                company_name TEXT NOT NULL,
                title TEXT NOT NULL,
                event_type TEXT NOT NULL,
                publication_date TEXT,
                source_url TEXT NOT NULL,
                attachment_url TEXT,
                local_path TEXT,
                checksum TEXT,
                extracted_text TEXT,
                material_facts TEXT NOT NULL DEFAULT '{}',
                retrieved_at TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_nse_disclosures_symbol_date
                ON nse_disclosures(symbol, publication_date);
            CREATE TABLE IF NOT EXISTS screener_peers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL,
                target_symbol TEXT NOT NULL,
                peer_symbol TEXT NOT NULL,
                company_name TEXT NOT NULL,
                screener_url TEXT NOT NULL,
                source_url TEXT NOT NULL,
                rank INTEGER,
                cmp REAL,
                pe REAL,
                market_cap_cr REAL,
                div_yield_pct REAL,
                qtr_profit_cr REAL,
                qtr_profit_var_pct REAL,
                qtr_sales_cr REAL,
                qtr_sales_var_pct REAL,
                roce_pct REAL,
                is_target INTEGER NOT NULL DEFAULT 0,
                retrieved_at TEXT NOT NULL,
                cache_expires_at TEXT NOT NULL,
                is_current INTEGER NOT NULL DEFAULT 1,
                UNIQUE(target_symbol, peer_symbol)
            );
            CREATE INDEX IF NOT EXISTS idx_screener_peers_target
                ON screener_peers(target_symbol, is_current);
            """
        )
        columns = {
            row["name"] for row in db.execute("PRAGMA table_info(reports)").fetchall()
        }
        migrations = {
            "report_title": "TEXT",
            "publication_date": "TEXT",
            "source_page": "TEXT",
            "source_kind": "TEXT NOT NULL DEFAULT 'manual_upload'",
            "download_status": "TEXT NOT NULL DEFAULT 'manual_upload'",
            "verification_json": "TEXT NOT NULL DEFAULT '{}'",
        }
        for name, definition in migrations.items():
            if name not in columns:
                db.execute(f"ALTER TABLE reports ADD COLUMN {name} {definition}")
        for company in COMPANIES:
            db.execute(
                """INSERT OR IGNORE INTO companies
                   (company_id, name, exchange, symbol, sector, metadata_source)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    company.company_id,
                    company.name,
                    company.exchange,
                    company.symbol,
                    company.sector,
                    company.metadata_source,
                ),
            )


def upsert_company(company: Company) -> None:
    with connect() as db:
        db.execute(
            """INSERT INTO companies
               (company_id, name, exchange, symbol, sector, metadata_source)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(company_id) DO UPDATE SET
                   name=excluded.name, symbol=excluded.symbol,
                   sector=excluded.sector""",
            (
                company.company_id,
                company.name,
                company.exchange,
                company.symbol,
                company.sector,
                company.metadata_source,
            ),
        )


def get_company(company_id: str) -> Company | None:
    initialize()
    with connect() as db:
        row = db.execute(
            "SELECT * FROM companies WHERE company_id = ?", (company_id,)
        ).fetchone()
    return Company(**dict(row)) if row else None


def search_companies(query: str = "") -> list[Company]:
    initialize()
    pattern = f"%{query.strip()}%"
    with connect() as db:
        rows = db.execute(
            """SELECT * FROM companies
               WHERE name LIKE ? OR symbol LIKE ? OR company_id LIKE ?
               ORDER BY name LIMIT 50""",
            (pattern, pattern, pattern),
        ).fetchall()
    return [Company(**dict(row)) for row in rows]


def list_reports(company_id: str | None = None) -> list[dict]:
    initialize()
    with connect() as db:
        if company_id:
            rows = db.execute(
                "SELECT * FROM reports WHERE company_id=? ORDER BY financial_year DESC",
                (company_id,),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT * FROM reports ORDER BY retrieval_timestamp DESC"
            ).fetchall()
    return [dict(row) for row in rows]


def get_report(report_id: str) -> dict | None:
    initialize()
    with connect() as db:
        row = db.execute(
            "SELECT * FROM reports WHERE report_id = ?", (report_id,)
        ).fetchone()
    return dict(row) if row else None


def get_verified_report(company_id: str, financial_year: str, statement_basis: str) -> dict | None:
    initialize()
    with connect() as db:
        rows = db.execute(
            """SELECT * FROM reports WHERE company_id=? AND financial_year=?
               AND statement_basis=? AND download_status='verified'
               ORDER BY retrieval_timestamp DESC""",
            (company_id, financial_year, statement_basis),
        ).fetchall()
    for row in rows:
        report = dict(row)
        if Path(report["local_path"]).is_file():
            return report
    return None


def save_discovery(
    discovery_id: str,
    company_id: str,
    financial_year: str,
    statement_basis: str,
    status: str,
    candidates: list[dict],
    errors: list[str],
    created_at: str,
) -> None:
    import json

    with connect() as db:
        db.execute(
            """INSERT INTO report_discoveries
               (discovery_id, company_id, financial_year, statement_basis,
                status, candidates_json, errors_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                discovery_id,
                company_id,
                financial_year,
                statement_basis,
                status,
                json.dumps(candidates),
                json.dumps(errors),
                created_at,
                created_at,
            ),
        )


def update_discovery(
    discovery_id: str,
    status: str,
    candidates: list[dict],
    errors: list[str],
) -> None:
    import json

    from datetime import datetime, timezone

    with connect() as db:
        db.execute(
            """UPDATE report_discoveries SET status=?, candidates_json=?,
               errors_json=?, updated_at=? WHERE discovery_id=?""",
            (
                status,
                json.dumps(candidates),
                json.dumps(errors),
                datetime.now(timezone.utc).isoformat(),
                discovery_id,
            ),
        )


def get_discovery(discovery_id: str) -> dict | None:
    import json

    initialize()
    with connect() as db:
        row = db.execute(
            "SELECT * FROM report_discoveries WHERE discovery_id=?",
            (discovery_id,),
        ).fetchone()
    if not row:
        return None
    discovery = dict(row)
    discovery["candidates"] = json.loads(discovery.pop("candidates_json"))
    discovery["errors"] = json.loads(discovery.pop("errors_json"))
    return discovery


# ---------------------------------------------------------------------------
# Conversational Research Sessions, Messages, Findings, Disclosures Persistence
# ---------------------------------------------------------------------------

def create_research_session(
    symbol: str | None = None,
    title: str | None = None,
    peers: list[str] | None = None,
    metadata: dict | None = None,
    target_company: str | None = None,
    session_peers: list[str] | None = None,
) -> dict:
    import uuid
    import json
    from datetime import datetime, timezone

    initialize()
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    clean_sym = (symbol or target_company or "").strip().upper()
    if not clean_sym:
        raise ValueError("A company symbol (symbol or target_company) is required to create a session.")
    auto_title = title or f"{clean_sym} Research Investigation"
    peer_list = peers if peers is not None else session_peers
    peers_json = json.dumps(peer_list or [])
    meta_json = json.dumps(metadata or {})

    with connect() as db:
        db.execute(
            """INSERT INTO research_sessions
               (session_id, title, company_symbol, selected_peers, created_at, updated_at, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (session_id, auto_title, clean_sym, peers_json, now, now, meta_json),
        )

    return get_research_session(session_id)


def get_research_session(session_id: str) -> dict | None:
    import json

    initialize()
    with connect() as db:
        row = db.execute(
            "SELECT * FROM research_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if not row:
        return None
    res = dict(row)
    peers = json.loads(res.get("selected_peers") or "[]")
    res["selected_peers"] = peers
    res["session_peers"] = peers
    res["target_company"] = res.get("company_symbol")
    res["metadata"] = json.loads(res.pop("metadata_json", "{}") or "{}")
    return res


def list_research_sessions(limit: int = 50) -> list[dict]:
    import json

    initialize()
    with connect() as db:
        rows = db.execute(
            "SELECT * FROM research_sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        peers = json.loads(d.get("selected_peers") or "[]")
        d["selected_peers"] = peers
        d["session_peers"] = peers
        d["target_company"] = d.get("company_symbol")
        d["metadata"] = json.loads(d.pop("metadata_json", "{}") or "{}")
        result.append(d)
    return result



def update_session_peers(session_id: str, peers: list[str]) -> dict | None:
    import json
    from datetime import datetime, timezone

    initialize()
    now = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        db.execute(
            "UPDATE research_sessions SET selected_peers = ?, updated_at = ? WHERE session_id = ?",
            (json.dumps(peers), now, session_id),
        )
    return get_research_session(session_id)


def delete_research_session(session_id: str) -> bool:
    initialize()
    with connect() as db:
        db.execute("DELETE FROM session_messages WHERE session_id = ?", (session_id,))
        db.execute("DELETE FROM session_findings WHERE session_id = ?", (session_id,))
        cur = db.execute("DELETE FROM research_sessions WHERE session_id = ?", (session_id,))
        return cur.rowcount > 0


def save_session_message(
    session_id: str,
    role: str,
    content: str,
    intent: str | None = None,
    evidence_references: list[str] | None = None,
    citations: list[dict] | None = None,
    structured_payload: dict | None = None,
) -> dict:
    import uuid
    import json
    from datetime import datetime, timezone

    initialize()
    message_id = f"msg_{uuid.uuid4().hex[:14]}"
    now = datetime.now(timezone.utc).isoformat()
    ev_json = json.dumps(evidence_references or [])
    cit_json = json.dumps(citations or [])
    payload_json = json.dumps(structured_payload or {})

    with connect() as db:
        db.execute(
            """INSERT INTO session_messages
               (message_id, session_id, role, content, intent, evidence_references, citations, structured_payload, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (message_id, session_id, role, content, intent, ev_json, cit_json, payload_json, now),
        )
        db.execute(
            "UPDATE research_sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id),
        )

    return {
        "message_id": message_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "intent": intent,
        "evidence_references": evidence_references or [],
        "citations": citations or [],
        "structured_payload": structured_payload or {},
        "metadata": structured_payload or {},
        "created_at": now,
    }


def get_session_messages(session_id: str, limit: int = 100) -> list[dict]:
    import json

    initialize()
    with connect() as db:
        rows = db.execute(
            "SELECT * FROM session_messages WHERE session_id = ? ORDER BY created_at ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["evidence_references"] = json.loads(d.get("evidence_references") or "[]")
        d["citations"] = json.loads(d.get("citations") or "[]")
        d["structured_payload"] = json.loads(d.get("structured_payload") or "{}")
        d["metadata"] = d["structured_payload"]
        result.append(d)
    return result


def save_session_finding(
    session_id: str,
    topic: str,
    claim: str,
    claim_type: str,
    supporting_evidence: list[dict] | None = None,
    confidence: float = 1.0,
) -> dict:
    import uuid
    import json
    from datetime import datetime, timezone

    initialize()
    finding_id = f"find_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    ev_json = json.dumps(supporting_evidence or [])

    with connect() as db:
        db.execute(
            """INSERT INTO session_findings
               (finding_id, session_id, topic, claim, claim_type, supporting_evidence, confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (finding_id, session_id, topic, claim, claim_type, ev_json, confidence, now),
        )

    return {
        "finding_id": finding_id,
        "session_id": session_id,
        "topic": topic,
        "claim": claim,
        "finding_text": claim,
        "claim_type": claim_type,
        "supporting_evidence": supporting_evidence or [],
        "confidence": confidence,
        "created_at": now,
    }


def get_session_findings(session_id: str) -> list[dict]:
    import json

    initialize()
    with connect() as db:
        rows = db.execute(
            "SELECT * FROM session_findings WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["supporting_evidence"] = json.loads(d.get("supporting_evidence") or "[]")
        d["finding_text"] = d.get("claim", "")
        result.append(d)
    return result


def save_nse_disclosure(
    symbol: str,
    company_name: str,
    title: str,
    event_type: str,
    source_url: str,
    publication_date: str | None = None,
    attachment_url: str | None = None,
    local_path: str | None = None,
    checksum: str | None = None,
    extracted_text: str | None = None,
    material_facts: dict | None = None,
    status: str = "discovered",
    error: str | None = None,
) -> dict:
    import uuid
    import json
    from datetime import datetime, timezone

    initialize()
    disclosure_id = f"nse_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    facts_json = json.dumps(material_facts or {})

    with connect() as db:
        db.execute(
            """INSERT INTO nse_disclosures
               (disclosure_id, symbol, company_name, title, event_type, publication_date,
                source_url, attachment_url, local_path, checksum, extracted_text,
                material_facts, retrieved_at, status, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                disclosure_id, symbol.upper(), company_name, title, event_type, publication_date,
                source_url, attachment_url, local_path, checksum, extracted_text,
                facts_json, now, status, error,
            ),
        )

    return {
        "disclosure_id": disclosure_id,
        "symbol": symbol.upper(),
        "company_name": company_name,
        "title": title,
        "event_type": event_type,
        "publication_date": publication_date,
        "source_url": source_url,
        "attachment_url": attachment_url,
        "local_path": local_path,
        "checksum": checksum,
        "extracted_text": extracted_text,
        "material_facts": material_facts or {},
        "retrieved_at": now,
        "status": status,
        "error": error,
    }


def list_nse_disclosures(
    symbol: str | None = None,
    event_type: str | None = None,
    limit: int = 50,
    company_symbol: str | None = None,
) -> list[dict]:
    import json

    initialize()
    target_sym = (symbol or company_symbol or "").strip().upper()
    query = "SELECT * FROM nse_disclosures WHERE 1=1"
    params = []
    if target_sym:
        query += " AND symbol = ?"
        params.append(target_sym)
    if event_type:
        query += " AND event_type = ?"
        params.append(event_type.strip())
    query += " ORDER BY publication_date DESC, retrieved_at DESC LIMIT ?"
    params.append(limit)

    with connect() as db:
        rows = db.execute(query, params).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["material_facts"] = json.loads(d.get("material_facts") or "{}")
        result.append(d)
    return result


def get_nse_disclosure(disclosure_id: str) -> dict | None:
    import json

    initialize()
    with connect() as db:
        row = db.execute(
            "SELECT * FROM nse_disclosures WHERE disclosure_id = ?",
            (disclosure_id,),
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["material_facts"] = json.loads(d.get("material_facts") or "{}")
    return d
