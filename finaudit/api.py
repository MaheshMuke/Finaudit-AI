import os
import sys
import json
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import uuid
import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from finaudit.activity_stream import get_event_hub

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finaudit.analysis_engine import AnalysisEngine
from finaudit.agent_orchestrator import AgentOrchestrator
from finaudit.service import (
    create_report_discovery,
    download_discovered_report,
    execute_report_discovery,
)
from finaudit.company_registry import CompanyRecord, get_company_registry
from finaudit.commodity_service import get_commodity_service
from finaudit.llm_gateway import get_llm_gateway
from finaudit.news_service import get_news_service
from finaudit.portfolio_impact_service import PortfolioImpactService
from finaudit.scenario_engine import ScenarioEngine
from finaudit.sector_intelligence import SectorIntelligence
from finaudit.sector_opportunity_service import SectorOpportunityService
from finaudit.watchlist_service import get_watchlist_service
from finaudit.database import (
    create_research_session,
    get_research_session,
    list_research_sessions,
    update_session_peers,
    delete_research_session,
    get_session_messages,
    get_session_findings,
    list_nse_disclosures,
    save_nse_disclosure,
)
from finaudit.macro_service import MacroMarketIntelligenceService
from finaudit.research_sources import OfficialQuarterlyDisclosureAdapter

import math

app = FastAPI(title="FinAudit AI API", version="1.0.0")

def sanitize_for_json(obj):
    """Recursively replaces NaN, Infinity, and -Infinity with None so json.dumps succeeds."""
    if obj is None:
        return None
    try:
        import numpy as np
        if isinstance(obj, (np.floating, float)):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return [sanitize_for_json(v) for v in obj.tolist()]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
    except Exception:
        pass

    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            # Replace NaNs/Infs in the dataframe before converting
            cleaned_df = obj.replace([float('inf'), float('-inf')], None).where(pd.notnull(obj), None)
            return sanitize_for_json(cleaned_df.to_dict(orient="records"))
        elif isinstance(obj, pd.Series):
            cleaned_s = obj.replace([float('inf'), float('-inf')], None).where(pd.notnull(obj), None)
            return sanitize_for_json(cleaned_s.to_dict())
    except Exception:
        pass

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]
    return obj

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = AnalysisEngine()
registry = get_company_registry()
commodity_service = get_commodity_service()
news_service = get_news_service()
watchlist_service = get_watchlist_service()
llm_gateway = get_llm_gateway()
impact_service = PortfolioImpactService()
radar_service = SectorOpportunityService()
agent_orchestrator = AgentOrchestrator()


class WatchlistAddRequest(BaseModel):
    symbol: str
    company_name: Optional[str] = None
    list_name: str = "default"
    quantity: Optional[float] = Field(default=None, ge=0)
    average_buy_price: Optional[float] = Field(default=None, ge=0)


class PortfolioImpactRequest(BaseModel):
    question: str
    symbols: Optional[List[str]] = None


class InvestigationRequest(BaseModel):
    question: str
    company: str


class ReportDiscoveryRequest(BaseModel):
    company_id: str
    financial_year: str
    statement_basis: str = "consolidated"


class ScenarioRequest(BaseModel):
    industry: str = "General"
    base_revenue: Optional[float] = None
    base_net_income: Optional[float] = None
    base_operating_income: Optional[float] = None
    base_total_debt: Optional[float] = None
    input_cost_shock_pct: float = 5.0
    revenue_shock_pct: float = -5.0
    borrowing_cost_shock_bps: float = 100.0


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None
    target_company: str
    session_peers: Optional[List[str]] = None


class SessionMessageRequest(BaseModel):
    content: str
    custom_peers: Optional[List[str]] = None


class UpdateSessionPeersRequest(BaseModel):
    peers: List[str]


class MultiDisclosureQueryRequest(BaseModel):
    companies: List[str]
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    event_types: Optional[List[str]] = None
    extract_content: bool = True



@app.get("/health")
def health():
    hw = llm_gateway.check_health()
    return {
        "status": "ok",
        "provider": "yfinance_local_dataset",
        "llm": hw,
    }


@app.post("/investigations/company")
def investigate_company(req: InvestigationRequest):
    question = req.question.strip()
    company = req.company.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not company:
        raise HTTPException(status_code=400, detail="Company cannot be empty.")
    return sanitize_for_json(
        agent_orchestrator.investigate_company(company, question=question)
    )


@app.post("/reports/discover", status_code=202)
def discover_reports(req: ReportDiscoveryRequest):
    try:
        discovery_id = create_report_discovery(
            req.company_id, req.financial_year, req.statement_basis
        )
        execute_report_discovery(discovery_id)
        return {"discovery_id": discovery_id, "status": "accepted"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/reports/discover/{discovery_id}")
def get_discovery_status(discovery_id: str):
    from finaudit.database import get_discovery

    discovery = get_discovery(discovery_id)
    if not discovery:
        raise HTTPException(status_code=404, detail="Discovery not found")
    return sanitize_for_json(discovery)


@app.post("/reports/discover/{discovery_id}/candidates/{candidate_id}/download", status_code=202)
def download_report_candidate(discovery_id: str, candidate_id: str):
    try:
        return sanitize_for_json(download_discovered_report(discovery_id, candidate_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/companies/search")
def search_companies(q: str = ""):
    results = registry.search(q, limit=20)
    return [r.dict() for r in results]


@app.get("/companies/{symbol}")
def get_company(symbol: str):
    rec = registry.get_by_symbol(symbol)
    if not rec:
        raise HTTPException(status_code=404, detail="Company not found")
    return rec.dict()


@app.get("/financial-data/{symbol}/status")
def financial_data_status(symbol: str):
    company = registry.get_by_symbol(symbol)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    facts = engine.data_manager.screener.get_cached_facts(company.symbol)
    fresh_facts = engine.data_manager.screener.get_cached_facts(
        company.symbol, fresh_only=True
    )
    return sanitize_for_json(
        {
            "symbol": company.symbol,
            "provider": "screener" if facts else "unknown",
            "original_source": "Screener" if facts else None,
            "cache_layer": "local_financial_facts" if facts else "none",
            "cache_status": "fresh" if fresh_facts else "stale_or_missing",
            "fact_count": len(facts),
            "fresh_fact_count": len(fresh_facts),
            "sections": sorted({fact["statement"] for fact in facts}),
            "period_types": sorted({fact["period_type"] for fact in facts}),
            "last_retrieved_at": max(
                (fact["retrieved_at"] for fact in facts), default=None
            ),
            "warnings": sorted(
                {
                    warning
                    for fact in facts
                    for warning in json.loads(
                        fact.get("data_quality_warnings") or "[]"
                    )
                }
            ),
        }
    )


@app.get("/analyses/{symbol}")
def analyze_company(symbol: str):
    result = engine.run_full_analysis(symbol)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error"))

    comp = result.get("company", {})
    calc = result.get("calculations", {})
    ratios = calc.get("latest_ratios", {})

    # Sector Framework
    sector_fw = SectorIntelligence.get_framework(comp.get("industry", "General"))

    # Initial Scenarios
    scenarios_data = ScenarioEngine.run_sensitivity(
        base_revenue=ratios.get("revenue"),
        base_net_income=ratios.get("net_income"),
        base_operating_income=ratios.get("operating_income"),
        base_total_debt=ratios.get("total_debt"),
        industry=comp.get("industry", "General"),
    )

    # Convert any raw statements DataFrames to dicts for clean JSON serialization
    raw_stmts = result.get("statements", {})
    serialized_stmts = {}
    if isinstance(raw_stmts, dict):
        for k, v in raw_stmts.items():
            if hasattr(v, "to_dict"):
                serialized_stmts[k] = v.to_dict(orient="records")
            else:
                serialized_stmts[k] = v

    ai_intel = result.get("ai_analysis") or {}

    response_payload = {
        "success": True,
        "company": comp,
        "market_data": result.get("market_data", {}),
        "calculations": calc,
        "competitors": result.get("competitors", {}),
        "sector_framework": sector_fw,
        "scenarios": scenarios_data,
        "ai_intelligence": ai_intel,
        "statements": serialized_stmts,
        "llm_provider": result.get("llm_provider"),
        "llm_metadata": result.get("llm_metadata"),
        "ai_error": result.get("ai_error"),
        "data_metadata": {
            "financial": calc.get("metric_metadata", {}),
            "market": {
                "source": result.get("market_data", {}).get("source"),
                "retrieved_at": result.get("market_data", {}).get("retrieved_at"),
                "status": result.get("market_data", {}).get("status"),
            },
        },
    }
    return sanitize_for_json(response_payload)


@app.get("/commodities")
def get_commodities():
    return commodity_service.get_indicators()


@app.get("/news")
def get_news(limit: int = 20):
    return news_service.get_recent_articles(limit=limit)


@app.post("/news/refresh")
def refresh_news():
    added = news_service.fetch_latest_feeds()
    return {"status": "ok", "new_articles": added}


@app.get("/watchlist")
def get_watchlist(list_name: str = "default"):
    return watchlist_service.get_items(list_name=list_name)


@app.post("/watchlist")
def add_to_watchlist(req: WatchlistAddRequest):
    rec = registry.resolve(req.symbol)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Company '{req.symbol}' was not found.")
    company_name = rec.company_name
    watchlist_service.add_item(
        symbol=req.symbol,
        company_name=company_name,
        list_name=req.list_name,
        quantity=req.quantity,
        average_buy_price=req.average_buy_price,
    )
    return {"status": "success", "symbol": req.symbol, "company_name": company_name}


@app.delete("/watchlist/{symbol}")
def delete_from_watchlist(symbol: str, list_name: str = "default"):
    watchlist_service.remove_item(symbol, list_name=list_name)
    return {"status": "success", "removed": symbol}


@app.post("/portfolio-impact")
def analyze_portfolio_impact(req: PortfolioImpactRequest):
    symbols = req.symbols
    if not symbols:
        items = watchlist_service.get_items("default")
        symbols = [item["symbol"] for item in items]
    if not symbols:
        raise HTTPException(
            status_code=400,
            detail="No companies specified and watchlist is empty. Please add companies first."
        )
    return impact_service.analyze_impact(req.question, symbols)


@app.get("/sector-opportunities")
def get_sector_opportunities():
    return sanitize_for_json(radar_service.get_sector_radar())


@app.post("/scenarios/simulate")
def simulate_scenario(req: ScenarioRequest):
    res = ScenarioEngine.run_sensitivity(
        base_revenue=req.base_revenue,
        base_net_income=req.base_net_income,
        base_operating_income=req.base_operating_income,
        base_total_debt=req.base_total_debt,
        industry=req.industry,
        input_cost_shock_pct=req.input_cost_shock_pct,
        revenue_shock_pct=req.revenue_shock_pct,
        rate_hike_bps=req.borrowing_cost_shock_bps,
    )
    return sanitize_for_json(res)


@app.post("/sessions")
def create_session_endpoint(req: CreateSessionRequest):
    comp = registry.resolve(req.target_company)
    symbol = comp.symbol if comp else req.target_company.strip().upper()
    title = req.title or f"Research: {symbol}"
    sess = create_research_session(
        title=title,
        target_company=symbol,
        session_peers=req.session_peers,
    )
    return sanitize_for_json(sess)


@app.get("/sessions")
def list_sessions_endpoint():
    return sanitize_for_json(list_research_sessions())


@app.get("/sessions/{session_id}")
def get_session_endpoint(session_id: str):
    sess = get_research_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = get_session_messages(session_id)
    findings = get_session_findings(session_id)
    return sanitize_for_json({
        "session": sess,
        "messages": messages,
        "findings": findings,
    })


@app.delete("/sessions/{session_id}")
def delete_session_endpoint(session_id: str):
    sess = get_research_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    delete_research_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.put("/sessions/{session_id}/peers")
def update_peers_endpoint(session_id: str, req: UpdateSessionPeersRequest):
    sess = get_research_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    clean_peers = [p.strip().upper() for p in req.peers if p.strip()]
    update_session_peers(session_id, clean_peers)
    return {"status": "success", "session_id": session_id, "session_peers": clean_peers}


@app.post("/sessions/{session_id}/messages")
def post_session_message_endpoint(session_id: str, req: SessionMessageRequest):
    sess = get_research_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    content = req.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    target_comp = sess.get("target_company", "")
    if not target_comp:
        raise HTTPException(status_code=400, detail="Session has no target company set.")
    res = agent_orchestrator.investigate_company(
        query=target_comp,
        question=content,
        session_id=session_id,
        custom_peers=req.custom_peers or sess.get("session_peers"),
    )
    return sanitize_for_json(res)


@app.post("/sessions/{session_id}/messages/stream")
def post_session_message_stream_endpoint(session_id: str, req: SessionMessageRequest):
    """Stream live research activity, tool events, document extraction, and final answer via SSE."""
    sess = get_research_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    content = req.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    request_id = f"req_{uuid.uuid4().hex[:12]}"
    target_comp = sess.get("target_company", "")
    if not target_comp:
        raise HTTPException(status_code=400, detail="Session has no target company set.")
    event_hub = get_event_hub()

    def run_investigation():
        try:
            agent_orchestrator.investigate_company(
                query=target_comp,
                question=content,
                session_id=session_id,
                custom_peers=req.custom_peers or sess.get("session_peers"),
                request_id=request_id,
            )
        except Exception as exc:
            event_hub.emit("activity_failed", session_id, request_id, {
                "activity_id": f"act_err_{uuid.uuid4().hex[:6]}",
                "source_name": "Agent Orchestrator",
                "source_category": "calculations",
                "task": "Investigating company",
                "status": "extraction_failed",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "error_details": str(exc),
                "limitations": [f"Encountered error during investigation: {exc}"],
            })
            event_hub.emit("done", session_id, request_id, {})

    worker = threading.Thread(target=run_investigation, daemon=True)
    worker.start()

    return StreamingResponse(
        event_hub.stream_sse(session_id, request_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/sessions/{session_id}/findings")
def get_session_findings_endpoint(session_id: str):
    return sanitize_for_json(get_session_findings(session_id))


@app.get("/sessions/{session_id}/disclosures")
def get_session_disclosures_endpoint(session_id: str):
    sess = get_research_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    sym = sess.get("target_company")
    disclosures = list_nse_disclosures(company_symbol=sym)
    return sanitize_for_json(disclosures)


@app.post("/disclosures/multi-query")
def multi_disclosure_query_endpoint(req: MultiDisclosureQueryRequest):
    if not req.companies:
        raise HTTPException(status_code=400, detail="Companies list cannot be empty")
    adapter = OfficialQuarterlyDisclosureAdapter()
    result = adapter.discover_multi(
        companies=req.companies,
        from_date=req.from_date,
        to_date=req.to_date,
        event_types=req.event_types,
        extract_content=req.extract_content,
    )
    for doc in result.get("documents", []):
        try:
            save_nse_disclosure(
                company_symbol=doc.get("company_symbol") or doc.get("symbol", ""),
                document_id=doc.get("document_id") or doc.get("title", ""),
                event_type=doc.get("event_type", "General Disclosure"),
                title=doc.get("title", ""),
                published_at=doc.get("date"),
                file_url=doc.get("file_url") or doc.get("original_url"),
                content_hash=doc.get("content_hash"),
                extracted_text=doc.get("extracted_content") or doc.get("text"),
                page_count=doc.get("page_count", 0),
                material_facts=doc.get("material_facts"),
            )
        except Exception:
            pass
    return sanitize_for_json(result)


@app.get("/macro/snapshot")
def get_macro_snapshot_endpoint():
    service = MacroMarketIntelligenceService()
    return sanitize_for_json(service.get_macro_snapshot())


@app.get("/macro/relative-performance")
def get_relative_performance_endpoint(target: str, peers: str = "", period: str = "1mo"):
    service = MacroMarketIntelligenceService()
    peer_list = [p.strip().upper() for p in peers.split(",") if p.strip()]
    return sanitize_for_json(service.get_peer_relative_performance(target.upper(), peer_list, period=period))


class RAGQueryRequest(BaseModel):
    query: str
    symbol: Optional[str] = None
    document_types: Optional[List[str]] = None
    collections: Optional[List[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    limit: int = 5


class RAGIndexRequest(BaseModel):
    symbol: Optional[str] = None
    index_news: bool = False


@app.get("/rag/status")
def get_rag_status_endpoint():
    from finaudit.rag import get_vector_store
    vs = get_vector_store()
    return sanitize_for_json(vs.get_status())


@app.post("/rag/query")
def post_rag_query_endpoint(req: RAGQueryRequest):
    from finaudit.rag import get_hybrid_retriever
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    retriever = get_hybrid_retriever()
    passages = retriever.search(
        query=req.query,
        symbol=req.symbol,
        document_types=req.document_types,
        collections=req.collections,
        date_from=req.date_from,
        date_to=req.date_to,
        limit=req.limit,
    )
    return sanitize_for_json({
        "query": req.query,
        "symbol": req.symbol,
        "count": len(passages),
        "passages": [
            {
                "chunk_id": p.chunk_id,
                "document_id": p.document_id,
                "title": p.title,
                "company_name": p.company_name,
                "symbol": p.symbol,
                "document_type": p.document_type,
                "publication_date": p.publication_date,
                "page_number": p.page_number,
                "section_heading": p.section_heading,
                "semantic_similarity": p.semantic_similarity,
                "keyword_score": p.keyword_score,
                "hybrid_score": p.hybrid_score,
                "source_url": p.source_url,
                "text": p.text,
            }
            for p in passages
        ],
        "citations": [p.to_citation() for p in passages],
    })


@app.post("/rag/index")
def post_rag_index_endpoint(req: RAGIndexRequest):
    from finaudit.rag import get_ingestion_service
    ingestion = get_ingestion_service()
    result: Dict[str, Any] = {"status": "success"}
    if req.symbol:
        res = ingestion.ingest_all_for_company(req.symbol)
        result["company_ingestion"] = res
    if req.index_news:
        news_res = ingestion.ingest_recent_news()
        result["news_ingestion"] = news_res
    return sanitize_for_json(result)

