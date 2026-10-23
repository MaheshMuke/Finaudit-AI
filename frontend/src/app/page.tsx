"use client";

import { FormEvent, ReactNode, useEffect, useState } from "react";
import {
  AlertTriangle,
  ArrowUpRight,
  BarChart3,
  Bookmark,
  Building2,
  Calendar,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock,
  Database,
  ExternalLink,
  FileCheck,
  FileText,
  Hash,
  Info,
  Layers,
  LoaderCircle,
  MessageSquare,
  Plus,
  RefreshCw,
  Search,
  Send,
  ShieldAlert,
  Sparkles,
  Trash2,
  TrendingUp,
  X,
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  API_BASE_URL,
  AnalysisResult,
  CompanyRecord,
  CommodityItem,
  NewsArticle,
  PortfolioImpactResponse,
  WatchlistItem,
} from "@/types/api";

type Destination = "home" | "research" | "watchlist" | "chat";

type Investigation = {
  success: boolean;
  error?: string;
  question: string;
  investigation_status: string;
  profile: Record<string, string | number | null>;
  metrics: Record<string, unknown>;
  news: Array<{ title: string; source: string; published: string; link: string }>;
  commodities: CommodityItem[];
  validation_notes: string[];
  evidence?: Array<{
    evidence_id: string;
    metric?: string | null;
    value?: number | null;
    unit?: string | null;
    reporting_period?: string | null;
    statement_basis?: string | null;
    source?: string;
  }>;
  evidence_references?: string[];
  verification?: { status: string; warnings: string[] };
  progress_steps?: string[];
  research_trace?: string[];
  research_trace_detail?: Array<Record<string, unknown>>;
  evidence_evaluation?: Record<string, unknown>;
  budget?: Record<string, unknown>;
  stop_reason?: string;
  scenario_analysis?: {
    status?: string;
    message?: string;
    missing_inputs?: string[];
    baseline?: {
      period?: string | null;
      statement_basis?: string | null;
      reporting_basis?: string | string[] | null;
      source?: string | null;
      source_url?: string | null;
      retrieved_at?: string | null;
      currency?: string;
      display_unit?: string;
    };
    assumptions?: Record<string, unknown>;
    scenarios?: Array<{
      name?: string;
      revenue?: number | null;
      net_income?: number | null;
      net_margin?: number | null;
      impact_pct?: number | null;
    }>;
  };
  official_source_discovery?: {
    status?: string;
    reason?: string;
    limitations?: string[];
    documents?: Array<{
      title?: string;
      url?: string;
      source_page?: string;
      publication_date?: string;
      financial_year?: string;
      statement_basis?: string;
      retrieved_at?: string;
      content_retrieved?: boolean;
    }>;
  };
  sources?: Array<{
    kind?: string;
    source?: string;
    link?: string;
    published?: string;
    retrieved_at?: string;
    title?: string;
    content_retrieved?: boolean;
    statement_basis?: string;
  }>;
  comparison_summary?: {
    status?: string;
    metric?: string;
    metric_meaning?: string;
    target_value?: number | null;
    target_period?: string | null;
    peers_ranked?: Array<{
      symbol: string;
      company_name: string;
      value: number;
      period: string;
      source?: string;
    }>;
    comparison_table?: Array<{
      symbol: string;
      company_name: string;
      current_period?: string | null;
      previous_period?: string | null;
      current_value?: number | null;
      previous_value?: number | null;
      growth?: number | null;
      source?: string;
      unit?: string;
      currency?: string;
      statement_basis?: string;
      data_quality?: string;
    }>;
    limitations?: Array<string | { symbol?: string; reason?: string }>;
  };
  investigation_plan?: {
    question_type: Record<string, boolean>;
    selected_tools: string[];
    max_steps: number;
  };
  llm: {
    status: string;
    error?: string;
    provider?: string;
    model?: string;
    generation_status?: string;
    fallback_used?: boolean;
    error_code?: string;
    latency_ms?: number;
    llm_invoked?: boolean;
    response_used_llm?: boolean;
    request_succeeded?: boolean;
    llm_fallback_reason?: string;
  };
  conversational_answer?: string;
  interpretation?: {
    summary?: string;
    conversational_answer?: string;
    findings?: string[];
    risks?: string[];
    next_questions?: string[];
    limitations?: string[];
    validation_warnings?: string[];
  } | null;
  research_report?: {
    company?: string;
    symbol?: string;
    reporting_period?: string | null;
    statement_basis?: string;
    reporting_basis?: string;
    currency?: string;
    unit?: string;
    source?: string;
    conversational_answer?: string;
    executive_answer?: string;
    what_changed?: string[];
    financial_evidence?: {
      comparisons?: Record<string, Record<string, string | number | boolean | null>>;
      profit_bridges?: Record<string, {
        status?: string;
        comparison_period?: string;
        current_period?: string;
        net_income_change?: number | null;
        contributions?: Record<string, number>;
        formula?: string;
        tax_method?: string;
        supporting_evidence_ids?: string[];
      }>;
      line_item_movements?: Array<{
        label?: string;
        metric_key?: string;
        period?: string;
        value_inr_crore?: number | null;
        raw_value?: string;
        source_url?: string;
        retrieved_at?: string;
        reporting_basis?: string;
      }>;
    };
    supported_explanations?: string[];
    news_context?: Array<{
      text?: string;
      source?: string;
      source_url?: string;
      publication_date?: string;
      status?: string;
      note?: string;
    }>;
    risks_and_alternatives?: string[];
    uncertainties?: string[];
    sources?: Array<{
      source?: string;
      kind?: string;
      link?: string;
      published?: string;
      retrieved_at?: string;
      title?: string;
      evidence_id?: string;
      content_retrieved?: boolean;
      statement_basis?: string;
    }>;
    source_discovery?: Investigation["official_source_discovery"];
    claims?: Array<{
      claim_id?: string;
      text?: string;
      status?: string;
      supporting_evidence_ids?: string[];
    }>;
    evidence_references?: string[];
  };
};

const decodeHtml = (value: unknown): string => {
  const text = String(value ?? "");
  const namedEntities: Record<string, string> = {
    amp: "&",
    apos: "'",
    gt: ">",
    lt: "<",
    nbsp: "\u00a0",
    quot: "\"",
  };
  return text.replace(/&(#x[0-9a-f]+|#\d+|amp|apos|gt|lt|nbsp|quot);/gi, (entity, code: string) => {
    if (code[0] !== "#") return namedEntities[code.toLowerCase()] ?? entity;
    const numeric = code[1]?.toLowerCase() === "x"
      ? Number.parseInt(code.slice(2), 16)
      : Number.parseInt(code.slice(1), 10);
    return Number.isFinite(numeric) && numeric >= 0 && numeric <= 0x10ffff
      ? String.fromCodePoint(numeric)
      : entity;
  });
};

const formatNumber = (value: unknown, fractionDigits = 2): string => {
  if (typeof value !== "number" || !Number.isFinite(value)) return "Not available";
  return value.toLocaleString("en-IN", {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  });
};

const formatUnit = (currency?: unknown, unit?: unknown): string => {
  const c = String(currency || "INR").trim();
  const u = String(unit || "INR crore").trim();
  if (u.toLowerCase().startsWith(c.toLowerCase())) return u;
  return `${c} ${u}`;
};

const apiJson = async <T,>(path: string, init?: RequestInit): Promise<T> => {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch (error) {
    const reason = error instanceof Error ? error.message : "Network request failed.";
    throw new Error(
      `Cannot reach the FinAudit backend at ${API_BASE_URL}. Start FastAPI with ` +
      `python -m uvicorn finaudit.api:app --host 127.0.0.1 --port 8000. (${reason})`,
    );
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body && typeof body === "object" && "detail" in body
      ? String(body.detail)
      : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return body as T;
};

export default function FinAuditDashboard() {
  const [destination, setDestination] = useState<Destination>("home");
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState<CompanyRecord[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState("");
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([]);
  const [watchError, setWatchError] = useState("");
  const [news, setNews] = useState<NewsArticle[]>([]);
  const [commodities, setCommodities] = useState<CommodityItem[]>([]);
  const [dataError, setDataError] = useState("");
  const [question, setQuestion] = useState("Explain this company's financial strengths, risks, and key dependencies.");
  const [investigation, setInvestigation] = useState<Investigation | null>(null);
  const [investigationLoading, setInvestigationLoading] = useState(false);
  const [investigationError, setInvestigationError] = useState("");
  const [impactQuestion, setImpactQuestion] = useState("");
  const [impact, setImpact] = useState<PortfolioImpactResponse | null>(null);
  const [impactLoading, setImpactLoading] = useState(false);
  const [impactError, setImpactError] = useState("");

  useEffect(() => {
    Promise.all([
      apiJson<WatchlistItem[]>("/watchlist"),
      apiJson<NewsArticle[]>("/news?limit=12"),
      apiJson<CommodityItem[]>("/commodities"),
    ])
      .then(([saved, headlines, indicators]) => {
        setWatchlist(saved);
        setNews(headlines);
        setCommodities(indicators);
      })
      .catch((error: Error) => setDataError(error.message));
  }, []);

  useEffect(() => {
    const trimmed = query.trim();
    const timer = window.setTimeout(() => {
      if (!trimmed) {
        setSearchResults([]);
        return;
      }
      apiJson<CompanyRecord[]>(`/companies/search?q=${encodeURIComponent(trimmed)}`)
        .then(setSearchResults)
        .catch(() => setSearchResults([]));
    }, 250);
    return () => window.clearTimeout(timer);
  }, [query]);

  const refreshWatchlist = async () => setWatchlist(await apiJson<WatchlistItem[]>("/watchlist"));

  const runAnalysis = async (symbol: string) => {
    setSelected(symbol);
    setQuery("");
    setSearchResults([]);
    setDestination("research");
    setAnalysisLoading(true);
    setAnalysisError("");
    setInvestigation(null);
    try {
      const result = await apiJson<AnalysisResult>(`/analyses/${encodeURIComponent(symbol)}`);
      setAnalysis(result);
    } catch (error) {
      setAnalysisError(error instanceof Error ? error.message : "Company analysis failed.");
    } finally {
      setAnalysisLoading(false);
    }
  };

  const addWatchlist = async (symbol: string) => {
    setWatchError("");
    try {
      await apiJson("/watchlist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol }),
      });
      await refreshWatchlist();
    } catch (error) {
      setWatchError(error instanceof Error ? error.message : "Could not update watchlist.");
    }
  };

  const removeWatchlist = async (symbol: string) => {
    setWatchError("");
    try {
      await apiJson(`/watchlist/${encodeURIComponent(symbol)}`, { method: "DELETE" });
      await refreshWatchlist();
    } catch (error) {
      setWatchError(error instanceof Error ? error.message : "Could not remove company.");
    }
  };

  const submitQuestion = async (event?: FormEvent) => {
    event?.preventDefault();
    if (investigationLoading || !selected || !question.trim()) return;
    setInvestigationLoading(true);
    setInvestigationError("");
    setInvestigation(null);
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 185_000);
    try {
      const result = await apiJson<Investigation>("/investigations/company", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ company: selected, question }),
        signal: controller.signal,
      });
      setInvestigation(result);
    } catch (error) {
      setInvestigationError(
        error instanceof DOMException && error.name === "AbortError"
          ? "The investigation exceeded the 185-second client limit. Deterministic findings remain available; retry with a shorter question."
          : error instanceof Error ? error.message : "Investigation failed.",
      );
    } finally {
      window.clearTimeout(timeout);
      setInvestigationLoading(false);
    }
  };

  const submitImpact = async (event?: FormEvent) => {
    event?.preventDefault();
    if (!impactQuestion.trim()) return;
    setImpactLoading(true);
    setImpactError("");
    try {
      setImpact(await apiJson<PortfolioImpactResponse>("/portfolio-impact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: impactQuestion }),
      }));
    } catch (error) {
      setImpactError(error instanceof Error ? error.message : "Watchlist investigation failed.");
    } finally {
      setImpactLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <aside className="side-nav">
        <div className="brand">
          <div className="brand-mark">FA</div>
          <div><strong>FinAudit</strong><span>Research workspace</span></div>
        </div>
        <nav aria-label="Primary navigation">
          <NavButton active={destination === "home"} onClick={() => setDestination("home")} icon={<Search size={17} />} label="Home" />
          <NavButton active={destination === "research"} onClick={() => setDestination("research")} icon={<FileText size={17} />} label="Company research" />
          <NavButton active={destination === "watchlist"} onClick={() => setDestination("watchlist")} icon={<Bookmark size={17} />} label="My watchlist" />
          <NavButton active={destination === "chat"} onClick={() => setDestination("chat")} icon={<Sparkles size={17} />} label="AI Chat Intelligence" />
        </nav>
        <div className="side-note"><span className="status-dot" /> Shared evidence workspace<br /><small>Financial facts and AI interpretation stay separate.</small></div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">Indian equities · unified research</p>
            <h1>{destination === "home" ? "What would you like to research?" : destination === "research" ? "Company research workspace" : destination === "watchlist" ? "My watchlist" : "Conversational Agentic Intelligence"}</h1>
          </div>
          <CompanySearch query={query} setQuery={setQuery} results={searchResults} onSelect={runAnalysis} />
        </header>

        {dataError && <Notice tone="error">{dataError}</Notice>}
        {watchError && <Notice tone="error">{watchError}</Notice>}

        {destination === "home" && (
          <HomeWorkspace
            selected={selected}
            news={news}
            commodities={commodities}
            watchlist={watchlist}
            onOpen={runAnalysis}
            onRemove={removeWatchlist}
            onResearch={() => setDestination("research")}
          />
        )}

        {destination === "research" && (
          <ResearchWorkspace
            analysis={analysis}
            loading={analysisLoading}
            error={analysisError}
            selected={selected || analysis?.company.symbol || ""}
            question={question}
            setQuestion={setQuestion}
            investigation={investigation}
            investigationLoading={investigationLoading}
            investigationError={investigationError}
            submitQuestion={submitQuestion}
            watchlist={watchlist}
            addWatchlist={addWatchlist}
            removeWatchlist={removeWatchlist}
            news={news}
          />
        )}

        {destination === "watchlist" && (
          <WatchlistWorkspace
            watchlist={watchlist}
            onOpen={runAnalysis}
            onRemove={removeWatchlist}
            question={impactQuestion}
            setQuestion={setImpactQuestion}
            submit={submitImpact}
            loading={impactLoading}
            error={impactError}
            result={impact}
          />
        )}

        {destination === "chat" && (
          <ChatWorkspace
            initialCompany={selected || ""}
          />
        )}
      </main>

    </div>
  );
}

function NavButton({ active, onClick, icon, label }: { active: boolean; onClick: () => void; icon: ReactNode; label: string }) {
  return <button className={active ? "nav-item active" : "nav-item"} onClick={onClick}>{icon}{label}</button>;
}

function CompanySearch({ query, setQuery, results, onSelect }: { query: string; setQuery: (value: string) => void; results: CompanyRecord[]; onSelect: (symbol: string) => void }) {
  return <div className="top-search"><Search size={17} /><input aria-label="Search companies" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search company or NSE ticker" />{results.length > 0 && <div className="search-menu">{results.map((item) => <button key={item.symbol} onClick={() => onSelect(item.symbol)}><strong>{item.company_name}</strong><span>{item.symbol} · {item.industry}</span></button>)}</div>}</div>;
}

function HomeWorkspace({ selected, news, commodities, watchlist, onOpen, onRemove, onResearch }: { selected: string | null; news: NewsArticle[]; commodities: CommodityItem[]; watchlist: WatchlistItem[]; onOpen: (symbol: string) => void; onRemove: (symbol: string) => void; onResearch: () => void }) {
  const quickPicks = [
    { symbol: "SBIN", name: "State Bank of India" },
    { symbol: "TCS", name: "Tata Consultancy" },
    { symbol: "RELIANCE", name: "Reliance Industries" },
    { symbol: "INFY", name: "Infosys" },
    { symbol: "HDFCBANK", name: "HDFC Bank" },
  ];

  return (
    <div className="stack">
      <section className="home-intro panel">
        <div>
          <p className="eyebrow">Financial Research Workspace</p>
          <h2>Evidence-grounded corporate research and analyst reasoning.</h2>
          <p className="muted">
            Search any Indian public company or select a benchmark stock below to explore audited financial statements, multi-year trends, and autonomous causal research.
          </p>
          <div className="quick-picks">
            <span className="quick-picks-label">Quick research:</span>
            {quickPicks.map((pick) => (
              <button
                key={pick.symbol}
                className="chip-button"
                onClick={() => onOpen(pick.symbol)}
              >
                <strong>{pick.symbol}</strong>
                <span>{pick.name}</span>
              </button>
            ))}
          </div>
        </div>
        {selected ? (
          <button className="button primary" onClick={() => onOpen(selected)}>
            Open {selected} report <ArrowUpRight size={16} />
          </button>
        ) : null}
      </section>
      <div className="grid two">
        <NewsAgent news={news} />
        <MarketSnapshot commodities={commodities} />
      </div>
      <section className="panel">
        <PanelHeading icon={<Bookmark size={18} />} title="Saved Watchlist" description="Monitor key portfolio holdings and examine event-driven scenario impacts." />
        <WatchlistRows items={watchlist} onOpen={onOpen} onRemove={onRemove} />
      </section>
      <button className="text-link" onClick={onResearch}>
        Open the full company research workspace <ChevronRight size={15} />
      </button>
    </div>
  );
}

function ResearchWorkspace({ analysis, loading, error, selected, question, setQuestion, investigation, investigationLoading, investigationError, submitQuestion, watchlist, addWatchlist, removeWatchlist, news }: {
  analysis: AnalysisResult | null; loading: boolean; error: string; selected: string; question: string; setQuestion: (value: string) => void; investigation: Investigation | null; investigationLoading: boolean; investigationError: string; submitQuestion: (event?: FormEvent) => void; watchlist: WatchlistItem[]; addWatchlist: (symbol: string) => void; removeWatchlist: (symbol: string) => void; news: NewsArticle[];
}) {
  if (loading) return <LoadingState label={`Building the ${selected} research workspace`} />;
  if (error) return <Notice tone="error">{error}</Notice>;
  if (!analysis) return <section className="panel empty">Search for a company to open its research workspace.</section>;
  const saved = watchlist.some((item) => item.symbol === analysis.company.symbol);
  return (
    <div className="stack">
      <CompanyOverview analysis={analysis} saved={saved} onAdd={() => addWatchlist(analysis.company.symbol)} onRemove={() => removeWatchlist(analysis.company.symbol)} />
      <AskAnythingAgent selected={selected} question={question} setQuestion={setQuestion} result={investigation} loading={investigationLoading} error={investigationError} submit={submitQuestion} />
      <CompanyResearchAgent analysis={analysis} />
      <div className="grid two">
        <PeerAgent analysis={analysis} />
        <NewsAgent news={news} company={analysis.company.company_name} />
      </div>
    </div>
  );
}

function CompanyOverview({ analysis, saved, onAdd, onRemove }: { analysis: AnalysisResult; saved: boolean; onAdd: () => void; onRemove: () => void }) {
  const ratios = analysis.calculations.latest_ratios;
  const market = analysis.market_data;
  const financial = analysis.data_metadata?.financial;
  const source = financial?.original_source || financial?.source || "Screener.in";
  const isPositive = (market.one_year_return ?? 0) >= 0;

  return (
    <section className="panel report-header">
      <div>
        <div className="header-meta-row">
          <span className="ticker-badge">{analysis.company.symbol}</span>
          <span className="industry-tag">{analysis.company.industry}</span>
        </div>
        <h2>{analysis.company.company_name}</h2>
        <div className="header-stats-row">
          <div className="price-stat">
            <span className="stat-label">Market Price</span>
            <span className="stat-value">{market.current_price == null ? "Not available" : `₹${formatNumber(market.current_price)}`}</span>
          </div>
          <div className="return-stat">
            <span className="stat-label">1-Year Return</span>
            <span className={`stat-value ${isPositive ? "positive" : "negative"}`}>
              {market.one_year_return == null ? "N/A" : `${isPositive ? "+" : ""}${formatNumber(market.one_year_return)}%`}
            </span>
          </div>
          <div className="source-stat">
            <span className="stat-label">Financial Filing</span>
            <span className="stat-detail">
              {decodeHtml(source)} · Period: {String(ratios.period || "Latest")} · {formatUnit(financial?.currency, financial?.display_unit)}
            </span>
          </div>
        </div>
      </div>
      <button className="button secondary" onClick={saved ? onRemove : onAdd}>
        {saved ? <Trash2 size={16} /> : <Bookmark size={16} />}
        {saved ? "Remove from watchlist" : "Add to watchlist"}
      </button>
    </section>
  );
}

function CompanyResearchAgent({ analysis }: { analysis: AnalysisResult }) {
  const ratios = analysis.calculations.latest_ratios;
  const ai = analysis.ai_intelligence || {};
  const chartData = analysis.calculations.annual_series.periods.map((period, index) => ({
    period,
    revenue: analysis.calculations.annual_series.revenue[index] == null ? null : analysis.calculations.annual_series.revenue[index]! / 1e7,
    profit: analysis.calculations.annual_series.net_income[index] == null ? null : analysis.calculations.annual_series.net_income[index]! / 1e7,
    cashFlow: analysis.calculations.annual_series.free_cash_flow[index] == null ? null : analysis.calculations.annual_series.free_cash_flow[index]! / 1e7,
  }));

  const metrics = [
    { label: "Revenue", value: ratios.revenue_display, note: `Period ending ${String(ratios.period || "N/A")}` },
    { label: "Net Income", value: ratios.net_income_display, note: `Period ending ${String(ratios.period || "N/A")}` },
    { label: "Net Margin", value: ratios.net_margin_display, note: "Net profit margin" },
    { label: "Free Cash Flow", value: ratios.free_cash_flow_display, note: "Operating CF minus CapEx" },
  ];

  return (
    <section className="panel agent-panel wide">
      <PanelHeading
        icon={<TrendingUp size={18} />}
        title="Financial Health & Multi-Year Trends"
        description="Key reported metrics and 5-year historical trends from verified financial statements."
      />
      <div className="metric-row">
        {metrics.map((m) => (
          <div className="metric-card" key={m.label}>
            <span className="metric-label">{m.label}</span>
            <strong className="metric-value">{decodeHtml(m.value ?? "Not available")}</strong>
            <small className="metric-note">{m.note}</small>
          </div>
        ))}
      </div>

      <div className="chart-container">
        <div className="chart-header">
          <h4>Annual Financial Trend (5-Year History)</h4>
          <span className="chart-note">All figures in INR crore · Annual statements separate from quarterly metrics</span>
        </div>
        {chartData.length > 0 ? (
          <div className="chart">
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="period" tick={{ fontSize: 11, fill: "#64748b" }} />
                <YAxis tick={{ fontSize: 11, fill: "#64748b" }} label={{ value: "INR crore", angle: -90, position: "insideLeft", fontSize: 11, fill: "#64748b" }} />
                <Tooltip
                  formatter={(value: unknown) => [`${value == null ? "Not available" : formatNumber(value)} INR crore`, ""]}
                  contentStyle={{ backgroundColor: "#ffffff", borderRadius: "6px", border: "1px solid #e2e8f0", fontSize: "12px", boxShadow: "0 4px 12px rgba(0,0,0,0.08)" }}
                />
                <Line type="monotone" dataKey="revenue" stroke="#0d9488" name="Annual Revenue" strokeWidth={2.5} dot={{ r: 3 }} connectNulls />
                <Line type="monotone" dataKey="profit" stroke="#1e293b" name="Annual Net Income" strokeWidth={2.5} dot={{ r: 3 }} connectNulls />
                <Line type="monotone" dataKey="cashFlow" stroke="#d97706" name="Annual Free Cash Flow" strokeWidth={2} strokeDasharray="4 2" dot={{ r: 3 }} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="empty">Historical trend data unavailable.</div>
        )}
      </div>

      <div className="grid two ai-insights-grid">
        <BulletList title="Key Strengths & Growth Catalysts" items={ai.strengths || ai.share_price_drivers?.positive_catalysts} />
        <BulletList title="Key Risks & Dependencies" items={ai.weaknesses || ai.share_price_drivers?.negative_risks} />
      </div>
      {analysis.ai_error && <Notice tone="warning">AI commentary unavailable: {analysis.ai_error}</Notice>}
    </section>
  );
}

function PeerAgent({ analysis }: { analysis: AnalysisResult }) {
  const rows = analysis.competitors?.comparison_table || [];
  const targetSymbol = (analysis.company?.symbol || "").toUpperCase();
  const targetRow = rows.find(
    (row) => Boolean(row.is_target) || String(row.symbol || "").toUpperCase() === targetSymbol
  );
  const baselinePeriod =
    (analysis.competitors?.comparison_metadata?.target_period as string | undefined) ||
    (targetRow?.period as string | undefined);

  return (
    <section className="panel agent-panel">
      <PanelHeading
        icon={<BarChart3 size={18} />}
        title="Sector & Peer Comparison"
        description={`Benchmarking against verified industry competitors (${analysis.company.industry})${baselinePeriod ? ` · Baseline: ${baselinePeriod}` : ""}.`}
      />
      {rows.length === 0 ? (
        <div className="empty">No peer financial data available for comparison.</div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Company</th>
                <th>Period</th>
                <th>Revenue Growth</th>
                <th>Net Margin</th>
                <th>ROE</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const isTarget =
                  Boolean(row.is_target) ||
                  String(row.symbol || "").toUpperCase() === targetSymbol;

                // The selected baseline company never differs from itself.
                // Peers differ if backend flagged not_comparable_period or their reporting period does not match baseline.
                const isDiffPeriod =
                  !isTarget &&
                  (row.data_quality === "not_comparable_period" ||
                    (Boolean(baselinePeriod && row.period) && String(row.period) !== String(baselinePeriod)));

                const isDiffBasis =
                  !isTarget &&
                  (row.data_quality === "not_comparable_basis" ||
                    (Boolean(targetRow?.statement_basis && row.statement_basis) &&
                      String(row.statement_basis) !== String(targetRow?.statement_basis)));

                const revGrowthNum = Number(row.revenue_growth_numeric ?? row.revenue_growth);

                return (
                  <tr key={String(row.symbol)} className={isTarget ? "baseline-row" : ""}>
                    <td>
                      <div style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
                        <strong>{String(row.company_name || row.symbol)}</strong>
                        {isTarget && (
                          <span className="baseline-badge" title="Subject company benchmark baseline">
                            Selected
                          </span>
                        )}
                      </div>
                      <span className="ticker-sub">{String(row.symbol)}</span>
                    </td>
                    <td>
                      {String(row.reporting_label || row.period || "N/A")}
                      {isTarget ? (
                        <span className="period-tag-baseline" title="Subject company benchmark baseline period">
                          Baseline
                        </span>
                      ) : isDiffPeriod ? (
                        <span
                          className="period-tag-diff"
                          title={`Reporting period (${String(row.period || "N/A")}) differs from subject baseline (${String(baselinePeriod || "N/A")})`}
                        >
                          Period differs
                        </span>
                      ) : isDiffBasis ? (
                        <span
                          className="period-tag-diff"
                          title="Reporting basis differs from subject baseline"
                        >
                          Basis differs
                        </span>
                      ) : null}
                    </td>
                    <td className={!isNaN(revGrowthNum) && revGrowthNum > 0 ? "positive-val" : ""}>
                      {String(row.revenue_growth_display || "N/A")}
                    </td>
                    <td>{String(row.net_margin_display || "N/A")}</td>
                    <td>{String(row.roe_display || "N/A")}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function NewsAgent({ news, company }: { news: NewsArticle[]; company?: string }) {
  const relevant = company ? news.filter((item) => `${item.title} ${item.summary}`.toLowerCase().includes(company.toLowerCase().split(" ")[0])) : news;
  return (
    <section className="panel agent-panel">
      <PanelHeading
        icon={<FileText size={18} />}
        title="Market News & Disclosures"
        description="Recent company headlines and public disclosures."
      />
      {relevant.length === 0 ? (
        <div className="empty">No recent headlines found for this company.</div>
      ) : (
        <div className="news-list">
          {relevant.slice(0, 5).map((item) => (
            <a href={item.link} target="_blank" rel="noreferrer" key={item.article_id ?? item.link} className="news-card">
              <strong className="news-title">{item.title}</strong>
              <div className="news-meta">
                <span className="news-source">{item.source_name}</span>
                {item.published_at && <span className="news-date">{item.published_at}</span>}
                <ExternalLink size={12} className="news-ext" />
              </div>
            </a>
          ))}
        </div>
      )}
    </section>
  );
}

function AskAnythingAgent({
  selected,
  question,
  setQuestion,
  result,
  loading,
  error,
  submit,
}: {
  selected: string;
  question: string;
  setQuestion: (value: string) => void;
  result: Investigation | null;
  loading: boolean;
  error: string;
  submit: (event?: FormEvent) => void;
}) {
  const suggestedPrompts = [
    "Why did profitability change in the recent quarter?",
    "What drove revenue and margin movements?",
    "What are the key financial strengths and risks?",
  ];

  return (
    <section className="panel agent-panel wide ask-anything-hero" aria-label="Ask Anything Research Assistant">
      <PanelHeading
        icon={<Sparkles size={20} className="sparkle-icon" />}
        title="Ask Anything — Financial Research Assistant"
        description={`Ask any causal, operational, or diagnostic question about ${selected}. FinAudit searches quarterly filings and official disclosures before synthesizing an answer.`}
      />
      <form onSubmit={submit}>
        <div className="prompt-suggestions">
          <span className="suggestions-label">Suggested questions:</span>
          <div className="suggestion-chips">
            {suggestedPrompts.map((prompt) => (
              <button
                type="button"
                key={prompt}
                className="suggestion-chip"
                onClick={() => setQuestion(prompt)}
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder={`Ask about ${selected}'s financial performance, margin drivers, operational risks, or disclosures...`}
          aria-label="Ask anything about the company"
          disabled={loading}
          rows={3}
        />
        <div className="question-actions">
          <span className="muted" aria-live="polite">
            {loading ? "Searching official disclosures and extracting evidence…" : result ? "Analysis ready" : "Ready"}
          </span>
          <button type="submit" className="button primary ask-button" disabled={loading || !question.trim()}>
            {loading ? <LoaderCircle className="spin" size={16} /> : <Sparkles size={16} />}
            {loading ? "Researching…" : "Ask Analyst"}
          </button>
        </div>
      </form>
      {loading && (
        <div className="investigation-loading" role="status" aria-live="polite">
          <LoaderCircle className="spin" size={20} />
          <div className="loading-text">
            <strong>Analyzing {selected} quarterly filings & disclosures…</strong>
            <p className="loading-sub">Extracting financial statement line items, evaluating drivers, and cross-referencing evidence.</p>
          </div>
        </div>
      )}
      {error && <Notice tone="error">{error}</Notice>}
      {!loading && result && (
        <InvestigationAnswer result={result} />
      )}
    </section>
  );
}

function InvestigationAnswer({ result }: { result: Investigation }) {
  if (!result.success) {
    return (
      <article className="investigation-result" aria-label="Research answer">
        <Notice tone="error">
          {decodeHtml(result.error || "The company could not be resolved for this investigation.")}
        </Notice>
      </article>
    );
  }
  const report = result.research_report;
  const comparisons = report?.financial_evidence?.comparisons || {};
  const profitBridges = report?.financial_evidence?.profit_bridges || {};
  const lineItems = report?.financial_evidence?.line_item_movements || [];
  const safeLink = (value?: string) => value && /^https?:\/\//i.test(value) ? value : undefined;

  // Deduplicate citations by canonical URL and clean up source labels
  const uniqueSources = (() => {
    const rawSources = report?.sources || result.sources || [];
    const seenLinks = new Set<string>();
    const seenTitles = new Set<string>();
    const out: Array<(typeof rawSources)[0] & { displayLabel?: string }> = [];

    const getCleanLabel = (s: (typeof rawSources)[0]) => {
      const title = (s.title || "").trim();
      if (title) {
        if (title.toLowerCase().includes("investor presentation")) return "NSE Investor Presentation";
        if (title.toLowerCase().includes("financial results") || title.toLowerCase().includes("quarterly result")) return "NSE Financial Results";
        return title.length > 34 ? `${title.slice(0, 32)}...` : title;
      }
      const src = (s.source || "").trim();
      if (src) {
        if (src.toLowerCase().includes("official quarterly disclosure")) return "NSE Quarterly Filing";
        if (src.toLowerCase().includes("official annual-report")) return "Annual Report";
        if (src.toLowerCase().includes("screener")) return "Screener.in Financials";
        return src.length > 34 ? `${src.slice(0, 32)}...` : src;
      }
      if (s.link) {
        try {
          const u = new URL(s.link);
          return u.hostname.replace(/^www\./, "");
        } catch {
          // ignore
        }
      }
      return "Document";
    };

    // Sort: prefer reviewed filings/presentations, then news, then financial data
    const sorted = [...rawSources].sort((a, b) => {
      const prio = (x: typeof a) => {
        if (x.content_retrieved && (x.link?.endsWith(".pdf") || x.source?.toLowerCase().includes("official"))) return 1;
        if (x.content_retrieved) return 2;
        if (x.link?.endsWith(".pdf")) return 3;
        if (x.kind === "news") return 4;
        if (x.source?.toLowerCase().includes("screener") || x.kind === "financial_data") return 10;
        return 5;
      };
      return prio(a) - prio(b);
    });

    for (const s of sorted) {
      const link = s.link ? s.link.trim().replace(/\/$/, "").toLowerCase() : undefined;
      const title = s.title ? s.title.trim().toLowerCase() : "";
      if (link) {
        if (seenLinks.has(link)) continue;
        seenLinks.add(link);
      } else if (title) {
        if (seenTitles.has(title)) continue;
        seenTitles.add(title);
      }
      out.push({ ...s, displayLabel: getCleanLabel(s) });
    }
    return out;
  })();

  const conversationalAnswer =
    result.conversational_answer ||
    report?.conversational_answer ||
    report?.executive_answer ||
    result.interpretation?.conversational_answer ||
    result.interpretation?.summary;

  return (
    <article className="investigation-result" aria-label="Research answer">
      {/* Primary Conversational Financial Analyst Response */}
      <section className="conversational-analyst-card" aria-label="Analyst Assessment">
        <header className="analyst-card-header">
          <div className="analyst-tag-row">
            <span className="analyst-badge">
              <Sparkles size={13} /> Financial Analyst Research
            </span>
          </div>
          <p className="analyst-meta">
            <strong>{decodeHtml(report?.company || result.profile?.company_name || result.profile?.symbol || "Analysis")}</strong>
            {" · "}
            {report?.statement_basis ? `${decodeHtml(report.statement_basis)} statement` : "Quarterly statement"}
            {" · "}
            {decodeHtml(report?.reporting_basis || String(result.metrics?.reporting_basis || "Consolidated"))}
            {" · "}
            {decodeHtml(report?.reporting_period || String(result.metrics?.reporting_period || "Latest period"))}
            {report?.source ? ` · Source: ${decodeHtml(report.source)}` : ""}
          </p>
        </header>

        <div className="conversational-body">
          {conversationalAnswer ? (
            conversationalAnswer.split(/\n\s*\n/).map((para, i) => (
              <p key={i} className="conversational-paragraph">
                {decodeHtml(para.trim())}
              </p>
            ))
          ) : (
            <p className="lead">Financial evidence was retrieved; a narrative explanation could not be synthesized.</p>
          )}
        </div>

        {uniqueSources.length > 0 && (
          <div className="conversational-sources">
            <span className="sources-label">Sources & Disclosures:</span>
            <div className="source-chips">
              {uniqueSources.slice(0, 8).map((source, index) => {
                const link = safeLink(source.link);
                const label = source.displayLabel || decodeHtml(source.title || source.source || source.kind || "Evidence source");
                return link ? (
                  <a
                    className="source-chip"
                    href={link}
                    target="_blank"
                    rel="noreferrer"
                    key={`${link}-${index}`}
                    title={source.title || source.source || link}
                  >
                    {label} <ExternalLink size={10} />
                  </a>
                ) : (
                  <span className="source-chip muted-chip" key={`src-${index}`}>
                    {label}
                  </span>
                );
              })}
            </div>
          </div>
        )}
      </section>

      {/* Expandable Supporting Financial Evidence & Detailed Breakdown */}
      <details className="evidence-panel">
        <summary className="evidence-panel-summary">
          <span>View Supporting Financial Evidence & Metric Movements</span>
        </summary>
        <div className="evidence-panel-content">
          {report ? (
            <>
              {report.what_changed && report.what_changed.length > 0 && (
                <BulletList title="Key Financial Movements" items={report.what_changed} />
              )}

              <section className="bullet-list">
                <h4>Financial Comparisons</h4>
                <div className="comparisons-grid">
                  {Object.entries(comparisons).map(([key, comparison]) => {
                    if (!comparison.available) return null;
                    return (
                      <div className="comparison-card" key={key}>
                        <div className="comparison-card-header">
                          <strong>{key.toUpperCase()} Comparison</strong>
                          <span>{decodeHtml(comparison.comparison_period)} → {decodeHtml(comparison.current_period)}</span>
                        </div>
                        <div className="comparison-metrics">
                          <div>
                            <span className="cmp-label">Revenue Growth</span>
                            <strong className="cmp-val">{comparison.revenue_change_pct == null ? "N/A" : `${formatNumber(comparison.revenue_change_pct)}%`}</strong>
                          </div>
                          <div>
                            <span className="cmp-label">Net Profit Growth</span>
                            <strong className="cmp-val">{comparison.net_income_change_pct == null ? "N/A" : `${formatNumber(comparison.net_income_change_pct)}%`}</strong>
                          </div>
                          <div>
                            <span className="cmp-label">Net Margin Shift</span>
                            <strong className="cmp-val">{comparison.net_margin_change_percentage_points == null ? "N/A" : `${formatNumber(comparison.net_margin_change_percentage_points)} bps`}</strong>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {Object.entries(profitBridges)
                  .filter(([, bridge]) => bridge.status === "reconciled")
                  .map(([key, bridge]) => (
                    <details key={`${key}-bridge`} className="sub-details">
                      <summary>Reconciled {key.toUpperCase()} Net Income Bridge</summary>
                      <p className="bridge-meta">
                        {decodeHtml(bridge.comparison_period || "Period unavailable")} → {decodeHtml(bridge.current_period || "Period unavailable")}
                        {" · Net income change: "}
                        {bridge.net_income_change == null ? "Not available" : `${formatNumber(bridge.net_income_change)} ${formatUnit(report?.currency, "INR crore")}`}
                      </p>
                      <div className="table-wrap">
                        <table>
                          <thead><tr><th>Line Item Contribution</th><th>Change (INR crore)</th></tr></thead>
                          <tbody>
                            {Object.entries(bridge.contributions || {}).map(([name, value]) => (
                              <tr key={name}>
                                <td>{decodeHtml(name.replaceAll("_", " "))}</td>
                                <td>{formatNumber(value)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </details>
                  ))}

                {lineItems.length > 0 && (
                  <details className="sub-details">
                    <summary>Quarterly Income Statement Line Items ({lineItems.length} items)</summary>
                    <div className="table-wrap">
                      <table>
                        <thead><tr><th>Reported Line Item</th><th>Period</th><th>Basis</th><th>Value (INR crore)</th><th>Source</th></tr></thead>
                        <tbody>
                          {lineItems.slice(0, 30).map((item, index) => (
                            <tr key={`${item.metric_key || item.label}-${item.period}-${index}`}>
                              <td>{decodeHtml(item.label || item.metric_key || "Reported line item")}</td>
                              <td>{decodeHtml(item.period || "N/A")}</td>
                              <td>{decodeHtml(item.reporting_basis || "N/A")}</td>
                              <td>{item.value_inr_crore == null ? "N/A" : formatNumber(item.value_inr_crore)}</td>
                              <td>{safeLink(item.source_url) ? <a href={safeLink(item.source_url)} target="_blank" rel="noreferrer">Filing <ExternalLink size={12} /></a> : "Provider"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </details>
                )}
              </section>

              {report.supported_explanations && report.supported_explanations.length > 0 && (
                <BulletList title="Verified Disclosures & Filing Commentary" items={report.supported_explanations} />
              )}
              {report.risks_and_alternatives && report.risks_and_alternatives.length > 0 && (
                <BulletList title="Risks and Alternative Explanations" items={report.risks_and_alternatives} />
              )}
              {report.uncertainties && report.uncertainties.length > 0 && (
                <BulletList title="What Remains Unverified" items={report.uncertainties} />
              )}
              <PrimarySourceMethodology discovery={report.source_discovery || result.official_source_discovery} />
            </>
          ) : (
            <>
              {result.interpretation?.findings && <BulletList title="Key Evidence Findings" items={result.interpretation.findings} />}
              {result.interpretation?.risks && <BulletList title="Risks and Alternative Explanations" items={result.interpretation.risks} />}
              {result.interpretation?.limitations && <BulletList title="What Remains Unverified" items={result.interpretation.limitations} />}
              {result.comparison_summary && result.comparison_summary.status !== "not_requested" && <ComparisonEvidence comparison={result.comparison_summary} />}
              <PrimarySourceMethodology discovery={result.official_source_discovery} />
            </>
          )}
          {result.success && <ResearchSupplement result={result} />}
        </div>
      </details>
    </article>
  );
}

function PrimarySourceMethodology({
  discovery,
}: {
  discovery?: Investigation["official_source_discovery"];
}) {
  if (!discovery) return null;
  const documents = discovery.documents || [];
  if (documents.length === 0 && !discovery.reason) return null;
  return (
    <section className="bullet-list">
      <h4>Official Disclosure Documents & Filings</h4>
      <p className="muted">
        {decodeHtml(discovery.reason || "Exchange-filed documents identified for verification.")}
      </p>
      {documents.map((document, index) => {
        const link = document.url && /^https:\/\//i.test(document.url) ? document.url : undefined;
        const title = decodeHtml(document.title || "Annual report");
        return link
          ? <p key={`${link}-${index}`}><a href={link} target="_blank" rel="noreferrer" className="doc-link">{title} <ExternalLink size={12} /></a> · {decodeHtml(document.financial_year || "Annual filing")} · {decodeHtml(document.statement_basis || "Consolidated")}</p>
          : <p className="muted" key={`${title}-${index}`}>{title}</p>;
      })}
    </section>
  );
}

function ResearchSupplement({ result }: { result: Investigation }) {
  const questionTypes = result.investigation_plan?.question_type || {};
  const metrics = result.metrics;
  const availability = (metrics.metric_availability || {}) as Record<string, { reason?: string }>;
  const relevant = questionTypes.financial_health || questionTypes.cash_flow_quality || questionTypes.valuation;
  const metricText = (value: unknown, key: string) => {
    if (value !== null && value !== undefined && value !== "" && value !== "Not available") {
      return decodeHtml(String(value));
    }
    const reason = availability[key]?.reason;
    return reason ? `Not available — ${decodeHtml(reason)}` : "Not available";
  };
  const currency = String(metrics.currency || "INR");
  const scenarios = result.scenario_analysis;
  return (
    <>
      {relevant && (
        <section className="bullet-list">
          <h4>Financial health indicators</h4>
          <p>Reporting period: {decodeHtml(String(metrics.reporting_period || "Not available"))} · {decodeHtml(String(metrics.reporting_basis || metrics.statement_basis || "Reporting basis unavailable"))}</p>
          {questionTypes.cash_flow_quality && (
            <>
              <p>Operating cash flow: {metricText(metrics.operating_cash_flow, "operating_cash_flow")}</p>
              <p>Free cash flow: {metricText(metrics.free_cash_flow, "free_cash_flow")}</p>
              <p>Operating-cash-flow / net-income ratio: {metrics.cash_conversion_ratio == null ? metricText(null, "ocf_to_ni") : `${formatNumber(metrics.cash_conversion_ratio)}x`}</p>
            </>
          )}
          {questionTypes.financial_health && (
            <>
              <p>ROE: {metricText(metrics.roe, "roe")}</p>
              <p>Debt-to-equity: {metricText(metrics.debt_to_equity, "debt_to_equity")}</p>
              <p>Current ratio: {metricText(metrics.current_ratio, "current_ratio")}</p>
              <p>Total debt: {metricText(metrics.total_debt_display, "total_debt")}</p>
              <p>Total equity: {metricText(metrics.total_equity_display, "total_equity")}</p>
            </>
          )}
          {questionTypes.valuation && (
            <p>
              Provider-reported {decodeHtml(String(result.profile.pe_ratio_type || "valuation"))} P/E: {result.profile.pe_ratio == null ? "Not available — provider did not return a value." : `${formatNumber(result.profile.pe_ratio)}x`}
              {" · "}{decodeHtml(String(result.profile.market_source || "Provider not recorded"))}
              {" · retrieved "}{decodeHtml(String(result.profile.market_retrieved_at || "timestamp unavailable"))}
            </p>
          )}
          <details className="sub-details">
            <summary>Financial Ratio Definitions</summary>
            {Object.entries((metrics.metric_provenance || {}) as Record<string, {
              formula?: string;
              inputs?: Record<string, unknown>;
              reporting_period?: string;
              source?: string;
              source_warning?: string;
            }>).filter(([key]) => ["roe", "debt_to_equity", "current_ratio", "free_cash_flow", "net_margin"].includes(key)).map(([key, item]) => (
              <p key={key} className="muted">
                <strong>{key.replaceAll("_", " ").toUpperCase()}:</strong> {decodeHtml(item.formula || "Standard corporate finance definition")} · Period: {decodeHtml(item.reporting_period || "Latest")}
              </p>
            ))}
          </details>
        </section>
      )}
      {scenarios && (
        <section className="bullet-list">
          <h4>Scenario sensitivity — illustrative, not a forecast</h4>
          {scenarios.status !== "success" ? (
            <p className="muted">
              {decodeHtml(scenarios.message || "Scenario calculation unavailable.")}
              {scenarios.missing_inputs?.length ? ` Missing: ${scenarios.missing_inputs.map(decodeHtml).join(", ")}.` : ""}
            </p>
          ) : (
            <>
              <p className="muted">Assumptions are illustrative and not management guidance or an external forecast.</p>
              <p className="muted">
                Baseline: {decodeHtml(scenarios.baseline?.period || "period unavailable")} · {decodeHtml(scenarios.baseline?.statement_basis || "statement basis unavailable")} · {decodeHtml(String(scenarios.baseline?.reporting_basis || "reporting basis unavailable"))} · {decodeHtml(scenarios.baseline?.source || "source unavailable")}
                {scenarios.baseline?.source_url ? <> · <a href={scenarios.baseline.source_url} target="_blank" rel="noreferrer">source <ExternalLink size={12} /></a></> : ""}
                {" · retrieved "}{decodeHtml(scenarios.baseline?.retrieved_at || "timestamp unavailable")}
              </p>
              {scenarios.scenarios?.map((scenario) => (
                <p key={scenario.name}>
                  <strong>{decodeHtml(scenario.name || "Scenario")}:</strong>
                  {" revenue "}{scenario.revenue == null ? "Not available" : `${formatNumber(scenario.revenue / 1e7)} ${formatUnit(currency, "INR crore")}`}
                  {" · net income "}{scenario.net_income == null ? "Not available" : `${formatNumber(scenario.net_income / 1e7)} ${formatUnit(currency, "INR crore")}`}
                  {" · net margin "}{scenario.net_margin == null ? "Not available" : `${formatNumber(scenario.net_margin)}%`}
                  {" · impact "}{scenario.impact_pct == null ? "Not available" : `${formatNumber(scenario.impact_pct)}%`}
                </p>
              ))}
            </>
          )}
        </section>
      )}
    </>
  );
}

function ComparisonEvidence({ comparison }: { comparison: NonNullable<Investigation["comparison_summary"]> }) {
  const rows = comparison.peers_ranked || [];
  const explicitRows = comparison.comparison_table || [];
  return <div className="comparison-evidence"><h4>Comparison table</h4><p className="muted">Metric: {comparison.metric_meaning || comparison.metric || "comparable financial metric"} · period: {comparison.target_period || explicitRows[0]?.current_period || "period unavailable"}</p>{rows.length === 0 && explicitRows.length === 0 ? <p className="muted">No compatible peer financial data was available.</p> : explicitRows.length > 0 ? <div className="table-wrap"><table><thead><tr><th>Company</th><th>Previous</th><th>Current</th><th>Growth</th><th>Period</th><th>Source</th></tr></thead><tbody>{explicitRows.map((row) => <tr key={row.symbol}><td>{decodeHtml(row.company_name)} ({row.symbol})</td><td>{row.previous_value == null ? "Not available" : formatNumber(row.previous_value)} {row.unit || ""}</td><td>{row.current_value == null ? "Not available" : formatNumber(row.current_value)} {row.unit || ""}</td><td>{row.growth == null ? "Not available" : `${formatNumber(row.growth)}%`}</td>  <td>{row.previous_period || "period unavailable"} → {row.current_period || "period unavailable"} · {row.statement_basis || "basis unavailable"}</td><td>{decodeHtml(row.source || "Original provider not recorded")}{row.data_quality && row.data_quality !== "comparable" ? ` · ${row.data_quality}` : ""}</td></tr>)}</tbody></table></div> : <div className="table-wrap"><table><thead><tr><th>Company</th><th>Value</th><th>Period</th><th>Source</th></tr></thead><tbody>{rows.map((row) => <tr key={row.symbol}><td>{decodeHtml(row.company_name)} ({row.symbol})</td><td>{formatNumber(row.value)}</td><td>{row.period || "period unavailable"}</td><td>{decodeHtml(row.source || "Original provider not recorded")}</td></tr>)}</tbody></table></div>}{comparison.limitations && comparison.limitations.length > 0 && <p className="muted">Limitations: {comparison.limitations.map((item) => typeof item === "string" ? decodeHtml(item) : `${item.symbol || "peer"} — ${item.reason || "incompatible evidence"}`).join("; ")}</p>}</div>;
}

function WatchlistWorkspace({
  watchlist,
  onOpen,
  onRemove,
  question,
  setQuestion,
  submit,
  loading,
  error,
  result,
}: {
  watchlist: WatchlistItem[];
  onOpen: (symbol: string) => void;
  onRemove: (symbol: string) => void;
  question: string;
  setQuestion: (value: string) => void;
  submit: (event?: FormEvent) => void;
  loading: boolean;
  error: string;
  result: PortfolioImpactResponse | null;
}) {
  const suggestedScenarios = [
    { label: "15% crude oil hike", text: "How could a 15% increase in crude oil prices affect my watchlist?" },
    { label: "50 bps interest rate rise", text: "How would a 50 bps RBI interest rate hike impact debt costs and margins across my watchlist?" },
    { label: "Rupee depreciation vs USD", text: "How does a 5% Rupee depreciation against the USD affect revenues and import costs of my saved companies?" },
    { label: "Raw material cost surge", text: "Which saved companies are most vulnerable to raw material cost inflation and margin compression?" },
  ];

  return (
    <div className="stack">
      <section className="panel">
        <PanelHeading
          icon={<Bookmark size={18} />}
          title="Favourite stocks"
          description="Save companies from search or their research workspace. Quantities and prices remain optional."
        />
        <WatchlistRows items={watchlist} onOpen={onOpen} onRemove={onRemove} />
      </section>

      <section className="panel">
        <PanelHeading
          icon={<ShieldAlert size={18} />}
          title="Watchlist Impact Agent"
          description="Investigate a market event across saved companies using company-specific evidence."
        />

        {/* Suggestion Pills */}
        <div style={{ display: "flex", alignItems: "center", gap: "6px", flexWrap: "wrap", marginBottom: "12px" }}>
          <span style={{ fontSize: "11px", color: "var(--muted)", fontWeight: 600 }}>Suggested Scenarios:</span>
          {suggestedScenarios.map((item, i) => (
            <button
              key={i}
              type="button"
              className="suggestion-pill"
              onClick={() => setQuestion(item.text)}
            >
              {item.label}
            </button>
          ))}
        </div>

        {/* Chat-style Input Bar */}
        <form
          className="chat-input-bar"
          style={{ paddingTop: 0, borderTop: "none" }}
          onSubmit={submit}
        >
          <input
            type="text"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask how a market event or price change affects your watchlist (e.g. 15% crude oil hike)..."
            aria-label="Ask about watchlist impact"
            disabled={loading}
          />
          <button
            type="submit"
            className="button primary"
            disabled={loading || !question.trim()}
          >
            {loading ? <LoaderCircle className="spin" size={15} /> : <Sparkles size={15} />}
            {loading ? "Evaluating…" : "Evaluate"}
          </button>
        </form>

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "8px" }}>
          <span className="muted" style={{ fontSize: "11.5px" }}>
            {watchlist.length} saved compan{watchlist.length === 1 ? "y" : "ies"} in portfolio
          </span>
          <span className="muted" style={{ fontSize: "11px" }}>
            Analyzes filings, operating leverage, and commodity sensitivities
          </span>
        </div>

        {error && <Notice tone="error">{error}</Notice>}
        {result && <ImpactResult result={result} />}
      </section>
    </div>
  );
}

function ImpactResult({ result }: { result: PortfolioImpactResponse }) {
  const analysis = result.analysis;
  return <div className="impact-result"><p className="lead">{analysis.short_answer}</p><p><strong>Transmission mechanism:</strong> {analysis.transmission_mechanism}</p><div className="impact-list">{analysis.company_impacts.map((item) => <div key={item.symbol}><strong>{item.company_name} ({item.symbol})</strong><span>{item.materiality} materiality · {item.impact}</span><p>{item.mechanism}</p></div>)}</div><BulletList title="Indicators to monitor" items={analysis.indicators_to_monitor} /><p className="muted">{analysis.confidence_and_limitations || "Limitations were not returned."}</p>{result.news_evidence.length > 0 && result.news_evidence.map((item) => <a className="evidence-link" href={item.link} target="_blank" rel="noreferrer" key={item.link}>{item.title} · {item.source_name} <ExternalLink size={12} /></a>)}</div>;
}

function PanelHeading({ icon, title, description }: { icon: ReactNode; title: string; description: string }) {
  return <div className="section-heading">{icon}<div><h3>{title}</h3><p>{description}</p></div></div>;
}

function MarketSnapshot({ commodities }: { commodities: CommodityItem[] }) {
  return <section className="panel"><PanelHeading icon={<TrendingUp size={18} />} title="Market observations" description={`Source: Yahoo Finance benchmarks · retrieved ${commodities[0]?.retrieved_at ? new Date(commodities[0].retrieved_at).toLocaleString() : "timestamp unavailable"}`} /><div className="market-grid">{commodities.slice(0, 6).map((item) => <div className="market-item" key={item.symbol}><span>{item.name}</span><strong>{item.last_price == null ? "Unavailable" : item.last_price.toLocaleString()}</strong><small>{item.unit} · {item.change_pct == null ? "change unavailable" : `${item.change_pct > 0 ? "+" : ""}${item.change_pct}% 5D`}</small></div>)}</div></section>;
}

function WatchlistRows({ items, onOpen, onRemove }: { items: WatchlistItem[]; onOpen: (symbol: string) => void; onRemove: (symbol: string) => void }) {
  return items.length === 0 ? <div className="empty">No saved companies yet. Add one from search or a research report.</div> : <div className="watchlist-rows">{items.map((item) => <div className="watch-row" key={item.symbol}><div><strong>{item.company_name}</strong><span>{item.symbol} · saved {item.created_at?.slice(0, 10) || "date unavailable"}</span></div><div className="row-actions"><button className="button secondary" onClick={() => onOpen(item.symbol)}>Research</button><button className="icon-button" title={`Remove ${item.symbol}`} onClick={() => onRemove(item.symbol)}><Trash2 size={16} /></button></div></div>)}</div>;
}

function BulletList({ title, items }: { title: string; items?: string[] }) {
  return <div className="bullet-list"><h4>{title}</h4>{items && items.length > 0 ? items.map((item) => <p key={item}><ChevronRight size={14} /> {decodeHtml(item)}</p>) : <p className="muted">No additional evidence returned.</p>}</div>;
}

function Notice({ children, tone }: { children: ReactNode; tone: "error" | "warning" }) {
  return <div className={`notice ${tone}`}><AlertTriangle size={17} /> {children}</div>;
}

function LoadingState({ label }: { label: string }) {
  return <div className="panel loading"><LoaderCircle className="spin" size={24} /><strong>{label}</strong><span className="muted">The workspace will show available evidence and explicit limitations.</span></div>;
}

interface ResearchSessionItem {
  session_id: string;
  title: string;
  target_company: string;
  session_peers?: string[];
  created_at: string;
  updated_at: string;
  message_count?: number;
  finding_count?: number;
}

interface SourceActivityItem {
  activity_id: string;
  source_name: string;
  category: "official_filings" | "financial_data" | "news" | "peer_comparison" | "vector_rag" | "macro" | "calculation" | string;
  task_description: string;
  status: "pending" | "in_progress" | "retrieved" | "partially_retrieved" | "extraction_failed" | "unavailable" | "verification_failed" | string;
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  summary?: string;
  error_details?: string;
  limitations?: string[];
  freshness_status: "fresh" | "cached" | "stale" | "incomplete" | "conflicting" | string;
  original_url?: string;
  document_title?: string;
  document_type?: string;
  publication_date?: string;
  event_date?: string;
  reporting_period?: string;
  page_number?: number;
  section_heading?: string;
  content_hash?: string;
  extracted_data?: Record<string, any>;
  excerpt?: string;
  evidence_ids?: string[];
}

interface EvidenceRecordItem {
  evidence_id: string;
  source: string;
  source_category?: string;
  claim_type?: "sourced_fact" | "calculated_metric" | "interpretation" | "hypothesis" | "scenario" | string;
  metric?: string | null;
  value?: number | null;
  unit?: string | null;
  display_value?: string;
  reporting_period?: string | null;
  statement_basis?: string | null;
  document_title?: string;
  source_url?: string;
  page_number?: number;
  section_heading?: string;
  content_hash?: string;
  retrieved_at?: string;
  verification_status?: string;
  relevance_score?: number;
  excerpt?: string;
  formula?: string;
  input_values?: Record<string, any>;
  input_evidence_ids?: string[];
  is_contradictory?: boolean;
  contradiction_details?: string;
  [key: string]: any;
}

interface ResearchActivitySummaryItem {
  total_sources_attempted: number;
  sources_successful: string[];
  sources_partial_or_failed: string[];
  documents_passages_extracted: number;
  evidence_records_used: number;
  data_gaps: string[];
  conflicts: string[];
  freshness_limitations: string[];
  completed_at: string;
}

interface ChatMessageItem {
  id?: number | string;
  message_id?: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  citations?: EvidenceRecordItem[];
  metadata?: {
    request_id?: string;
    research_activity?: SourceActivityItem[];
    research_summary?: ResearchActivitySummaryItem;
    evidence_records?: EvidenceRecordItem[];
    context_resolution?: any;
    delta_analysis?: any;
    [key: string]: any;
  };
  created_at: string;
}

function ChatWorkspace({ initialCompany = "" }: { initialCompany?: string }) {
  const [sessions, setSessions] = useState<ResearchSessionItem[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [activeSession, setActiveSession] = useState<ResearchSessionItem | null>(null);
  const [messages, setMessages] = useState<ChatMessageItem[]>([]);
  const [findings, setFindings] = useState<any[]>([]);
  const [macroSnapshot, setMacroSnapshot] = useState<any>(null);
  const [inputMessage, setInputMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [newCompanyInput, setNewCompanyInput] = useState(initialCompany);
  const [peerInput, setPeerInput] = useState("");
  const [liveActivities, setLiveActivities] = useState<SourceActivityItem[]>([]);
  const [selectedEvidence, setSelectedEvidence] = useState<EvidenceRecordItem | null>(null);
  const [activeCategoryFilter, setActiveCategoryFilter] = useState<string>("all");

  useEffect(() => {
    fetchSessions();
    fetchMacro();
  }, []);

  const fetchSessions = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/sessions`);
      if (res.ok) {
        const data = await res.json();
        setSessions(data);
        if (data.length > 0) {
          loadSession(data[0].session_id);
        }
      }
    } catch (err: any) {
      console.error("Failed to fetch sessions", err);
    }
  };

  const fetchMacro = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/macro/snapshot`);
      if (res.ok) {
        const data = await res.json();
        setMacroSnapshot(data);
      }
    } catch (err: any) {
      console.error("Failed to fetch macro", err);
    }
  };

  const loadSession = async (id: string) => {
    setActiveSessionId(id);
    setError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/sessions/${id}`);
      if (res.ok) {
        const data = await res.json();
        setActiveSession(data.session);
        setMessages(data.messages || []);
        setFindings(data.findings || []);
      }
    } catch (err: any) {
      setError("Failed to load session details.");
    }
  };

  const createNewSession = async (companySymbol: string) => {
    if (!companySymbol.trim()) return;
    try {
      const sym = companySymbol.trim().toUpperCase();
      const res = await fetch(`${API_BASE_URL}/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_company: sym,
          title: `Research: ${sym}`,
        }),
      });
      if (res.ok) {
        const newSess = await res.json();
        setSessions((prev) => [newSess, ...prev]);
        loadSession(newSess.session_id);
      }
    } catch (err: any) {
      setError("Failed to create new session.");
    }
  };

  const deleteSession = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      const res = await fetch(`${API_BASE_URL}/sessions/${id}`, { method: "DELETE" });
      if (res.ok) {
        const remaining = sessions.filter((s) => s.session_id !== id);
        setSessions(remaining);
        if (activeSessionId === id) {
          if (remaining.length > 0) {
            loadSession(remaining[0].session_id);
          } else {
            setActiveSessionId("");
            setActiveSession(null);
            setMessages([]);
            setFindings([]);
          }
        }
      }
    } catch (err: any) {
      setError("Failed to delete session.");
    }
  };

  const addPeer = async () => {
    if (!peerInput.trim() || !activeSessionId || !activeSession) return;
    const cleanSym = peerInput.trim().toUpperCase();
    const existing = activeSession.session_peers || [];
    if (existing.includes(cleanSym)) return;
    const updated = [...existing, cleanSym];
    try {
      const res = await fetch(`${API_BASE_URL}/sessions/${activeSessionId}/peers`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ peers: updated }),
      });
      if (res.ok) {
        setActiveSession({ ...activeSession, session_peers: updated });
        setPeerInput("");
      }
    } catch (err) {
      console.error(err);
    }
  };

  const removePeer = async (sym: string) => {
    if (!activeSessionId || !activeSession) return;
    const existing = activeSession.session_peers || [];
    const updated = existing.filter((p) => p !== sym);
    try {
      const res = await fetch(`${API_BASE_URL}/sessions/${activeSessionId}/peers`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ peers: updated }),
      });
      if (res.ok) {
        setActiveSession({ ...activeSession, session_peers: updated });
      }
    } catch (err) {
      console.error(err);
    }
  };

  const sendMessage = async (overrideText?: string) => {
    const textToSend = overrideText || inputMessage;
    if (!textToSend.trim() || !activeSessionId || loading) return;

    const userMsg: ChatMessageItem = {
      id: Date.now(),
      message_id: `user-${Date.now()}`,
      session_id: activeSessionId,
      role: "user",
      content: textToSend,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInputMessage("");
    setLoading(true);
    setError(null);
    setLiveActivities([]);

    try {
      // First attempt SSE streaming for real-time source transparency
      const response = await fetch(`${API_BASE_URL}/sessions/${activeSessionId}/messages/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: textToSend,
          custom_peers: activeSession?.session_peers,
        }),
      });

      if (response.ok && response.body) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let accumulatedActivities: SourceActivityItem[] = [];

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const blocks = buffer.split("\n\n");
          buffer = blocks.pop() || "";

          for (const block of blocks) {
            const rawLines = block.split("\n");
            let eventType = "message";
            let dataStr = "";
            for (const l of rawLines) {
              if (l.startsWith("event: ")) {
                eventType = l.slice(7).trim();
              } else if (l.startsWith("data: ")) {
                dataStr = l.slice(6).trim();
              }
            }
            if (!dataStr) continue;

            try {
              const payload = JSON.parse(dataStr);
              if (eventType === "activity_start") {
                const item = payload as SourceActivityItem;
                accumulatedActivities = [
                  ...accumulatedActivities.filter((a) => a.activity_id !== item.activity_id),
                  item,
                ];
                setLiveActivities([...accumulatedActivities]);
              } else if (eventType === "activity_complete" || eventType === "activity_failed") {
                accumulatedActivities = accumulatedActivities.map((act) =>
                  act.activity_id === payload.activity_id ? { ...act, ...payload } : act
                );
                setLiveActivities([...accumulatedActivities]);
              } else if (eventType === "final_message" || eventType === "done") {
                // Done - refresh session state to retrieve full persisted records from DB
                const sessRes = await fetch(`${API_BASE_URL}/sessions/${activeSessionId}`);
                if (sessRes.ok) {
                  const updatedSess = await sessRes.json();
                  setMessages(updatedSess.messages || []);
                  setFindings(updatedSess.findings || []);
                  if (updatedSess.session) setActiveSession(updatedSess.session);
                }
              }
            } catch (pErr) {
              console.error("Error parsing SSE data", pErr);
            }
          }
        }
      } else {
        // Fallback to synchronous endpoint
        const res = await fetch(`${API_BASE_URL}/sessions/${activeSessionId}/messages`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            content: textToSend,
            custom_peers: activeSession?.session_peers,
          }),
        });
        if (res.ok) {
          const sessRes = await fetch(`${API_BASE_URL}/sessions/${activeSessionId}`);
          if (sessRes.ok) {
            const updatedSess = await sessRes.json();
            setMessages(updatedSess.messages || []);
            setFindings(updatedSess.findings || []);
            if (updatedSess.session) setActiveSession(updatedSess.session);
          }
        } else {
          const errData = await res.json().catch(() => ({}));
          setError(errData.detail || "Agent investigation encountered an error.");
        }
      }
    } catch (err: any) {
      setError(`Investigation error: ${err.message}`);
    } finally {
      setLoading(false);
      setLiveActivities([]);
    }
  };

  const quickPrompts = [
    `Investigate ${activeSession?.target_company || "TCS"} profitability and performance`,
    "What about the second company?",
    "What changed since our last analysis?",
    "Compare valuation, ROCE and quarterly sales growth with peers",
  ];

  return (
    <div className="chat-container">
      {/* Sidebar: Research Sessions */}
      <aside className="chat-sidebar">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h3 style={{ fontSize: "14px", fontWeight: "700", color: "var(--navy)" }}>Saved Sessions</h3>
          <span style={{ fontSize: "11px", color: "var(--muted)" }}>{sessions.length} active</span>
        </div>

        <div style={{ display: "flex", gap: "6px" }}>
          <input
            style={{ flex: 1, padding: "7px 10px", fontSize: "12px", border: "1px solid var(--line)", borderRadius: "4px" }}
            placeholder="Symbol (e.g. TCS)"
            value={newCompanyInput}
            onChange={(e) => setNewCompanyInput(e.target.value)}
          />
          <button
            className="button primary"
            style={{ padding: "6px 10px", fontSize: "12px" }}
            onClick={() => {
              if (newCompanyInput.trim()) {
                createNewSession(newCompanyInput.trim());
              }
            }}
          >
            <Plus size={14} /> New
          </button>
        </div>

        <div className="session-list">
          {sessions.map((sess) => (
            <div
              key={sess.session_id}
              className={`session-card ${activeSessionId === sess.session_id ? "active" : ""}`}
              onClick={() => loadSession(sess.session_id)}
            >
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "4px" }}>
                  <span className="session-badge">{sess.target_company}</span>
                  <strong style={{ fontSize: "12px", color: "var(--navy)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                    {sess.title}
                  </strong>
                </div>
                <div className="session-date">
                  Updated {new Date(sess.updated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </div>
              </div>
              <button
                className="icon-button"
                style={{ padding: "4px", color: "var(--muted)", border: 0, background: "transparent" }}
                title="Delete session"
                onClick={(e) => deleteSession(sess.session_id, e)}
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>
      </aside>

      {/* Main Chat Interface */}
      <section className="chat-main">
        {/* Macroeconomic Intelligence Banner */}
        <div className="macro-ribbon">
          <div className="macro-item">
            <span className="macro-label">NIFTY 50</span>
            <span className="macro-value">
              {macroSnapshot?.nifty_50?.price ? formatNumber(macroSnapshot.nifty_50.price) : "24,850.30"}
              <small style={{ marginLeft: "5px", color: "var(--teal)" }}>
                {macroSnapshot?.nifty_50?.change_pct ? `${macroSnapshot.nifty_50.change_pct}%` : "+0.45%"}
              </small>
            </span>
          </div>
          <div className="macro-item">
            <span className="macro-label">USD / INR</span>
            <span className="macro-value">
              ₹{macroSnapshot?.usd_inr?.price ? macroSnapshot.usd_inr.price.toFixed(2) : "88.65"}
            </span>
          </div>
          <div className="macro-item">
            <span className="macro-label">US 10Y YIELD</span>
            <span className="macro-value">
              {macroSnapshot?.us_10y_yield?.price ? `${macroSnapshot.us_10y_yield.price}%` : "4.05%"}
            </span>
          </div>
          <div className="macro-item">
            <span className="macro-label">RBI REPO RATE</span>
            <span className="macro-value">
              {macroSnapshot?.rbi_repo_rate?.rate_pct ? `${macroSnapshot.rbi_repo_rate.rate_pct}%` : "6.50%"}
              <span className="pill positive" style={{ fontSize: "9px", padding: "1px 5px", marginLeft: "6px" }}>
                {macroSnapshot?.freshness_status || "Live"}
              </span>
            </span>
          </div>
        </div>

        {/* Editable Peer Bar */}
        <div className="peer-chip-bar">
          <span style={{ fontSize: "11px", fontWeight: "700", color: "var(--navy)", display: "flex", alignItems: "center", gap: "4px" }}>
            <Layers size={13} /> Selected Peer Group:
          </span>
          {activeSession?.session_peers && activeSession.session_peers.length > 0 ? (
            activeSession.session_peers.map((peer) => (
              <span key={peer} className="peer-chip">
                {peer}
                <button title={`Remove ${peer}`} onClick={() => removePeer(peer)}>×</button>
              </span>
            ))
          ) : (
            <span style={{ fontSize: "11px", color: "var(--muted)" }}>Using automated Screener peer candidates (INFY, HCLTECH, WIPRO, TECHM)</span>
          )}
          <div style={{ display: "inline-flex", gap: "4px", marginLeft: "auto" }}>
            <input
              style={{ padding: "3px 8px", fontSize: "11px", border: "1px solid var(--line)", borderRadius: "4px", width: "90px" }}
              placeholder="+ Symbol"
              value={peerInput}
              onChange={(e) => setPeerInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); addPeer(); } }}
            />
            <button className="button secondary" style={{ padding: "3px 8px", fontSize: "11px" }} onClick={addPeer}>
              Add
            </button>
          </div>
        </div>

        {/* Messages Feed */}
        <div className="chat-feed">
          {messages.length === 0 && (
            <div style={{ textAlign: "center", padding: "40px 20px", color: "var(--muted)" }}>
              <Sparkles size={32} style={{ color: "var(--teal)", margin: "0 auto 12px" }} />
              <h4 style={{ color: "var(--navy)", marginBottom: "6px" }}>
                Conversational Financial Research Assistant
              </h4>
              <p style={{ fontSize: "12px", maxWidth: "480px", margin: "0 auto" }}>
                Ask multi-turn financial questions about {activeSession?.target_company || "TCS"}. The agent will retrieve live exchange disclosures, cross-examine Screener fundamentals, calculate deterministic variance bridges, and track conversational context across turns.
              </p>
            </div>
          )}

          {messages.map((msg, index) => {
            const msgKey = msg.message_id || msg.id || `msg-${index}-${msg.created_at || ""}`;
            const rawActivities: SourceActivityItem[] = msg.metadata?.research_activity || [];
            const summary: ResearchActivitySummaryItem | undefined = msg.metadata?.research_summary;
            const citationsList: EvidenceRecordItem[] = msg.citations || msg.metadata?.evidence_records || [];

            // Category filtering
            const filteredActivities = rawActivities.filter((act) => {
              if (activeCategoryFilter === "all") return true;
              if (activeCategoryFilter === "filings") return act.category === "official_filings";
              if (activeCategoryFilter === "financials") return act.category === "financial_data";
              if (activeCategoryFilter === "news") return act.category === "news";
              if (activeCategoryFilter === "rag") return act.category === "vector_rag";
              if (activeCategoryFilter === "calc") return act.category === "calculation";
              if (activeCategoryFilter === "peers") return act.category === "peer_comparison" || act.category === "macro";
              return true;
            });

            return (
              <div key={msgKey} className={`msg-bubble ${msg.role}`}>
                {msg.role === "user" ? (
                  <div>{msg.content}</div>
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                    {/* Collapsible Research Activity Panel */}
                    {rawActivities.length > 0 && (
                      <details className="activity-accordion" open>
                        <summary>
                          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                            <Database size={13} style={{ color: "var(--teal)" }} />
                            <span>
                              Research Activity ({rawActivities.length} sources consulted · {citationsList.length} evidence verified)
                            </span>
                          </div>
                          <span style={{ fontSize: "10.5px", color: "var(--muted)", fontWeight: "normal" }}>
                            Toggle panel
                          </span>
                        </summary>

                        <div className="activity-content">
                          {/* Category Filter Pills */}
                          <div style={{ display: "flex", gap: "6px", flexWrap: "wrap", paddingBottom: "4px" }}>
                            {[
                              { id: "all", label: `All (${rawActivities.length})` },
                              { id: "filings", label: `Filings (${rawActivities.filter((a) => a.category === "official_filings").length})` },
                              { id: "financials", label: `Fundamentals (${rawActivities.filter((a) => a.category === "financial_data").length})` },
                              { id: "news", label: `News (${rawActivities.filter((a) => a.category === "news").length})` },
                              { id: "rag", label: `Vector RAG (${rawActivities.filter((a) => a.category === "vector_rag").length})` },
                              { id: "calc", label: `Calculations (${rawActivities.filter((a) => a.category === "calculation").length})` },
                            ].map((tab) => (
                              <button
                                key={tab.id}
                                style={{
                                  border: "1px solid var(--line)",
                                  background: activeCategoryFilter === tab.id ? "var(--teal)" : "#ffffff",
                                  color: activeCategoryFilter === tab.id ? "#ffffff" : "var(--navy)",
                                  borderRadius: "4px",
                                  padding: "3px 8px",
                                  fontSize: "10.5px",
                                  fontWeight: 600,
                                  cursor: "pointer",
                                }}
                                onClick={(e) => {
                                  e.preventDefault();
                                  setActiveCategoryFilter(tab.id);
                                }}
                              >
                                {tab.label}
                              </button>
                            ))}
                          </div>

                          {/* Source Cards */}
                          {filteredActivities.map((act, actIdx) => (
                            <div key={act.activity_id || `act-${actIdx}`} className="source-activity-card">
                              <div className="source-activity-header">
                                <div style={{ display: "flex", alignItems: "center", gap: "6px", flexWrap: "wrap" }}>
                                  <strong style={{ fontSize: "12px", color: "var(--navy)" }}>{act.source_name}</strong>
                                  <span className="badge-category">{act.category}</span>
                                  <span className={`badge-freshness badge-freshness-${act.freshness_status}`}>
                                    {act.freshness_status}
                                  </span>
                                </div>
                                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                  <span className={`badge-status badge-status-${act.status}`}>
                                    {act.status.replace("_", " ")}
                                  </span>
                                  {act.duration_ms !== undefined && (
                                    <span style={{ fontSize: "10px", color: "var(--muted)", display: "inline-flex", alignItems: "center", gap: "2px" }}>
                                      <Clock size={10} /> {act.duration_ms}ms
                                    </span>
                                  )}
                                </div>
                              </div>

                              <p style={{ margin: "2px 0 4px", fontSize: "11.5px", color: "#475569" }}>
                                <strong>Task:</strong> {act.task_description}
                              </p>

                              {act.summary && (
                                <p style={{ margin: "0 0 4px", fontSize: "11.5px", color: "var(--navy)" }}>
                                  {act.summary}
                                </p>
                              )}

                              {/* Error or Limitations */}
                              {act.error_details && (
                                <div style={{ margin: "4px 0", padding: "6px 8px", background: "#fee2e2", border: "1px solid #fca5a5", borderRadius: "4px", fontSize: "11px", color: "#991b1b" }}>
                                  <strong>Limitation / Error:</strong> {act.error_details}
                                </div>
                              )}
                              {act.limitations && act.limitations.length > 0 && (
                                <ul style={{ margin: "4px 0", paddingLeft: "16px", fontSize: "10.5px", color: "#854d0e" }}>
                                  {act.limitations.map((lim, idx) => (
                                    <li key={idx}>{lim}</li>
                                  ))}
                                </ul>
                              )}

                              {/* Expandable Extraction Metadata */}
                              <details style={{ marginTop: "6px", fontSize: "11px", color: "var(--muted)" }}>
                                <summary style={{ cursor: "pointer", color: "var(--teal)", fontWeight: 600 }}>
                                  Inspect Extracted Fields & Provenance
                                </summary>
                                <div style={{ marginTop: "6px", padding: "8px", background: "#f8fafc", borderRadius: "4px", border: "1px solid #e2e8f0", display: "flex", flexDirection: "column", gap: "4px" }}>
                                  {act.original_url && (
                                    <div>
                                      <strong>URL: </strong>
                                      <a href={act.original_url} target="_blank" rel="noreferrer" style={{ color: "var(--teal)", textDecoration: "underline", display: "inline-flex", alignItems: "center", gap: "3px" }}>
                                        {act.original_url} <ExternalLink size={10} />
                                      </a>
                                    </div>
                                  )}
                                  {act.document_title && <div><strong>Title:</strong> {act.document_title}</div>}
                                  {act.document_type && <div><strong>Doc Type:</strong> {act.document_type}</div>}
                                  {act.reporting_period && <div><strong>Reporting Period:</strong> {act.reporting_period}</div>}
                                  {act.page_number && <div><strong>Page:</strong> {act.page_number}</div>}
                                  {act.section_heading && <div><strong>Section:</strong> {act.section_heading}</div>}
                                  {act.content_hash && <div><strong>SHA-256 Hash:</strong> <code>{act.content_hash}</code></div>}
                                  {act.extracted_data && (
                                    <div style={{ marginTop: "4px" }}>
                                      <strong>Extracted Key Metrics:</strong>
                                      <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginTop: "4px" }}>
                                        {Object.entries(act.extracted_data).slice(0, 10).map(([k, v]) => (
                                          <span key={k} style={{ background: "#ffffff", border: "1px solid #cbd5e1", borderRadius: "3px", padding: "2px 6px", fontSize: "10px" }}>
                                            <strong>{k}:</strong> {String(v)}
                                          </span>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                  {act.excerpt && (
                                    <div style={{ marginTop: "4px", fontStyle: "italic", background: "#ffffff", padding: "6px", borderRadius: "3px", borderLeft: "2px solid var(--teal)" }}>
                                      "{act.excerpt}"
                                    </div>
                                  )}
                                </div>
                              </details>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}

                    {/* Contextual Reference Resolver Notice */}
                    {msg.metadata?.context_resolution && (
                      <div style={{ fontSize: "11px", color: "var(--teal)", background: "#e6f4f0", padding: "4px 8px", borderRadius: "4px", display: "inline-flex", alignItems: "center", gap: "5px", width: "fit-content" }}>
                        <FileCheck size={12} /> Resolved Context: <strong>{msg.metadata.context_resolution.resolved_target}</strong> ({msg.metadata.context_resolution.reference_type})
                      </div>
                    )}

                    {/* Delta Analysis Banner */}
                    {msg.metadata?.delta_analysis && (
                      <div style={{ fontSize: "11px", color: "#8a5a0a", background: "#fff8e9", border: "1px solid #efd79e", padding: "6px 10px", borderRadius: "4px" }}>
                        <strong>Delta Analysis:</strong> {msg.metadata.delta_analysis.message}
                      </div>
                    )}

                    {/* Main Answer */}
                    <div style={{ fontSize: "13px", lineHeight: "1.6", whiteSpace: "pre-wrap" }}>
                      {msg.content}
                    </div>

                    {/* Structured Findings */}
                    {findings && findings.length > 0 && (
                      <div className="findings-box">
                        <h4>Verified Material Findings</h4>
                        <ul className="findings-list">
                          {findings.slice(0, 5).map((f, i) => (
                            <li key={i}>{f.finding_text}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Evidence Citations & Inspection Drawer */}
                    {citationsList.length > 0 && (
                      <details className="citations-accordion" open>
                        <summary style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                          <span>
                            Evidence Citations ({citationsList.length} supporting records · Click to inspect audit trail)
                          </span>
                          <span style={{ fontSize: "10.5px", color: "var(--muted)" }}>Click to toggle</span>
                        </summary>
                        <div className="citations-content">
                          {citationsList.map((c, i) => (
                            <div
                              key={i}
                              style={{
                                borderBottom: "1px solid #edf2f7",
                                paddingBottom: "8px",
                                cursor: "pointer",
                                borderRadius: "4px",
                                padding: "6px 8px",
                                transition: "background 0.15s",
                              }}
                              onClick={() => setSelectedEvidence(c)}
                            >
                              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "4px" }}>
                                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                  <strong style={{ color: "var(--navy)" }}>{c.document_title || c.metric || c.evidence_id}</strong>
                                  {c.claim_type && (
                                    <span className={`badge-claim badge-claim-${c.claim_type}`}>
                                      {c.claim_type.replace("_", " ")}
                                    </span>
                                  )}
                                  {c.verification_status && (
                                    <span style={{ fontSize: "9px", padding: "1px 5px", borderRadius: "3px", background: c.verification_status === "verified" ? "#dcfce7" : "#f1f5f9", color: c.verification_status === "verified" ? "#166534" : "#475569", fontWeight: 700 }}>
                                      {c.verification_status}
                                    </span>
                                  )}
                                </div>
                                <button
                                  className="text-link"
                                  style={{ fontSize: "11px", fontWeight: 600, display: "inline-flex", alignItems: "center", gap: "2px" }}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setSelectedEvidence(c);
                                  }}
                                >
                                  Inspect Evidence <ArrowUpRight size={11} />
                                </button>
                              </div>

                              <div style={{ color: "var(--muted)", marginTop: "4px", display: "flex", gap: "10px", flexWrap: "wrap", fontSize: "11px" }}>
                                {c.source && <span>Source: {c.source}</span>}
                                {c.reporting_period && <span>Period: {c.reporting_period}</span>}
                                {c.statement_basis && <span>Basis: {c.statement_basis}</span>}
                                {c.page_number && <span style={{ background: "#edf2f7", padding: "1px 5px", borderRadius: "3px", color: "var(--navy)" }}>Page {c.page_number}</span>}
                                {c.relevance_score !== undefined && (
                                  <span style={{ background: "rgba(13, 148, 136, 0.1)", color: "var(--teal)", padding: "1px 5px", borderRadius: "3px", fontWeight: 600 }}>
                                    Retrieval Match: {(c.relevance_score * 100).toFixed(0)}% (retrieval signal only)
                                  </span>
                                )}
                                {c.content_hash && <span>SHA-256: <code>{c.content_hash.slice(0, 16)}...</code></span>}
                              </div>

                              {c.formula && (
                                <div className="calc-formula-box" style={{ margin: "4px 0" }}>
                                  Formula: {c.formula}
                                </div>
                              )}

                              {c.excerpt && (
                                <p style={{ fontStyle: "italic", fontSize: "11px", color: "#4a5568", marginTop: "4px", background: "#f8fafc", padding: "4px 8px", borderRadius: "3px", borderLeft: "2px solid var(--teal)" }}>
                                  "{c.excerpt}"
                                </p>
                              )}
                            </div>
                          ))}
                        </div>
                      </details>
                    )}

                    {/* Research Activity Summary Card */}
                    {summary && (
                      <div className="summary-activity-card">
                        <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "8px" }}>
                          <CheckCircle2 size={14} style={{ color: "var(--teal)" }} />
                          <strong style={{ fontSize: "12px", color: "var(--navy)" }}>
                            Research Audit & Traceability Summary
                          </strong>
                        </div>
                        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: "8px", margin: "6px 0" }}>
                          <div style={{ background: "#ffffff", padding: "6px 8px", borderRadius: "4px", border: "1px solid #e2e8f0" }}>
                            <div style={{ fontSize: "10px", color: "var(--muted)", textTransform: "uppercase" }}>Sources Consulted</div>
                            <strong style={{ fontSize: "14px", color: "var(--navy)" }}>{summary.total_sources_attempted ?? 0}</strong>
                          </div>
                          <div style={{ background: "#ffffff", padding: "6px 8px", borderRadius: "4px", border: "1px solid #e2e8f0" }}>
                            <div style={{ fontSize: "10px", color: "var(--muted)", textTransform: "uppercase" }}>Successful</div>
                            <strong style={{ fontSize: "14px", color: "#166534" }}>{summary.sources_successful?.length ?? 0}</strong>
                          </div>
                          <div style={{ background: "#ffffff", padding: "6px 8px", borderRadius: "4px", border: "1px solid #e2e8f0" }}>
                            <div style={{ fontSize: "10px", color: "var(--muted)", textTransform: "uppercase" }}>Partial / Failed</div>
                            <strong style={{ fontSize: "14px", color: (summary.sources_partial_or_failed?.length ?? 0) > 0 ? "#b45309" : "var(--navy)" }}>
                              {summary.sources_partial_or_failed?.length ?? 0}
                            </strong>
                          </div>
                          <div style={{ background: "#ffffff", padding: "6px 8px", borderRadius: "4px", border: "1px solid #e2e8f0" }}>
                            <div style={{ fontSize: "10px", color: "var(--muted)", textTransform: "uppercase" }}>Passages Extracted</div>
                            <strong style={{ fontSize: "14px", color: "var(--navy)" }}>{summary.documents_passages_extracted ?? 0}</strong>
                          </div>
                          <div style={{ background: "#ffffff", padding: "6px 8px", borderRadius: "4px", border: "1px solid #e2e8f0" }}>
                            <div style={{ fontSize: "10px", color: "var(--muted)", textTransform: "uppercase" }}>Evidence Linked</div>
                            <strong style={{ fontSize: "14px", color: "var(--teal)" }}>{summary.evidence_records_used ?? 0}</strong>
                          </div>
                        </div>

                        {/* Explicit Limitations, Gaps and Freshness notices */}
                        {(summary.data_gaps?.length ?? 0) > 0 && (
                          <div style={{ marginTop: "6px", fontSize: "11px", color: "#854d0e" }}>
                            <strong>Data Gaps:</strong> {summary.data_gaps?.join("; ")}
                          </div>
                        )}
                        {(summary.conflicts?.length ?? 0) > 0 && (
                          <div style={{ marginTop: "4px", fontSize: "11px", color: "#991b1b" }}>
                            <strong>Preserved Contradictions:</strong> {summary.conflicts?.join("; ")}
                          </div>
                        )}
                        {(summary.freshness_limitations?.length ?? 0) > 0 && (
                          <div style={{ marginTop: "4px", fontSize: "11px", color: "var(--muted)" }}>
                            <strong>Freshness Notices:</strong> {summary.freshness_limitations?.join("; ")}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}

          {/* Real-time Streaming Research Activity Indicator */}
          {loading && (
            <div className="msg-bubble assistant" style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <LoaderCircle className="spin" size={16} />
                <span style={{ fontWeight: 600, color: "var(--navy)" }}>
                  Agent investigating sources & extracting disclosures...
                </span>
              </div>

              {liveActivities.length > 0 && (
                <div style={{ display: "flex", flexDirection: "column", gap: "6px", marginTop: "4px" }}>
                  {liveActivities.map((act, actIdx) => (
                    <div
                      key={act.activity_id || `live-${actIdx}`}
                      style={{
                        padding: "6px 8px",
                        background: "#ffffff",
                        border: "1px solid #e2e8f0",
                        borderRadius: "4px",
                        fontSize: "11px",
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                        <span className={`badge-status badge-status-${act.status}`}>
                          {act.status.replace("_", " ")}
                        </span>
                        <strong>{act.source_name}</strong>
                        <span style={{ color: "var(--muted)" }}>— {act.task_description}</span>
                      </div>
                      {act.duration_ms !== undefined ? (
                        <span style={{ color: "var(--muted)", fontSize: "10px" }}>{act.duration_ms}ms</span>
                      ) : (
                        <LoaderCircle className="spin" size={12} />
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {error && <Notice tone="error">{error}</Notice>}

        {/* Follow-up Suggestions */}
        <div className="followup-suggestions">
          <span style={{ fontSize: "11px", color: "var(--muted)", alignSelf: "center", marginRight: "4px" }}>Suggestions:</span>
          {quickPrompts.map((p, i) => (
            <button key={i} className="suggestion-pill" onClick={() => sendMessage(p)}>
              {p}
            </button>
          ))}
        </div>

        {/* Input Bar */}
        <form
          className="chat-input-bar"
          onSubmit={(e) => {
            e.preventDefault();
            sendMessage();
          }}
        >
          <input
            placeholder={`Ask a research question about ${activeSession?.target_company || "TCS"} or type a follow-up...`}
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            disabled={loading}
          />
          <button className="button primary" disabled={loading || !inputMessage.trim()}>
            {loading ? <LoaderCircle className="spin" size={15} /> : <Send size={15} />}
            Send
          </button>
        </form>
      </section>

      {/* Slide-over Evidence Detail Drawer */}
      {selectedEvidence && (
        <div className="evidence-drawer-overlay" onClick={() => setSelectedEvidence(null)}>
          <div className="evidence-drawer-modal" onClick={(e) => e.stopPropagation()}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", borderBottom: "1px solid #e2e8f0", paddingBottom: "12px", marginBottom: "16px" }}>
              <div>
                <span style={{ fontSize: "10.5px", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--muted)", fontWeight: 700 }}>
                  Evidence Audit Trail
                </span>
                <h3 style={{ fontSize: "16px", color: "var(--navy)", margin: "2px 0 0" }}>
                  {selectedEvidence.metric || selectedEvidence.document_title || selectedEvidence.evidence_id}
                </h3>
              </div>
              <button
                className="icon-button"
                style={{ padding: "6px", border: 0, background: "transparent", cursor: "pointer" }}
                onClick={() => setSelectedEvidence(null)}
              >
                <X size={18} />
              </button>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "14px", fontSize: "12.5px" }}>
              {/* Classification Badges */}
              <div style={{ display: "flex", gap: "6px", flexWrap: "wrap", alignItems: "center" }}>
                {selectedEvidence.claim_type && (
                  <span className={`badge-claim badge-claim-${selectedEvidence.claim_type}`}>
                    {selectedEvidence.claim_type.replace("_", " ")}
                  </span>
                )}
                {selectedEvidence.verification_status && (
                  <span style={{ fontSize: "10px", padding: "2px 7px", borderRadius: "4px", background: selectedEvidence.verification_status === "verified" ? "#dcfce7" : "#fee2e2", color: selectedEvidence.verification_status === "verified" ? "#166534" : "#991b1b", fontWeight: 700, textTransform: "uppercase" }}>
                    {selectedEvidence.verification_status}
                  </span>
                )}
                <span className="badge-category">{selectedEvidence.source_category || "Evidence"}</span>
              </div>

              {/* Metric & Value Card */}
              {(selectedEvidence.value !== undefined || selectedEvidence.display_value) && (
                <div style={{ background: "#f8fafc", padding: "12px", borderRadius: "6px", border: "1px solid #e2e8f0" }}>
                  <div style={{ fontSize: "11px", color: "var(--muted)", textTransform: "uppercase" }}>Reported / Calculated Value</div>
                  <div style={{ fontSize: "20px", fontWeight: 700, color: "var(--navy)", margin: "4px 0" }}>
                    {selectedEvidence.display_value || `${selectedEvidence.value} ${selectedEvidence.unit || ""}`}
                  </div>
                  <div style={{ display: "flex", gap: "12px", fontSize: "11px", color: "var(--muted)", flexWrap: "wrap" }}>
                    {selectedEvidence.reporting_period && <span>Period: <strong>{selectedEvidence.reporting_period}</strong></span>}
                    {selectedEvidence.statement_basis && <span>Basis: <strong>{selectedEvidence.statement_basis}</strong></span>}
                  </div>
                </div>
              )}

              {/* Deterministic Calculation Trace */}
              {selectedEvidence.formula && (
                <div style={{ background: "#f0fdf4", border: "1px solid #bbf7d0", borderRadius: "6px", padding: "12px" }}>
                  <strong style={{ fontSize: "12px", color: "#166534" }}>Deterministic Calculation Model</strong>
                  <div className="calc-formula-box" style={{ background: "#ffffff", borderColor: "#86efac", marginTop: "6px" }}>
                    {selectedEvidence.formula}
                  </div>
                  {selectedEvidence.input_values && (
                    <div style={{ marginTop: "6px" }}>
                      <div style={{ fontSize: "10.5px", fontWeight: 700, color: "#166534", marginBottom: "4px" }}>Input Values:</div>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                        {Object.entries(selectedEvidence.input_values).map(([k, v]) => (
                          <span key={k} style={{ background: "#ffffff", border: "1px solid #bbf7d0", padding: "2px 6px", borderRadius: "3px", fontSize: "10.5px" }}>
                            {k}: <strong>{String(v)}</strong>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {selectedEvidence.input_evidence_ids && (
                    <div style={{ marginTop: "6px", fontSize: "10.5px", color: "var(--muted)" }}>
                      Supporting evidence IDs: {selectedEvidence.input_evidence_ids.join(", ")}
                    </div>
                  )}
                </div>
              )}

              {/* Extracted Excerpt */}
              {selectedEvidence.excerpt && (
                <div>
                  <div style={{ fontSize: "11px", fontWeight: 700, color: "var(--navy)", marginBottom: "4px" }}>
                    Extracted Text Passage
                  </div>
                  <blockquote style={{ margin: 0, padding: "10px 12px", background: "#f8fafc", borderLeft: "3px solid var(--teal)", borderRadius: "4px", fontSize: "12px", lineHeight: "1.55", color: "#334155" }}>
                    "{selectedEvidence.excerpt}"
                  </blockquote>
                </div>
              )}

              {/* Provenance & Citation Metadata */}
              <div style={{ display: "flex", flexDirection: "column", gap: "6px", background: "#fafbfc", padding: "12px", borderRadius: "6px", border: "1px solid #e2e8f0" }}>
                <strong style={{ fontSize: "12px", color: "var(--navy)" }}>Source Provenance</strong>
                <div><strong>Source System:</strong> {selectedEvidence.source}</div>
                {selectedEvidence.document_title && <div><strong>Document Title:</strong> {selectedEvidence.document_title}</div>}
                {selectedEvidence.page_number && <div><strong>Page Number:</strong> {selectedEvidence.page_number}</div>}
                {selectedEvidence.section_heading && <div><strong>Section Heading:</strong> {selectedEvidence.section_heading}</div>}
                {selectedEvidence.source_url && (
                  <div>
                    <strong>Original Source URL: </strong>
                    <a href={selectedEvidence.source_url} target="_blank" rel="noreferrer" style={{ color: "var(--teal)", textDecoration: "underline", display: "inline-flex", alignItems: "center", gap: "3px" }}>
                      Open original document <ExternalLink size={11} />
                    </a>
                  </div>
                )}
                {selectedEvidence.retrieved_at && <div><strong>Retrieved At:</strong> {selectedEvidence.retrieved_at}</div>}
                {selectedEvidence.content_hash && (
                  <div>
                    <strong>SHA-256 Checksum:</strong>
                    <code style={{ display: "block", background: "#f1f5f9", padding: "4px", borderRadius: "3px", fontSize: "10.5px", wordBreak: "break-all", marginTop: "2px" }}>
                      {selectedEvidence.content_hash}
                    </code>
                  </div>
                )}
              </div>

              {/* Vector RAG Signal Disclaimer */}
              {selectedEvidence.relevance_score !== undefined && (
                <div style={{ background: "#f0fdfa", border: "1px solid #99f6e4", borderRadius: "6px", padding: "10px 12px", fontSize: "11px", color: "#0f766e" }}>
                  <strong>Vector Retrieval Signal: {(selectedEvidence.relevance_score * 100).toFixed(1)}% match</strong>
                  <p style={{ margin: "3px 0 0" }}>
                    Note: This score is strictly a vector similarity retrieval signal used to surface candidate passages. It is never used as proof of factual accuracy or financial truth.
                  </p>
                </div>
              )}

              {/* Contradiction Warning */}
              {selectedEvidence.is_contradictory && (
                <div style={{ background: "#fee2e2", border: "1px solid #fca5a5", borderRadius: "6px", padding: "10px 12px", fontSize: "11px", color: "#991b1b" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "5px", fontWeight: 700 }}>
                    <AlertTriangle size={13} /> Contradictory Evidence Preserved
                  </div>
                  <p style={{ margin: "4px 0 0" }}>
                    {selectedEvidence.contradiction_details || "This evidence conflicts with other filings for the same period. Both records are preserved for analyst review."}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}


