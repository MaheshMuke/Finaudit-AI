# FinAudit AI — Agentic Financial Intelligence & Portfolio Impact Platform

An agentic financial intelligence platform for Indian equities (Nifty 500). Combines:

- Verified Nifty 500 financial statements pre-downloaded by `trail.py`.
- On-demand `yfinance` market indicators and live price feeds.
- Real-time RSS financial news aggregation and SQLite deduplication.
- Global commodity futures (Crude, Gold, Silver, Natural Gas, Copper) and FX (USD/INR) tracking.
- Local private **Ollama** (`qwen3:8b`) as the primary reasoning engine, with configurable cloud API fallback.
- Sector-specific financial frameworks (Banking vs. IT vs. Auto vs. Energy).
- **Portfolio Impact Intelligence**: Ask natural-language questions about your watchlist holdings against macro shocks.
- **Sector Opportunity Radar**: Median constituent growth and profitability scoring.

---

## 🧭 Navigation & Core Features

1. **Overview**: Macro commodity benchmarks, latest financial disclosures, and quick watchlist.
2. **Company Analysis**: 1-click comprehensive financial diagnostic:
   - Executive Overview & sector profile
   - Interactive financial trend charts & full statements (Annual/Quarterly)
   - 1-Year stock performance, volatility, and max drawdown
   - Peer competitor benchmarking & rankings
   - Deterministic stress testing (Input cost shocks & demand contraction)
   - AI-powered business model, dependencies, and catalysts
3. **My Watchlist**: Manage persistent holdings and company watchlists in SQLite.
4. **Portfolio Impact**: Ask natural-language questions (e.g. *"How could rising crude prices affect my watchlist?"*) and receive structured transmission mechanisms and company-by-company exposure breakdowns.
5. **News & Commodities**: Dedicated RSS feed aggregator (Economic Times, Livemint, Business Standard) and live commodity futures.
6. **Sector Opportunities**: Comparative industry opportunity scoring radar.
7. **Settings**: LLM Gateway status (Ollama health check, model configuration, optional cloud fallback).

### Evidence-first research workflow

Ask Anything now separates deterministic measurement from model interpretation:

- Explicitly named companies are resolved independently from the company registry.
- Revenue, net-income, margin, and peer growth are calculated in Python from compatible periods.
- Causal questions run deterministic change and variance-breakdown checks before interpretation.
- News/disclosure evidence is searched for causal questions; unsupported causes are labelled unverified.
- Results include reporting periods, statement basis, units, sources, research trace, and limitations.
- If Ollama fails or returns malformed structured output, deterministic findings remain available.

Investigation responses also include stable `evidence` records and
`evidence_references`. Each record carries the company, metric, value, unit,
reporting period, statement basis, and source. The independent verification
status reports unresolved references and inconsistent period metadata without
replacing unavailable values with zero.

### On-demand Screener financial data

When a supported company has no usable local statement data, FinAudit retrieves
that company's consolidated Screener page on demand, normalizes the available
quarterly results, annual profit and loss, balance sheet, cash flow, and ratio
tables, and stores the facts in SQLite. Repeated requests reuse fresh cached
facts; failed refreshes preserve the last valid observations. The freshness
window can be configured with `SCREENER_FINANCIAL_TTL_SECONDS` and the request
timeout with `SCREENER_REQUEST_TIMEOUT`.

Cache status is available from `GET /financial-data/{symbol}/status`. Screener
retrieval is company-scoped and does not preload the Nifty 500.

For example, a question comparing Hindustan Copper with Hindalco Industries retrieves both
companies and produces a comparable FY2025-26 versus FY2024-25 table before asking Ollama
to explain the measured difference.

---

## 🛠️ Installation & Setup

### 1. Python Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pip install feedparser yfinance openpyxl
Copy-Item .env.example .env
```

### 2. Ollama Setup (Local LLM)
1. Download Ollama from [ollama.com](https://ollama.com).
2. Start the service and pull the configured model:
   ```powershell
   ollama serve
   ollama pull qwen3:4b
   ```
   The backend uses a 120-second Ollama generation limit and a bounded 768-token JSON response. The Next.js investigation request allows 185 seconds so the backend can return a classified timeout without the browser aborting first. The default local configuration uses `qwen3:4b` for faster interactive responses; `qwen3:8b` remains supported if you prefer its larger model.

---

## 🚀 Launching the Applications

### Option A: Modern Next.js Frontend (Recommended)

1. **Start the FastAPI Backend** (in terminal 1):
   ```powershell
   python -m uvicorn finaudit.api:app --host 127.0.0.1 --port 8000
   ```
2. **Start the Next.js Frontend** (in terminal 2):
   ```powershell
   cd frontend
   npm run dev
   ```
   Open your browser to **`http://localhost:3000`**.

   If the browser reports that a Recharts/Turbopack module factory is unavailable,
   stop the frontend dev server, remove only the frontend cache, restart it, and
   hard-refresh the page:
   ```powershell
   Remove-Item -Recurse -Force .next
   npm run dev
   ```
   Also unregister any service worker for `localhost:3000` if one is present.

---

### Option B: Streamlit Dashboard

From the project root:
```powershell
python -m streamlit run finaudit/dashboard.py
```
Open your browser to **`http://localhost:8501`**.

---

## 🧪 Automated Testing

Execute the unit test suites:

```powershell
python -m pytest tests/test_agentic_features.py -v
python -m pytest tests/test_financial_platform.py -v
```

All network calls, RSS feeds, and LLM queries in the tests are mocked for reliable offline test execution.
