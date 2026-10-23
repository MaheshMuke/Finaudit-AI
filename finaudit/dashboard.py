import os
import sys
from pathlib import Path

# Ensure project root is in sys.path when running via streamlit
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from finaudit.analysis_engine import AnalysisEngine
from finaudit.commodity_service import get_commodity_service
from finaudit.company_registry import get_company_registry
from finaudit.llm_gateway import get_llm_gateway
from finaudit.news_service import get_news_service
from finaudit.portfolio_impact_service import PortfolioImpactService
from finaudit.scenario_engine import ScenarioEngine
from finaudit.sector_intelligence import SectorIntelligence
from finaudit.sector_opportunity_service import SectorOpportunityService
from finaudit.watchlist_service import get_watchlist_service

load_dotenv()

st.set_page_config(
    page_title="FinAudit AI — Agentic Financial Intelligence & Portfolio Impact",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 20px;
        font-weight: 700;
    }
    .metric-sub {
        color: #38bdf8;
        font-size: 11px;
        margin-top: 2px;
    }
    .badge-tag {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .impact-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize core services
registry = get_company_registry()
engine = AnalysisEngine()
watchlist_service = get_watchlist_service()
news_service = get_news_service()
commodity_service = get_commodity_service()
impact_service = PortfolioImpactService()
radar_service = SectorOpportunityService()
llm_gateway = get_llm_gateway()

# Sidebar Navigation
with st.sidebar:
    st.markdown("### ⚡ FinAudit AI")
    st.caption("Agentic Financial Intelligence & Portfolio Impact Platform")
    st.divider()

    nav_selection = st.radio(
        "Navigation",
        [
            "Overview",
            "Company Analysis",
            "My Watchlist",
            "Portfolio Impact",
            "News & Commodities",
            "Sector Opportunities",
            "Settings",
        ],
        index=0,
    )

    st.divider()
    hw = llm_gateway.check_health()
    if hw["ollama"]["available"]:
        st.success(f"🟢 Ollama Active (`{hw['ollama']['target_model']}`)")
    else:
        st.error("🔴 Ollama Offline")
        st.caption("Start Ollama: `ollama run qwen3:8b`")
        if hw["fallback"]["enabled"] and hw["fallback"]["configured"]:
            st.info(f"Using Cloud Fallback: `{hw['fallback']['provider']}`")
        else:
            st.caption("Deterministic metrics and charts remain active!")


# -------------------------------------------------------------
# 1. OVERVIEW PAGE
# -------------------------------------------------------------
if nav_selection == "Overview":
    st.title("Market Overview & Research Dashboard")
    st.markdown(
        "Welcome to **FinAudit AI** — an autonomous financial intelligence platform "
        "synthesizing verified statements, real-time market indicators, and portfolio sensitivities."
    )

    col_q1, col_q2 = st.columns([4, 1])
    with col_q1:
        quick_query = st.text_input(
            "Quick Search Any Indian Company",
            value="Reliance Industries",
            placeholder="Search by company name or symbol (e.g. Reliance, TCS, INFY, HDFC)...",
            label_visibility="collapsed",
        )
    with col_q2:
        if st.button("Deep Dive Analysis", type="primary", use_container_width=True):
            st.session_state["company_query"] = quick_query
            st.session_state["run_analysis_now"] = True
            st.rerun()

    st.divider()

    # Top Macro & Commodity Ticker Row
    st.markdown("##### 🌐 Global Macro & Commodity Benchmarks")
    comms = commodity_service.get_indicators()
    cols_com = st.columns(len(comms))
    for i, c in enumerate(comms):
        with cols_com[i]:
            chg = c.get("change_pct")
            delta_str = f"{chg:+.2f}% 5D" if chg is not None else None
            st.metric(
                label=f"{c['name']} ({c['unit']})",
                value=f"{c['last_price']}" if c['last_price'] else "N/A",
                delta=delta_str,
            )

    st.divider()

    col_ov_left, col_ov_right = st.columns([3, 2])
    with col_ov_left:
        st.markdown("##### 📰 Latest Market Disclosures & Financial News")
        articles = news_service.get_recent_articles(limit=5)
        if articles:
            for a in articles:
                with st.container():
                    st.markdown(f"**[{a['title']}]({a['link']})**")
                    st.caption(f"{a['source_name']} • {a['category']} • Published: {a['published_at'][:16]}")
        else:
            st.caption("Fetching latest RSS feeds...")

    with col_ov_right:
        st.markdown("##### 📌 My Quick Watchlist")
        w_items = watchlist_service.get_items("default")
        if w_items:
            for item in w_items[:5]:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<span class='badge-tag'>{item['symbol']}</span> **{item['company_name']}**"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Watchlist is empty. Add companies in the 'My Watchlist' section.")


# -------------------------------------------------------------
# 2. COMPANY ANALYSIS PAGE
# -------------------------------------------------------------
elif nav_selection == "Company Analysis":
    st.title("Company Financial & Business Intelligence")
    st.caption("Deterministic accounting analysis + Bounded local agent reasoning")

    default_q = st.session_state.get("company_query", "Reliance Industries")
    col_s1, col_s2 = st.columns([4, 1])
    with col_s1:
        query_input = st.text_input("Enter Company Name or Ticker", value=default_q, label_visibility="collapsed")
    with col_s2:
        run_now = st.button("Analyse Company", type="primary", use_container_width=True)

    if run_now or st.session_state.get("run_analysis_now"):
        st.session_state["run_analysis_now"] = False
        st.session_state["company_query"] = query_input

        with st.status(f"Investigating {query_input}...", expanded=True) as status:
            prog = st.empty()

            def cb(m):
                prog.write(f"🔍 {m}")

            res = engine.run_full_analysis(query_input, progress_callback=cb)
            st.session_state["last_company_res"] = res
            if res.get("success"):
                status.update(label=f"Investigation completed for {res['company']['company_name']}!", state="complete")
            else:
                status.update(label=f"Error: {res.get('error')}", state="error")

    res = st.session_state.get("last_company_res")
    if res and res.get("success"):
        comp = res["company"]
        mkt = res["market_data"]
        calc = res["calculations"]
        ratios = calc.get("latest_ratios", {})
        metric_basis = calc.get("metric_metadata", {}).get("statement_basis", "unknown")
        metric_period = ratios.get("reporting_period", ratios.get("period", "unavailable"))
        series = calc.get("annual_series", {})
        peers = res["competitors"]
        ai = res["ai_analysis"]
        ai_err = res["ai_error"]
        raw = calc.get("raw_tables", {})

        st.divider()

        # Company Header Bar
        col_h1, col_h2, col_h3, col_h4 = st.columns([3, 2, 2, 2])
        with col_h1:
            st.subheader(comp["company_name"])
            st.markdown(
                f"<span class='badge-tag'>{comp['symbol']}</span> • "
                f"**Industry:** {comp['industry']} • **Ticker:** `{comp['yahoo_ticker']}`",
                unsafe_allow_html=True,
            )
            # Add to watchlist button
            if st.button("⭐ Add to Watchlist", key="btn_add_wl"):
                watchlist_service.add_item(comp["symbol"], comp["company_name"])
                st.toast(f"Added {comp['symbol']} to watchlist!")

        with col_h2:
            st.metric("Market Price", f"₹{mkt['current_price']:,.2f}" if mkt['current_price'] else "N/A", delta=f"{mkt['one_year_return']:+.2f}% 1Y" if mkt['one_year_return'] is not None else None)
        with col_h3:
            st.metric(f"Revenue ({metric_basis.title()})", ratios.get("revenue_display", "N/A"), delta=ratios.get("revenue_growth_display"))
        with col_h4:
            st.metric(f"Net Profit (PAT, {metric_basis.title()})", ratios.get("net_income_display", "N/A"), delta=ratios.get("net_income_growth_display"))

        # Tabs
        t_overview, t_fin, t_stock, t_peers, t_scenarios, t_intel = st.tabs(
            [
                "📋 Executive Overview",
                "📊 Financial Statements",
                "📈 Stock & Valuation",
                "🏢 Peer Competitors",
                "🎯 Scenario Sensitivity",
                "🧠 AI Intelligence & Catalysts",
            ]
        )

        with t_overview:
            # Sector-specific framework
            framework = SectorIntelligence.get_framework(comp["industry"])
            st.markdown(f"**Sector Profile:** `{comp['industry']}`")
            st.info(f"**Core Sector Drivers:** {framework.get('primary_driver')}")
            st.caption(
                f"Latest financial metrics: {metric_basis} statement ending {metric_period}. "
                f"Currency: {calc.get('metric_metadata', {}).get('currency', 'INR')}; "
                f"unit: {calc.get('metric_metadata', {}).get('display_unit', 'INR crore')}."
            )

            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            with col_m1:
                st.markdown(f"<div class='metric-card'><div class='metric-title'>Operating Margin ({metric_basis})</div><div class='metric-value'>{ratios.get('operating_margin_display', 'N/A')}</div></div>", unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"<div class='metric-card'><div class='metric-title'>Net Margin ({metric_basis})</div><div class='metric-value'>{ratios.get('net_margin_display', 'N/A')}</div></div>", unsafe_allow_html=True)
            with col_m3:
                st.markdown(f"<div class='metric-card'><div class='metric-title'>Debt to Equity ({metric_basis})</div><div class='metric-value'>{ratios.get('debt_to_equity_display', 'N/A')}</div></div>", unsafe_allow_html=True)
            with col_m4:
                st.markdown(f"<div class='metric-card'><div class='metric-title'>ROE ({metric_basis})</div><div class='metric-value'>{ratios.get('roe_display', 'N/A')}</div></div>", unsafe_allow_html=True)
            with col_m5:
                st.markdown(f"<div class='metric-card'><div class='metric-title'>Free Cash Flow ({metric_basis})</div><div class='metric-value'>{ratios.get('free_cash_flow_display', 'N/A')}</div></div>", unsafe_allow_html=True)

            if mkt.get("business_summary"):
                st.markdown("##### Business Summary")
                st.caption(mkt["business_summary"])

        with t_fin:
            st.markdown("#### Multi-Year Performance Trend")
            periods = series.get("periods", [])
            rev_vals = series.get("revenue", [])
            ni_vals = series.get("net_income", [])
            if periods and any(v is not None for v in rev_vals):
                fig = go.Figure()
                fig.add_trace(go.Bar(x=periods, y=[v / 1e7 if v else None for v in rev_vals], name="Revenue (₹ Cr)", marker_color="#38bdf8"))
                fig.add_trace(go.Scatter(x=periods, y=[v / 1e7 if v else None for v in ni_vals], name="Net Profit (₹ Cr)", line=dict(color="#4ade80", width=3), mode="lines+markers"))
                fig.update_layout(title="Annual Financial Trajectory (₹ Crores)", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("Income Statement (Annual & Quarterly)"):
                st.dataframe(raw.get("annual_income", pd.DataFrame()).style.format("{:,.0f}", na_rep="N/A"), use_container_width=True)
            with st.expander("Balance Sheet (Annual & Quarterly)"):
                st.dataframe(raw.get("annual_balance", pd.DataFrame()).style.format("{:,.0f}", na_rep="N/A"), use_container_width=True)
            with st.expander("Cash Flow Statement (Annual & Quarterly)"):
                st.dataframe(raw.get("annual_cashflow", pd.DataFrame()).style.format("{:,.0f}", na_rep="N/A"), use_container_width=True)

        with t_stock:
            st.markdown("#### Historical 1-Year Stock Performance")
            h_dates = mkt.get("history_dates", [])
            h_prices = mkt.get("history_prices", [])
            if h_dates and h_prices:
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.metric("1Y Return", f"{mkt['one_year_return']:+.2f}%" if mkt['one_year_return'] is not None else "N/A")
                with col_s2:
                    st.metric("Annualized Volatility", f"{mkt['annualized_volatility']:.2f}%" if mkt['annualized_volatility'] is not None else "N/A")
                with col_s3:
                    st.metric("Max Drawdown", f"{mkt['max_drawdown']:.2f}%" if mkt['max_drawdown'] is not None else "N/A")
                with col_s4:
                    st.metric("52-Week Range", f"₹{mkt['fifty_two_week_low']} - ₹{mkt['fifty_two_week_high']}" if mkt['fifty_two_week_low'] else "N/A")

                df_p = pd.DataFrame({"Date": pd.to_datetime(h_dates), "Price": h_prices})
                fig_p = px.line(df_p, x="Date", y="Price", title="1Y Adjusted Close", template="plotly_dark")
                fig_p.update_traces(line_color="#38bdf8")
                st.plotly_chart(fig_p, use_container_width=True)
            else:
                st.info("Market price data unavailable.")

        with t_peers:
            st.markdown(f"#### Peer Competitor Benchmarking ({comp['industry']})")
            c_table = peers.get("comparison_table", [])
            rankings = peers.get("rankings", {})
            if rankings:
                c_ranks = st.columns(len(rankings))
                for i, (m_label, r_data) in enumerate(rankings.items()):
                    with c_ranks[i]:
                        st.markdown(
                            f"<div class='metric-card'><div class='metric-title'>{m_label} Rank</div>"
                            f"<div class='metric-value'>#{r_data['rank']} / {r_data['out_of']}</div>"
                            f"<div class='metric-sub'>Leader: {r_data['leader']}</div></div>",
                            unsafe_allow_html=True,
                        )
            st.dataframe(pd.DataFrame(c_table), use_container_width=True)

        with t_scenarios:
            st.markdown("#### Deterministic Stress-Testing & Sensitivity Engine")
            st.caption("Evaluate how input inflation or demand contractions impact Net Margins.")

            col_sc1, col_sc2 = st.columns(2)
            with col_sc1:
                cost_shock = st.slider("Input Cost Shock (%)", min_value=0.0, max_value=25.0, value=5.0, step=1.0)
            with col_sc2:
                demand_shock = st.slider("Demand / Revenue Contraction (%)", min_value=-25.0, max_value=0.0, value=-5.0, step=1.0)

            sc_res = ScenarioEngine.run_sensitivity(
                base_revenue=ratios.get("revenue"),
                base_net_income=ratios.get("net_income"),
                base_operating_income=ratios.get("operating_income"),
                base_total_debt=ratios.get("total_debt"),
                industry=comp["industry"],
                input_cost_shock_pct=cost_shock,
                revenue_shock_pct=demand_shock,
            )

            if sc_res.get("status") == "success":
                sc_df = pd.DataFrame(sc_res["scenarios"])
                sc_df["Revenue (₹ Cr)"] = sc_df["revenue"] / 1e7
                sc_df["Net Profit (₹ Cr)"] = sc_df["net_income"] / 1e7
                sc_df["Net Margin (%)"] = sc_df["net_margin"].round(2)
                st.dataframe(sc_df[["name", "Revenue (₹ Cr)", "Net Profit (₹ Cr)", "Net Margin (%)"]], use_container_width=True)

        with t_intel:
            st.markdown("#### AI Business Intelligence & Share Price Drivers")
            if ai:
                with st.expander("💼 Business Model & Monetization", expanded=True):
                    st.write(ai.get("business_model", {}).get("summary", "N/A"))
                    st.write(f"**Revenue Generation:** {ai.get('business_model', {}).get('revenue_generation', 'N/A')}")
                with st.expander("🔗 Business Dependencies", expanded=True):
                    bd = ai.get("business_dependencies", {})
                    st.write(f"- **Customers:** {bd.get('customers_and_concentration', 'N/A')}")
                    st.write(f"- **Suppliers & Input Costs:** {bd.get('suppliers_and_input_costs', 'N/A')}")
                    st.write(f"- **Commodity & FX:** {bd.get('macro_commodities_fx_rates', 'N/A')}")
                with st.expander("⚡ Share Price Drivers & Catalysts", expanded=True):
                    spd = ai.get("share_price_drivers", {})
                    col_cat1, col_cat2 = st.columns(2)
                    with col_cat1:
                        st.markdown("**Positive Catalysts:**")
                        for p in spd.get("positive_catalysts", []):
                            st.markdown(f"- {p}")
                    with col_cat2:
                        st.markdown("**Negative Risks:**")
                        for r in spd.get("negative_risks", []):
                            st.markdown(f"- {r}")
            else:
                st.warning(f"AI Narrative unavailable: {ai_err}")


# -------------------------------------------------------------
# 3. MY WATCHLIST PAGE
# -------------------------------------------------------------
elif nav_selection == "My Watchlist":
    st.title("My Watchlist & Portfolio Holdings")
    st.caption("Manage saved companies, monitor exposures, and track portfolio allocations.")

    col_w1, col_w2 = st.columns([3, 1])
    with col_w1:
        add_sym = st.text_input("Add Company by Symbol or Name", placeholder="e.g. TCS, INFY, TITAN...")
    with col_w2:
        if st.button("Add to Watchlist", type="primary", use_container_width=True):
            resolved = registry.resolve(add_sym)
            if resolved:
                watchlist_service.add_item(resolved.symbol, resolved.company_name)
                st.success(f"Added {resolved.company_name}!")
                st.rerun()
            else:
                st.error("Company not found.")

    items = watchlist_service.get_items("default")
    if items:
        st.markdown(f"**Saved Holdings ({len(items)} companies):**")
        for item in items:
            with st.container():
                col_i1, col_i2, col_i3 = st.columns([3, 2, 1])
                with col_i1:
                    st.markdown(f"**{item['company_name']}** (`{item['symbol']}`)")
                with col_i2:
                    st.caption(f"Added: {item['created_at'][:10]}")
                with col_i3:
                    if st.button("Remove", key=f"del_{item['symbol']}"):
                        watchlist_service.remove_item(item["symbol"])
                        st.rerun()
                st.divider()
    else:
        st.info("Your watchlist is currently empty. Add your key holdings above!")


# -------------------------------------------------------------
# 4. PORTFOLIO IMPACT INTELLIGENCE
# -------------------------------------------------------------
elif nav_selection == "Portfolio Impact":
    st.title("Portfolio Impact Intelligence")
    st.caption("Ask natural-language questions about your holdings and macro shocks.")

    items = watchlist_service.get_items("default")
    symbols = [it["symbol"] for it in items]

    if not symbols:
        st.warning("Please add at least one company to 'My Watchlist' before running Portfolio Impact analysis.")
    else:
        st.markdown(f"**Active Watchlist Scope:** {', '.join(symbols)}")

        sample_questions = [
            "How could rising crude oil prices affect my watchlist?",
            "Which companies could be affected by an RBI interest rate change?",
            "How does a weakening Indian Rupee (USD/INR) impact these holdings?",
            "What are the major commodity dependencies across my portfolio?",
        ]
        chosen_sample = st.selectbox("Or select an example question:", ["Custom Query..."] + sample_questions)

        user_q = st.text_area(
            "Natural Language Question",
            value="" if chosen_sample == "Custom Query..." else chosen_sample,
            placeholder="Type your macro or portfolio question...",
        )

        if st.button("Analyse Watchlist Impact", type="primary"):
            if not user_q.strip():
                st.error("Please enter a question.")
            else:
                with st.spinner("Agent investigating cross-company transmission mechanisms..."):
                    impact_res = impact_service.analyze_impact(user_q, symbols)
                    st.session_state["impact_res"] = impact_res

        res = st.session_state.get("impact_res")
        if res and res.get("success"):
            data = res["analysis"]
            st.divider()
            st.subheader("Impact Executive Summary")
            st.info(data.get("short_answer"))

            st.markdown(f"**Transmission Mechanism:** {data.get('transmission_mechanism')}")

            st.markdown("##### Company-by-Company Exposure Breakdown")
            for c_imp in data.get("company_impacts", []):
                imp_val = c_imp.get("impact", "Mixed")
                border_color = "#4ade80" if "Positive" in imp_val else ("#f87171" if "Negative" in imp_val else "#facc15")

                st.markdown(
                    f"<div class='impact-card' style='border-left: 5px solid {border_color};'>"
                    f"**{c_imp.get('company_name')} ({c_imp.get('symbol')})** — "
                    f"<span style='color: {border_color}; font-weight: bold;'>{imp_val}</span> "
                    f"(Materiality: {c_imp.get('materiality', 'Medium')})<br/>"
                    f"<span style='color: #cbd5e1; font-size: 13px;'>{c_imp.get('mechanism')}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("##### Key Indicators to Monitor Next")
            for ind in data.get("indicators_to_monitor", []):
                st.markdown(f"- {ind}")


# -------------------------------------------------------------
# 5. NEWS & COMMODITIES PAGE
# -------------------------------------------------------------
elif nav_selection == "News & Commodities":
    st.title("Market Disclosures, Feeds & Commodity Intelligence")

    t_n1, t_n2 = st.tabs(["📰 Real-Time RSS News Feeds", "🌐 Commodity & FX Benchmarks"])

    with t_n1:
        if st.button("🔄 Refresh Latest Feeds"):
            with st.spinner("Fetching RSS feeds..."):
                added = news_service.fetch_latest_feeds()
                st.toast(f"Fetched {added} new articles!")

        articles = news_service.get_recent_articles(limit=25)
        for a in articles:
            with st.container():
                st.markdown(f"**[{a['title']}]({a['link']})**")
                st.caption(f"{a['source_name']} • {a['category']} • {a['published_at'][:16]}")
                if a.get("summary"):
                    st.write(a["summary"][:200] + "...")
                st.divider()

    with t_n2:
        st.markdown("##### Global Futures & Benchmark Indicators")
        comms = commodity_service.get_indicators()
        st.dataframe(pd.DataFrame(comms), use_container_width=True)


# -------------------------------------------------------------
# 6. SECTOR OPPORTUNITIES PAGE
# -------------------------------------------------------------
elif nav_selection == "Sector Opportunities":
    st.title("Sector Opportunity Radar")
    st.caption("Comparative industry scoring based on median constituent revenue growth and profit margins.")

    radar_data = radar_service.get_sector_radar()
    if radar_data:
        df_radar = pd.DataFrame(radar_data)
        fig_radar = px.bar(
            df_radar,
            x="opportunity_score",
            y="industry",
            orientation="h",
            title="Sector Opportunity Score (0 to 100)",
            template="plotly_dark",
            color="opportunity_score",
            color_continuous_scale="Viridis",
        )
        fig_radar.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_radar, use_container_width=True)

        st.dataframe(
            df_radar[
                [
                    "industry",
                    "opportunity_score",
                    "median_revenue_growth",
                    "median_net_margin",
                    "median_roe",
                    "company_count",
                ]
            ].rename(
                columns={
                    "industry": "Industry",
                    "opportunity_score": "Score",
                    "median_revenue_growth": "Median YoY Growth (%)",
                    "median_net_margin": "Median Margin (%)",
                    "median_roe": "Median ROE (%)",
                    "company_count": "Constituents",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# -------------------------------------------------------------
# 7. SETTINGS PAGE
# -------------------------------------------------------------
elif nav_selection == "Settings":
    st.title("Platform Configuration & Model Settings")

    st.markdown("#### LLM Gateway Configuration")
    hw = llm_gateway.check_health()
    st.write(f"Primary Provider: `{hw['primary']}`")
    st.write(f"Active Provider: `{hw['active_provider']}`")
    st.write(f"Configured Model: `{hw['ollama'].get('target_model')}`")

    st.divider()
    st.markdown("#### Cloud Fallback Settings")
    st.write(f"Fallback Enabled: `{hw['fallback']['enabled']}`")
    st.write(f"Fallback Provider: `{hw['fallback']['provider']}`")
    st.write(f"API Key Configured: `{hw['fallback']['configured']}`")

    st.caption("To change model settings, update `.env` or set environment variables before starting Streamlit.")
