"""Demonstration of end-to-end TCS research session with multi-turn contextual follow-ups.

Executes:
Turn 1: Initial TCS profitability & quarterly performance research
Turn 2: Follow-up question with ordinal coreference ("What were the profit margins and valuation of the second company?")
Turn 3: Follow-up question with delta analysis ("What changed since our last analysis?")
"""

import sys
import json
from datetime import datetime

# Ensure utf-8 encoding for currency symbols like ₹ on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from finaudit.agent_orchestrator import AgentOrchestrator
from finaudit.memory_service import MemoryService
from finaudit.macro_service import MacroMarketIntelligenceService
from finaudit.research_sources import OfficialQuarterlyDisclosureAdapter


def run_tcs_demonstration():
    print("=" * 80)
    print("FINAUDIT AI: PERSISTENT CONVERSATIONAL AGENTIC INTELLIGENCE SYSTEM")
    print("TCS End-to-End Multi-Turn Research Demonstration")
    print("=" * 80)

    mem = MemoryService()
    macro = MacroMarketIntelligenceService()
    orchestrator = AgentOrchestrator()

    # Step 0: Create Persistent Research Session
    print("\n[Step 0] Initializing Persistent Research Session...")
    sess = mem.create_session(
        title="TCS Q1 Strategic & Financial Intelligence Audit",
        target_company="TCS",
        session_peers=["INFY", "HCLTECH", "WIPRO", "TECHM"],
    )
    session_id = sess["session_id"]
    peers = sess.get("session_peers") or sess.get("selected_peers") or ["INFY", "HCLTECH", "WIPRO"]
    print(f"  Session Created: ID={session_id}")
    print(f"  Target Company: {sess.get('target_company') or sess.get('company_symbol')}")
    print(f"  Active Peer Group: {peers}")

    # Fetch Macro Benchmark Snapshot
    macro_snap = macro.get_macro_snapshot()
    print(f"\n[Environment] Macro Benchmarks ({macro_snap.get('freshness_status')}):")
    for k in ["nifty_50", "usd_inr", "us_10y_yield"]:
        b = macro_snap.get(k, {})
        print(f"  - {k.upper()}: Current={b.get('current')}, 1Y Return={b.get('1y_return_pct')}%")
    repo = macro_snap.get("rbi_repo_rate", {})
    print(f"  - RBI Repo Rate: {repo.get('rate_pct')}% ({repo.get('stance')})")

    # -------------------------------------------------------------------------
    # TURN 1: Initial Investigation on TCS
    # -------------------------------------------------------------------------
    q1 = "Analyze TCS quarterly profitability, revenue growth, and key operating drivers."
    print("\n" + "=" * 80)
    print(f"TURN 1: USER INQUIRY: \"{q1}\"")
    print("=" * 80)

    turn1_res = orchestrator.investigate_company(
        query="TCS",
        question=q1,
        session_id=session_id,
        custom_peers=peers,
    )

    print(f"\n[Turn 1 Response Status]: {turn1_res.get('investigation_status')}")
    print(f"[Resolved Query]: {turn1_res.get('question')}")
    conv_ans1 = turn1_res.get("conversational_answer") or turn1_res.get("summary") or "Analysis completed."
    print(f"\n[Conversational Answer]:\n{conv_ans1}")

    # Display Metrics & Bridge
    metrics = turn1_res.get("metrics", {})
    print(f"\n[Reported Financial Facts for TCS]:")
    print(f"  - Reporting Period: {metrics.get('reporting_period')}")
    print(f"  - Revenue: {metrics.get('revenue')}")
    print(f"  - Net Income: {metrics.get('net_income')}")
    print(f"  - Operating Margin: {metrics.get('operating_margin')}")
    print(f"  - Net Margin: {metrics.get('net_margin')}")

    # Display Citations & Evidence
    evidence = turn1_res.get("evidence", [])
    print(f"\n[Evidence & Primary Source Citations ({len(evidence)} items)]:")
    for i, ev in enumerate(evidence[:4], 1):
        print(f"  [{i}] Source: {ev.get('source')} | Title: {ev.get('title')}")
        print(f"      Link: {ev.get('link') or ev.get('url')} | Published: {ev.get('published') or ev.get('date')}")

    # -------------------------------------------------------------------------
    # TURN 2: Contextual Follow-Up with Ordinal Reference
    # -------------------------------------------------------------------------
    q2 = "What were the profit margins and valuation of the second company?"
    print("\n" + "=" * 80)
    print(f"TURN 2: CONTEXTUAL FOLLOW-UP: \"{q2}\"")
    print("=" * 80)

    turn2_res = orchestrator.investigate_company(
        query="TCS",
        question=q2,
        session_id=session_id,
        custom_peers=peers,
    )

    ctx_res = turn2_res.get("context_resolution", {})
    print(f"\n[Context Resolution]:")
    print(f"  - Reference Type: {ctx_res.get('reference_type')}")
    print(f"  - Resolved Target Symbol: {ctx_res.get('resolved_target')}")
    print(f"  - Resolved Company Name: {ctx_res.get('referenced_company', {}).get('name')}")
    print(f"  - Rewritten Query: \"{turn2_res.get('question')}\"")

    conv_ans2 = turn2_res.get("conversational_answer") or turn2_res.get("summary") or "Peer analysis complete."
    print(f"\n[Conversational Answer]:\n{conv_ans2}")

    peer_metrics = turn2_res.get("metrics", {})
    print(f"\n[Financial Facts for {ctx_res.get('resolved_target')}]:")
    print(f"  - Reporting Period: {peer_metrics.get('reporting_period')}")
    print(f"  - Revenue: {peer_metrics.get('revenue')}")
    print(f"  - Net Margin: {peer_metrics.get('net_margin')}")
    print(f"  - PE Ratio: {peer_metrics.get('pe_ratio')}")

    # -------------------------------------------------------------------------
    # TURN 3: Contextual Follow-Up with Temporal Delta Query
    # -------------------------------------------------------------------------
    q3 = "What changed since our last analysis?"
    print("\n" + "=" * 80)
    print(f"TURN 3: CONTEXTUAL DELTA: \"{q3}\"")
    print("=" * 80)

    turn3_res = orchestrator.investigate_company(
        query="TCS",
        question=q3,
        session_id=session_id,
        custom_peers=peers,
    )

    ctx_res3 = turn3_res.get("context_resolution", {})
    delta_res = turn3_res.get("delta_analysis", {})
    print(f"\n[Context Resolution]:")
    print(f"  - Reference Type: {ctx_res3.get('reference_type')}")
    print(f"  - Is Delta Query: {ctx_res3.get('is_delta_query')}")
    print(f"  - Prior Findings Count: {delta_res.get('prior_findings_count')}")

    conv_ans3 = turn3_res.get("conversational_answer") or turn3_res.get("summary") or "Delta check complete."
    print(f"\n[Conversational Answer]:\n{conv_ans3}")

    # Step 4: Verify Full Trajectory from Database
    print("\n" + "=" * 80)
    print("PERSISTENT SESSION AUDIT LOG (SQLite Database)")
    print("=" * 80)
    all_msgs = mem.get_messages(session_id)
    print(f"Total Persisted Messages: {len(all_msgs)}")
    for idx, m in enumerate(all_msgs, 1):
        content_preview = m['content'][:120] + "..." if len(m['content']) > 120 else m['content']
        print(f"  Message #{idx} [{m['role'].upper()}] @ {m['created_at']}: {content_preview}")
        if m.get("citations"):
            print(f"    Citations Attached: {len(m['citations'])}")

    all_findings = mem.get_findings(session_id)
    print(f"\nTotal Persisted Findings: {len(all_findings)}")
    for f in all_findings[:5]:
        print(f"  - [{f.get('topic')}] {f.get('claim')} (Confidence: {f.get('confidence')})")

    print("\n[SUCCESS] End-to-end multi-turn research demonstration completed successfully!")


if __name__ == "__main__":
    run_tcs_demonstration()
