"""Persistent conversational and research memory service for FinAudit AI.

Handles multi-turn session tracking, message history, structured findings storage,
and anaphora/context resolution for references like "that announcement", "the second company",
and "what changed since our last analysis?".
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from finaudit import database
from finaudit.company_registry import get_company_registry

logger = logging.getLogger(__name__)


class MemoryService:
    """Manages persistent conversational sessions and context resolution."""

    def __init__(self):
        self.registry = get_company_registry()

    def create_session(
        self,
        *args: Any,
        symbol: Optional[str] = None,
        title: Optional[str] = None,
        peers: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        target_company: Optional[str] = None,
        session_peers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a new persistent research session."""
        resolved_sym = target_company or symbol or kwargs.get("company_symbol")
        resolved_title = title
        resolved_peers = session_peers or peers or kwargs.get("selected_peers")

        if len(args) == 1:
            if not resolved_sym:
                resolved_sym = args[0]
            elif not resolved_title:
                resolved_title = args[0]
        elif len(args) == 2:
            if args[0].isupper() and len(args[0]) <= 10:
                resolved_sym = args[0]
                resolved_title = args[1]
            else:
                resolved_title = args[0]
                resolved_sym = args[1]
        elif len(args) >= 3:
            if args[0].isupper() and len(args[0]) <= 10 and isinstance(args[2], list):
                resolved_sym = args[0]
                resolved_title = args[1]
                resolved_peers = args[2]
            else:
                resolved_title = args[0]
                resolved_sym = args[1]
                resolved_peers = args[2]

        resolved_sym = resolved_sym or "GENERAL"
        return database.create_research_session(
            symbol=resolved_sym,
            title=resolved_title,
            peers=resolved_peers,
            metadata=metadata,
        )

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve research session by ID."""
        return database.get_research_session(session_id)

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List active research sessions ordered by last update."""
        return database.list_research_sessions(limit=limit)

    def update_session_peers(self, session_id: str, peers: List[str]) -> Optional[Dict[str, Any]]:
        """Update selected peer group for a session."""
        return database.update_session_peers(session_id, peers)

    def delete_session(self, session_id: str) -> bool:
        """Delete session and all its associated messages and findings."""
        return database.delete_research_session(session_id)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: Optional[str] = None,
        evidence_references: Optional[List[str]] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
        structured_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a user or assistant message with full citations and evidence links."""
        return database.save_session_message(
            session_id=session_id,
            role=role,
            content=content,
            intent=intent,
            evidence_references=evidence_references,
            citations=citations,
            structured_payload=structured_payload,
        )

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: Optional[str] = None,
        evidence_references: Optional[List[str]] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        structured_payload: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Alias for add_message with flexible metadata handling."""
        payload = structured_payload or metadata or {}
        return self.add_message(
            session_id=session_id,
            role=role,
            content=content,
            intent=intent,
            evidence_references=evidence_references,
            citations=citations,
            structured_payload=payload,
        )

    def get_messages(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve full message trajectory for a session."""
        return database.get_session_messages(session_id=session_id, limit=limit)

    def add_finding(
        self,
        session_id: str,
        topic: str,
        claim: str,
        claim_type: str,
        supporting_evidence: Optional[List[Dict[str, Any]]] = None,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """Store a structured finding with timestamp and citations."""
        return database.save_session_finding(
            session_id=session_id,
            topic=topic,
            claim=claim,
            claim_type=claim_type,
            supporting_evidence=supporting_evidence,
            confidence=confidence,
        )

    def save_findings(
        self,
        session_id: str,
        company_symbol: str,
        findings: List[str],
        citations: Optional[List[Dict[str, Any]]] = None,
        reporting_period: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Save a collection of structured findings."""
        results = []
        for f in findings:
            finding_rec = self.add_finding(
                session_id=session_id,
                topic=company_symbol,
                claim=str(f),
                claim_type="fact",
                supporting_evidence=citations,
            )
            results.append(finding_rec)

        # Index qualitative findings in ChromaDB vector store
        try:
            from finaudit.rag import get_ingestion_service
            ingestion = get_ingestion_service()
            for finding_rec in results:
                ingestion.index_session_finding(
                    finding=finding_rec,
                    session_id=session_id,
                    company_symbol=company_symbol,
                )
        except Exception as exc:
            logger.warning("Optional vector indexing of session finding failed: %s", exc)

        return results

    def get_findings(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all findings captured in this session."""
        return database.get_session_findings(session_id=session_id)

    def resolve_contextual_references(
        self,
        session_id: Optional[str] = None,
        question: Optional[str] = None,
        current_symbol: Optional[str] = None,
        query: Optional[str] = None,
        custom_peers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Resolve conversational anaphora and contextual follow-up questions.
        
        Handles:
        1. 'that announcement' / 'the filing' / 'the attachment' -> latest disclosure mentioned
        2. 'the second company' / 'the first peer' / 'the third peer' -> indexed company from active peer set
        3. 'what changed since our last analysis?' / 'what has changed?' -> delta comparison request
        """
        raw_q = (question or query or "").strip()
        q_lower = raw_q.lower()
        resolved_info: Dict[str, Any] = {
            "original_question": raw_q,
            "resolved_question": raw_q,
            "resolved_query": raw_q,
            "target_symbol": current_symbol,
            "resolved_target": current_symbol,
            "reference_type": "direct",
            "referenced_disclosure": None,
            "referenced_company": None,
            "is_delta_query": False,
            "resolution_notes": [],
        }

        if not session_id:
            return resolved_info

        session = self.get_session(session_id)
        if not session:
            return resolved_info

        session_symbol = session.get("company_symbol") or session.get("target_company") or current_symbol
        resolved_info["target_symbol"] = session_symbol
        resolved_info["resolved_target"] = session_symbol
        peers: List[str] = custom_peers or session.get("selected_peers") or session.get("session_peers") or []

        messages = self.get_messages(session_id, limit=20)
        findings = self.get_findings(session_id)

        # 1. Resolve company ordinal references (e.g. "the second company", "first peer")
        peer_ordinal_patterns = [
            (r"\b(?:the\s+)?(?:first|1st)\s+(?:company|peer|competitor)\b", 0),
            (r"\b(?:the\s+)?(?:second|2nd)\s+(?:company|peer|competitor)\b", 1),
            (r"\b(?:the\s+)?(?:third|3rd)\s+(?:company|peer|competitor)\b", 2),
            (r"\b(?:the\s+)?(?:fourth|4th)\s+(?:company|peer|competitor)\b", 3),
            (r"\b(?:the\s+)?last\s+(?:company|peer|competitor)\b", -1),
        ]
        for pattern, idx in peer_ordinal_patterns:
            if re.search(pattern, q_lower):
                if peers and (idx == -1 or idx < len(peers)):
                    resolved_sym = peers[idx]
                    rec = self.registry.get_by_symbol(resolved_sym)
                    resolved_name = rec.company_name if rec else resolved_sym
                    resolved_info["reference_type"] = "ordinal_peer"
                    resolved_info["referenced_company"] = {"symbol": resolved_sym, "name": resolved_name, "index": idx}
                    resolved_info["target_symbol"] = resolved_sym
                    resolved_info["resolved_target"] = resolved_sym
                    # Rewrite question to explicitly state the resolved company name
                    resolved_q = re.sub(pattern, f"{resolved_name} ({resolved_sym})", raw_q, flags=re.I)
                    resolved_info["resolved_question"] = resolved_q
                    resolved_info["resolved_query"] = resolved_q
                    resolved_info["resolution_notes"].append(
                        f"Resolved peer reference to {resolved_name} ({resolved_sym})."
                    )
                    break

        # 2. Resolve disclosure references (e.g. "that announcement", "the filing", "the attachment")
        disclosure_patterns = [
            r"\bthat\s+(?:announcement|filing|disclosure|notice|attachment)\b",
            r"\bthe\s+(?:announcement|filing|disclosure|notice|attachment)\b",
            r"\bthis\s+(?:announcement|filing|disclosure|notice|attachment)\b",
        ]
        if any(re.search(pat, q_lower) for pat in disclosure_patterns):
            resolved_info["reference_type"] = "announcement"
            latest_disc = None
            for msg in reversed(messages):
                citations = msg.get("citations") or []
                for c in citations:
                    if c.get("source_kind") in ("nse_corporate_disclosure_candidate", "nse_disclosure") or "nse" in str(c.get("source")).lower() or ".pdf" in str(c.get("url") or c.get("link")).lower():
                        latest_disc = c
                        break
                if latest_disc:
                    break

            if not latest_disc:
                disclosures = database.list_nse_disclosures(company_symbol=session_symbol, limit=5)
                if disclosures:
                    latest_disc = disclosures[0]

            if latest_disc:
                title = latest_disc.get("title") or "NSE announcement"
                date_str = latest_disc.get("publication_date") or ""
                resolved_info["referenced_disclosure"] = latest_disc
                resolved_q = raw_q
                for pat in disclosure_patterns:
                    resolved_q = re.sub(pat, f"the disclosure '{title}' ({date_str})", resolved_q, flags=re.I)
                resolved_info["resolved_question"] = resolved_q
                resolved_info["resolved_query"] = resolved_q
                resolved_info["resolution_notes"].append(
                    f"Resolved disclosure reference to '{title}' ({date_str})."
                )

        # 3. Resolve change comparison: "what changed since our last analysis?"
        change_patterns = [
            r"what\s+changed\s+since\s+(?:our\s+)?(?:last|previous)\s+analysis",
            r"what\s+(?:has\s+)?changed",
            r"how\s+has\s+it\s+changed\s+since\s+last\s+time",
        ]
        if any(re.search(pat, q_lower) for pat in change_patterns):
            resolved_info["is_delta_query"] = True
            resolved_info["reference_type"] = "delta_query"
            resolved_info["resolution_notes"].append(
                f"Detected delta comparison against prior session findings for {session_symbol}."
            )

        resolved_info["resolved_target"] = resolved_info.get("resolved_target") or resolved_info["target_symbol"]
        resolved_info["resolved_query"] = resolved_info.get("resolved_query") or resolved_info["resolved_question"]
        return resolved_info


    def compute_findings_delta(
        self,
        session_id: str,
        current_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute delta between earlier findings/metrics and current state."""
        findings = self.get_findings(session_id)
        if not findings:
            return {
                "status": "no_prior_findings",
                "delta_summary": "No prior findings were recorded in this research session.",
                "changes": [],
            }

        changes = []
        cur_rev = current_metrics.get("revenue_numeric") or current_metrics.get("revenue")
        cur_ni = current_metrics.get("net_income_numeric") or current_metrics.get("net_income")

        for f in findings:
            changes.append({
                "topic": f.get("topic"),
                "previous_claim": f.get("claim"),
                "recorded_at": f.get("created_at"),
                "claim_type": f.get("claim_type"),
            })

        return {
            "status": "delta_available",
            "prior_findings_count": len(findings),
            "changes": changes,
            "current_period": current_metrics.get("reporting_period") or current_metrics.get("period"),
            "delta_summary": (
                f"Comparing against {len(findings)} prior findings from session {session_id}. "
                f"Current reporting period is {current_metrics.get('period', 'N/A')}."
            ),
        }

    def get_delta_analysis(self, session_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve delta analysis against earlier session findings."""
        findings = self.get_findings(session_id)
        if symbol:
            matching = [f for f in findings if f.get("topic") == symbol]
            if matching:
                findings = matching
        if not findings:
            return {
                "status": "no_prior_findings",
                "prior_findings_count": 0,
                "delta_summary": "No prior findings were recorded for this company in this research session.",
                "changes": [],
            }
        return {
            "status": "delta_available",
            "prior_findings_count": len(findings),
            "changes": [
                {
                    "topic": f.get("topic"),
                    "claim": f.get("claim"),
                    "claim_type": f.get("claim_type"),
                    "recorded_at": f.get("created_at"),
                }
                for f in findings
            ],
            "delta_summary": f"Identified {len(findings)} prior findings recorded in this session for comparison.",
        }


_memory_service_instance: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    global _memory_service_instance
    if _memory_service_instance is None:
        _memory_service_instance = MemoryService()
    return _memory_service_instance
