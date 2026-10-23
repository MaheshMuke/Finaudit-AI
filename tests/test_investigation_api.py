from fastapi.testclient import TestClient

from finaudit import api
from finaudit.analysis_engine import AnalysisEngine


def test_company_investigation_returns_deterministic_and_llm_sections(monkeypatch):
    def fake_investigate(query, question=None):
        return {
            "success": True,
            "question": question or query,
            "investigation_status": "complete",
            "profile": {"symbol": "AXISBANK"},
            "metrics": {"period": "2026-03-31", "revenue": "₹1 Cr"},
            "news": [],
            "commodities": [],
            "validation_notes": [],
            "llm": {"status": "available"},
            "interpretation": {
                "summary": "Evidence-backed summary.",
                "findings": ["Reported metrics were available."],
                "risks": ["Peer period coverage is limited."],
                "limitations": ["No current news evidence."],
            },
        }

    monkeypatch.setattr(api.agent_orchestrator, "investigate_company", fake_investigate)
    with TestClient(api.app) as client:
        response = client.post(
            "/investigations/company",
            json={"company": "AXISBANK", "question": "Explain its strengths and risks."},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["llm"]["status"] == "available"
    assert payload["interpretation"]["summary"] == "Evidence-backed summary."
    assert payload["question"] == "Explain its strengths and risks."


def test_analysis_engine_uses_configured_gemini_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    engine = AnalysisEngine()
    assert engine.gemini_provider is not None
    assert engine.gemini_provider.model_name == "gemini-2.5-flash-lite"
