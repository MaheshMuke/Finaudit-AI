import json
import logging
import os
import time
from typing import Any, Dict, Optional
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
OLLAMA_GENERATION_LIMIT = int(os.getenv("OLLAMA_NUM_PREDICT", "768"))


def get_ollama_base_url() -> str:
    url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip()
    return url.rstrip("/")


def get_ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()


def parse_ollama_json(raw_text: str) -> Dict[str, Any]:
    """Accept JSON objects wrapped in a code fence or brief surrounding prose."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("Ollama returned no JSON object.")
    try:
        parsed, _ = json.JSONDecoder(strict=False).raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        raise ValueError("Ollama returned malformed JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Ollama returned JSON with an unexpected shape.")
    return parsed


def check_ollama_health(timeout: int = 5) -> Dict[str, Any]:
    """
    Check if Ollama server is running and which models are installed.
    """
    base_url = get_ollama_base_url()
    target_model = get_ollama_model()
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name") for m in data.get("models", [])]
            # Match model name with or without tag
            has_target = any(
                m == target_model
                or m.startswith(f"{target_model}:")
                or target_model.startswith(m)
                for m in models
            )
            return {
                "available": True,
                "base_url": base_url,
                "target_model": target_model,
                "model_installed": has_target,
                "installed_models": models,
                "message": (
                    f"Ollama running with model '{target_model}'."
                    if has_target
                    else f"Ollama running, but configured model '{target_model}' is not installed. Run: ollama run {target_model}"
                ),
            }
        return {
            "available": False,
            "base_url": base_url,
            "target_model": target_model,
            "model_installed": False,
            "installed_models": [],
            "message": f"Ollama returned HTTP {resp.status_code}. Ensure Ollama is running.",
        }
    except Exception as e:
        return {
            "available": False,
            "base_url": base_url,
            "target_model": target_model,
            "model_installed": False,
            "installed_models": [],
            "message": f"Could not reach Ollama at {base_url} ({type(e).__name__}). Start it with: ollama serve",
        }


class OllamaProvider:
    """
    Client for interacting with local Ollama instance for structured financial intelligence.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.base_url = (base_url or get_ollama_base_url()).rstrip("/")
        self.model = model or get_ollama_model()
        env_timeout = os.getenv("OLLAMA_TIMEOUT")
        configured_timeout = int(env_timeout) if env_timeout and env_timeout.isdigit() else DEFAULT_TIMEOUT
        self.timeout = timeout or max(30, min(configured_timeout, 180))

    def analyze_company(self, compact_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate structured narrative analysis for a company using Ollama.
        Returns a dict containing:
        - business_model: { summary, revenue_generation, segments, products_services }
        - business_dependencies: { customers, suppliers_raw_materials, macro_commodities_fx, regulation_geopolitics, tech_capex }
        - share_price_drivers: { positive_catalysts, negative_risks, company_specific, macro_sector, indicators_to_monitor }
        - competitive_position: { strengths, weaknesses, peer_positioning }
        - overall_assessment: { health_summary, verdict, confidence_note }
        """
        prompt = self._build_prompt(compact_context)
        system_msg = (
            "You are a rigorous financial intelligence assistant. "
            "Analyze the provided structured company financial summary, historical growth metrics, "
            "and peer comparison data. "
            "Rules:\n"
            "1. Ground all commentary strictly in the provided data. Do not invent named customers, contracts, or unstated numbers.\n"
            "2. Distinguish confirmed data facts from analytical inferences.\n"
            "3. If information is missing or not provided, explicitly state 'Data not available in reported dataset'.\n"
            "4. Return ONLY a valid JSON object matching the requested schema, with concise "
            "single-sentence fields and short arrays. Do not use markdown fences or reasoning."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": 0.2,
                "num_predict": OLLAMA_GENERATION_LIMIT,
                "keep_alive": "10m",
            },
        }

        started = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=(5, self.timeout),
            )
        except requests.exceptions.Timeout as exc:
            elapsed = time.perf_counter() - started
            logger.warning("Ollama timed out after %.1fs", elapsed)
            raise RuntimeError(
                f"Ollama timed out after {elapsed:.1f}s (limit {self.timeout}s). "
                "Try a shorter question or confirm the local model is responsive."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url}. Start Ollama and retry."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        elapsed = time.perf_counter() - started
        logger.info("Ollama %s completed in %.1fs", self.model, elapsed)
        try:
            if resp.status_code != 200:
                if resp.status_code == 404 and "model" in resp.text.lower():
                    raise RuntimeError(
                        f"Configured Ollama model '{self.model}' is unavailable. "
                        f"Install it with: ollama pull {self.model}"
                    )
                raise RuntimeError(
                    f"Ollama request failed with HTTP {resp.status_code}: {resp.text[:300]}"
                )
            data = resp.json()
            raw_text = data.get("message", {}).get("content", "").strip()
            if not raw_text:
                raise ValueError("Ollama returned an empty response.")
            parsed = parse_ollama_json(raw_text)
            return self._validate_and_sanitize(parsed)
        except (ValueError, RuntimeError):
            raise
        except Exception as exc:
            logger.warning("Ollama analysis failed (%s): %s", type(exc).__name__, exc)
            raise

    def _build_prompt(self, ctx: Dict[str, Any]) -> str:
        company_info = ctx.get("company", {})
        metrics = ctx.get("metrics", {})
        competitors = ctx.get("competitors", [])[:4]
        raw_trends = ctx.get("trends", {})
        trends = {
            "periods": raw_trends.get("periods", [])[-5:],
            "revenue": raw_trends.get("revenue", [])[-5:],
            "net_income": raw_trends.get("net_income", [])[-5:],
        }
        price_perf = ctx.get("stock_performance", {})

        return f"""
Analyze this company based on its verified financial data:
Company: {company_info.get('name')} (Symbol: {company_info.get('symbol')}, Sector: {company_info.get('sector')}, Industry: {company_info.get('industry')})
Summary Description: {str(company_info.get('business_summary', 'N/A'))[:800]}

Key Financial Metrics (Latest Annual / TTM):
- Revenue: {metrics.get('revenue_display', 'N/A')} (YoY Growth: {metrics.get('revenue_growth_display', 'N/A')})
- Net Income: {metrics.get('net_income_display', 'N/A')} (YoY Growth: {metrics.get('net_income_growth_display', 'N/A')})
- Operating Margin: {metrics.get('operating_margin_display', 'N/A')}
- Net Margin: {metrics.get('net_margin_display', 'N/A')}
- Debt to Equity: {metrics.get('debt_to_equity_display', 'N/A')}
- Return on Equity (ROE): {metrics.get('roe_display', 'N/A')}
- Operating Cash Flow to Net Income: {metrics.get('ocf_to_ni_display', 'N/A')}
- Free Cash Flow: {metrics.get('fcf_display', 'N/A')}

Stock Performance:
- 1-Year Return: {price_perf.get('one_year_return_display', 'N/A')}
- 52-Week Range: {price_perf.get('range_52w_display', 'N/A')}
- P/E Ratio: {price_perf.get('pe_ratio_display', 'N/A')}
- Market Cap: {price_perf.get('market_cap_display', 'N/A')}

Competitor Context ({len(competitors)} peers analyzed):
{json.dumps(competitors, indent=2)}

Multi-Year Trends:
{json.dumps(trends, indent=2)}

Data Limitations / Missing: {ctx.get('data_limitations', [])}

Produce a JSON object with this EXACT structure:
{{
  "business_model": {{
    "summary": "...",
    "products_and_services": ["..."],
    "revenue_generation": "...",
    "key_segments": ["..."]
  }},
  "business_dependencies": {{
    "customers_and_concentration": "...",
    "suppliers_and_input_costs": "...",
    "macro_commodities_fx_rates": "...",
    "regulation_and_government_policy": "...",
    "technology_and_capex": "..."
  }},
  "share_price_drivers": {{
    "positive_catalysts": ["..."],
    "negative_risks": ["..."],
    "company_specific_factors": ["..."],
    "macro_and_sector_factors": ["..."],
    "indicators_to_monitor": ["..."]
  }},
  "competitive_position": {{
    "strengths_vs_peers": ["..."],
    "weaknesses_vs_peers": ["..."],
    "relative_ranking_summary": "..."
  }},
  "overall_assessment": {{
    "financial_health_summary": "...",
    "profitability_and_growth_verdict": "...",
    "key_uncertainties": "..."
  }}
}}
"""

    def _validate_and_sanitize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all required sections exist and have standard fallback keys."""
        sections = [
            "business_model",
            "business_dependencies",
            "share_price_drivers",
            "competitive_position",
            "overall_assessment",
        ]
        sanitized = {}
        for sec in sections:
            sanitized[sec] = data.get(sec, {})

        return sanitized
