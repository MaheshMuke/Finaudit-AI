import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))
OLLAMA_GENERATION_LIMIT = int(os.getenv("OLLAMA_NUM_PREDICT", "768"))
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_COMPATIBILITY_FALLBACK_MODEL = "gemini-2.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _parse_json_object(raw_text: str) -> Dict[str, Any]:
    """Parse one JSON object, allowing a code fence or short trailing prose."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    if not cleaned:
        raise ValueError("Ollama returned an empty response.")
    try:
        parsed, _ = json.JSONDecoder(strict=False).raw_decode(cleaned)
    except json.JSONDecodeError:
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


class LLMGenerationError(RuntimeError):
    """A classified local-model failure that can be shown without claiming success."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")


def get_ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()


class LLMGateway:
    """
    Unified LLM Gateway providing Ollama as primary local LLM,
    with configurable, explicit fallback (e.g., OpenAI or custom provider)
    when enabled and configured.
    """

    def __init__(self):
        self.primary_provider = os.getenv("LLM_PRIMARY", "ollama").lower()
        self.ollama_base_url = get_ollama_base_url()
        self.ollama_model = get_ollama_model()
        self.fallback_enabled = os.getenv("LLM_FALLBACK_ENABLED", "false").lower() == "true"
        self.fallback_provider = os.getenv("LLM_FALLBACK_PROVIDER", "openai").lower()
        self.fallback_model = os.getenv("LLM_FALLBACK_MODEL", "gpt-4o-mini")
        self.timeout = max(120, int(os.getenv("OLLAMA_TIMEOUT", "180")))
        self.gemini_model = (
            os.getenv("GEMINI_MODEL", os.getenv("LLM_GEMINI_MODEL", DEFAULT_GEMINI_MODEL))
            .strip()
            .lower()
            .replace(" ", "-")
        )
        self.gemini_timeout = max(30, int(os.getenv("GEMINI_TIMEOUT", "90")))
        self.last_call_metadata: Dict[str, Any] = {
            "provider": None,
            "model": None,
            "generation_status": "not_started",
            "fallback_used": False,
        }

    def _begin_call(self, provider: str, model: str) -> str:
        correlation_id = str(uuid.uuid4())
        self.last_call_metadata = {
            "request_id": correlation_id,
            "provider": provider,
            "model": model,
            "generation_status": "in_progress",
            "fallback_used": False,
            "llm_invoked": True,
            "llm_provider": provider,
            "llm_model": model,
            "llm_stage": "interpretation",
            "llm_request_status": "in_progress",
            "llm_fallback_reason": None,
            "provider_configured": True,
            "provider_available": True,
            "request_succeeded": False,
            "response_used_llm": False,
        }
        self.last_call_started = time.perf_counter()
        logger.info(
            "LLM generation started request_id=%s provider=%s model=%s",
            correlation_id,
            provider,
            model,
        )
        return correlation_id

    def _finish_call(
        self,
        status: str,
        error_code: Optional[str] = None,
        fallback_used: Optional[bool] = None,
    ) -> None:
        metadata = dict(self.last_call_metadata)
        metadata["generation_status"] = status
        metadata["latency_ms"] = round(
            (time.perf_counter() - getattr(self, "last_call_started", time.perf_counter()))
            * 1000
        )
        if error_code:
            metadata["error_code"] = error_code
        if fallback_used is not None:
            metadata["fallback_used"] = fallback_used
        if status == "success":
            metadata["llm_request_status"] = "succeeded"
            metadata["request_succeeded"] = True
            metadata["response_used_llm"] = True
            metadata["llm_fallback_reason"] = None
        else:
            metadata["llm_request_status"] = "failed"
            metadata["request_succeeded"] = False
            metadata["response_used_llm"] = False
            metadata["llm_fallback_reason"] = error_code or "LLM generation failed"
        self.last_call_metadata = metadata
        logger.info(
            "LLM generation finished request_id=%s provider=%s model=%s status=%s latency_ms=%s error_code=%s fallback_used=%s",
            metadata.get("request_id"),
            metadata.get("provider"),
            metadata.get("model"),
            status,
            metadata.get("latency_ms"),
            metadata.get("error_code"),
            metadata.get("fallback_used"),
        )

    def check_health(self) -> Dict[str, Any]:
        """Check status of primary and fallback LLMs without exposing secrets."""
        ollama_status = {"available": False, "installed_models": [], "message": ""}
        try:
            resp = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m.get("name") for m in resp.json().get("models", [])]
                has_model = any(
                    m == self.ollama_model or m.startswith(f"{self.ollama_model}:")
                    for m in models
                )
                ollama_status = {
                    "available": True,
                    "model_installed": has_model,
                    "target_model": self.ollama_model,
                    "installed_models": models,
                    "message": f"Ollama online with model '{self.ollama_model}'",
                }
        except Exception as e:
            ollama_status["message"] = f"Ollama offline: {type(e).__name__}"

        gemini_configured = bool(os.getenv("GEMINI_API_KEY", "").strip())
        gemini_status = {
            "configured": gemini_configured,
            "model": self.gemini_model,
        }
        fallback_status = {
            "enabled": self.fallback_enabled,
            "provider": self.fallback_provider if self.fallback_enabled else None,
            "configured": (
                bool(os.getenv("OPENAI_API_KEY"))
                if self.fallback_provider == "openai"
                else gemini_configured
                if self.fallback_provider == "gemini"
                else False
            ),
        }

        active_provider = self.primary_provider if (
            self.primary_provider == "gemini" and gemini_configured
        ) else "ollama" if ollama_status["available"] else (
            self.fallback_provider if (self.fallback_enabled and fallback_status["configured"]) else "none"
        )

        return {
            "primary": self.primary_provider,
            "active_provider": active_provider,
            "ollama": ollama_status,
            "gemini": gemini_status,
            "fallback": fallback_status,
        }

    def generate_json(
        self,
        prompt: str,
        system_instruction: str = "You are a professional financial analyst.",
        temperature: float = 0.2,
        required_keys: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Attempt structured JSON generation using primary Ollama first.
        Falls back to cloud provider only if enabled and Ollama fails.
        """
        if required_keys is None:
            required_keys = {"summary", "findings", "risks", "next_questions", "limitations"}

        if self.primary_provider == "gemini":
            self._begin_call("gemini", self.gemini_model)
            try:
                result = self._generate_gemini_json(
                    prompt, system_instruction, temperature, required_keys=required_keys
                )
                self._finish_call("success")
                return result
            except LLMGenerationError as exc:
                self._finish_call("failed", exc.kind)
                if not self.fallback_enabled:
                    raise
            except Exception as exc:
                logger.warning("Gemini call failed (%s): %s", type(exc).__name__, exc)
                if not self.fallback_enabled:
                    raise LLMGenerationError("provider_error", f"Gemini request failed: {exc}") from exc

        # Ollama primary or explicit fallback
        self._begin_call("ollama", self.ollama_model)
        try:
            payload = {
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "format": "json",
                "think": False,
                "options": {"temperature": temperature, "num_predict": max(OLLAMA_GENERATION_LIMIT, 2048)},
                "keep_alive": "10m",
            }
            try:
                resp = requests.post(
                    f"{self.ollama_base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.exceptions.Timeout as exc:
                raise LLMGenerationError(
                    "timeout",
                    f"Ollama did not complete within {self.timeout} seconds.",
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                raise LLMGenerationError(
                    "connection",
                    f"Could not connect to Ollama at {self.ollama_base_url}.",
                ) from exc
            if resp.status_code == 200:
                response_data = resp.json()
                message = response_data.get("message", {})
                raw_text = message.get("content", "")
                logger.info(
                    "Ollama response status=%s done_reason=%s structure_keys=%s message_keys=%s excerpt=%r",
                    resp.status_code,
                    response_data.get("done_reason"),
                    sorted(response_data.keys()),
                    sorted(message.keys()) if isinstance(message, dict) else [],
                    str(raw_text)[:240],
                )
                if response_data.get("done_reason") == "length":
                    raise LLMGenerationError(
                        "malformed_response",
                        "Ollama truncated the structured response before completing JSON.",
                    )
                try:
                    parsed = _parse_json_object(raw_text)
                except ValueError as exc:
                    raise LLMGenerationError("malformed_response", str(exc)) from exc
                if required_keys:
                    missing = sorted(required_keys.difference(parsed))
                    if missing:
                        raise LLMGenerationError(
                            "malformed_response",
                            f"Ollama JSON is missing required fields: {', '.join(missing)}.",
                        )
                self._finish_call("success")
                return parsed
            if resp.status_code in (404, 400) and "model" in resp.text.lower():
                raise LLMGenerationError(
                    "model_unavailable",
                    f"Configured Ollama model '{self.ollama_model}' is unavailable.",
                )
            logger.warning("Ollama HTTP %s: %s", resp.status_code, resp.text[:200])
        except Exception as exc:
            logger.warning("Ollama call failed (%s): %s", type(exc).__name__, exc)
            self._finish_call(
                "failed",
                exc.kind if isinstance(exc, LLMGenerationError) else type(exc).__name__.lower(),
            )
            if isinstance(exc, LLMGenerationError):
                raise

        # Cloud fallback
        if self.fallback_enabled and self.fallback_provider == "gemini" and os.getenv("GEMINI_API_KEY"):
            self._begin_call("gemini", self.gemini_model)
            result = self._generate_gemini_json(
                prompt, system_instruction, temperature, required_keys=required_keys
            )
            self._finish_call("success", fallback_used=True)
            return result

        if self.fallback_enabled and self.fallback_provider == "openai" and os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI

                client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=self.timeout)
                res = client.chat.completions.create(
                    model=self.fallback_model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = res.choices[0].message.content or "{}"
                return json.loads(text)
            except Exception as e:
                logger.error("Cloud fallback LLM failed (%s): %s", type(e).__name__, e)
                raise RuntimeError(f"Both Ollama and Fallback LLM failed: {e}") from e

        raise LLMGenerationError(
            "unavailable",
            "Local Ollama is unavailable or returned invalid output, and cloud fallback is disabled."
        )

    def _generate_gemini_json(
        self,
        prompt: str,
        system_instruction: str,
        temperature: float,
        required_keys: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise LLMGenerationError("not_configured", "GEMINI_API_KEY is not configured.")
        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
                "maxOutputTokens": max(OLLAMA_GENERATION_LIMIT, 2048),
            },
        }
        models = [self.gemini_model]
        for compat in ("gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-2.0-flash"):
            if compat not in models:
                models.append(compat)
        response = None
        for model in models:
            self.last_call_metadata["model"] = model
            try:
                response = requests.post(
                    f"{GEMINI_BASE_URL}/{model}:generateContent",
                    params={"key": api_key},
                    json=payload,
                    timeout=self.gemini_timeout,
                )
            except requests.exceptions.Timeout as exc:
                raise LLMGenerationError(
                    "timeout", f"Gemini did not complete within {self.gemini_timeout} seconds."
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                raise LLMGenerationError("connection", "Could not connect to Gemini.") from exc
            if response.status_code not in (404, 429, 503):
                break
            self.last_call_metadata["fallback_used"] = True
            logger.warning("Gemini model '%s' returned HTTP %s; trying compatibility model.", model, response.status_code)
        if response is None or response.status_code != 200:
            detail = response.text[:300] if response is not None else "No response."
            kind = "model_unavailable" if response is not None and response.status_code == 404 else "provider_error"
            raise LLMGenerationError(kind, f"Gemini HTTP {response.status_code if response is not None else 'unknown'}: {detail}")
        try:
            data = response.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = _parse_json_object(raw_text)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMGenerationError(
                "malformed_response", "Gemini returned an invalid structured response."
            ) from exc
        if required_keys:
            missing = sorted(required_keys.difference(parsed))
            if missing:
                raise LLMGenerationError(
                    "malformed_response",
                    f"Gemini JSON is missing required fields: {', '.join(missing)}.",
                )
        logger.info("Gemini response model=%s keys=%s", self.last_call_metadata.get("model"), sorted(parsed))
        return parsed


_gateway_instance: Optional[LLMGateway] = None


def get_llm_gateway() -> LLMGateway:
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = LLMGateway()
    return _gateway_instance
