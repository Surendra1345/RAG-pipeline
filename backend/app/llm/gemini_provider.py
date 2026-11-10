"""Gemini (fallback LLM)."""
import time

from app.core.config import settings
from app.llm.base import LLMError, LLMResult


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = settings.gemini_api_key if api_key is None else api_key
        self.model = model or settings.gemini_model

    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 700) -> LLMResult:
        if not self.available():
            raise LLMError("GEMINI_API_KEY is not set")
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model, system_instruction=system)
            start = time.perf_counter()
            reply = model.generate_content(
                user,
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
                request_options={"timeout": settings.llm_timeout_s},
            )
            text = (reply.text or "").strip()
        except Exception as exc:  # network, auth, blocked reply, quota
            raise LLMError(f"gemini failed: {type(exc).__name__}: {exc}") from exc
        if not text:
            raise LLMError("gemini returned an empty reply")
        return LLMResult(text=text, provider=self.name, model=self.model, elapsed_ms=(time.perf_counter() - start) * 1000)
