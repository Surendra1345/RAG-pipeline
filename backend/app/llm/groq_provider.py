"""Groq (primary LLM)."""
import time

from app.core.config import settings
from app.llm.base import LLMError, LLMResult


class GroqProvider:
    name = "groq"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = settings.groq_api_key if api_key is None else api_key
        self.model = model or settings.groq_model
        self._client = None

    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 700) -> LLMResult:
        if not self.available():
            raise LLMError("GROQ_API_KEY is not set")
        try:
            if self._client is None:
                from groq import Groq

                self._client = Groq(api_key=self.api_key, timeout=settings.llm_timeout_s, max_retries=0)
            start = time.perf_counter()
            extra = {}
            if "gpt-oss" in self.model:
                # Reasoning model: think briefly, and leave room for the visible reply.
                extra["reasoning_effort"] = "low"
                max_tokens += 300
            reply = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )
            text = (reply.choices[0].message.content or "").strip()
        except Exception as exc:  # network, auth, rate limit
            raise LLMError(f"groq failed: {type(exc).__name__}: {exc}") from exc
        if not text:
            raise LLMError("groq returned an empty reply")
        return LLMResult(text=text, provider=self.name, model=self.model, elapsed_ms=(time.perf_counter() - start) * 1000)
