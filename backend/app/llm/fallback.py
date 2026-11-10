"""Try each provider in order; the first one that works wins."""
import logging
from functools import lru_cache

from app.llm.base import LLMError, LLMProvider, LLMResult
from app.llm.gemini_provider import GeminiProvider
from app.llm.groq_provider import GroqProvider


log = logging.getLogger(__name__)


class FallbackLLM:
    def __init__(self, providers: list[LLMProvider] | None = None) -> None:
        self.providers = providers if providers is not None else [GroqProvider(), GeminiProvider()]

    def available(self) -> bool:
        return any(p.available() for p in self.providers)

    def generate(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 700) -> LLMResult:
        errors: list[str] = []
        for provider in self.providers:
            if not provider.available():
                errors.append(f"{provider.name}: no API key")
                log.warning("LLM provider %s skipped: no API key", provider.name)
                continue
            try:
                result = provider.generate(system, user, temperature, max_tokens)
                log.info("LLM %s/%s answered in %.0f ms (%d chars)", result.provider, result.model, result.elapsed_ms, len(result.text))
                return result
            except LLMError as exc:
                errors.append(str(exc))
                log.warning("LLM provider %s failed, trying the next one: %s", provider.name, exc)
        log.error("All LLM providers failed: %s", "; ".join(errors))
        raise LLMError("; ".join(errors) or "no LLM providers configured")


@lru_cache(maxsize=1)
def get_llm() -> FallbackLLM:
    return FallbackLLM()
