"""Common shape for every LLM provider."""
from dataclasses import dataclass
from typing import Protocol


class LLMError(Exception):
    """A provider failed (no key, network error, rate limit, empty reply)."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    provider: str
    model: str
    elapsed_ms: float


class LLMProvider(Protocol):
    name: str

    def available(self) -> bool:
        """True when the provider has what it needs (an API key)."""

    def generate(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 700) -> LLMResult:
        """Return the model's reply, or raise LLMError."""
