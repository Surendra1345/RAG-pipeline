"""Turn the user's question into a cleaner search query. Falls back to the original on any problem."""
import logging

from app.llm.base import LLMError
from app.llm.fallback import FallbackLLM, get_llm
from app.llm.prompts import REWRITE_SYSTEM


log = logging.getLogger(__name__)


class QueryRewriter:
    def __init__(self, llm: FallbackLLM | None = None) -> None:
        self.llm = llm or get_llm()

    def rewrite(self, question: str) -> str:
        question = question.strip()
        if not question or not self.llm.available():
            log.info("Query rewrite skipped (no question or no LLM key)")
            return question
        try:
            text = self.llm.generate(REWRITE_SYSTEM, question, temperature=0.0, max_tokens=60).text
        except LLMError:
            log.warning("Query rewrite failed, using the original question")
            return question
        rewritten = text.splitlines()[0].strip().strip('"').strip()
        # a rewrite that is empty or absurdly long is not a search query
        if not 0 < len(rewritten) <= 4 * max(len(question), 40):
            log.warning("Query rewrite rejected (empty or too long), using the original question")
            return question
        log.info("Query rewritten: %r -> %r", question, rewritten)
        return rewritten
