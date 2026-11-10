"""Question -> rewrite -> retrieve -> LLM answer with citations.

Try it from the backend folder (needs GROQ_API_KEY or GEMINI_API_KEY in backend/.env):

    python -m app.services.answer_service "How many days of annual leave do employees get?"
"""
import logging
import re
import sys

from app.core.constants import DEFAULT_CANDIDATES, DEFAULT_TOP_K
from app.core.logging import setup_logging
from app.llm.base import LLMError
from app.llm.fallback import FallbackLLM, get_llm
from app.llm.prompts import ANSWER_SYSTEM, NOT_FOUND, build_answer_prompt
from app.retrieval.query_rewriter import QueryRewriter
from app.schemas.answer import Answer, Citation
from app.services.search_service import SearchService

log = logging.getLogger(__name__)

_CITE = re.compile(r"[\[【](\d+)[\]】]")


def extract_citations(text: str, n_passages: int) -> list[int]:
    """Passage numbers cited in the text, in order of first use, ignoring numbers that do not exist."""
    seen: list[int] = []
    for m in _CITE.finditer(text):
        n = int(m.group(1))
        if 1 <= n <= n_passages and n not in seen:
            seen.append(n)
    return seen


class AnswerService:
    def __init__(self, search: SearchService | None = None, llm: FallbackLLM | None = None) -> None:
        self.search = search or SearchService()
        self.llm = llm or get_llm()
        self.rewriter = QueryRewriter(self.llm)

    def answer(self, question: str, k: int = DEFAULT_TOP_K, candidates: int = DEFAULT_CANDIDATES) -> Answer:
        log.info("Question received: %r", question)
        rewritten = self.rewriter.rewrite(question)
        response = self.search.search(rewritten, k=k, candidates=candidates)
        passages = response.final

        provider = model = None
        llm_ms = 0.0
        try:
            result = self.llm.generate(ANSWER_SYSTEM, build_answer_prompt(question, passages), max_tokens=500)
            text, provider, model, llm_ms = result.text, result.provider, result.model, result.elapsed_ms
        except LLMError as exc:
            text = f"The language model is unavailable ({exc}). The retrieved passages are shown below."

        citations = [
            Citation(
                number=n,
                chunk_id=passages[n - 1].chunk.id,
                doc_name=passages[n - 1].chunk.doc_name,
                section_title=passages[n - 1].chunk.section_title,
                text=passages[n - 1].chunk.text,
            )
            for n in extract_citations(text, len(passages))
        ]
        if text.strip() == NOT_FOUND:
            citations = []
        log.info("Answer ready: provider=%s, %d cited passage(s): %s", provider, len(citations), [c.chunk_id for c in citations])
        return Answer(
            question=question, rewritten_query=rewritten, answer=text, citations=citations,
            provider=provider, model=model, llm_ms=llm_ms, search=response,
        )


if __name__ == "__main__":
    setup_logging()
    service = AnswerService()
    for q in sys.argv[1:] or ["How many days of annual leave do employees get?"]:
        a = service.answer(q)
        print(f"\nQuestion : {a.question}\nSearched : {a.rewritten_query}\nVia      : {a.provider} / {a.model} ({a.llm_ms:.0f} ms)\n\n{a.answer}\n")
        for c in a.citations:
            print(f"  [{c.number}] {c.chunk_id}  {c.doc_name} > {c.section_title}")
