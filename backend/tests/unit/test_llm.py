import pytest

from app.llm.base import LLMError, LLMResult
from app.llm.fallback import FallbackLLM
from app.llm.groq_provider import GroqProvider
from app.llm.prompts import build_context
from app.retrieval.hybrid import to_ranked
from app.retrieval.query_rewriter import QueryRewriter
from app.schemas.document import Chunk
from app.services.answer_service import AnswerService, extract_citations


class Fake:
    def __init__(self, name, reply=None, error=None, key=True):
        self.name, self.reply, self.error, self.key, self.calls = name, reply, error, key, 0

    def available(self):
        return self.key

    def generate(self, system, user, temperature=0.0, max_tokens=700):
        self.calls += 1
        if self.error:
            raise LLMError(self.error)
        return LLMResult(text=self.reply, provider=self.name, model="m", elapsed_ms=1.0)


def chunk(cid, text="some text"):
    return Chunk(id=cid, doc_name="d.txt", section_title="1. S", parent_id="d:0", text=text, parent_text=text, chunk_index=0)


def passages(n=3):
    return to_ranked([(chunk(f"c{i}", f"text {i}"), 0.0) for i in range(1, n + 1)], "reranked")


class FakeSearch:
    def __init__(self):
        self.queries = []

    def search(self, query, k=5, candidates=20, rerank=True):
        from app.schemas.search import SearchResponse, StageResult

        self.queries.append(query)
        return SearchResponse(query=query, k=k, stages=[StageResult(stage="reranked", results=passages(), elapsed_ms=1.0)])


def test_first_working_provider_is_used_and_second_is_skipped():
    a, b = Fake("a", "from a"), Fake("b", "from b")
    assert FallbackLLM([a, b]).generate("s", "u").text == "from a"
    assert b.calls == 0


def test_falls_back_when_first_fails():
    a, b = Fake("a", error="boom"), Fake("b", "from b")
    assert FallbackLLM([a, b]).generate("s", "u").provider == "b"


def test_provider_without_key_is_skipped():
    a, b = Fake("a", "x", key=False), Fake("b", "from b")
    assert FallbackLLM([a, b]).generate("s", "u").text == "from b"
    assert a.calls == 0


def test_all_failing_raises_with_every_reason():
    with pytest.raises(LLMError, match="boom.*no API key"):
        FallbackLLM([Fake("a", error="boom"), Fake("b", key=False)]).generate("s", "u")


def test_groq_without_key_raises_cleanly():
    p = GroqProvider(api_key="")
    assert not p.available()
    with pytest.raises(LLMError):
        p.generate("s", "u")


def test_context_numbers_passages_from_one():
    ctx = build_context(passages(2))
    assert ctx.startswith("[1] (d.txt > 1. S)\ntext 1") and "[2] " in ctx


def test_rewriter_uses_first_line_and_strips_quotes():
    assert QueryRewriter(FallbackLLM([Fake("a", '"annual leave days"\nextra')])).rewrite("how many leaves?") == "annual leave days"


def test_rewriter_falls_back_to_original():
    assert QueryRewriter(FallbackLLM([Fake("a", error="x")])).rewrite("hello there") == "hello there"
    assert QueryRewriter(FallbackLLM([Fake("a", key=False)])).rewrite("hello there") == "hello there"
    assert QueryRewriter(FallbackLLM([Fake("a", "word " * 200)])).rewrite("hi") == "hi"


def test_extract_citations_ignores_unknown_and_repeats():
    assert extract_citations("A [2][1]. B [2] [9] [0]", 3) == [2, 1]


def test_answer_has_only_cited_passages_and_searches_the_rewrite():
    llm = FallbackLLM([Fake("a", "24 days [2].")])
    search = FakeSearch()
    service = AnswerService(search=search, llm=llm)
    service.rewriter = QueryRewriter(FallbackLLM([Fake("r", "annual leave")]))
    result = service.answer("how many leaves?")
    assert search.queries == ["annual leave"]
    assert [c.chunk_id for c in result.citations] == ["c2"]
    assert result.provider == "a"


def test_not_found_has_no_citations():
    service = AnswerService(search=FakeSearch(), llm=FallbackLLM([Fake("a", "I could not find this in the documents.")]))
    assert service.answer("q").citations == []


def test_llm_down_still_returns_the_passages():
    service = AnswerService(search=FakeSearch(), llm=FallbackLLM([Fake("a", error="down")]))
    result = service.answer("q")
    assert result.provider is None and "unavailable" in result.answer
    assert len(result.search.final) == 3


def test_extract_citations_accepts_fullwidth_brackets():
    assert extract_citations("Uses resources【2】 and more [1].", 3) == [2, 1]
