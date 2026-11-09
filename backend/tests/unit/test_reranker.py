import pytest

from app.retrieval.hybrid import to_ranked
from app.retrieval.reranker import get_reranker
from app.schemas.document import Chunk


def chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, doc_name="d.txt", section_title="1. S", parent_id="d:0", text=text, parent_text=text, chunk_index=0)


@pytest.fixture(scope="module")
def reranker():
    return get_reranker()


def candidates(*pairs: tuple[str, str]):
    return to_ranked([(chunk(cid, text), 0.0) for cid, text in pairs], "fused")


def test_relevant_chunk_moves_above_irrelevant_one(reranker):
    cands = candidates(
        ("cricket", "The ball is red in Test cricket and white in limited-overs matches."),
        ("leave", "Employees receive 24 days of paid annual leave every calendar year."),
    )
    result = reranker.rerank("How many days of annual leave do employees get?", cands)
    assert result[0].chunk.id == "leave"
    assert result[0].score > result[1].score


def test_previous_rank_and_source_are_recorded(reranker):
    cands = candidates(("a", "The pitch is 22 yards long."), ("b", "Annual leave is 24 days."))
    result = reranker.rerank("How long is annual leave?", cands)
    assert all(r.source == "reranked" for r in result)
    assert {r.chunk.id: r.previous_rank for r in result} == {"a": 1, "b": 2}
    assert [r.rank for r in result] == [1, 2]


def test_top_k_limits_the_output(reranker):
    cands = candidates(("a", "x one"), ("b", "y two"), ("c", "z three"))
    assert len(reranker.rerank("anything", cands, top_k=2)) == 2


def test_empty_candidates(reranker):
    assert reranker.rerank("anything", []) == []
