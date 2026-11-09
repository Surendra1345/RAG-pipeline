import pytest

from app.retrieval.hybrid import reciprocal_rank_fusion, to_ranked
from app.schemas.document import Chunk


def chunk(cid: str) -> Chunk:
    return Chunk(id=cid, doc_name="d.txt", section_title="1. S", parent_id="d:0", text=cid, parent_text=cid, chunk_index=0)


def ranking(ids: list[str], source="dense"):
    return to_ranked([(chunk(i), 1.0 / n) for n, i in enumerate(ids, start=1)], source)


def test_to_ranked_numbers_from_one():
    r = ranking(["a", "b", "c"])
    assert [x.rank for x in r] == [1, 2, 3]
    assert all(x.source == "dense" for x in r)


def test_rrf_formula_for_a_single_list():
    fused = reciprocal_rank_fusion([ranking(["a", "b"])], c=60)
    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[1].score == pytest.approx(1 / 62)


def test_chunk_in_both_lists_beats_chunk_first_in_only_one():
    dense = ranking(["x", "shared", "y"])
    keyword = ranking(["z", "shared", "w"], "bm25")
    fused = reciprocal_rank_fusion([dense, keyword])
    assert fused[0].chunk.id == "shared"
    assert fused[0].score == pytest.approx(2 / 62)


def test_fused_results_are_marked_and_renumbered():
    fused = reciprocal_rank_fusion([ranking(["a", "b"]), ranking(["b", "c"], "bm25")])
    assert [r.rank for r in fused] == [1, 2, 3]
    assert all(r.source == "fused" for r in fused)
    assert {r.chunk.id for r in fused} == {"a", "b", "c"}


def test_top_n_limits_the_output():
    fused = reciprocal_rank_fusion([ranking(["a", "b", "c", "d"])], top_n=2)
    assert [r.chunk.id for r in fused] == ["a", "b"]


def test_ties_are_broken_deterministically():
    # a and b each appear once at rank 1 in different lists: equal score, equal best rank -> by id
    fused = reciprocal_rank_fusion([ranking(["b"]), ranking(["a"], "bm25")])
    assert [r.chunk.id for r in fused] == ["a", "b"]


def test_one_empty_list_falls_back_to_the_other():
    fused = reciprocal_rank_fusion([ranking(["a", "b"]), []])
    assert [r.chunk.id for r in fused] == ["a", "b"]


def test_no_lists_gives_empty_result():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
