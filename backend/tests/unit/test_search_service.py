"""End-to-end retrieval tests. They need DATABASE_URL and the chunks table to be indexed."""
import pytest

from app.core.config import settings
from app.services.search_service import SearchService

pytestmark = pytest.mark.skipif(not settings.database_url, reason="DATABASE_URL is not set")


@pytest.fixture(scope="module")
def service():
    return SearchService()


def test_all_four_stages_come_back_in_order(service):
    response = service.search("How many days of annual leave do employees get?", k=5)
    assert [s.stage for s in response.stages] == ["dense", "bm25", "fused", "reranked"]
    assert all(s.elapsed_ms >= 0 for s in response.stages)
    assert len(response.final) == 5


def test_leave_question_finds_the_leave_section(service):
    response = service.search("How many days of annual leave do employees get?")
    assert response.final[0].chunk.parent_id == "hr_handbook:3"


def test_reranked_results_point_back_to_their_fused_rank(service):
    response = service.search("What is offside in football?")
    fused_rank = {r.chunk.id: r.rank for r in response.stage("fused").results}
    for r in response.stage("reranked").results:
        assert r.previous_rank == fused_rank[r.chunk.id]


def test_hybrid_rescues_an_acronym_that_dense_search_gets_wrong(service):
    response = service.search("What is the DLS method?")
    dense_first = response.stage("dense").results[0].chunk.id
    fused_first = response.stage("fused").results[0].chunk.id
    assert dense_first != "cricket_guide:10:0"  # dense confuses DLS with DRS
    assert fused_first == "cricket_guide:10:0"  # BM25 finds the real DLS chunk


def test_rerank_can_be_switched_off(service):
    response = service.search("What is offside in football?", rerank=False)
    assert [s.stage for s in response.stages] == ["dense", "bm25", "fused"]
    assert response.stage("reranked") is None
    assert response.final == response.stage("fused").results


def test_candidates_limit_is_respected(service):
    response = service.search("overs", candidates=7)
    assert len(response.stage("dense").results) == 7
    assert len(response.stage("fused").results) <= 7
