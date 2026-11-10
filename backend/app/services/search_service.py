"""Run one question through every retrieval stage and return each stage's ranked list.

    dense (vector search) + BM25 (keyword search)  ->  fused (RRF)  ->  reranked (cross-encoder)

Query rewriting is added in the LLM step. Try it from the backend folder:

    python -m app.services.search_service "How many days of annual leave do employees get?"
"""
import logging
import sys
import time

from app.core.constants import DEFAULT_CANDIDATES, DEFAULT_TOP_K
from app.core.logging import setup_logging
from app.retrieval.bm25_index import BM25Index
from app.retrieval.embedder import Embedder, get_embedder
from app.retrieval.hybrid import reciprocal_rank_fusion, to_ranked
from app.retrieval.reranker import Reranker, get_reranker
from app.retrieval.vector_store import PgVectorStore
from app.schemas.search import RankedChunk, SearchResponse, Stage, StageResult


log = logging.getLogger(__name__)


class SearchService:
    def __init__(
        self,
        store: PgVectorStore | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.store = store or PgVectorStore()
        self.embedder = embedder or get_embedder()
        self._reranker = reranker
        self.bm25 = BM25Index(self.store.all_chunks())

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = get_reranker()
        return self._reranker

    def search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        candidates: int = DEFAULT_CANDIDATES,
        rerank: bool = True,
    ) -> SearchResponse:
        """Return the ranked list of every stage. The last stage is the final answer set."""
        log.info("Search: %r (k=%d, candidates=%d, rerank=%s)", query, k, candidates, rerank)
        stages: list[StageResult] = []

        def timed(stage: Stage, fn) -> list[RankedChunk]:
            start = time.perf_counter()
            results = fn()
            ms = (time.perf_counter() - start) * 1000
            stages.append(StageResult(stage=stage, results=results, elapsed_ms=ms))
            top = results[0].chunk.id if results else "-"
            log.info("Search stage %-8s %3d results in %6.0f ms, top: %s", stage, len(results), ms, top)
            return results

        dense = timed("dense", lambda: to_ranked(self.store.search(self.embedder.embed_query(query), candidates), "dense"))
        keyword = timed("bm25", lambda: to_ranked(self.bm25.search(query, candidates), "bm25"))
        fused = timed("fused", lambda: reciprocal_rank_fusion([dense, keyword], top_n=candidates))
        if rerank:
            timed("reranked", lambda: self.reranker.rerank(query, fused, top_k=k))
        return SearchResponse(query=query, k=k, stages=stages)


def _print(response: SearchResponse, show: int = 5) -> None:
    print(f"\nQuestion: {response.query}")
    for stage in response.stages:
        print(f"\n== {stage.stage.upper()}  ({stage.elapsed_ms:.0f} ms, top {show} of {len(stage.results)})")
        for r in stage.results[:show]:
            moved = ""
            if r.previous_rank is not None:
                moved = f"  (was #{r.previous_rank})"
            print(f"  #{r.rank:<2} {r.score:8.3f}  {r.chunk.id:22s} {r.chunk.section_title[:30]:30s}{moved}")


if __name__ == "__main__":
    setup_logging()
    questions = sys.argv[1:] or ["How many days of annual leave do employees get?"]
    service = SearchService()
    for q in questions:
        _print(service.search(q))
