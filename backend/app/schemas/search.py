"""Models for search results: one ranked list per retrieval stage."""
from typing import Literal

from pydantic import BaseModel

from app.schemas.document import Chunk

Stage = Literal["dense", "bm25", "fused", "reranked"]


class RankedChunk(BaseModel):
    """A chunk with its position and score in one stage's ranking."""

    chunk: Chunk
    score: float  # meaning depends on the stage (cosine, BM25, RRF, cross-encoder)
    rank: int  # 1 = best
    source: Stage
    previous_rank: int | None = None  # rank in the stage before (set by the reranker)


class StageResult(BaseModel):
    stage: Stage
    results: list[RankedChunk]
    elapsed_ms: float


class SearchResponse(BaseModel):
    query: str
    k: int
    stages: list[StageResult]

    def stage(self, name: Stage) -> StageResult | None:
        return next((s for s in self.stages if s.stage == name), None)

    @property
    def final(self) -> list[RankedChunk]:
        """The last stage's results: reranked if present, else fused."""
        return self.stages[-1].results if self.stages else []
