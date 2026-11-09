"""Hybrid retrieval: merge the dense and BM25 rankings with Reciprocal Rank Fusion (RRF).

    RRF(d) = sum over rankings of  1 / (c + rank(d))

A chunk ranked well by both lists beats one ranked first by only one. Only ranks are
used, so the two score scales (cosine vs BM25) never need to be compared.
"""
from app.core.constants import RRF_C
from app.schemas.document import Chunk
from app.schemas.search import RankedChunk, Stage


def to_ranked(pairs: list[tuple[Chunk, float]], source: Stage) -> list[RankedChunk]:
    """Turn (chunk, score) pairs, best first, into RankedChunks with 1-based ranks."""
    return [
        RankedChunk(chunk=chunk, score=score, rank=i, source=source)
        for i, (chunk, score) in enumerate(pairs, start=1)
    ]


def reciprocal_rank_fusion(
    rankings: list[list[RankedChunk]], c: int = RRF_C, top_n: int | None = None
) -> list[RankedChunk]:
    """Fuse several ranked lists into one. Returns RankedChunks with source 'fused'.

    Ties are broken by the best rank any list gave the chunk, then by chunk id, so the
    result is deterministic.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for item in ranking:
            cid = item.chunk.id
            chunks[cid] = item.chunk
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (c + item.rank)
            best_rank[cid] = min(best_rank.get(cid, item.rank), item.rank)

    order = sorted(scores, key=lambda cid: (-scores[cid], best_rank[cid], cid))
    if top_n is not None:
        order = order[:top_n]
    return [
        RankedChunk(chunk=chunks[cid], score=scores[cid], rank=i, source="fused")
        for i, cid in enumerate(order, start=1)
    ]
