"""Cross-encoder reranking.

The retriever compares a question vector with chunk vectors that were made separately,
which is fast but coarse. A cross-encoder reads the question and one chunk TOGETHER and
scores how well the chunk answers it, which is slower but much more precise. We only run
it on the few candidates the retrievers returned.
"""
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.schemas.search import RankedChunk


class Reranker:
    def __init__(self, model_name: str = settings.reranker_model) -> None:
        self.model_name = model_name
        self._model = CrossEncoder(model_name)  # downloads weights on first use

    def rerank(self, query: str, candidates: list[RankedChunk], top_k: int | None = None) -> list[RankedChunk]:
        """Rescore candidates and return them best first as 'reranked' RankedChunks.

        Each result keeps its position in the incoming list as `previous_rank`, so the UI can
        show how far the chunk moved. `score` is the raw cross-encoder score (higher is better;
        it is not a probability and can be negative).
        """
        if not candidates:
            return []
        pairs = [(query, c.chunk.embed_text()) for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i].rank))
        if top_k is not None:
            order = order[:top_k]
        return [
            RankedChunk(
                chunk=candidates[i].chunk,
                score=float(scores[i]),
                rank=new_rank,
                source="reranked",
                previous_rank=candidates[i].rank,
            )
            for new_rank, i in enumerate(order, start=1)
        ]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """Load the model once and reuse it for the life of the process."""
    return Reranker()
