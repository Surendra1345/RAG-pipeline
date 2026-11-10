"""BM25 keyword search over the chunks.

Dense search matches meaning; BM25 matches the actual words, so it catches exact terms
such as "DLS", "doosra" or "G6" that an embedding can blur. It is built in memory from
the chunks (fast for hundreds of chunks).
"""
import logging

import numpy as np
from rank_bm25 import BM25Okapi

from app.schemas.document import Chunk
from app.utils.text_utils import tokenize


log = logging.getLogger(__name__)


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        # Index the same text the embedder sees: document and section name + the chunk.
        corpus = [tokenize(c.embed_text()) for c in self.chunks]
        self._bm25 = BM25Okapi(corpus) if self.chunks else None
        log.info("BM25 index built over %d chunks", len(self.chunks))

    def search(self, query: str, k: int = 20) -> list[tuple[Chunk, float]]:
        """Return up to k chunks as (chunk, BM25 score), best first.

        Chunks sharing no word with the query (score 0) are left out.
        """
        tokens = tokenize(query)
        if not tokens or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokens)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]
