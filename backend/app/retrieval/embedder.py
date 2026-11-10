"""Turn text into vectors that capture meaning (384 numbers per text).

Vectors are L2-normalised, so a plain dot product between two vectors equals their
cosine similarity. That keeps the vector store simple and fast.
"""
import logging
import time
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.schemas.document import Chunk


log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str = settings.embedding_model) -> None:
        self.model_name = model_name
        start = time.perf_counter()
        self._model = SentenceTransformer(model_name)  # downloads weights on first use
        log.info("Embedding model %s loaded in %.1f s", model_name, time.perf_counter() - start)

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed many texts. Returns a float32 array of shape (len(texts), dimension)."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        start = time.perf_counter()
        vectors = self._model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        log.debug("Embedded %d texts in %.0f ms", len(texts), (time.perf_counter() - start) * 1000)
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one question. Returns a vector of shape (dimension,)."""
        return self.embed_texts([query])[0]

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        """Embed chunks using their embed_text() (document and section name + text).

        Row i of the result belongs to chunks[i].
        """
        return self.embed_texts([c.embed_text() for c in chunks])


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Load the model once and reuse it for the life of the process."""
    return Embedder()
