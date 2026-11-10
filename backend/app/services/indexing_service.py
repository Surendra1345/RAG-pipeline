"""Build the index: load documents, chunk, embed, and store chunks + vectors in Postgres.

Run it from the backend folder:  python -m app.services.indexing_service
"""
import logging
import time

from app.core.logging import setup_logging
from app.retrieval.embedder import get_embedder
from app.retrieval.vector_store import PgVectorStore
from app.services.document_service import load_all_chunks


log = logging.getLogger(__name__)


def index_documents(store: PgVectorStore | None = None) -> int:
    """Re-index everything in the documents folder. Returns the number of chunks stored."""
    start = time.perf_counter()
    log.info("Indexing started")
    store = store or PgVectorStore()
    chunks = load_all_chunks()
    vectors = get_embedder().embed_chunks(chunks)
    store.ensure_schema()
    stored = store.replace_all(chunks, vectors)
    log.info("Indexing finished: %d chunks stored in %.1f s", stored, time.perf_counter() - start)
    return stored


if __name__ == "__main__":
    setup_logging()
    stored = index_documents()
    print(f"Stored {stored} chunks with their embeddings.")
