"""Tests for the Postgres vector store. They need DATABASE_URL and use a throwaway table."""
import numpy as np
import pytest

from app.core.config import settings
from app.retrieval.embedder import get_embedder
from app.retrieval.vector_store import PgVectorStore
from app.services.document_service import load_all_chunks

pytestmark = pytest.mark.skipif(not settings.database_url, reason="DATABASE_URL is not set")


@pytest.fixture(scope="module")
def loaded():
    chunks = load_all_chunks()
    vectors = get_embedder().embed_chunks(chunks)
    store = PgVectorStore(table="chunks_test")
    store.ensure_schema()
    store.replace_all(chunks, vectors)
    yield store, chunks, vectors
    store.drop()


def test_every_chunk_is_stored(loaded):
    store, chunks, _ = loaded
    assert store.count() == len(chunks)


def test_replace_all_is_repeatable(loaded):
    store, chunks, vectors = loaded
    assert store.replace_all(chunks, vectors) == len(chunks)
    assert store.count() == len(chunks)  # no duplicates


def test_a_chunk_finds_itself_first(loaded):
    store, chunks, vectors = loaded
    results = store.search(vectors[10], k=3)
    assert results[0][0].id == chunks[10].id
    assert results[0][1] == pytest.approx(1.0, abs=1e-4)


def test_results_are_ranked_best_first(loaded):
    store, _, _ = loaded
    q = get_embedder().embed_query("How many days of annual leave do employees get?")
    scores = [s for _, s in store.search(q, k=10)]
    assert scores == sorted(scores, reverse=True)
    assert len(scores) == 10


def test_stored_chunk_matches_original(loaded):
    store, chunks, vectors = loaded
    chunk, _ = store.search(vectors[5], k=1)[0]
    assert chunk == chunks[5]


def test_mismatched_inputs_are_rejected(loaded):
    store, chunks, vectors = loaded
    with pytest.raises(ValueError):
        store.replace_all(chunks[:3], vectors[:2])
    with pytest.raises(ValueError):
        store.replace_all(chunks[:2], np.zeros((2, 10), dtype=np.float32))
