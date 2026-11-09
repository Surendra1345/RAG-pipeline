import numpy as np
import pytest

from app.retrieval.embedder import get_embedder
from app.services.document_service import load_all_chunks


@pytest.fixture(scope="module")
def embedder():
    return get_embedder()


def test_shape_dtype_and_unit_length(embedder):
    vectors = embedder.embed_texts(["a cricket match", "a football match", "annual leave"])
    assert vectors.shape == (3, 384)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_dimension_property(embedder):
    assert embedder.dimension == 384


def test_empty_input(embedder):
    assert embedder.embed_texts([]).shape == (0, 384)


def test_similar_meaning_scores_higher_than_unrelated(embedder):
    q = embedder.embed_query("How many days of annual leave do employees get?")
    related = embedder.embed_texts(["Employees receive 24 days of paid annual leave each year."])[0]
    unrelated = embedder.embed_texts(["The ball is red in Test cricket."])[0]
    assert float(q @ related) > float(q @ unrelated)


def test_embedding_is_deterministic(embedder):
    a = embedder.embed_query("What is offside?")
    b = embedder.embed_query("What is offside?")
    assert np.allclose(a, b, atol=1e-6)


def test_all_real_chunks_embed_in_order(embedder):
    chunks = load_all_chunks()
    vectors = embedder.embed_chunks(chunks)
    assert vectors.shape == (len(chunks), 384)
    # row i belongs to chunk i: a chunk is closest to its own text
    probe = embedder.embed_texts([chunks[10].embed_text()])[0]
    assert int(np.argmax(vectors @ probe)) == 10
