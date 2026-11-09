from app.retrieval.bm25_index import BM25Index
from app.services.document_service import load_all_chunks
from app.utils.text_utils import tokenize


def test_tokenize_lowercases_and_drops_stopwords_and_punctuation():
    assert tokenize("What is the T20 format?") == ["t20", "format"]
    assert tokenize("What is the T20 format?", drop_stopwords=False) == ["what", "is", "the", "t20", "format"]


def test_exact_term_finds_its_chunk():
    index = BM25Index(load_all_chunks())
    results = index.search("doosra")
    assert results, "expected at least one hit"
    assert "doosra" in results[0][0].text.lower()


def test_acronym_that_dense_search_confuses_is_found():
    index = BM25Index(load_all_chunks())
    assert index.search("What is the DLS method?")[0][0].id == "cricket_guide:10:0"


def test_scores_are_sorted_and_k_is_respected():
    results = BM25Index(load_all_chunks()).search("overs bowler innings", k=5)
    scores = [s for _, s in results]
    assert len(results) <= 5
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)


def test_no_matching_words_returns_nothing():
    assert BM25Index(load_all_chunks()).search("zzzzqqqq") == []


def test_empty_or_stopword_only_query_returns_nothing():
    index = BM25Index(load_all_chunks())
    assert index.search("") == []
    assert index.search("what is the") == []


def test_empty_index_is_safe():
    assert BM25Index([]).search("anything") == []
