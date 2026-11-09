import pytest

from app.ingestion.chunker import chunk_document
from app.schemas.document import RawDocument, Section
from app.services.document_service import load_all_chunks


def make_doc(text: str) -> RawDocument:
    return RawDocument(name="sample.txt", file_type="txt", sections=[Section(title="1. Sample", text=text)])


def test_chunks_never_exceed_size():
    text = " ".join(f"Sentence number {i} says something useful." for i in range(60))
    chunks = chunk_document(make_doc(text), chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_overlap_repeats_text_between_neighbours():
    text = " ".join(f"Sentence number {i} says something useful." for i in range(30))
    chunks = chunk_document(make_doc(text), chunk_size=200, chunk_overlap=60)
    first_end = chunks[0].text.split(". ")[-1]
    assert first_end.rstrip(".") in chunks[1].text


def test_no_text_is_lost():
    text = " ".join(f"Fact {i} is here." for i in range(50))
    chunks = chunk_document(make_doc(text), chunk_size=150, chunk_overlap=30)
    joined = " ".join(c.text for c in chunks)
    assert all(f"Fact {i} is here." in joined for i in range(50))


def test_single_long_sentence_is_hard_split():
    chunks = chunk_document(make_doc("word " * 300), chunk_size=100, chunk_overlap=10)
    assert all(len(c.text) <= 100 for c in chunks)


def test_ids_and_parent_links_are_stable():
    chunks = chunk_document(make_doc("Short text."))
    assert chunks[0].id == "sample:0:0"
    assert chunks[0].parent_id == "sample:0"
    assert chunks[0].parent_text == "Short text."


def test_bad_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_document(make_doc("x"), chunk_size=100, chunk_overlap=100)


def test_real_documents_produce_unique_ids():
    chunks = load_all_chunks()
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert {c.doc_name for c in chunks} == {"cricket_guide.txt", "football_guide.docx", "hr_handbook.pdf"}
    assert all(len(c.text) <= 500 for c in chunks)
