import shutil
from pathlib import Path

import pytest

from app.core.config import settings
from app.core.exceptions import UnsupportedFileError
from app.ingestion.loaders import load_document
from app.utils.file_utils import detect_file_type
from app.utils.text_utils import split_sections

DOCS = settings.documents_dir


def test_split_sections_finds_numbered_headings():
    text = "Title line\n\n1. First part\n-------\nSome text.\n\n2. Second part\nMore text."
    sections = split_sections(text)
    assert [s.title for s in sections] == ["Introduction", "1. First part", "2. Second part"]
    assert sections[1].text == "Some text."


def test_numbered_sentence_is_not_a_heading():
    sections = split_sections("1. Intro\n2. This is a sentence that ends with a full stop.")
    assert len(sections) == 1
    assert sections[0].title == "1. Intro"
    assert sections[0].text.startswith("2. This is a sentence")


def test_txt_loads_with_sections():
    doc = load_document(DOCS / "cricket_guide.txt")
    assert doc.file_type == "txt"
    assert any(s.title.startswith("5. Ways of getting out") for s in doc.sections)


def test_pdf_loads_and_strips_running_header():
    doc = load_document(DOCS / "hr_handbook.pdf")
    assert doc.file_type == "pdf"
    assert not any("Confidential | Page" in s.text for s in doc.sections)
    assert len(doc.sections) >= 17


def test_docx_tables_become_labelled_rows():
    doc = load_document(DOCS / "football_guide.docx")
    assert doc.file_type == "docx"
    text = " ".join(s.text for s in doc.sections)
    assert "Law: 11; Subject: Offside" in text


def test_docx_renamed_to_pdf_is_detected_by_content(tmp_path: Path):
    fake_pdf = tmp_path / "football_guide.pdf"
    shutil.copy(DOCS / "football_guide.docx", fake_pdf)
    assert detect_file_type(fake_pdf) == "docx"
    assert load_document(fake_pdf).file_type == "docx"


def test_binary_junk_is_rejected(tmp_path: Path):
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"\x89\x00\xff\xfe\x01\x02")
    with pytest.raises(UnsupportedFileError):
        detect_file_type(junk)
