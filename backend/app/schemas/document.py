"""Data models shared by ingestion, retrieval and the API."""
from typing import Literal

from pydantic import BaseModel

FileType = Literal["txt", "pdf", "docx"]


class Section(BaseModel):
    """One titled block of a document (becomes a parent chunk)."""

    title: str
    text: str


class RawDocument(BaseModel):
    """A loaded file: its name, real type and ordered sections."""

    name: str
    file_type: FileType
    sections: list[Section]


class Chunk(BaseModel):
    """A small searchable piece of text, linked to its larger parent section."""

    id: str  # stable: "<doc_stem>:<section_idx>:<chunk_idx>"
    doc_name: str
    section_title: str
    parent_id: str  # "<doc_stem>:<section_idx>"
    text: str  # the child chunk (what gets embedded and retrieved)
    parent_text: str  # the whole section (what the LLM can be shown)
    chunk_index: int  # position of this chunk inside its section

    def embed_text(self) -> str:
        """Text used for embedding/BM25: the chunk plus its location for context."""
        return f"{self.doc_name} > {self.section_title}\n{self.text}"
