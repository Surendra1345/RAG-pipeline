"""What the answer step returns."""
from pydantic import BaseModel

from app.schemas.search import SearchResponse


class Citation(BaseModel):
    number: int  # the [n] the model wrote
    chunk_id: str
    doc_name: str
    section_title: str
    text: str


class Answer(BaseModel):
    question: str
    rewritten_query: str
    answer: str
    citations: list[Citation]  # only the passages the answer actually cites
    provider: str | None  # None when no LLM was used
    model: str | None
    llm_ms: float
    search: SearchResponse
