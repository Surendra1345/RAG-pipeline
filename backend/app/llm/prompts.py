"""Prompt text and the helper that lays the retrieved chunks out for the model."""
from app.schemas.search import RankedChunk

REWRITE_SYSTEM = (
    "You rewrite a user's question into a short search query for a document search engine. "
    "Spell out acronyms only if you are sure what they mean, fix typos, and keep names, numbers and "
    "grades exactly as written. Do not answer the question. Reply with the query only, on one line."
)

ANSWER_SYSTEM = (
    "You answer questions using ONLY the numbered context passages. "
    "Cite every claim with the passage number in square brackets, like [1] or [2][3]. "
    "If the passages do not contain the answer, reply exactly: I could not find this in the documents. "
    "Do not use outside knowledge. Keep the answer short and direct."
)

NOT_FOUND = "I could not find this in the documents."


def build_context(chunks: list[RankedChunk]) -> str:
    parts = []
    for n, r in enumerate(chunks, start=1):
        c = r.chunk
        parts.append(f"[{n}] ({c.doc_name} > {c.section_title})\n{c.text}")
    return "\n\n".join(parts)


def build_answer_prompt(question: str, chunks: list[RankedChunk]) -> str:
    return f"Context passages:\n\n{build_context(chunks)}\n\nQuestion: {question}\nAnswer:"
