"""Parent-child chunking.

Parent = a whole section. Child = a small chunk of at most `chunk_size` characters
packed from whole sentences/lines, with the last few sentences repeated as overlap.
Children are retrieved; the parent is kept so the LLM can be given full context.
"""
from pathlib import Path

from app.core.config import settings
from app.schemas.document import Chunk, RawDocument
from app.utils.text_utils import split_sentences


def chunk_document(
    doc: RawDocument,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    stem = Path(doc.name).stem
    chunks: list[Chunk] = []
    for s_idx, section in enumerate(doc.sections):
        parent_id = f"{stem}:{s_idx}"
        for c_idx, text in enumerate(_split_section(section.text, chunk_size, chunk_overlap)):
            chunks.append(
                Chunk(
                    id=f"{parent_id}:{c_idx}",
                    doc_name=doc.name,
                    section_title=section.title,
                    parent_id=parent_id,
                    text=text,
                    parent_text=section.text,
                    chunk_index=c_idx,
                )
            )
    return chunks


def _split_section(text: str, size: int, overlap: int) -> list[str]:
    units = _to_units(text, size)
    chunks: list[list[tuple[str, bool]]] = []
    current: list[tuple[str, bool]] = []

    for unit in units:
        if current and _length(current) + len(unit[0]) + 1 > size:
            chunks.append(current)
            current = _overlap_tail(current, overlap)
            # drop the overlap if it would stop the new unit from fitting
            if current and _length(current) + len(unit[0]) + 1 > size:
                current = []
        current.append(unit)
    if current:
        chunks.append(current)
    return [_join(c) for c in chunks]


def _to_units(text: str, size: int) -> list[tuple[str, bool]]:
    """Break text into (piece, starts_new_line) units no longer than `size`."""
    units: list[tuple[str, bool]] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        pieces = [line] if len(line) <= size else split_sentences(line)
        first = True
        for piece in pieces:
            for part in _hard_split(piece, size):
                units.append((part, first))
                first = False
    return units


def _hard_split(piece: str, size: int) -> list[str]:
    """Last resort for a single sentence longer than `size`: cut on word boundaries."""
    if len(piece) <= size:
        return [piece]
    out, cur = [], ""
    for word in piece.split(" "):
        if cur and len(cur) + len(word) + 1 > size:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return out


def _overlap_tail(units: list[tuple[str, bool]], overlap: int) -> list[tuple[str, bool]]:
    tail: list[tuple[str, bool]] = []
    total = 0
    for unit in reversed(units[1:]):  # never repeat the whole previous chunk
        if total + len(unit[0]) + 1 > overlap:
            break
        tail.insert(0, unit)
        total += len(unit[0]) + 1
    return tail


def _length(units: list[tuple[str, bool]]) -> int:
    return len(_join(units))


def _join(units: list[tuple[str, bool]]) -> str:
    out = ""
    for i, (text, new_line) in enumerate(units):
        out += text if i == 0 else ("\n" if new_line else " ") + text
    return out
