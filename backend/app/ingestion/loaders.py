"""Read TXT, PDF and DOCX files into a RawDocument made of titled sections."""
import re
from collections import Counter
from pathlib import Path

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from app.core.exceptions import DocumentLoadError
from app.schemas.document import RawDocument, Section
from app.utils.file_utils import detect_file_type
from app.utils.text_utils import clean_text, split_sections


def load_document(path: Path) -> RawDocument:
    """Load one file. The type is detected from its content, not its extension."""
    path = Path(path)
    file_type = detect_file_type(path)
    loader = {"txt": _load_txt, "pdf": _load_pdf, "docx": _load_docx}[file_type]
    try:
        sections = loader(path)
    except Exception as e:  # corrupt or unreadable file
        raise DocumentLoadError(f"{path.name}: could not read {file_type} file ({e})") from e
    if not sections:
        raise DocumentLoadError(f"{path.name}: no readable text found")
    return RawDocument(name=path.name, file_type=file_type, sections=sections)


def _load_txt(path: Path) -> list[Section]:
    return split_sections(clean_text(path.read_text(encoding="utf-8")))


def _load_pdf(path: Path) -> list[Section]:
    pages = [(p.extract_text() or "") for p in PdfReader(path).pages]
    pages = _strip_repeated_lines(pages)
    sections = split_sections(clean_text("\n".join(pages)))
    # PDF text is hard-wrapped at line ends; join each section's lines into flowing text.
    return [Section(title=s.title, text=re.sub(r"\s*\n\s*", " ", s.text)) for s in sections]


def _strip_repeated_lines(pages: list[str]) -> list[str]:
    """Remove running headers/footers: lines that repeat (ignoring digits) on most pages."""
    if len(pages) < 3:
        return pages
    counts: Counter[str] = Counter()
    for page in pages:
        for line in {re.sub(r"\d+", "#", ln.strip()) for ln in page.split("\n") if ln.strip()}:
            counts[line] += 1
    repeated = {line for line, n in counts.items() if n >= max(3, len(pages) * 0.6)}
    return [
        "\n".join(ln for ln in page.split("\n") if re.sub(r"\d+", "#", ln.strip()) not in repeated)
        for page in pages
    ]


def _load_docx(path: Path) -> list[Section]:
    """Walk paragraphs and tables in order. Headings start sections; table rows become
    'Column: value; Column: value' lines so their facts stay searchable."""
    document = docx.Document(str(path))
    sections: list[Section] = []
    title, lines = "Introduction", []

    def flush() -> None:
        body = clean_text("\n".join(lines))
        if body:
            sections.append(Section(title=title, text=body))

    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            text = block.text.strip()
            style = block.style.name if block.style is not None else ""
            if not text or style == "Title":
                continue
            if style.startswith("Heading"):
                flush()
                title, lines = text, []
            else:
                lines.append(text)
        elif isinstance(block, Table):
            lines.extend(_table_rows(block))
    flush()
    return sections


def _table_rows(table: Table) -> list[str]:
    rows = [[c.text.strip() for c in r.cells] for r in table.rows]
    if len(rows) < 2:
        return ["; ".join(c for r in rows for c in r if c)] if rows else []
    header, body = rows[0], rows[1:]
    return ["; ".join(f"{h}: {v}" for h, v in zip(header, r) if v) for r in body]
