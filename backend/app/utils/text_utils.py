"""Text cleaning and section/sentence splitting."""
import re

from app.schemas.document import Section

# A heading looks like "4. Company holidays for 2026": number, dot, a capital letter,
# short, and no full stop at the end (so numbered sentences are not mistaken for headings).
_HEADING_RE = re.compile(r"^\d{1,2}\.\s+[A-Z][^\n]{0,90}$")
_UNDERLINE_RE = re.compile(r"^[-=]{3,}\s*$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Very common words carry no signal for keyword search.
STOPWORDS = frozenset(
    "a an the and or of to in on at for from by with is are was were be been it its this that "
    "what which who how many much do does did can".split()
)


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Lower-case word tokens for keyword search ('T20' -> 't20', punctuation dropped)."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS] if drop_stopwords else tokens


def clean_text(text: str) -> str:
    """Normalise whitespace: unify newlines, drop trailing spaces, collapse blank runs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def is_heading(line: str) -> bool:
    line = line.strip()
    return bool(_HEADING_RE.match(line)) and not line.endswith(".")


def split_sections(text: str, default_title: str = "Introduction") -> list[Section]:
    """Split text into sections at numbered heading lines such as '3. Match formats'.

    Text before the first heading becomes a section called `default_title`.
    Underline rows ('-----', '=====') are dropped. Empty sections are skipped.
    """
    sections: list[Section] = []
    title, buf = default_title, []

    def flush() -> None:
        body = clean_text("\n".join(buf))
        if body:
            sections.append(Section(title=title, text=body))

    for line in text.split("\n"):
        if _UNDERLINE_RE.match(line):
            continue
        if is_heading(line):
            flush()
            title, buf = line.strip(), []
        else:
            buf.append(line)
    flush()
    return sections


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences (keeps decimals like 20.12 intact)."""
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
