"""File helpers. File type is detected from the bytes, never from the extension."""
import zipfile
from pathlib import Path

from app.core.exceptions import UnsupportedFileError
from app.schemas.document import FileType


def detect_file_type(path: Path) -> FileType:
    """Return 'pdf', 'docx' or 'txt' based on the file's content.

    A Word file renamed to .pdf is still detected as docx.
    """
    with open(path, "rb") as f:
        head = f.read(8)

    if head.startswith(b"%PDF"):
        return "pdf"

    if head.startswith(b"PK"):
        try:
            with zipfile.ZipFile(path) as z:
                if "word/document.xml" in z.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            pass
        raise UnsupportedFileError(f"{path.name}: zip archive that is not a DOCX file")

    try:
        path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as e:
        raise UnsupportedFileError(f"{path.name}: not a PDF, DOCX or UTF-8 text file") from e
    return "txt"


def list_document_files(directory: Path) -> list[Path]:
    """Visible files in a folder, sorted by name (skips .gitkeep and hidden files)."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and not p.name.startswith("."))
