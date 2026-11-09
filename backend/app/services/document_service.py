"""Find the documents on disk, load them, and turn them into chunks."""
import logging
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import DocumentLoadError, UnsupportedFileError
from app.ingestion.chunker import chunk_document
from app.ingestion.loaders import load_document
from app.schemas.document import Chunk, RawDocument
from app.utils.file_utils import list_document_files

log = logging.getLogger(__name__)


def load_all_documents(directory: Path = settings.documents_dir) -> list[RawDocument]:
    """Load every readable file in the folder. Bad files are logged and skipped."""
    docs: list[RawDocument] = []
    for path in list_document_files(directory):
        try:
            docs.append(load_document(path))
        except (UnsupportedFileError, DocumentLoadError) as e:
            log.warning("Skipping %s: %s", path.name, e)
    return docs


def load_all_chunks(directory: Path = settings.documents_dir) -> list[Chunk]:
    """Load and chunk every document in the folder."""
    return [c for doc in load_all_documents(directory) for c in chunk_document(doc)]
