"""Custom errors raised by the ingestion layer."""


class UnsupportedFileError(Exception):
    """The file is not a TXT, PDF or DOCX file."""


class DocumentLoadError(Exception):
    """The file was recognised but its text could not be read."""
