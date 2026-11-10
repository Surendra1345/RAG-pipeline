"""App-wide settings, read from environment variables or backend/.env."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # Paths
    documents_dir: Path = BASE_DIR / "data" / "documents"
    index_dir: Path = BASE_DIR / "data" / "index"

    # Postgres + pgvector (set in backend/.env, never in code)
    database_url: str = ""

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # LLM (keys go in backend/.env, never in code). Groq is tried first, Gemini is the fallback.
    groq_api_key: str = ""
    gemini_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    gemini_model: str = "gemini-3.5-flash-lite"
    llm_timeout_s: float = 30.0

    # Logging
    log_level: str = "INFO"

    # Chunking (characters)
    chunk_size: int = 500
    chunk_overlap: int = 80


settings = Settings()
