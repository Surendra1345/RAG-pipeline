"""Build the index: load documents, chunk, embed, and store chunks + vectors in Postgres.

Run it from the backend folder:  python -m app.services.indexing_service
"""
from app.retrieval.embedder import get_embedder
from app.retrieval.vector_store import PgVectorStore
from app.services.document_service import load_all_chunks


def index_documents(store: PgVectorStore | None = None) -> int:
    """Re-index everything in the documents folder. Returns the number of chunks stored."""
    store = store or PgVectorStore()
    chunks = load_all_chunks()
    vectors = get_embedder().embed_chunks(chunks)
    store.ensure_schema()
    return store.replace_all(chunks, vectors)


if __name__ == "__main__":
    stored = index_documents()
    print(f"Stored {stored} chunks with their embeddings.")
