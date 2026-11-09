"""Postgres + pgvector store for chunks and their embeddings.

One row per chunk. The `embedding` column holds the 384-number vector. Search uses
cosine distance (`<=>`); vectors are already normalised, so score = 1 - distance is the
cosine similarity (higher is better).
"""
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

from app.core.config import settings
from app.core.constants import EMBEDDING_DIM
from app.schemas.document import Chunk


class PgVectorStore:
    def __init__(self, database_url: str = settings.database_url, table: str = "chunks") -> None:
        if not database_url:
            raise ValueError(
                "DATABASE_URL is not set. Copy backend/.env.example to backend/.env and fill it in."
            )
        self.database_url = database_url
        self.table = table

    # -- connection and schema -------------------------------------------------

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self.database_url)
        register_vector(conn)  # needs the vector extension to exist already
        return conn

    def ensure_schema(self) -> None:
        """Create the extension, the table and the search index if they are missing."""
        with psycopg.connect(self.database_url) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        id            text PRIMARY KEY,
                        doc_name      text NOT NULL,
                        section_title text NOT NULL,
                        parent_id     text NOT NULL,
                        text          text NOT NULL,
                        parent_text   text NOT NULL,
                        chunk_index   integer NOT NULL,
                        embedding     vector({dim}) NOT NULL
                    )
                    """
                ).format(table=sql.Identifier(self.table), dim=sql.Literal(EMBEDDING_DIM))
            )
            conn.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {idx} ON {table} USING hnsw (embedding vector_cosine_ops)"
                ).format(
                    idx=sql.Identifier(f"{self.table}_embedding_hnsw"),
                    table=sql.Identifier(self.table),
                )
            )

    def drop(self) -> None:
        """Delete the table (used by tests)."""
        with psycopg.connect(self.database_url) as conn:
            conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(self.table)))

    # -- writing ---------------------------------------------------------------

    def replace_all(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        """Make the table hold exactly these chunks (one transaction). Returns the row count."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if len(vectors) and vectors.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"expected {EMBEDDING_DIM}-dimension vectors, got {vectors.shape[1]}")

        insert = sql.SQL(
            """
            INSERT INTO {table}
                (id, doc_name, section_title, parent_id, text, parent_text, chunk_index, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
        ).format(table=sql.Identifier(self.table))
        rows = [
            (c.id, c.doc_name, c.section_title, c.parent_id, c.text, c.parent_text, c.chunk_index, v)
            for c, v in zip(chunks, vectors.astype(np.float32))
        ]
        with self._connect() as conn:
            conn.execute(sql.SQL("TRUNCATE {}").format(sql.Identifier(self.table)))
            with conn.cursor() as cur:
                cur.executemany(insert, rows)
        return len(rows)

    # -- reading ---------------------------------------------------------------

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(self.table))).fetchone()
        return int(row[0])

    def all_chunks(self) -> list[Chunk]:
        """Every stored chunk (without vectors), ordered by id. Used to build the BM25 index."""
        stmt = sql.SQL(
            """
            SELECT id, doc_name, section_title, parent_id, text, parent_text, chunk_index
            FROM {table} ORDER BY id
            """
        ).format(table=sql.Identifier(self.table))
        with self._connect() as conn:
            rows = conn.execute(stmt).fetchall()
        return [
            Chunk(id=r[0], doc_name=r[1], section_title=r[2], parent_id=r[3],
                  text=r[4], parent_text=r[5], chunk_index=r[6])
            for r in rows
        ]

    def search(self, query_vector: np.ndarray, k: int = 20) -> list[tuple[Chunk, float]]:
        """Return the k nearest chunks as (chunk, cosine similarity), best first."""
        query = np.asarray(query_vector, dtype=np.float32)
        stmt = sql.SQL(
            """
            SELECT id, doc_name, section_title, parent_id, text, parent_text, chunk_index,
                   1 - (embedding <=> %s) AS score
            FROM {table}
            ORDER BY embedding <=> %s
            LIMIT %s
            """
        ).format(table=sql.Identifier(self.table))
        with self._connect() as conn:
            rows = conn.execute(stmt, (query, query, k)).fetchall()
        return [
            (
                Chunk(
                    id=r[0], doc_name=r[1], section_title=r[2], parent_id=r[3],
                    text=r[4], parent_text=r[5], chunk_index=r[6],
                ),
                float(r[7]),
            )
            for r in rows
        ]
