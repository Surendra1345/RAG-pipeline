# RAG Explorer — Advanced RAG Pipeline with Visible Retrieval Metrics

 **how a RAG system finds the correct information**. A user asks a question about the loaded documents, and the UI shows every retrieval stage and how well each one performed.

> **Status:** Steps 1 and 2 are done (folder structure; document loading and chunking). Step 3 (retrieval) is done: the embedder, the Postgres/pgvector vector store (95 chunks stored), BM25 keyword search, hybrid fusion (RRF), the cross-encoder reranker, and a search service that returns every stage's ranked list. 51 tests pass. Files marked **Implemented** below contain working code. Everything else is still an **empty placeholder**, and its "Planned contents" describe what it will hold.

---

## 1. What the app does

For every question the UI will show:

1. **Retriever output:** top-k chunks from dense (embedding) search, BM25 (keyword) search, and the hybrid fusion of both.
2. **Reranker output:** the same candidates reordered by a cross-encoder, with arrows showing how far each chunk moved.
3. **Metrics:** precision@k, recall@k, MRR, and NDCG@k, before and after reranking.
4. **Worked calculation:** a step-by-step breakdown of how each metric was computed for that query.
5. **Answer:** an LLM answer built from the top chunks, with citations.

## 2. Pipeline

```
Indexing (once):  Documents (TXT/PDF/DOCX) -> Chunker -> Embedder -> Vector store (+ BM25 index)
Query (each):     Question -> Query rewrite -> Hybrid retrieve (dense + BM25, RRF) -> Rerank -> LLM answer
Evaluation:       Relevance labels (test set / LLM judge / manual) -> precision@k, recall@k, MRR, NDCG@k
```

Advanced-RAG techniques: hybrid retrieval with Reciprocal Rank Fusion, query rewriting / HyDE, cross-encoder reranking, parent-child chunking, cited answers.

## 3. Tech stack

| Part | Choice |
|---|---|
| Backend | Python, FastAPI |
| Embeddings / reranker | sentence-transformers (local, free) |
| Keyword search | rank-bm25 |
| Vector store | PostgreSQL 18 + pgvector 0.8.7 (HNSW index, cosine distance) |
| LLM | Groq (primary), Gemini (automatic fallback) |
| Frontend | React + TypeScript (Vite), Tailwind CSS, Recharts |
| Deployment | Backend on Hugging Face Spaces (Docker), frontend on Vercel/Netlify |

### 3.1 Dependencies and what each one does

All dependencies are installed. The backend uses a Python 3.11 virtual environment at `backend/.venv`. The frontend packages are in `frontend/node_modules`. Both folders are git-ignored.

**Backend** (`backend/requirements.txt`)

| Package | What it does in this project |
|---|---|
| `fastapi` | Web framework for the API: routes, request validation, JSON responses |
| `uvicorn[standard]` | Server that runs the FastAPI app |
| `python-multipart` | Lets FastAPI accept file uploads (TXT/PDF/DOCX) |
| `pydantic-settings` | Loads settings and API keys from environment variables / `.env` |
| `pypdf` | Extracts text from PDF files |
| `python-docx` | Extracts text and tables from Word (DOCX) files |
| `sentence-transformers` | Runs the local embedding model (384-dimension vectors) and the cross-encoder reranker. Installs `torch` and `transformers` with it. |
| `rank-bm25` | BM25 keyword search, the keyword half of hybrid retrieval |
| `faiss-cpu` | Installed but not used now: the vector store is Postgres + pgvector. Kept in case a FAISS option is added later. |
| `psycopg[binary]` | Python driver for PostgreSQL (version 3) |
| `pgvector` | Lets psycopg read and write the Postgres `vector` type from NumPy arrays |
| `numpy` | Vector math: cosine similarity, scores, and metric calculations |
| `groq` | Client for the Groq API, the primary LLM (query rewriting, judge, answers) |
| `google-generativeai` | Client for the Gemini API, the automatic fallback LLM |
| `pytest` | Test runner for `backend/tests/` |
| `httpx` | HTTP client that FastAPI's test client needs for the integration tests |

**Frontend** (`frontend/package.json`)

| Package | What it does in this project |
|---|---|
| `react`, `react-dom` | UI library and its browser renderer |
| `recharts` | Charts for the metrics panel (before/after reranking) |
| `vite` | Dev server and production build tool |
| `@vitejs/plugin-react` | Lets Vite compile React (JSX) |
| `typescript` | Type checking |
| `@types/react`, `@types/react-dom` | TypeScript type definitions for React |
| `tailwindcss` | Utility-class styling |
| `postcss`, `autoprefixer` | Process the Tailwind CSS and add browser prefixes |

**Not downloaded yet**
- Nothing else to download for retrieval: both models (`all-MiniLM-L6-v2` embedder and `ms-marco-MiniLM-L-6-v2` reranker) are downloaded into the Hugging Face cache on first use. The very first reranked search takes about 40 seconds while the reranker loads; after that a rerank takes tens to a few hundred milliseconds.
- API keys (`GROQ_API_KEY`, `GEMINI_API_KEY`) go in a local `backend/.env`, which git ignores. They are needed from step 4.

**Database setup (PostgreSQL + pgvector)**

The vector store needs a PostgreSQL database with the pgvector extension. Done on this project's machine (PostgreSQL 18.4 from the standard installer, at `/Library/PostgreSQL/18`):

1. Create a database (here it is named `RAG`).
2. Install pgvector into that Postgres. The installer does not include it, and Homebrew's `brew install pgvector` only targets Homebrew's own Postgres, so build it from source against the installed one:
   ```
   git clone --branch v0.8.7 https://github.com/pgvector/pgvector.git
   cd pgvector
   make PG_CONFIG=/Library/PostgreSQL/18/bin/pg_config PG_SYSROOT=$(xcrun --show-sdk-path)
   sudo make install PG_CONFIG=/Library/PostgreSQL/18/bin/pg_config PG_SYSROOT=$(xcrun --show-sdk-path)
   ```
   `PG_SYSROOT` is needed because the installer's build settings point at a macOS 14 SDK that may not exist; without it the build fails with `inttypes.h file not found`. `PG_CONFIG` is needed on every `make` command because Postgres is not on the PATH.
3. Enable it in the database: `CREATE EXTENSION IF NOT EXISTS vector;` (the indexer also does this if it is missing). Check with `SELECT extname, extversion FROM pg_extension;`.
4. Copy `backend/.env.example` to `backend/.env` and set `DATABASE_URL=postgresql://USER:PASSWORD@localhost:5432/RAG`. A password with special characters must be URL-encoded. `.env` is git-ignored.
5. Load the data: `cd backend && python -m app.services.indexing_service`. It creates the `chunks` table and HNSW index if needed, then replaces the table's rows with every chunk and its embedding. Re-running does not create duplicates.

**Database schema (the `chunks` table)**

One row per chunk. The chunk text, its parent section and its embedding sit in the same row, so a search returns everything needed without a second lookup. `PgVectorStore.ensure_schema()` creates all of this if it is missing, and is safe to run repeatedly.

| Column | Type | Meaning |
|---|---|---|
| `id` | text, primary key | Stable chunk ID: `<doc_stem>:<section>:<chunk>`, for example `cricket_guide:3:1` |
| `doc_name` | text | File the chunk came from |
| `section_title` | text | Heading of its section, for example `4. Offside` |
| `parent_id` | text | `<doc_stem>:<section>`; groups the chunks of one section |
| `text` | text | The child chunk (at most 500 characters). This is what is embedded and retrieved |
| `parent_text` | text | The whole section, kept so the LLM can be shown fuller context |
| `chunk_index` | integer | Position of the chunk inside its section |
| `embedding` | `vector(384)` | The normalised `all-MiniLM-L6-v2` vector for the chunk |

Indexes: `chunks_pkey` (B-tree on `id`) and `chunks_embedding_hnsw` (HNSW on `embedding` with cosine distance).

A search is one query. `<=>` is pgvector's cosine distance, so `1 - distance` is the cosine similarity (higher is better):
```sql
SELECT id, doc_name, section_title, text, 1 - (embedding <=> :query_vector) AS score
FROM chunks
ORDER BY embedding <=> :query_vector
LIMIT :k;
```

How it behaves:
- **Loading is all or nothing.** `replace_all` empties the table and inserts every chunk inside one transaction. If anything fails, the table is left as it was. Running it again never creates duplicates.
- **Exact results at this size.** With 95 rows, `EXPLAIN` shows Postgres doing a sequential scan, so every search compares against every vector and is exact. The HNSW index is approximate (pgvector's default `hnsw.ef_search` is 40) and only starts to matter as the table grows.
- **Vector size is checked twice.** The code rejects anything that is not 384 numbers wide before writing, and the column type enforces it too.
- **BM25 is not in the database.** It is built in memory from `all_chunks()` when the search service starts.
- **Connections.** `DATABASE_URL` comes from `backend/.env`. Each operation opens its own connection; there is no pool yet (planned with the API in step 5).
- **Size.** About 552 kB for 95 rows including the indexes.

Look at the data yourself in pgAdmin (on the `RAG` database):
```sql
SELECT count(*) FROM chunks;                                    -- 95
SELECT doc_name, count(*) FROM chunks GROUP BY 1 ORDER BY 1;    -- cricket 26, football 24, hr 45
SELECT id, section_title, left(text, 60), vector_dims(embedding) FROM chunks ORDER BY id LIMIT 10;
SELECT extname, extversion FROM pg_extension;                   -- includes vector 0.8.7
```

**Setup commands**
```
# Backend
cd backend
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

**Known note:** `npm audit` reports a moderate `esbuild` advisory. It only affects the local dev server, not the deployed app. The fix means a major Vite upgrade, so it is left alone for now.

## 4. Build steps

| # | Step | Status |
|---|---|---|
| 1 | Project scaffold (folders and empty files) | Done |
| 2 | Document loading and chunking | Done (14 tests pass) |
| 3 | Embeddings, BM25, hybrid search, reranker | Done (51 tests pass): embedder, Postgres/pgvector store with 95 chunks, BM25, hybrid RRF, reranker, search service. Query rewriting is part of step 4. |
| 4 | Metrics and LLM judge, with fallback | Pending |
| 5 | FastAPI endpoints | Pending |
| 6 | React UI | Pending |
| 7 | Deployment setup | Pending |

The README is updated after every step.

---

## 5. Folder structure

```
RAG-pipeline/
├── README.md
├── .gitignore
├── backend/
└── frontend/
```

### 5.1 Backend (`backend/`)

Request flow: **routes -> services -> (ingestion, retrieval, llm, evaluation)**. Routes only handle HTTP. Services orchestrate. The domain packages each do one job.

| Path | Purpose | Planned contents |
|---|---|---|
| `requirements.txt` | Python dependencies | fastapi, uvicorn, sentence-transformers, rank-bm25, faiss-cpu, psycopg, pgvector, pypdf, python-docx, groq, google-generativeai |
| `.env.example` | **Implemented.** Template for secrets | Contains a `DATABASE_URL` placeholder. Copy it to `.env` and fill it in. The real `.env` is git-ignored and must never be committed. More variables (API keys) come in step 4. |
| `data/documents/` | Source documents to index | **Loaded:** `cricket_guide.txt` (13 sections), `football_guide.docx` (12 sections, 6 tables), `hr_handbook.pdf` (18 sections, 6 pages, fictional company "Aurora Systems"). Together they produce 95 chunks. |
| `data/index/` | Reserved for saved index files | Currently unused: chunks and embeddings are stored in PostgreSQL. Git-ignored. |
| `data/eval/` | Evaluation test set | Questions mapped to gold chunks |

#### `app/main.py`
Creates the FastAPI app, enables CORS for the frontend, registers the routers, and loads the index at startup.

#### `app/api/` (HTTP layer)
| File | Planned contents |
|---|---|
| `deps.py` | Shared dependencies, such as access to the services |
| `routes/health.py` | `GET /health` liveness check |
| `routes/documents.py` | List, upload, and delete documents; trigger re-indexing |
| `routes/search.py` | Run retrieval + reranking and return every stage's ranked chunks |
| `routes/answer.py` | Generate the cited LLM answer |
| `routes/evaluation.py` | Return metrics and the worked calculation; run the test set |

#### `app/core/` (app-wide basics)
| File | Planned contents |
|---|---|
| `config.py` | **Implemented.** `Settings` class (pydantic) reading env vars / `backend/.env`. Holds `database_url` (read from `.env`), `documents_dir`, `index_dir`, `embedding_model` (all-MiniLM-L6-v2), `embedding_batch_size` (32), `chunk_size` (500 characters) and `chunk_overlap` (80). The shared `settings` object is imported everywhere. More settings are added in later steps. |
| `logging.py` | Logging setup |
| `exceptions.py` | **Implemented (partly).** `UnsupportedFileError` (not a TXT/PDF/DOCX file) and `DocumentLoadError` (file recognised but unreadable). HTTP mappings come with the routes in step 5. |
| `constants.py` | **Implemented (partly).** `EMBEDDING_DIM = 384`, the vector size the database column must match. `RRF_C = 60` (the fusion constant), `DEFAULT_CANDIDATES = 20` (how many each retriever returns and the reranker rescores) and `DEFAULT_TOP_K = 5`. The relevance scale is added with the metrics. |

#### `app/schemas/` (request/response models)
| File | Planned contents |
|---|---|
| `document.py` | **Implemented.** Three models. `Section` (title + text). `RawDocument` (name, real file type, list of sections). `Chunk` (stable `id` like `cricket_guide:3:1`, `doc_name`, `section_title`, `parent_id`, child `text`, full `parent_text`, `chunk_index`). `Chunk.embed_text()` returns "doc > section" plus the text, which is what gets embedded and indexed in step 3. |
| `search.py` | **Implemented (results part).** `RankedChunk` (chunk, score, 1-based rank, source stage, and `previous_rank` set by the reranker so the UI can show movement), `StageResult` (stage name, ranked chunks, elapsed milliseconds) and `SearchResponse` (the query, k, and one `StageResult` per stage; `.stage(name)` looks one up and `.final` is the last stage's list). The search request model comes with the API in step 5. |
| `answer.py` | Answer and citation models |
| `evaluation.py` | Metric results and calculation-step models |

#### `app/services/` (orchestration)
| File | Planned contents |
|---|---|
| `document_service.py` | **Implemented.** `load_all_documents(directory)` loads every readable file in the folder and logs and skips bad ones, so one broken file never stops the rest. `load_all_chunks(directory)` loads and chunks everything. Upload/delete come in step 5. |
| `indexing_service.py` | **Implemented (vector store part).** `index_documents()` loads and chunks the documents, embeds every chunk, makes sure the schema exists, and stores chunks and vectors with `replace_all`. Returns the number of chunks stored. Run it with `python -m app.services.indexing_service`. The BM25 index is added later. |
| `search_service.py` | **Implemented (without query rewriting).** `SearchService.search(query, k=5, candidates=20, rerank=True)` runs dense search, BM25, RRF fusion and the reranker, times each stage, and returns every stage's ranked list. It builds the BM25 index from the chunks stored in Postgres. Try it: `cd backend && python -m app.services.search_service "your question"` prints the top 5 of each stage, with "(was #n)" showing how far the reranker moved each chunk. |
| `answer_service.py` | Build the prompt from top chunks and get the cited answer |
| `evaluation_service.py` | Get relevance labels and compute metrics for each stage |

#### `app/ingestion/` (getting text in)
| File | Planned contents |
|---|---|
| `loaders.py` | **Implemented.** `load_document(path)` detects the real file type, then reads it into a `RawDocument` of sections. TXT: splits at numbered headings. PDF: `_strip_repeated_lines` removes running page headers (any line repeated on most pages, ignoring digits), then splits at headings and rejoins hard-wrapped lines. DOCX: `_load_docx` walks paragraphs and tables in order, starts a section at each Heading style, and `_table_rows` turns each table row into `Column: value; Column: value` so facts keep their labels. |
| `chunker.py` | **Implemented.** Parent-child chunking. `chunk_document(doc)` makes each section a parent and splits it into child chunks of at most 500 characters. Chunks are packed from whole sentences or lines (`_split_section`), a single over-long sentence is cut on word boundaries (`_hard_split`), and the last sentences of a chunk repeat at the start of the next as overlap (`_overlap_tail`). Chunk IDs are stable and unique. |

#### `app/retrieval/` (finding relevant chunks)
| File | Planned contents |
|---|---|
| `embedder.py` | **Implemented.** Turns text into 384-number vectors with `all-MiniLM-L6-v2`. `Embedder.embed_texts(texts)` embeds many texts in batches and returns a float32 array of shape (n, 384). `embed_query(q)` embeds one question. `embed_chunks(chunks)` embeds each chunk's `embed_text()` (document and section name plus text); row i belongs to chunk i. All vectors are L2-normalised, so a plain dot product equals cosine similarity. `dimension` reports the vector size. `get_embedder()` loads the model once and reuses it. Measured on this machine: model load about 6 seconds, all 95 chunks embedded in under 1 second. |
| `vector_store.py` | **Implemented.** `PgVectorStore` keeps chunks and embeddings in a Postgres table (default name `chunks`). `ensure_schema()` creates the extension, the table (text fields plus `embedding vector(384)`) and an HNSW cosine index if missing. `replace_all(chunks, vectors)` truncates and inserts everything in one transaction, rejecting mismatched counts or wrong vector sizes. `search(query_vector, k)` returns the k nearest chunks as `(chunk, score)`, best first, where score = 1 - cosine distance (the cosine similarity). `count()` and `drop()` (for tests) round it out. `all_chunks()` returns every stored chunk (used to build the BM25 index). Raises a clear error if `DATABASE_URL` is not set. |
| `bm25_index.py` | **Implemented.** `BM25Index(chunks)` builds a BM25 (Okapi) index in memory over each chunk's `embed_text()`, tokenised with `tokenize` (lower-case, stop words dropped). `search(query, k)` returns up to k `(chunk, score)` pairs, best first, leaving out chunks that share no word with the query. It finds exact terms such as "DLS" or "G6" that embeddings can blur. |
| `hybrid.py` | **Implemented.** `to_ranked(pairs, source)` turns `(chunk, score)` pairs into numbered `RankedChunk`s. `reciprocal_rank_fusion(rankings, c=60, top_n)` merges the dense and BM25 lists: a chunk's score is the sum of `1 / (c + rank)` over the lists it appears in, so a chunk ranked well by both beats one ranked first by only one. Ties break by best rank, then chunk id, so the result is deterministic. |
| `reranker.py` | **Implemented.** `Reranker.rerank(query, candidates, top_k)` scores each (question, chunk) pair together with the cross-encoder `ms-marco-MiniLM-L-6-v2`, sorts best first, and records each chunk's old position as `previous_rank`. Scores are raw model scores (higher is better, can be negative, not probabilities). `get_reranker()` loads the model once. |
| `query_rewriter.py` | LLM query rewriting / HyDE |

#### `app/llm/` (LLM access)
| File | Planned contents |
|---|---|
| `base.py` | Common interface every provider follows |
| `groq_provider.py` | Primary provider |
| `gemini_provider.py` | Fallback provider |
| `fallback.py` | Try Groq, switch to Gemini on rate limit or error |
| `prompts.py` | Prompt templates (rewrite, judge, answer) |

#### `app/evaluation/` (measuring quality)
| File | Planned contents |
|---|---|
| `metrics.py` | precision@k, recall@k, MRR, NDCG@k |
| `judge.py` | LLM relevance scoring of chunks (one batched request) |
| `calculation_steps.py` | Turn a metric computation into human-readable steps |

#### `app/utils/` (small helpers)
| File | Planned contents |
|---|---|
| `file_utils.py` | **Implemented.** `detect_file_type(path)` reads the file's first bytes: `%PDF` means PDF, a zip containing `word/document.xml` means DOCX, valid UTF-8 means TXT, anything else raises `UnsupportedFileError`. The extension is ignored, so a Word file renamed `.pdf` still loads correctly. `list_document_files(dir)` lists visible files. |
| `text_utils.py` | **Implemented.** `clean_text` (normalise whitespace), `is_heading` (a line like `4. Company holidays`, not ending in a full stop), `split_sections` (cut text into titled sections; text before the first heading becomes "Introduction"), `split_sentences` (keeps decimals like 20.12 intact), and `tokenize` (lower-case word tokens with a small stop-word list, used by BM25: "What is the T20 format?" becomes `t20`, `format`). |
| `timing.py` | Stage timing |

#### `tests/`
| Path | Purpose |
|---|---|
| `conftest.py` | Shared test fixtures |
| `unit/test_loaders.py` | **Implemented (7 tests).** Section splitting, the three real documents, running-header removal, DOCX table rows, a Word file renamed to `.pdf`, and rejection of binary junk |
| `unit/test_chunker.py` | **Implemented (7 tests).** Size limit, overlap, no lost text, long-sentence splitting, stable IDs, bad settings rejected, unique IDs across the real documents |
| `unit/test_embedder.py` | **Implemented (6 tests).** Shape (n, 384), float32 and unit length; dimension; empty input; a related sentence scores higher than an unrelated one; same text gives the same vector; all 95 real chunks embed in order |
| `unit/test_vector_store.py` | **Implemented (6 tests).** Needs `DATABASE_URL`; skipped otherwise. Uses a throwaway table `chunks_test`, so it never touches the real `chunks` table. Checks: every chunk stored, repeatable `replace_all` with no duplicates, a chunk finds itself first with score 1.0, results ranked best first, a stored chunk equals the original, mismatched inputs rejected |
| `unit/test_bm25.py` | **Implemented (7 tests).** Tokenising; exact-term lookup ("doosra"); the acronym "DLS" finds the right chunk; sorted scores and the k limit; no-match, empty and stop-word-only queries; empty index |
| `unit/test_hybrid.py` | **Implemented (8 tests).** The RRF formula; a chunk in both lists beats one first in only one; renumbering and source marking; `top_n`; deterministic ties; an empty list; no lists |
| `unit/test_reranker.py` | **Implemented (4 tests).** A relevant chunk moves above an irrelevant one; `previous_rank` and source recorded; `top_k`; empty input. Uses the real model |
| `unit/test_search_service.py` | **Implemented (6 tests).** Needs `DATABASE_URL`. All four stages in order; the leave question finds the leave section; reranked results point back to their fused rank; hybrid rescues "DLS" (dense search puts the DRS chunk first, fusion puts the DLS chunk first); reranking can be switched off; the candidates limit |
| `unit/test_metrics.py` | Empty until step 4 |

**Retrieval quality is mixed, and that is expected.** On "What is the DLS method?", BM25 and fusion put the right chunk first, but the reranker moved the DRS chunk above it. On "What is the bonus target for grade G6?", the top reranked chunk straddles the end of the G5 paragraph and the start of G6, so the exact answer sits one chunk lower. Step 4's metrics (precision@k, recall@k, MRR, NDCG@k) exist to measure this properly, per stage, instead of judging by eye.

Run the tests: `cd backend && .venv/bin/python -m pytest tests -q`

**Known limitation:** tables inside PDFs come out as flat text with the column labels lost, for example "Study leave 8 No For examinations only". The facts are still searchable. DOCX tables keep their labels because Word stores them as real tables.
| `integration/test_search_api.py` | Test the search endpoint end to end |

### 5.2 Frontend (`frontend/`)

| Path | Purpose |
|---|---|
| `index.html` | HTML entry page |
| `package.json` | Dependencies and scripts |
| `vite.config.ts`, `tsconfig.json` | Build and TypeScript config |
| `tailwind.config.js`, `postcss.config.js` | Styling config |
| `public/` | Static assets |

#### `src/`
| Path | Planned contents |
|---|---|
| `main.tsx`, `App.tsx` | React entry point and top-level layout/routing |
| `vite-env.d.ts` | Vite type declarations |
| `api/` | `client.ts` (shared fetch wrapper) plus `search.ts`, `documents.ts`, `answer.ts`, `evaluation.ts` (one per backend route) |
| `pages/` | `ExplorerPage.tsx` (ask and inspect), `DocumentsPage.tsx` (upload/manage) |
| `components/layout/` | `Header`, `Sidebar`, `ThemeToggle` (light/dark) |
| `components/search/` | `SearchBar`, `SettingsPanel` (k slider, retrieval mode) |
| `components/pipeline/` | `PipelineView`, `StageColumn`, `ChunkCard`, `RankMovement` (rank-change arrows) |
| `components/metrics/` | `MetricsPanel`, `MetricsChart` (before/after reranking), `CalculationSteps` (worked math) |
| `components/answer/` | `AnswerPanel`, `Citation` |
| `components/documents/` | `DocumentList`, `UploadDropzone` |
| `components/common/` | Reusable `Button`, `Card`, `Spinner`, `ErrorBanner` |
| `hooks/` | `useSearch`, `useTheme`, `useDocuments` |
| `types/` | TypeScript types matching the backend schemas |
| `utils/format.ts` | Formatting helpers |
| `styles/index.css` | Global styles and Tailwind directives |

---

## 6. Rules for this project

- Secrets (API keys) live only in a local `.env`, never in code or git.
- Nothing is committed or changed without explicit approval.
- The README is updated after every step.
