# RAG Explorer — Advanced RAG Pipeline with Visible Retrieval Metrics

 **how a RAG system finds the correct information**. A user asks a question about the loaded documents, and the UI shows every retrieval stage and how well each one performed.

> **Status:** Steps 1 and 2 are done (folder structure; document loading and chunking). Step 3 is in progress: the embedder is built. Files marked **Implemented** below contain working code. Everything else is still an **empty placeholder**, and its "Planned contents" describe what it will hold.

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
| Vector store | NumPy / FAISS |
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
| `faiss-cpu` | Fast vector similarity search (the alternative to the plain NumPy store) |
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
- The reranker model (`ms-marco-MiniLM-L-6-v2`) downloads automatically the first time the reranker runs. The embedding model (`all-MiniLM-L6-v2`) is already downloaded into the Hugging Face cache, on first use of the embedder.
- API keys (`GROQ_API_KEY`, `GEMINI_API_KEY`) go in a local `backend/.env`, which git ignores. They are needed from step 4.

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
| 3 | Embeddings, BM25, hybrid search, reranker | In progress: embedder done (20 tests pass in total); vector store, BM25, hybrid search and reranker pending |
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
| `requirements.txt` | Python dependencies | fastapi, uvicorn, sentence-transformers, rank-bm25, faiss-cpu, pypdf, python-docx, groq, google-generativeai |
| `.env.example` | Template for secrets/settings | Variable names only (API keys, model names, chunk size). The real `.env` is git-ignored. |
| `data/documents/` | Source documents to index | **Loaded:** `cricket_guide.txt` (13 sections), `football_guide.docx` (12 sections, 6 tables), `hr_handbook.pdf` (18 sections, 6 pages, fictional company "Aurora Systems"). Together they produce 95 chunks. |
| `data/index/` | Saved embeddings and indexes | Generated; git-ignored |
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
| `config.py` | **Implemented.** `Settings` class (pydantic) reading env vars / `backend/.env`. Holds `documents_dir`, `index_dir`, `embedding_model` (all-MiniLM-L6-v2), `embedding_batch_size` (32), `chunk_size` (500 characters) and `chunk_overlap` (80). The shared `settings` object is imported everywhere. More settings are added in later steps. |
| `logging.py` | Logging setup |
| `exceptions.py` | **Implemented (partly).** `UnsupportedFileError` (not a TXT/PDF/DOCX file) and `DocumentLoadError` (file recognised but unreadable). HTTP mappings come with the routes in step 5. |
| `constants.py` | Fixed values (default k, RRF constant, relevance scale) |

#### `app/schemas/` (request/response models)
| File | Planned contents |
|---|---|
| `document.py` | **Implemented.** Three models. `Section` (title + text). `RawDocument` (name, real file type, list of sections). `Chunk` (stable `id` like `cricket_guide:3:1`, `doc_name`, `section_title`, `parent_id`, child `text`, full `parent_text`, `chunk_index`). `Chunk.embed_text()` returns "doc > section" plus the text, which is what gets embedded and indexed in step 3. |
| `search.py` | Search request; ranked-chunk and per-stage results |
| `answer.py` | Answer and citation models |
| `evaluation.py` | Metric results and calculation-step models |

#### `app/services/` (orchestration)
| File | Planned contents |
|---|---|
| `document_service.py` | **Implemented.** `load_all_documents(directory)` loads every readable file in the folder and logs and skips bad ones, so one broken file never stops the rest. `load_all_chunks(directory)` loads and chunks everything. Upload/delete come in step 5. |
| `indexing_service.py` | Load -> chunk -> embed -> build the vector store and BM25 index |
| `search_service.py` | Rewrite query -> hybrid retrieve -> rerank; return all stages |
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
| `vector_store.py` | Store vectors and run similarity search (NumPy/FAISS) |
| `bm25_index.py` | Keyword search index |
| `hybrid.py` | Merge dense and BM25 rankings with Reciprocal Rank Fusion |
| `reranker.py` | Cross-encoder rescoring of the top candidates |
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
| `text_utils.py` | **Implemented.** `clean_text` (normalise whitespace), `is_heading` (a line like `4. Company holidays`, not ending in a full stop), `split_sections` (cut text into titled sections; text before the first heading becomes "Introduction"), `split_sentences` (keeps decimals like 20.12 intact). |
| `timing.py` | Stage timing |

#### `tests/`
| Path | Purpose |
|---|---|
| `conftest.py` | Shared test fixtures |
| `unit/test_loaders.py` | **Implemented (7 tests).** Section splitting, the three real documents, running-header removal, DOCX table rows, a Word file renamed to `.pdf`, and rejection of binary junk |
| `unit/test_chunker.py` | **Implemented (7 tests).** Size limit, overlap, no lost text, long-sentence splitting, stable IDs, bad settings rejected, unique IDs across the real documents |
| `unit/test_embedder.py` | **Implemented (6 tests).** Shape (n, 384), float32 and unit length; dimension; empty input; a related sentence scores higher than an unrelated one; same text gives the same vector; all 95 real chunks embed in order |
| `unit/test_hybrid.py`, `test_metrics.py` | Empty until steps 3 and 4 |

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
