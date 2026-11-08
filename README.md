# RAG Explorer — Advanced RAG Pipeline with Visible Retrieval Metrics

 **how a RAG system finds the correct information**. A user asks a question about the loaded documents, and the UI shows every retrieval stage and how well each one performed.

> **Status:** Step 1 (folder structure) is done. All code files exist but are **empty placeholders**. The "Planned contents" below describe what each file will hold; they are not implemented yet.

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

## 4. Build steps

| # | Step | Status |
|---|---|---|
| 1 | Project scaffold (folders and empty files) | Done |
| 2 | Document loading and chunking | Pending |
| 3 | Embeddings, BM25, hybrid search, reranker | Pending |
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
| `data/documents/` | Source documents to index | Your TXT/PDF/DOCX files |
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
| `config.py` | Settings loaded from environment variables |
| `logging.py` | Logging setup |
| `exceptions.py` | Custom errors and their HTTP mappings |
| `constants.py` | Fixed values (default k, RRF constant, relevance scale) |

#### `app/schemas/` (request/response models)
| File | Planned contents |
|---|---|
| `document.py` | Document and chunk models |
| `search.py` | Search request; ranked-chunk and per-stage results |
| `answer.py` | Answer and citation models |
| `evaluation.py` | Metric results and calculation-step models |

#### `app/services/` (orchestration)
| File | Planned contents |
|---|---|
| `document_service.py` | Manage the document files on disk |
| `indexing_service.py` | Load -> chunk -> embed -> build the vector store and BM25 index |
| `search_service.py` | Rewrite query -> hybrid retrieve -> rerank; return all stages |
| `answer_service.py` | Build the prompt from top chunks and get the cited answer |
| `evaluation_service.py` | Get relevance labels and compute metrics for each stage |

#### `app/ingestion/` (getting text in)
| File | Planned contents |
|---|---|
| `loaders.py` | Read TXT, PDF, and DOCX into plain text |
| `chunker.py` | Parent-child chunking with overlap |

#### `app/retrieval/` (finding relevant chunks)
| File | Planned contents |
|---|---|
| `embedder.py` | Turn text into 384-dimension vectors |
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
| `file_utils.py` | File and path helpers |
| `text_utils.py` | Text cleaning and normalization |
| `timing.py` | Stage timing |

#### `tests/`
| Path | Purpose |
|---|---|
| `conftest.py` | Shared test fixtures |
| `unit/test_loaders.py`, `test_chunker.py`, `test_hybrid.py`, `test_metrics.py` | Test each module in isolation |
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
