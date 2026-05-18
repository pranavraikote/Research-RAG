# ResearchRAG Development Plan

This document tracks the development roadmap for ResearchRAG, an evidence-aware RAG system for querying and analysing scientific research papers.

## Technology Stack

| Component | Details |
| --- | --- |
| Framework | LangChain + LangGraph |
| Vector DB | FAISS (HNSW M=32, efC=64) |
| Embeddings | BAAI/bge-base-en-v1.5 (768d, 512 max tokens) |
| Sparse search | bm25s (sparse matrices, mmap persistence) |
| Reranker | BAAI/bge-reranker-v2-m3 (568M params, MPS/CUDA/CPU auto-detect) |
| Chunking | HybridStructuredChunker (section-aware, citation-aware, paragraph-based) |
| LLM | ChatOllama — default `qwen3:14b`; `qwen3:8b` lighter alternative |
| Agentic | LangGraph flat `StateGraph` + `MemorySaver` + `interrupt()` for HITL |
| Observability | LangSmith (automatic tracing via env vars) |
| Primary interface | Streamlit chat UI (`app.py`) |
| Corpus | ACL Anthology — 842 papers, 152K adaptive chunks |

---

## Development Phases

### Phase 1 — Traditional RAG Foundation

**Status**: Complete

- Project setup and structure
- ACL Anthology data integration and PDF processing pipeline
- Basic chunking, embedding generation, FAISS indexing
- Basic RAG query chain with semantic search
- CLI interface with streaming output, source citations, metadata filtering

### Phase 2 — Re-ranking and Improved Retrieval

**Status**: Complete

- Cross-encoder reranker (BAAI/bge-reranker-v2-m3) with min-max score normalisation
- Hybrid search (weighted combination + RRF fusion strategies)
- Metadata filtering with auto-parsing and FAISS IDSelector pre-filtering
- BM25 via bm25s (sparse matrices, mmap disk persistence)
- BAAI/bge-base-en-v1.5 embeddings (768d) with HNSW index
- Retrieval latency benchmark tooling

### Phase 3 — Advanced Chunking

**Status**: Complete

- HybridStructuredChunker: section-aware, citation-aware, paragraph-based chunking with automatic structure detection
- Adaptive retrieval with section filtering and context expansion via neighbouring chunks
- Reranker upgraded to BAAI/bge-reranker-v2-m3 (SOTA, 568M params)
- Full corpus indexed: 842 PDFs → 152K adaptive chunks (FAISS HNSW + BM25)
- Chunking ablation study: MRR, nDCG@10, P@5/P@10 across all configurations (see EVAL.md)

### Phase 4 — Conversational RAG

**Status**: Complete (module since removed in simplification)

- Multi-turn history with token counting and turn-based truncation
- Heuristic query rewriting: pronoun resolution, entity substitution, context expansion
- Citation reference resolution for follow-up queries
- LLM-based query rewriting for complex coreference

### Phase 5 — Agentic RAG

**Status**: Complete

**Motivation**: Precision eval revealed adaptive chunking fragments complex queries across section chunks. The fix is smarter querying via multi-query decomposition, not simpler chunking.

- LangGraph `create_react_agent` with `bind_tools` replacing custom orchestrator
- `search_papers`, `search_papers_in_section`, `search_papers_multi`, `detect_relevant_sections`, `search_pubmed` tools
- Retriever-agnostic tool layer — unified `_search()` routes across HybridRetriever, SemanticRetriever, BM25Retriever
- Conversational memory via `MemorySaver` checkpointer — full message history persisted per `thread_id`
- Query decomposition (LLM + heuristic fallback) with section hints
- Multi-query RRF retrieval via `search_papers_multi`
- Prompt injection protection via `safety.py`
- Reflection pass (`reflect_on_answer`) for post-answer citation checks
- PubMed integration via NCBI E-utilities

### Phase 6 — UI and Production

**Status**: Complete

- Streamlit chat UI (`app.py`) — streaming, live agent status, session management, model selector
- Reranker on MPS with float16 — ~2× faster on Apple Silicon
- Default LLM upgraded to `qwen3:14b`
- Reranker score cache — `(query, chunk_id)` keyed; skips redundant forward passes
- Thinking mode toggle — `/no_think` prefix for Qwen3; off by default
- Docker setup — `Dockerfile`, `docker-compose.yml`, Ollama sidecar

### Phase 7 — Evaluation and Credibility

**Status**: In Progress

- DeepEval suite with Gemini-as-judge (custom `DeepEvalBaseLLM` wrapper)
- Faithfulness 0.99, Answer Relevancy 0.87, Citation Quality 0.99, Refusal Correctness 0.90 (qwen3:14b, n=10)
- [ ] Full eval run at n=200 (currently validated at n=10–20)
- [ ] Human spot-checks on judge disagreements
- [ ] BEIR or MTEB retrieval benchmarks against published baselines
- [ ] Per-question thread isolation in `eval_agent.py` (shared thread_id currently causes memory reuse)

### Phase 8 — Controllable Agentic Execution

**Status**: Complete

This cycle: cleaned up the architecture in preparation for eventual API wrapping and multi-tenancy.

**Flat StateGraph** (replacing `create_react_agent`):

Pipeline made explicit as a flat `StateGraph` — no black-box nesting, every node visible and testable:

```text
START → decompose → retrieve → rerank → human_approval → llm_node ⇄ tool (max 5)
                                               │
                               (feedback, max 3 attempts)
                                               ↓
                                          decompose (retry)
                                               │
                                          (exhausted)
                                               ↓
                                             END
```

- `decompose` — `QueryDecomposer` breaks the query into focused sub-queries; resets ephemeral state; incorporates `retrieval_feedback` hint when retrying
- `retrieve` — `_search()` × N sub-queries, RRF merge
- `rerank` — `CrossEncoderReranker`, top 5 chunks
- `human_approval` — HITL gate via `interrupt()`; resume value: `True` = approve, `False` = abort, `str` = feedback hint → retry
- `llm_node` — `ChatOllama.bind_tools(tools)`, system message rebuilt each iteration from `reranked_chunks` (not stored in state)
- `tool` — `ToolNode`, appends `ToolMessage` to state
- `exhausted` — fires after 3 rejected retrievals; appends a graceful `AIMessage` and ends

**State fields** (`ResearchRAGState`):

| Field | Persisted | Description |
| --- | --- | --- |
| `messages` | ✓ MemorySaver | Full conversation history |
| `query` | ephemeral | Raw user query for this turn |
| `sub_queries` | ephemeral | Decomposed sub-queries |
| `chunks` | ephemeral | Raw retrieval results |
| `reranked_chunks` | ephemeral | Cross-encoder scored chunks |
| `tool_iterations` | ephemeral | Guard counter for tool loop |
| `human_approved` | ephemeral | Approval gate flag |
| `retrieval_feedback` | ephemeral | User hint for retry; cleared after decompose |
| `retrieval_attempts` | ephemeral | Retry loop counter; capped at 3 |

**`ResearchRAGGraph` class** (replacing module globals):

Components (`_llm_with_tools`, `_decomposer`, `_rag_chain`) are now instance attributes. Multiple graph instances in the same process are fully independent — required for any multi-tenant deployment.

**`make_tools(rag_chain)` factory** (replacing `set_rag_chain()` mutation):

All five tools are closures created per call, each capturing their own `rag_chain`. `_search()` takes `rag_chain` as an explicit parameter. No module-level state in `tools.py`.

**`checkpoint_reflection()`**:

After `reflect_on_answer()`, the critique is persisted into the MemorySaver checkpoint as a `SystemMessage` via `graph.update_state()`, making it visible to the LLM on the next turn.

---

## Credibility Checklist

Before ResearchRAG can be positioned as a production product:

1. **Complete the 200-query eval** — full run with human spot-checks on judge disagreements
2. **Scale proof** — demonstrate indexing and querying at 10K, 50K, 100K documents with latency and accuracy at each scale
3. **Production infrastructure** — API layer, auth, multi-tenancy; Docker done, single-user only
4. **Standardised benchmarks** — BEIR or MTEB retrieval results to compare against published baselines
5. **Domain validation** — deploy on a non-NLP corpus (legal, medical, financial) and prove section-aware retrieval generalises

---

## Data Sources

**Current corpus** (ACL Anthology, 2025 conferences, ~200 papers/venue):

- **Papers**: 842 — ACL, EMNLP, NAACL, EACL, COLING
- **Adaptive chunks**: 152,021 (HybridStructuredChunker, default)
- **Basic chunks**: 54,264 (fixed-size, benchmarking only)

**Future**: NeurIPS, ICML, arXiv (cs.CL, cs.AI, cs.LG)
