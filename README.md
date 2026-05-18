# ResearchRAG

Evidence-aware RAG system for querying ACL Anthology research papers with cited, section-targeted answers — designed for domain-specific retrieval where generic copilots fall short. Runs fully local (no data egress, GDPR-friendly).

## Features

- **Retrieval** — Hybrid search (FAISS HNSW + BM25/RRF), adaptive section filtering, FAISS IDSelector pre-filtering
- **Chunking** — HybridStructuredChunker: section-aware, citation-preserving, paragraph-based splitting
- **Re-ranking** — Cross-encoder re-ranking (BAAI/bge-reranker-v2-m3, 568M params, MPS/CUDA/CPU auto-detect)
- **Agentic** — Flat LangGraph StateGraph with query decomposition, multi-query RRF retrieval, tool loop, and MemorySaver conversational memory
- **HITL gate** — Human-in-the-loop approval before synthesis; annotated rejections loop back with a search hint; capped at 3 attempts
- **Observability** — LangSmith tracing (automatic for all LangChain/LangGraph calls via env vars)
- **Evaluation** — DeepEval suite with Gemini-as-judge; Faithfulness 0.99, Answer Relevancy 0.87 (n=10, qwen3:14b)

## Quick Start

```bash
git clone <repository-url> && cd ResearchRAG
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 1. Download papers

```bash
python ingestion/data/download_papers.py
# Downloads ACL, EMNLP, NAACL, EACL, COLING 2025 (~200 papers/venue → 842 total)
# Or: place PDFs manually in data/acl/
```

### 2. Build indices

```bash
# Adaptive chunking (152K section-aware chunks — default, recommended)
python ingestion/data/process_papers.py

# Pre-built artifacts are committed: artifacts/adaptive_chunks.json,
# adaptive_faiss_index, adaptive_bm25
```

### 3. Run the app

```bash
# Streamlit chat interface (primary)
streamlit run app.py
# Opens at http://localhost:8501
# Pick model and retrieval strategy in the sidebar.
```

Or use the CLI:

```bash
# Interactive conversational mode
python -m src.agentic_main

# Single-shot query
python -m src.agentic_main -q "What are the main approaches to efficient attention?"

# With reflection pass (post-answer citation check)
python -m src.agentic_main -q "..." --reflect

# Lighter model for faster responses
python -m src.agentic_main --ollama-model qwen3:8b
```

> **LangSmith tracing**: set these env vars to enable automatic observability:
>
> ```bash
> export LANGCHAIN_TRACING_V2=true
> export LANGCHAIN_API_KEY=<your_key>
> export LANGCHAIN_PROJECT=ResearchRAG
> ```

### 4. Build indices (Docker)

```bash
docker compose up --build
# Ollama sidecar mounts ~/.ollama; pull your model first with: ollama pull qwen3:14b
```

### 5. Evaluate

```bash
# DeepEval suite (Gemini-as-judge, requires GEMINI_API_KEY)
python experiments/eval_agent.py

# 10-turn conversational eval
python experiments/agent_eval_long.py

# Latency benchmark
python experiments/eval_retrieval.py
```

## Architecture

```text
Query
  │
  ▼
decompose ── QueryDecomposer (LLM + heuristic fallback) → sub-queries
  │
  ▼
retrieve ── _search() × N sub-queries → RRF merge
  │
  ▼
rerank ── CrossEncoderReranker (bge-reranker-v2-m3)
  │
  ▼
human_approval ── interrupt() ──► approved → llm_node
                              ──► feedback  → decompose (retry, max 3)
                              ──► abort     → END
  │
  ▼
llm_node ── ChatOllama + bind_tools → answer or tool call
  │  ▲
  │  └── tool ── ToolNode (search_papers_multi / search_papers /
  │               search_papers_in_section / detect_relevant_sections / search_pubmed)
  ▼
END
```

**State** (ResearchRAGState): `messages` persisted by MemorySaver across turns; all other fields are ephemeral and reset each turn.

## Project Structure

```
ResearchRAG/
├── app.py                        # Streamlit chat interface (primary)
├── src/
│   ├── agentic_main.py           # CLI entrypoint
│   ├── rag_chain.py              # RAGChain — embeddings, retriever, reranker, LLM
│   ├── embeddings.py             # BAAI/bge-base-en-v1.5 (768d, LRU cache)
│   ├── agentic/
│   │   ├── graph.py              # ResearchRAGGraph class, build_graph, run_graph,
│   │   │                         #   checkpoint_reflection, reflect_on_answer
│   │   ├── state.py              # ResearchRAGState TypedDict
│   │   ├── tools.py              # make_tools(rag_chain) factory — 5 tool closures
│   │   ├── decomposer.py         # QueryDecomposer (LLM + heuristic fallback)
│   │   └── safety.py             # Input sanitisation + injection-resistant wrapping
│   └── retrieval/
│       ├── semantic.py           # FAISS HNSW/FlatIP + IDSelector filtering
│       ├── bm25.py               # BM25 (bm25s, mmap persistence)
│       ├── hybrid.py             # HybridRetriever (RRF fusion)
│       ├── reranker.py           # CrossEncoderReranker (MPS/CUDA/CPU, score cache)
│       └── query_parser.py       # Metadata filter extraction from natural language
├── ingestion/                    # One-time index building — not needed at query time
│   ├── chunking/
│   │   ├── basic.py              # Fixed-size, paragraph-aware
│   │   ├── semantic.py           # Embedding-based boundary detection
│   │   └── hybrid_structured.py  # Section + citation aware (default)
│   └── data/
│       ├── acl_loader.py         # ACL Anthology metadata
│       ├── download_papers.py    # Batch PDF downloader
│       ├── loader.py             # PDF processing (PyMuPDF)
│       ├── structure_utils.py    # Section/citation/paragraph detection
│       └── process_papers.py    # Chunk → embed → index pipeline
├── experiments/                  # Eval scripts (not a test suite)
│   ├── eval_agent.py             # DeepEval suite (Gemini-as-judge)
│   ├── agent_eval_long.py        # 10-turn conversational eval
│   └── eval_retrieval.py         # Latency benchmarks
├── artifacts/                    # Generated — not in git
│   ├── adaptive_faiss_index      # 152K chunk HNSW index
│   ├── adaptive_chunks.json      # Chunk text + metadata
│   └── adaptive_bm25             # BM25 sparse index
└── data/                         # PDFs and metadata
```

## Tech Stack

| Component | Details |
|---|---|
| Vector DB | FAISS (HNSW M=32, efC=64) |
| Embeddings | BAAI/bge-base-en-v1.5 (768d, 512 tokens) |
| Sparse search | bm25s (sparse matrices, mmap) |
| Re-ranking | BAAI/bge-reranker-v2-m3 (568M params, MPS auto-detect) |
| Chunking | HybridStructuredChunker (section + citation aware) |
| LLM | ChatOllama — default `qwen3:14b`; `qwen3:8b` for lighter use |
| Agentic | LangGraph flat `StateGraph` + `MemorySaver` + `interrupt()` for HITL |
| Observability | LangSmith (automatic tracing via env vars) |
| Data | ACL Anthology — 842 papers, 152K adaptive chunks |
