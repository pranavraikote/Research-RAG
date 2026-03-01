# ResearchRAG

Evidence-aware RAG system for querying ACL Anthology research papers with cited answers.

## Features

- **Retrieval** — Hybrid search (FAISS HNSW + BM25/RRF), adaptive section filtering, FAISS IDSelector pre-filtering
- **Chunking** — HybridStructuredChunker: section-aware, citation-preserving, paragraph-based splitting with neighbor context expansion
- **Re-ranking** — Cross-encoder re-ranking (BAAI/bge-reranker-v2-m3, 568M params) with min-max normalized scores
- **Generation** — Cited streaming answers, auto LLM cascade (Ollama → HuggingFace), KV prompt caching
- **Conversational** — Multi-turn history, heuristic + LLM query rewriting, pronoun/citation reference resolution
- **Agentic** — LangGraph ReAct agent with query decomposition, multi-query RRF retrieval, citation enforcement, prompt injection protection, and conversational memory (MemorySaver)
- **Observability** — LangSmith tracing (automatic for all LangChain/LangGraph calls via env vars)
- **Evaluation** — Human-labeled precision eval (MRR, P@K, nDCG@10) and latency benchmarks — see [EVAL.md](EVAL.md)

## Quick Start

```bash
git clone <repository-url> && cd ResearchRAG
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 1. Download papers

```bash
python src/data/download_papers.py
# Downloads ACL, EMNLP, NAACL, EACL, COLING 2025 (200 papers/venue)
# Or: place PDFs manually in data/acl/
```

### 2. Build indices

```bash
# Basic chunking (54K chunks — better retrieval precision)
python src/data/process_papers.py

# Adaptive chunking (152K section-aware chunks — better generation quality)
# Pre-built: artifacts/adaptive_chunks.json, adaptive_faiss_index, adaptive_bm25
```

### 3. Query

```bash
# Standalone query (auto-detects Ollama, falls back to HuggingFace)
python src/main.py -q "What are the main approaches to few-shot learning?"

# Hybrid retrieval + re-ranking
python src/main.py -q "How do transformers handle long sequences?" \
  --retrieval hybrid --fusion rrf --top-k 5 --rerank-k 3

# With adaptive indices (auto-detects section keywords and filters)
python src/main.py -q "What methods are used for training transformers?" \
  --retrieval hybrid \
  --chunks-path artifacts/adaptive_chunks.json \
  --index-path artifacts/adaptive_faiss_index

# Metadata filtering (also auto-parsed from natural language)
python src/main.py -q "Find ACL 2025 papers about transformers"

# Conversational mode (multi-turn, with query rewriting)
python src/conversation_main.py --retrieval hybrid

# Agentic mode — interactive conversational ReAct agent (recommended)
python src/agentic_main.py --react --retrieval hybrid --ollama-model qwen2.5:7b

# Agentic mode — single-shot query
python src/agentic_main.py --react --retrieval hybrid --ollama-model qwen2.5:7b \
  -q "Compare linear attention and standard attention methods and results"
```

> **LangSmith tracing**: add the following to your `.envrc` or shell environment to enable automatic observability for all agent runs:
> ```bash
> export LANGCHAIN_TRACING_V2=true
> export LANGCHAIN_API_KEY=<your_key>   # smith.langchain.com → Settings → API Keys
> export LANGCHAIN_PROJECT=ResearchRAG
> ```

### 4. Evaluate

```bash
# Latency benchmark across all strategy × chunking combinations
python experiments/benchmark.py

# Precision eval — interactive human labeling then metric computation
python experiments/evaluate.py --label --retrieval hybrid
python experiments/evaluate.py --eval --labels experiments/labels.json

# Chunking comparison (basic vs adaptive)
python experiments/evaluate.py --compare
```

## Architecture

**Standalone / Conversational**
```
Query → Query Parser (metadata filters) → Hybrid Retriever (FAISS + BM25)
      → Re-ranker (cross-encoder) → LLM (cited streaming answer)
```

**Agentic (ReAct)**
```
Query → Safety sanitizer → ReAct Agent (LangGraph + MemorySaver)
          ├── search_papers_multi  →  Decomposer → N×Retriever → RRF merge → Reranker
          ├── search_papers        →  Retriever → Reranker
          ├── search_papers_in_section → Section-filtered Retriever → Reranker
          └── detect_relevant_sections → Keyword mapper
        → Cited structured answer (## Summary / ## Key Findings / ## Limitations)
```

## Project Structure

```
ResearchRAG/
├── src/
│   ├── main.py                    # Standalone query CLI
│   ├── conversation_main.py       # Conversational CLI
│   ├── agentic_main.py            # Agentic CLI
│   ├── rag_chain.py               # RAG pipeline (LangChain)
│   ├── embeddings.py              # BAAI/bge-base-en-v1.5 (768d)
│   ├── data/
│   │   ├── acl_loader.py          # ACL Anthology downloader
│   │   ├── download_papers.py     # Batch download script
│   │   ├── loader.py              # PDF processing (PyMuPDF)
│   │   ├── structure_utils.py     # Section/citation/paragraph detection
│   │   └── process_papers.py     # Chunk → embed → index pipeline
│   ├── retrieval/
│   │   ├── semantic.py            # FAISS HNSW/FlatIP + IDSelector filtering
│   │   ├── bm25.py                # BM25 (bm25s, mmap persistence)
│   │   ├── hybrid.py              # RRF + weighted fusion
│   │   ├── adaptive_retriever.py  # Section keyword detection + context expansion
│   │   ├── reranker.py            # Cross-encoder (bge-reranker-v2-m3)
│   │   └── query_parser.py        # Metadata filter parsing
│   ├── chunking/
│   │   ├── basic.py               # Fixed-size, paragraph-aware
│   │   └── hybrid_structured.py   # Section-aware, citation-aware
│   ├── conversation/              # Multi-turn history, heuristic + LLM query rewriting
│   └── agentic/
│       ├── react_agent.py         # LangGraph create_react_agent + MemorySaver
│       ├── tools.py               # search_papers_multi, search_papers, search_papers_in_section
│       ├── decomposer.py          # LLM query decomposition with heuristic fallback
│       ├── safety.py              # Input sanitization + injection-resistant content wrapping
│       └── orchestrator.py        # Legacy v1 two-agent orchestrator
├── experiments/
│   ├── benchmark.py               # Latency ablation
│   └── evaluate.py                # Precision eval (label + eval + compare)
├── artifacts/                     # FAISS index, BM25 index, chunks.json
└── data/                          # PDFs and metadata
```

## Tech Stack

| Component | Details |
|---|---|
| Vector DB | FAISS (HNSW M=32, efC=64; FlatIP fallback) |
| Embeddings | BAAI/bge-base-en-v1.5 (768d, 512 tokens) |
| Sparse search | bm25s (sparse matrices, mmap) |
| Re-ranking | BAAI/bge-reranker-v2-m3 (568M params) |
| Chunking | HybridStructuredChunker (section + citation aware) |
| LLM | ChatOllama (langchain-ollama) → ChatHuggingFace (fallback); both support bind_tools |
| Agentic | LangGraph `create_react_agent` + `MemorySaver` |
| Observability | LangSmith (automatic tracing via env vars) |
| Data | ACL Anthology — 842 papers, 152K adaptive chunks |
