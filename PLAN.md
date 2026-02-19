# ResearchRAG Development Plan

This document outlines the development roadmap for ResearchRAG, an evidence-aware RAG system for querying and analyzing scientific research papers.

## Project Overview

ResearchRAG is being developed incrementally, starting with traditional RAG and advancing through re-ranking, advanced chunking strategies, conversational enhancements, and finally to an agentic RAG version.

**Technology Stack**:
- **Framework**: LangChain
- **Vector Database**: FAISS (HNSW with M=32, efConstruction=64, FlatIP fallback)
- **Embeddings**: BAAI/bge-base-en-v1.5 (768d, 512 max tokens)
- **BM25**: bm25s (sparse matrices, mmap persistence)
- **Reranker**: BAAI/bge-reranker-v2-m3 (568M params, multilingual SOTA)
- **Chunking**: HybridStructuredChunker (section-aware, citation-aware, paragraph-based)
- **Primary Data Source**: ACL Anthology
- **Deployment**: Local CLI-based interface

## Development Phases

### Phase 1: Traditional RAG Foundation

**Status**: Completed

**Goal**: Build a basic RAG system that can ingest research papers and answer questions about them.

**Completed**:
- [x] Project setup and structure
- [x] ACL Anthology data integration
- [x] PDF processing pipeline
- [x] Basic chunking strategy
- [x] Embedding generation and FAISS indexing
- [x] Basic RAG query chain with semantic search
- [x] CLI interface with streaming output
- [x] Source citations and metadata filtering

### Phase 2: Re-ranking and Improved Retrieval

**Status**: Completed

**Goal**: Improve retrieval quality by re-ranking results and using hybrid search.

**Completed**:
- [x] Cross-encoder reranker with min-max score normalisation and original/reranked score display
- [x] Hybrid search with weighted combination and RRF fusion strategies
- [x] Metadata filtering with auto-parsing and FAISS IDSelector pre-filtering
- [x] BM25 via bm25s (sparse matrices, mmap disk persistence)
- [x] BAAI/bge-base-en-v1.5 embeddings (768d) with HNSW index and FlatIP fallback
- [x] Retrieval latency benchmark tooling (Semantic, BM25, Hybrid, Filtered)

**Future**:
- [ ] Query expansion and reformulation (see Phase 5 for evaluation-backed motivation)

### Phase 3: Advanced Chunking Strategies

**Status**: Completed

**Goal**: Implement and systematically evaluate intelligent chunking strategies that preserve semantic structure.

**Completed**:
- [x] Basic fixed-size chunking (1500 chars, 300 overlap) and semantic chunking implementations
- [x] PDF text extraction cleanup (preserve Figure/Table captions, fix hyphenation, preserve paragraph boundaries)
- [x] HybridStructuredChunker: section-aware, citation-aware, paragraph-based chunking with automatic structure detection
- [x] Adaptive retrieval with section filtering and context expansion via neighbouring chunks
- [x] Updated reranker to BAAI/bge-reranker-v2-m3 (SOTA, 568M params)
- [x] Built adaptive indices for full corpus (842 PDFs → 152K chunks) with BM25/FAISS persistence and incremental update support
- [x] Chunking ablation study framework (basic vs adaptive, 2×2 with reranking)
- [x] Evaluation metrics: MRR, nDCG@10, P@5, P@10 across all configurations (see EVAL.md)

### Phase 4: Conversational RAG Enhancement

**Status**: Completed

**Goal**: Transform the standalone single-query RAG system into a conversational system that maintains context across multiple turns.

**Completed**:
- [x] Conversation history with token counting, turn-based truncation, and message summarisation
- [x] Heuristic query rewriting: pronoun resolution, entity substitution, context keyword expansion
- [x] Citation reference resolution (paper N / [N] → actual paper title) for follow-up queries
- [x] Multi-turn retrieval: follow-up detection, query boosting with discussed paper titles
- [x] Conversation context injected into prompt; prompt caching (KV cache) for HuggingFace provider
- [x] Interactive CLI mode (`conversation_main.py`) with session export/load support

**Future**:
- [ ] LLM-based query rewriting for complex coreference (stub exists, `use_llm_rewriting=False`)

### Phase 5: Agentic RAG

**Status**: Future

**Goal**: Build an agentic system that can reason across papers, compare findings, and identify gaps.

**Motivation from evaluation**

Precision eval (see [EVAL.md](EVAL.md)) revealed a fundamental limitation of single-query retrieval against adaptive chunks: complex queries asking about two linked concepts (problem + solution, method + evaluation) get fragmented because adaptive chunking section-splits the corpus. The KBQA query dropped from MRR=1.0 on basic chunks to MRR=0.25 on adaptive — not because the content is missing, but because it lives in separate section chunks that each score too low individually.

The fix is not simpler chunking — it's smarter querying. Adaptive chunking's section-level granularity is actually an asset when the system can issue multiple targeted sub-queries, each hitting its natural section type (intro for problem framing, methods for technical approach, results for empirical evidence). A multi-query or agentic architecture unlocks this:

```
Complex query
    ↓ decompose (LLM)
Sub-query 1: "X problem definition"    → intro/background sections
Sub-query 2: "Y method approach"       → methods sections
Sub-query 3: "Z evaluation results"    → results sections
    ↓ parallel AdaptiveRetriever calls
    ↓ RRF merge + dedup
    ↓ LLM sees focused, section-aligned chunks
```

This is the key insight: basic chunking is forgiving of single-query retrieval because paragraphs co-locate related ideas. Adaptive chunking is the right design for high-quality generation, but it requires the retrieval layer to match its granularity — either via multi-query decomposition or iterative agent-driven retrieval.

**Planned Features**:
- [ ] Query decomposition — break complex multi-part queries into focused sub-queries before retrieval
- [ ] Multi-query retrieval with RRF merge — parallel AdaptiveRetriever calls, one per sub-query
- [ ] Multi-agent architecture (retriever, analyzer, comparator, synthesizer)
- [ ] Cross-paper reasoning and comparison
- [ ] Structured output generation (claims, evidence, limitations)
- [ ] Gap detection in literature
- [ ] Iterative refinement and follow-up question generation

## Data Sources

**Current Corpus** (ACL Anthology, 2025 conferences, 200 papers/venue):
- **Papers**: 842 — ACL, EMNLP, NAACL, EACL, COLING
- **Basic chunks**: 54,264 (fixed-size, 1500 chars, 300 overlap)
- **Adaptive chunks**: 152,021 (HybridStructuredChunker, section-aware)

**Future sources**: NeurIPS, ICML, arXiv (cs.CL, cs.AI, cs.LG)
