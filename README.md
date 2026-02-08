# ResearchRAG

An evidence-aware RAG (Retrieval-Augmented Generation) system for querying and analyzing scientific research papers with proper citations and source attribution.

## Overview

ResearchRAG enables you to query a collection of research papers and get well-cited, evidence-backed answers. The system uses hybrid retrieval (semantic + BM25), cross-encoder re-ranking, and metadata filtering to find the most relevant information, then generates comprehensive answers with proper source citations.

## Features

- **Hybrid Retrieval**: Semantic search (FAISS FlatIP) + BM25 (bm25s) with Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Re-ranking**: Uses cross-encoder models to improve retrieval precision with normalized scores
- **FAISS IDSelector Pre-filtering**: Metadata filters applied during search, not post-retrieval
- **Cited Answers**: Answers include numbered citations linking back to source papers
- **Streaming Output**: Real-time token streaming with markdown formatting and performance metrics
- **Multiple Retrieval Strategies**: Choose between semantic-only, BM25-only, or hybrid search
- **LLM Provider Auto-detection**: Tries Ollama first, falls back to HuggingFace automatically
- **KV Prompt Caching**: Cached system prompt KV for faster TFFT across conversation turns (HuggingFace)
- **Conversational RAG**: Multi-turn conversations with context preservation, query rewriting, and reference resolution
- **Agentic RAG**: Two-agent system for complex reasoning, comparison, and gap detection across papers
- **Retrieval Benchmarking**: Latency benchmarks for all retrieval paths (Semantic, BM25, Hybrid, Filtered)

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ResearchRAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Setup

1. **Set up HuggingFace token** (if using gated models):
   ```bash
   export HF_TOKEN="your_huggingface_token_here"
   # Or inline: HF_TOKEN="your_token" python src/main.py ...
   ```

2. **Download papers from ACL Anthology**:
   ```bash
   python src/data/download_papers.py
   ```
   Downloads papers from ACL, EMNLP, NAACL, EACL, and COLING (2025, 200 per venue).

   Or download programmatically:
   ```python
   from src.data.acl_loader import ACLAnthologyLoader

   loader = ACLAnthologyLoader(data_dir="data/acl")
   loader.download_by_venue(venue="ACL", year=2025, limit=200, save_metadata=True)
   ```

   Alternatively, manually download PDFs and place them in `ResearchRAG/data/acl/` directory.

3. **Process papers and build index**:
   ```bash
   python src/data/process_papers.py
   ```

   This will:
   - Extract text and metadata from PDFs in `data/acl/`
   - Chunk papers (1500 chars, 300 overlap) with paragraph-aware text cleaning
   - Generate embeddings (BAAI/bge-base-en-v1.5, 768d) and build FAISS FlatIP index
   - Build BM25 index (bm25s with disk persistence)
   - Save to `artifacts/` (faiss_index, bm25_index, chunks.json)

   Optional flags:
   ```bash
   python src/data/process_papers.py --chunk-size 1500 --chunk-overlap 300 --metric IP --index-type flat --limit 100
   ```

### Usage

**Basic query (auto-detects Ollama, falls back to HuggingFace):**
```bash
python src/main.py -q "What are the main approaches to few-shot learning?"
```

**Force a specific LLM provider:**
```bash
# Force Ollama
python src/main.py -q "What are transformers?" --llm-provider ollama --ollama-model qwen2:1.5b

# Force HuggingFace
python src/main.py -q "What are transformers?" --llm-provider huggingface --llm-model Qwen/Qwen2-1.5B-Instruct
```

**With hybrid retrieval (RRF fusion) and re-ranking:**
```bash
python src/main.py \
  -q "How do transformer models handle long sequences?" \
  --retrieval hybrid \
  --fusion rrf \
  --top-k 5 \
  --initial-retrieval-k 20 \
  --rerank-k 3
```

**With metadata filtering:**
```bash
python src/main.py \
  -q "What methods are used for evaluation in ACL 2025 papers?" \
  --conference "ACL" \
  --year "2025"
```

**Auto-parse filters from query:**
The system automatically detects filters in natural language queries:
```bash
python src/main.py -q "Find papers from ACL 2024 about transformers"
# Automatically filters: conference=ACL, year=2024, title contains "transformer"
```

**Conversational mode:**
```bash
python src/conversation_main.py --retrieval hybrid
```
Starts an interactive session with multi-turn conversations. Maintains context across turns and resolves references like "it", "that method", "the last 2 papers", etc. Commands: `clear`, `history`, `exit`.

**Agentic mode:**
```bash
python src/agentic_main.py -q "Compare GAPO and PPO methods"
```
Uses a two-agent system (Retriever + Reasoner) for complex reasoning tasks like comparison, gap detection, and synthesis across multiple papers.

**Retrieval benchmarking:**
```bash
python src/benchmark_retrieval.py
```
Benchmarks all retrieval paths (Semantic, BM25, Hybrid RRF, Filtered) with latency metrics (Avg, P50, P99).

**Note**: Streaming is enabled by default. All queries stream tokens in real-time with markdown formatting.

## Architecture

```
┌─────────────┐
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Query Parser   │ (extracts metadata filters)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Hybrid Retriever│ ──► Semantic (FAISS) + BM25
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   Re-ranker     │ (cross-encoder)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│      LLM        │ (generates answer with citations)
└─────────────────┘
```

## Project Structure

```
ResearchRAG/
├── src/
│   ├── main.py                # CLI interface (standalone mode)
│   ├── conversation_main.py   # CLI interface (conversational mode)
│   ├── agentic_main.py        # CLI interface (agentic mode)
│   ├── benchmark_retrieval.py # Retrieval latency benchmarks
│   ├── rag_chain.py           # RAG query chain (LangChain-based, auto provider cascade)
│   ├── embeddings.py          # Embedding generation (BAAI/bge-base-en-v1.5)
│   ├── utils.py               # Utility functions
│   ├── data/                  # Data ingestion and processing
│   │   ├── loader.py          # PDF processing (PyMuPDF)
│   │   ├── acl_loader.py      # ACL Anthology data loader
│   │   ├── download_papers.py # Paper download script (5 venues, 200/venue)
│   │   └── process_papers.py  # Paper processing pipeline (chunk + embed + index)
│   ├── retrieval/             # Retrieval strategies
│   │   ├── semantic.py        # Semantic search (FAISS FlatIP/HNSW, IDSelector filtering)
│   │   ├── bm25.py            # BM25 keyword search (bm25s, mmap persistence)
│   │   ├── hybrid.py          # Hybrid retrieval (RRF + weighted fusion)
│   │   ├── reranker.py        # Cross-encoder re-ranking
│   │   └── query_parser.py    # Metadata filter parsing
│   ├── conversation/          # Conversational RAG components
│   │   ├── history.py         # Conversation history management
│   │   ├── query_rewriter.py  # Query rewriting with context
│   │   └── conversation_rag.py # Conversational RAG chain wrapper
│   ├── agentic/               # Agentic RAG components
│   │   ├── base_agent.py      # Base agent framework
│   │   ├── retriever_agent.py # Retrieval agent
│   │   ├── reasoner_agent.py  # Reasoning agent
│   │   └── orchestrator.py    # Multi-agent orchestrator
│   └── chunking/              # Chunking strategies
│       ├── basic.py           # Fixed-size chunking (paragraph-aware cleaning)
│       └── semantic.py        # Semantic chunking
├── artifacts/                 # Generated files (faiss_index, bm25_index, chunks.json)
├── data/                      # PDF files and metadata
├── docs/                      # Documentation
└── requirements.txt
```

## 🔧 Configuration

### Retrieval Strategies

- `semantic`: Embedding-based similarity search (FAISS) - default
- `bm25`: Keyword-based BM25 search
- `hybrid`: Weighted combination of semantic + BM25

### Distance Metrics

- `IP`: Inner product (cosine similarity) - default, normalized scores 0-1
- `L2`: Euclidean distance

### Re-ranking

- Uses cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Scores are normalized to 0-1 range (min-max scaling)
- Both original retrieval scores and reranked scores are displayed

### LLM Provider Cascade

- **Default**: `auto` (tries Ollama first, falls back to HuggingFace)
- **Ollama**: `qwen2:1.5b` (default) - fast inference via llama.cpp backend
- **HuggingFace**: `Qwen/Qwen2-1.5B-Instruct` (default) - auto-downloads, MPS/CUDA/CPU support
- KV prompt caching available for HuggingFace models (faster TFFT on follow-up queries)

## Performance Metrics

- **Time to First Token (TFFT)**: Measures latency before first token generation
- **Total Generation Time**: End-to-end query processing time
- **Score Display**: Both original retrieval scores and normalized reranked scores (0-1 range)

## Tech Stack

- **Framework**: LangChain (chains for retrieval pipeline, wrappers for LLM)
- **Vector DB**: FAISS (FlatIP for cosine similarity, HNSW support)
- **Embeddings**: BAAI/bge-base-en-v1.5 (768d, 512 max tokens)
- **BM25**: bm25s (sparse matrices, disk persistence via mmap)
- **Re-ranking**: Cross-encoder models (cross-encoder/ms-marco-MiniLM-L-6-v2)
- **LLM**: Auto-cascade: Ollama (preferred) -> HuggingFace Transformers (fallback)
- **Data Source**: ACL Anthology (~842 papers, 54K chunks across 5 venues)

## Development Roadmap

For detailed development plans, experiments, and future features, see [PLAN.md](PLAN.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This is a research project focused on improving RAG systems for academic paper analysis. The system is designed to handle ACL Anthology papers but can be adapted for other paper collections.
