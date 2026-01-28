# ResearchRAG

An evidence-aware RAG (Retrieval-Augmented Generation) system for querying and analyzing scientific research papers with proper citations and source attribution.

## Overview

ResearchRAG enables you to query a collection of research papers and get well-cited, evidence-backed answers. The system uses hybrid retrieval (semantic + BM25), cross-encoder re-ranking, and metadata filtering to find the most relevant information, then generates comprehensive answers with proper source citations.

## Features

- **Hybrid Retrieval**: Combines semantic search (FAISS) and BM25 keyword search for better recall
- **Cross-Encoder Re-ranking**: Uses cross-encoder models to improve retrieval precision with normalized scores
- **Metadata Filtering**: Filter by conference, year, or paper title (with auto-parsing from queries)
- **Cited Answers**: Answers include numbered citations linking back to source papers
- **Streaming Output**: Real-time token streaming with markdown formatting and performance metrics
- **Multiple Retrieval Strategies**: Choose between semantic-only, BM25-only, or hybrid search
- **LangChain Integration**: Uses LangChain chains for composable retrieval pipeline
- **Conversational RAG**: Multi-turn conversations with context preservation and reference resolution
- **Agentic RAG**: Two-agent system for complex reasoning, comparison, and gap detection across papers

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
   
   You can download papers programmatically using Python:
   ```python
   from src.data.acl_loader import ACLAnthologyLoader
   
   loader = ACLAnthologyLoader(data_dir="data/acl")
   
   # Download papers from a specific venue and year
   loader.download_by_venue(venue="ACL", year=2025, limit=50)
   ```
   
   Or create a simple download script:
   ```python
   # download_papers.py
   from src.data.acl_loader import ACLAnthologyLoader
   
   loader = ACLAnthologyLoader(data_dir="data/acl")
   loader.download_by_venue(venue="ACL", year=2025, limit=100)
   ```
   
   **Note**: The `acl_anthology` package is required. Install it with:
   ```bash
   pip install acl-anthology
   ```
   
   Alternatively, manually download PDFs and place them in `ResearchRAG/data/acl/` directory.

3. **Process papers and build index**:
   ```bash
   python src/data/process_papers.py --data-dir data/acl --chunk-size 1000 --chunk-overlap 200
   ```
   
   This will:
   - Extract text and metadata from PDFs in `data/acl/`
   - Chunk the papers into smaller segments
   - Generate embeddings and build FAISS index
   - Save chunks to `artifacts/chunks.json` and index to `artifacts/faiss_index`

### Usage

**Basic query:**
```bash
python src/main.py --query "What are the main approaches to few-shot learning?"
```

**With hybrid retrieval and re-ranking:**
```bash
python src/main.py \
  --query "How do transformer models handle long sequences?" \
  --retrieval hybrid \
  --top-k 5 \
  --initial-retrieval-k 20 \
  --rerank-k 3
```

**With metadata filtering:**
```bash
python src/main.py \
  --query "What methods are used for evaluation in ACL 2025 papers?" \
  --conference "ACL" \
  --year "2025"
```

**Auto-parse filters from query:**
The system automatically detects filters in natural language queries:
```bash
python src/main.py --query "Find papers from ACL 2024 about transformers"
# Automatically filters: conference=ACL, year=2024, title contains "transformer"
```

**Conversational mode:**
```bash
python src/conversation_main.py
```
Starts an interactive session where you can have multi-turn conversations. The system maintains context across turns and resolves references like "it", "that method", "the last 2 papers", etc.

**Agentic mode:**
```bash
python src/agentic_main.py -q "Compare GAPO and PPO methods"
```
Uses a two-agent system (Retriever + Reasoner) for complex reasoning tasks like comparison, gap detection, and synthesis across multiple papers.

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
│   ├── main.py              # CLI interface (standalone mode)
│   ├── conversation_main.py # CLI interface (conversational mode)
│   ├── agentic_main.py      # CLI interface (agentic mode)
│   ├── rag_chain.py         # RAG query chain (LangChain-based)
│   ├── embeddings.py        # Embedding generation
│   ├── utils.py             # Utility functions
│   ├── data/                # Data ingestion and processing
│   │   ├── loader.py        # PDF processing
│   │   ├── acl_loader.py    # ACL Anthology data loader
│   │   └── process_papers.py # Paper processing pipeline
│   ├── retrieval/           # Retrieval strategies
│   │   ├── semantic.py      # Semantic search (FAISS)
│   │   ├── bm25.py          # BM25 keyword search
│   │   ├── hybrid.py        # Hybrid retrieval
│   │   ├── reranker.py      # Cross-encoder re-ranking
│   │   └── query_parser.py  # Metadata filter parsing
│   ├── conversation/        # Conversational RAG components
│   │   ├── history.py       # Conversation history management
│   │   ├── query_rewriter.py # Query rewriting with context
│   │   └── conversation_rag.py # Conversational RAG chain wrapper
│   ├── agentic/             # Agentic RAG components
│   │   ├── base_agent.py    # Base agent framework
│   │   ├── retriever_agent.py # Retrieval agent
│   │   ├── reasoner_agent.py # Reasoning agent
│   │   └── orchestrator.py  # Multi-agent orchestrator
│   └── chunking/            # Chunking strategies
│       ├── basic.py         # Fixed-size chunking
│       └── semantic.py      # Semantic chunking
├── artifacts/               # Generated files (index, chunks)
├── data/                    # PDF files and metadata
├── docs/                    # Documentation
├── examples/                # Example scripts
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

### LLM Models

- **Default**: `Qwen/Qwen2-1.5B-Instruct` (HuggingFace)
- Supports any HuggingFace model or Ollama models
- Prompt caching available for HuggingFace models (faster TFFT)

## Performance Metrics

- **Time to First Token (TFFT)**: Measures latency before first token generation
- **Total Generation Time**: End-to-end query processing time
- **Score Display**: Both original retrieval scores and normalized reranked scores (0-1 range)

## Tech Stack

- **Framework**: LangChain (chains for retrieval pipeline, wrappers for LLM)
- **Vector DB**: FAISS (with IP/L2 distance metrics)
- **Embeddings**: Sentence Transformers (HuggingFace)
- **Re-ranking**: Cross-encoder models (sentence-transformers)
- **LLM**: HuggingFace Transformers (or Ollama)

## Development Roadmap

For detailed development plans, experiments, and future features, see [PLAN.md](PLAN.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This is a research project focused on improving RAG systems for academic paper analysis. The system is designed to handle ACL Anthology papers but can be adapted for other paper collections.
