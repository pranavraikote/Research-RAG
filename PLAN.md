# ResearchRAG Development Plan

This document outlines the development roadmap for ResearchRAG, an evidence-aware RAG system for querying and analyzing scientific research papers.

## Project Overview

ResearchRAG is being developed incrementally, starting with traditional RAG and advancing through re-ranking, advanced chunking strategies, conversational enhancements, and finally to an agentic RAG version.

**Technology Stack**:
- **Framework**: LangChain
- **Vector Database**: FAISS (FlatIP, HNSW support)
- **Embeddings**: BAAI/bge-base-en-v1.5 (768d, 512 max tokens)
- **BM25**: bm25s (sparse matrices, disk persistence)
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
- [x] Cross-encoder re-ranking implementation
- [x] Score normalization (0-1 range)
- [x] Hybrid search (semantic + BM25)
- [x] Weighted combination strategy
- [x] Display of both original and reranked scores
- [x] Metadata filtering with auto-parsing
- [x] Reciprocal rank fusion (RRF) as alternative fusion method
- [x] Migrated BM25 to bm25s (sparse matrices, disk persistence via mmap)
- [x] FAISS IDSelector pre-filtering (filter during search, not post-retrieval)
- [x] Upgraded embedding model to BAAI/bge-base-en-v1.5 (768d, 512 max tokens)
- [x] HNSW index support with FlatIP fallback
- [x] Retrieval latency benchmark tooling (Semantic, BM25, Hybrid, Filtered)

**Future**:
- [ ] Query expansion and reformulation

### Phase 3: Advanced Chunking Strategies

**Status**: In Progress

**Goal**: Implement and systematically evaluate intelligent chunking strategies that preserve semantic structure.

**Completed**:
- [x] Semantic chunking implementation
- [x] Basic fixed-size chunking
- [x] Pre-chunking vs post-chunking text cleaning (preserve paragraph boundaries for recursive splitter)
- [x] Optimized chunk size (1500) and overlap (300) for research papers
- [x] PDF text extraction cleanup (preserve Figure/Table captions, fix hyphenation)

**Planned**:
- [ ] Citation-aware chunking (preserve citation context)
- [ ] Section-based chunking (abstract, intro, methods, etc.)
- [ ] Paragraph-based chunking with section context
- [ ] Chunking ablation study framework
- [ ] Evaluation metrics for chunking quality

**Evaluation Metrics**:
- Retrieval quality (MRR, NDCG, Precision@K, Recall@K)
- Chunk coherence (semantic similarity within chunks)
- Citation preservation (percentage not split across chunks)
- Context preservation (overlap effectiveness)

### Phase 4: Conversational RAG Enhancement

**Status**: In Progress

**Goal**: Transform the standalone single-query RAG system into a conversational system that maintains context across multiple turns.

**Completed**:
- [x] Conversation history management
- [x] Query rewriting for follow-up questions
- [x] Enhanced prompt with conversation context
- [x] Interactive CLI mode
- [x] Prompt caching for conversational mode
- [x] Multi-turn retrieval optimization (discussed paper tracking, follow-up detection, query boosting with paper titles)

**Design Considerations**:
- Balance between context preservation and token limits
- Query rewriting (heuristic vs. LLM-based)
- Context-aware retrieval strategies
- Session persistence (optional)

### Phase 5: Agentic RAG

**Status**: Future

**Goal**: Build an agentic system that can reason across papers, compare findings, and identify gaps.

**Planned Features**:
- [ ] Multi-agent architecture (retriever, analyzer, comparator, synthesizer)
- [ ] Cross-paper reasoning and comparison
- [ ] Structured output generation (claims, evidence, limitations)
- [ ] Gap detection in literature
- [ ] Iterative refinement and follow-up question generation

## Research Focus Areas

### 1. Chunking Strategy Ablations
Systematic evaluation of different chunking strategies:
- Fixed-size vs. semantic boundaries
- Impact of overlap on retrieval quality
- Citation-aware chunking effectiveness
- Section-based vs. paragraph-based chunking
- Optimal chunk sizes for academic papers

### 2. Retrieval Strategy Comparisons
Comprehensive comparison of retrieval approaches:
- Semantic search (embedding-based) baseline
- BM25 keyword search
- Hybrid search strategies
- Re-ranking impact on final results
- Query expansion and reformulation

### 3. Real-World Use Cases
Focus on evaluation scenarios that reflect real research needs:
- Finding papers that cite specific methods
- Comparing approaches across papers
- Identifying gaps in literature
- Extracting claims and supporting evidence
- Cross-paper reasoning tasks

## Key Considerations

### Performance
- Monitor latency at each step
- Optimize batch processing
- Cache embeddings and frequent queries
- Use async operations where possible
- Prompt caching for faster TFFT

### Scalability
- FAISS supports efficient similarity search at scale
- Implement batch processing for large collections
- Use FAISS index serialization for persistence
- Consider incremental indexing for new papers

### Accuracy
- Validate retrieval quality regularly
- Test with diverse query types
- Monitor hallucination rates
- Implement fact-checking where possible

### Maintainability
- Write clean, documented code
- Use type hints and linting
- Follow consistent code style
- Create comprehensive tests

## Evaluation Metrics

### Retrieval Quality
- **MRR** (Mean Reciprocal Rank): Average of reciprocal ranks of first relevant result
- **NDCG@K**: Ranking quality metric
- **Precision@K**: Fraction of retrieved items that are relevant
- **Recall@K**: Fraction of relevant items that are retrieved

### Answer Quality
- **Citation Accuracy**: Percentage of citations that are correct and relevant
- **Groundedness**: How well answers are supported by retrieved chunks
- **Completeness**: Whether all aspects of query are addressed

### System Performance
- **TFFT** (Time to First Token): Latency before first token generation
- **Total Generation Time**: End-to-end query processing time
- **Index Size**: FAISS index size and memory usage

## Data Sources

### Primary: ACL Anthology
The ACL Anthology is the primary data source, containing complete proceedings from:
- ACL (Association for Computational Linguistics) conferences
- EMNLP (Empirical Methods in Natural Language Processing)
- NAACL (North American Chapter of the ACL)
- EACL (European Chapter of the ACL)
- COLING (International Conference on Computational Linguistics)

**Current Corpus**: ~842 papers (2025), 54,264 chunks, 200 per venue limit

**Key Features**:
- Well-structured XML metadata
- Complete citation networks
- High-quality PDFs
- Consistent formatting
- Rich metadata (authors, venues, years, abstracts)

### Secondary Sources (Future)
- **NeurIPS**: Neural Information Processing Systems proceedings
- **ICML**: International Conference on Machine Learning proceedings
- **arXiv**: Preprints (cs.CL, cs.AI, cs.LG categories)

---

**Note**: This is a living document that will be updated as the project evolves. For current implementation status, see the checked items in each phase.
