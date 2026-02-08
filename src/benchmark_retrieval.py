import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import EmbeddingGenerator
from src.retrieval.semantic import SemanticRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever

project_root = Path(__file__).parent.parent
index_path = str(project_root / "artifacts" / "faiss_index")
chunks_path = str(project_root / "artifacts" / "chunks.json")
bm25_index_path = str(project_root / "artifacts" / "bm25_index")

# Test queries covering different retrieval patterns
queries = [
    "What are the main approaches to few-shot learning?",
    "How do transformer models handle long sequences?",
    "What evaluation metrics are used for text generation?",
    "Compare reinforcement learning methods for language models",
    "What is the role of attention mechanisms in NLP?",
]

def benchmark(name, search_fn, queries, top_k = 5, warmup = 1):
    """
    Benchmarking a retrieval function.

    Args:
        name: Name of the retriever
        search_fn: Search function to benchmark
        queries: List of query strings
        top_k: Number of results to retrieve
        warmup: Number of warmup runs
    """

    # Warmup runs (first query loads caches, tokenizers, etc.)
    for _ in range(warmup):
        search_fn(queries[0], top_k = top_k)

    latencies = []
    for query in queries:
        start = time.perf_counter()
        results = search_fn(query, top_k = top_k)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        print(f"  [{elapsed:6.1f}ms] {query[:50]}... -> {len(results)} results")

    avg = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p99 = sorted(latencies)[-1]

    print(f"\n  {name} Summary:")
    print(f"  Avg: {avg:.1f}ms | P50: {p50:.1f}ms | P99: {p99:.1f}ms\n")
    return latencies


if __name__ == "__main__":

    print("Loading embedding model...")
    embedding_gen = EmbeddingGenerator()

    # --- Semantic Retrieval ---
    print("\nLoading Semantic Retriever (FAISS)...")
    semantic = SemanticRetriever(index_path = index_path, chunks_path = chunks_path)
    print(f"Index size: {semantic.index.ntotal} vectors, dimension: {semantic.index.d}")

    # Wrapping search to include embedding time
    def semantic_search(query, top_k = 5):
        query_embedding = embedding_gen.embed_query(query)
        return semantic.search(query_embedding, top_k = top_k)

    print("\n--- Semantic (FAISS FlatIP) ---")
    benchmark("Semantic", semantic_search, queries)

    # --- BM25 Retrieval ---
    print("Loading BM25 Retriever...")
    bm25 = BM25Retriever(chunks_path = chunks_path, index_path = bm25_index_path)
    print(f"Corpus size: {len(bm25.chunks)} chunks")

    print("\n--- BM25 (bm25s) ---")
    benchmark("BM25", bm25.search, queries)

    # --- Hybrid Retrieval (RRF) ---
    print("--- Hybrid (RRF) ---")
    hybrid = HybridRetriever(semantic, bm25, fusion_method = "rrf")

    def hybrid_search(query, top_k = 5):
        query_embedding = embedding_gen.embed_query(query)
        return hybrid.search(query, query_embedding, top_k = top_k)

    benchmark("Hybrid RRF", hybrid_search, queries)

    # --- With metadata filter ---
    print("--- Semantic + Filter (conference=ACL) ---")
    def filtered_search(query, top_k = 5):
        query_embedding = embedding_gen.embed_query(query)
        return semantic.search(query_embedding, top_k = top_k, filters = {"conference": "ACL"})

    benchmark("Semantic+Filter", filtered_search, queries)
