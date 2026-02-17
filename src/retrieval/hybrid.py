import numpy as np
from typing import List, Dict, Optional, Any
from pydantic import ConfigDict
from .semantic import SemanticRetriever
from .bm25 import BM25Retriever

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

class HybridRetriever(BaseRetriever):

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, semantic_retriever, bm25_retriever, alpha = 0.7, beta = 0.3,
                 fusion_method = "rrf", rrf_k = 60, embedding_generator = None, top_k: int = 10, **kwargs):
        """
        Initialize hybrid retriever (LangChain-compatible).

        Args:
            semantic_retriever: Semantic retriever instance
            bm25_retriever: BM25 retriever instance
            alpha: Weight for semantic scores (weighted fusion only)
            beta: Weight for BM25 scores (weighted fusion only)
            fusion_method: Fusion strategy ("rrf" or "weighted")
            rrf_k: RRF constant (default 60, higher values = more emphasis on top ranks)
            embedding_generator: EmbeddingGenerator instance (required for LangChain compatibility)
            top_k: Default number of results to return (used by LangChain)
            **kwargs: Additional arguments for BaseRetriever
        """
        lc_kwargs = {k: v for k, v in kwargs.items() if k in ("tags",)}
        super().__init__(**lc_kwargs)

        self.semantic_retriever = semantic_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha
        self.beta = beta
        self.fusion_method = fusion_method.lower()
        self.rrf_k = rrf_k
        self.embedding_generator = embedding_generator
        self.top_k = top_k

    def search(self, query, query_embedding, top_k = 10, filters = None):
        """
        Hybrid searching function.

        Args:
            query: Query string for BM25 search
            query_embedding: Query embedding for semantic search
            top_k: Number of results to return
            filters: Dictionary of metadata filters

        Returns:
            results: List of retrieved chunks with combined scores
        """

        # Getting results from both retrievers
        semantic_results = self.semantic_retriever.search(query_embedding, top_k = top_k * 2, filters = filters)
        bm25_results = self.bm25_retriever.search(query, top_k = top_k * 2, filters = filters)

        # Fusing results based on chosen strategy
        if self.fusion_method == "rrf":
            return self._rrf_fusion(semantic_results, bm25_results, top_k)
        else:
            return self._weighted_fusion(semantic_results, bm25_results, top_k)

    def _rrf_fusion(self, semantic_results, bm25_results, top_k):
        """
        Reciprocal Rank Fusion function.

        Combines results using rank positions instead of raw scores,
        making it robust across different score distributions.

        RRF(d) = sum(1 / (k + rank(d))) for each retriever

        Args:
            semantic_results: Results from semantic retriever
            bm25_results: Results from BM25 retriever
            top_k: Number of results to return

        Returns:
            results: List of combined results with RRF scores
        """

        rrf_scores = {}
        all_chunks = {}

        # Accumulating RRF scores from semantic results
        for rank, result in enumerate(semantic_results, 1):
            chunk_id = result["metadata"].get("chunk_id", f"sem_{rank}")
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (self.rrf_k + rank)
            all_chunks[chunk_id] = result

        # Accumulating RRF scores from BM25 results
        for rank, result in enumerate(bm25_results, 1):
            chunk_id = result["metadata"].get("chunk_id", f"bm25_{rank}")
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (self.rrf_k + rank)
            if chunk_id not in all_chunks:
                all_chunks[chunk_id] = result

        # Sorting by RRF score (higher is better)
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse = True)[:top_k]

        # Final payload preparation :P
        results = []
        for rank, (chunk_id, score) in enumerate(sorted_chunks, 1):
            result = all_chunks[chunk_id].copy()
            result["score"] = score
            result["rank"] = rank
            results.append(result)

        return results

    def _weighted_fusion(self, semantic_results, bm25_results, top_k):
        """
        Weighted fusion function.

        Args:
            semantic_results: Results from semantic retriever
            bm25_results: Results from BM25 retriever
            top_k: Number of results to return

        Returns:
            results: List of combined results with weighted scores
        """

        # Normalizing scores
        semantic_scores = {r["metadata"].get("chunk_id", i): r["score"] for i, r in enumerate(semantic_results)}
        bm25_scores = {r["metadata"].get("chunk_id", i): r["score"] for i, r in enumerate(bm25_results)}

        # Normalizing to [0, 1] range
        max_semantic = max(semantic_scores.values()) if semantic_scores else 1
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1

        # Combining scores
        combined_scores = {}
        all_chunks = {}

        # Alpha * Norm score (higher semantic score = higher normalized score)
        for idx, result in enumerate(semantic_results):

            chunk_id = result["metadata"].get("chunk_id", idx)
            normalized_score = result["score"] / max_semantic if max_semantic > 0 else 0

            combined_scores[chunk_id] = self.alpha * normalized_score
            all_chunks[chunk_id] = result

        # Beta * Norm score
        for idx, result in enumerate(bm25_results):

            chunk_id = result["metadata"].get("chunk_id", idx)
            normalized_score = result["score"] / max_bm25 if max_bm25 > 0 else 0

            if chunk_id in combined_scores:
                combined_scores[chunk_id] += self.beta * normalized_score
            else:
                combined_scores[chunk_id] = self.beta * normalized_score
                all_chunks[chunk_id] = result

        # Sorting by combined score
        sorted_chunks = sorted(combined_scores.items(), key=lambda x: x[1], reverse = True)[:top_k]

        # Final payload preparation :P
        results = []
        for rank, (chunk_id, score) in enumerate(sorted_chunks, 1):
            result = all_chunks[chunk_id].copy()
            result["score"] = score
            result["rank"] = rank
            results.append(result)

        return results

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        top_k: Optional[int] = None,
        filters: Optional[Dict] = None
    ) -> List[Document]:
        """
        Retrieve documents relevant to a query (LangChain-compatible method).

        This is the required method for BaseRetriever that LangChain calls.
        It performs hybrid retrieval (semantic + BM25 fusion) and returns Document objects.

        Args:
            query: Query string (NOT embedding - we handle embedding internally)
            run_manager: Callback manager for logging/tracing (LangChain provides this)
            top_k: Number of results to return (overrides self.top_k if provided)
            filters: Metadata filters (conference, year, title, section_type)

        Returns:
            List of LangChain Document objects with page_content and metadata
        """
        # Use instance default if not specified
        k = top_k if top_k is not None else self.top_k

        if self.embedding_generator is None:
            raise ValueError(
                "embedding_generator is required for LangChain compatibility. "
                "Pass it to __init__: HybridRetriever(..., embedding_generator=embedding_gen)"
            )

        query_embedding = self.embedding_generator.embed_query(query)
        results = self.search(query, query_embedding, top_k=k, filters=filters)

        return [
            Document(
                page_content=result["text"],
                metadata={**result["metadata"], "score": result["score"], "rank": result["rank"], "retriever": f"hybrid_{self.fusion_method}"}
            )
            for result in results
        ]
