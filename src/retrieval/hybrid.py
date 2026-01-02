import numpy as np
from .semantic import SemanticRetriever
from .bm25 import BM25Retriever

class HybridRetriever:
    
    def __init__(self, semantic_retriever, bm25_retriever, alpha = 0.7, beta = 0.3):
        """
        Initialize hybrid retriever.
        
        Args:
            semantic_retriever: Semantic retriever instance
            bm25_retriever: BM25 retriever instance
            alpha: Weight for semantic scores
            beta: Weight for BM25 scores
        """

        self.semantic_retriever = semantic_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha
        self.beta = beta
    
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
        
        return self._weighted_fusion(semantic_results, bm25_results, top_k)
    
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
        
        # Alpha * Norm score
        for idx, result in enumerate(semantic_results):

            chunk_id = result["metadata"].get("chunk_id", idx)
            normalized_score = 1 - (result["score"] / max_semantic) if max_semantic > 0 else 0
            
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