import logging

import numpy as np
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:

    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        """
        Initialize cross-encoder reranker.

        Default model: BAAI/bge-reranker-v2-m3 (568M params, SOTA).
        Uses MPS on Apple Silicon with float16 for ~2x faster inference.

        Alternative models:
        - "BAAI/bge-reranker-v2-gemma" (2B, best quality)
        - "jina-ai/jina-reranker-v2-base-multilingual" (278M, faster)

        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.model = CrossEncoder(model_name, device=device)

        # float16 on MPS/CUDA for faster inference
        if device in ("mps", "cuda"):
            self.model.model.half()

        logger.info("Reranker loaded on %s (dtype=%s)", device, next(self.model.model.parameters()).dtype)
    
    def rerank(self, query, chunks, top_k = 3):
        """
        Re-ranking function.
        
        Args:
            query: Query text
            chunks: List of chunk dictionaries with 'text' key
            top_k: Number of top chunks 
        
        Returns:
            results: List of re-ranked chunks 
        """

        if not chunks:
            return []
        
        # Extracting chunk texts
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Creating query-chunk pairs for cross-encoder
        pairs = [[query, chunk_text] for chunk_text in chunk_texts]
        
        # Computing relevance score using cross-encoder
        scores = self.model.predict(pairs)
        
        # Converting to numpy array for normalization
        scores = np.array(scores)
        
        # Normalizing scores to 0-1 range using min-max scaling
        if len(scores) > 1:
            min_score = scores.min()
            max_score = scores.max()
            if max_score != min_score:
                normalized_scores = (scores - min_score) / (max_score - min_score)
            else:
                # All scores are the same, set to 1.0
                normalized_scores = np.ones_like(scores)
        else:
            # Single score, set to 1.0
            normalized_scores = np.array([1.0])
        
        # Creating list of (chunk, score) tuples
        chunk_scores = list(zip(chunks, normalized_scores))
        
        # Sorting by score (descending)
        chunk_scores.sort(key=lambda x: x[1], reverse = True)
        
        # Returning top_k chunks with updated scores
        reranked_chunks = []
        for rank, (chunk, normalized_score) in enumerate(chunk_scores[:top_k], 1):
            reranked_chunk = chunk.copy()
            
            # Preserving original retrieval score
            original_score = chunk.get('score', 0.0)
            reranked_chunk['original_score'] = float(original_score)
            
            # Adding normalized reranked score
            reranked_chunk['score'] = float(normalized_score)
            reranked_chunk['reranked_score'] = float(normalized_score)
            
            reranked_chunk['rank'] = rank
            reranked_chunk['reranked'] = True
            reranked_chunks.append(reranked_chunk)
        
        return reranked_chunks