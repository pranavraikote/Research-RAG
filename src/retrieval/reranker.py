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
        
        # Computing relevance scores using cross-encoder.
        # bge-reranker-v2-m3 outputs sigmoid scores in [0, 1] — these are
        # absolute relevance scores, not relative.  Do NOT min-max normalise;
        # downstream threshold checks (e.g. _RELEVANCE_THRESHOLD = 0.5)
        # depend on the raw sigmoid values.
        scores = self.model.predict(pairs)
        scores = np.array(scores, dtype=float)

        # Sort by raw sigmoid score (descending).
        chunk_scores = sorted(
            zip(chunks, scores), key=lambda x: x[1], reverse=True
        )

        reranked_chunks = []
        for rank, (chunk, raw_score) in enumerate(chunk_scores[:top_k], 1):
            reranked_chunk = chunk.copy()
            reranked_chunk["original_score"] = float(chunk.get("score", 0.0))
            reranked_chunk["score"] = float(raw_score)
            reranked_chunk["raw_score"] = float(raw_score)
            reranked_chunk["rank"] = rank
            reranked_chunk["reranked"] = True
            reranked_chunks.append(reranked_chunk)

        return reranked_chunks