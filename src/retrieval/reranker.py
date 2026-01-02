import numpy as np
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    
    def __init__(self, model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder re-ranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
        """

        # Loading the cross-encoder model: [CLS] query [SEP] chunk [SEP]
        self.model = CrossEncoder(model_name)
    
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