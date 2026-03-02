from functools import lru_cache

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingGenerator:

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", cache_size: int = 256):
        """
        Initialize embedding generator.

        Args:
            model_name: HuggingFace model name (default: BAAI/bge-base-en-v1.5)
            cache_size: Number of query embeddings to cache (LRU). Set to 0 to disable.
        """
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # Per-instance LRU cache for query embeddings.
        # Defined as a closure so lru_cache works on a plain function (avoids
        # the self-not-hashable problem with @lru_cache on instance methods).
        _embed_fn = self.embeddings.embed_query

        if cache_size > 0:
            @lru_cache(maxsize=cache_size)
            def _embed_cached(query: str) -> tuple:
                # Store as tuple so the result is hashable and immutable in cache;
                # callers receive a numpy array via embed_query().
                return tuple(_embed_fn(query))

            self._embed_cached = _embed_cached
        else:
            self._embed_cached = None

    def embed_texts(self, texts):
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings (n_texts, dimension)
        """
        embeddings = self.embeddings.embed_documents(texts)
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query, with LRU caching.

        Identical queries within the same process are served from cache,
        avoiding redundant model inference on repeated or similar tool calls.

        Args:
            query: Query string

        Returns:
            Numpy array of embedding (dimension,)
        """
        if self._embed_cached is not None:
            return np.array(self._embed_cached(query))
        return np.array(self.embeddings.embed_query(query))

    def cache_info(self):
        """Return lru_cache statistics (hits, misses, maxsize, currsize), or None if disabled."""
        if self._embed_cached is not None:
            return self._embed_cached.cache_info()
        return None

    def cache_clear(self):
        """Evict all cached query embeddings."""
        if self._embed_cached is not None:
            self._embed_cached.cache_clear()

    def encode(self, query: str) -> np.ndarray:
        """
        Alias for embed_query for compatibility with SentenceTransformer interface.

        Args:
            query: Query string

        Returns:
            Numpy array of embedding (dimension,)
        """
        return self.embed_query(query)