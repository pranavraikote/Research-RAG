import faiss
import numpy as np
from pathlib import Path
from typing import ClassVar, List, Dict, Optional, Any
from pydantic import ConfigDict
from .bm25 import BM25Retriever

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

class SemanticRetriever(BaseRetriever):

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # HNSW parameters
    HNSW_M: ClassVar[int] = 32                  # Number of connections per layer
    HNSW_EF_CONSTRUCTION: ClassVar[int] = 64    # Construction-time search depth
    HNSW_EF_SEARCH: ClassVar[int] = 32          # Query-time search depth
    HNSW_MIN_CHUNKS: ClassVar[int] = 1000       # Minimum chunks to use HNSW (Flat fallback below this)

    def __init__(self, index_path = None, chunks_path = None, dimension = None, metric = "IP",
                 index_type = "hnsw", hnsw_m = None, hnsw_ef_construction = None, hnsw_ef_search = None,
                 embedding_generator = None, top_k: int = 10, **kwargs):
        """
        Initialize semantic retriever (LangChain-compatible).

        Args:
            index_path: Path to saved FAISS index
            chunks_path: Path to chunks JSON file
            dimension: Dimension of embeddings
            metric: Distance metric ("IP" for cosine similarity, "L2" for Euclidean distance)
            index_type: Index type ("hnsw" for HNSW graph search, "flat" for brute-force)
            hnsw_m: HNSW connections per layer (default 32)
            hnsw_ef_construction: HNSW construction-time search depth (default 64)
            hnsw_ef_search: HNSW query-time search depth (default 32)
            embedding_generator: EmbeddingGenerator instance (required for LangChain compatibility)
            top_k: Default number of results to return (used by LangChain)
            **kwargs: Additional arguments for BaseRetriever
        """
        # Initialize BaseRetriever (LangChain parent class)
        # Only pass recognized LangChain fields (tags, metadata) from kwargs
        lc_kwargs = {k: v for k, v in kwargs.items() if k in ("tags",)}
        super().__init__(**lc_kwargs)

        self.dimension = dimension
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        self.metric = metric.upper()
        self.index_type = index_type.lower()
        self._active_index_type = None

        # LangChain-specific attributes
        self.embedding_generator = embedding_generator
        self.top_k = top_k

        # HNSW parameters (use class defaults if not specified)
        self.hnsw_m = hnsw_m or self.HNSW_M
        self.hnsw_ef_construction = hnsw_ef_construction or self.HNSW_EF_CONSTRUCTION
        self.hnsw_ef_search = hnsw_ef_search or self.HNSW_EF_SEARCH

        # Loading the index and chunks
        if index_path and Path(index_path).exists():
            self.load_index(index_path)
            self.dimension = self.index.d

            if chunks_path is None:
                chunks_path = str(Path(index_path).parent / "chunks.json")

            if Path(chunks_path).exists():
                self.load_chunks(chunks_path)

    def _create_index(self, dimension, n_chunks = 0):
        """
        Creating FAISS index based on configuration function.

        Uses HNSW for large collections (fast approximate search),
        falls back to Flat for small ones (exact brute-force search).

        Args:
            dimension: Embedding dimension
            n_chunks: Number of chunks to be indexed (for auto-selection)
        """

        # HNSW for large collections, Flat as fallback for small ones
        if self.index_type == "hnsw" and n_chunks >= self.HNSW_MIN_CHUNKS:

            if self.metric == "IP":
                self.index = faiss.IndexHNSWFlat(dimension, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            else:
                self.index = faiss.IndexHNSWFlat(dimension, self.hnsw_m)

            # Setting construction and search parameters
            self.index.hnsw.efConstruction = self.hnsw_ef_construction
            self.index.hnsw.efSearch = self.hnsw_ef_search
            self._active_index_type = "hnsw"

            print(f"Using HNSW index (M={self.hnsw_m}, efConstruction={self.hnsw_ef_construction}, efSearch={self.hnsw_ef_search})")

        else:
            # Flat index for small collections or explicit flat request
            if self.metric == "IP":
                self.index = faiss.IndexFlatIP(dimension)
            else:
                self.index = faiss.IndexFlatL2(dimension)

            self._active_index_type = "flat"

            if self.index_type == "hnsw" and n_chunks < self.HNSW_MIN_CHUNKS:
                print(f"Using Flat index (collection size {n_chunks} below HNSW threshold {self.HNSW_MIN_CHUNKS})")
            else:
                print(f"Using Flat index")

    def add_chunks(self, embeddings, chunks, metadata):
        """
        Add new chunks to the FAISS index (supports efficient incremental updates).

        FAISS automatically assigns sequential IDs (0, 1, 2, ...) to added vectors.
        Since we extend chunks/metadata arrays in the same order, the FAISS IDs
        align perfectly with array indices - no manual ID management needed!

        Args:
            embeddings: Numpy array of embeddings (n_chunks, dimension)
            chunks: List of chunk texts
            metadata: List of metadata dictionaries
        """

        if self.index is None:
            self.dimension = embeddings.shape[1]
            self._create_index(self.dimension, n_chunks = len(chunks))

        elif embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch :(")

        embeddings = embeddings.astype('float32')

        # Normalizing the vectors if IP chosen
        if self.metric == "IP":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms

        # FAISS efficiently adds vectors with sequential IDs
        # IDs match array indices: if we have 1000 chunks, new chunks get IDs 1000, 1001, 1002...
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self.chunk_metadata.extend(metadata)

    def load_chunks(self, chunks_path):
        """
        Load chunks and build index function.

        Args:
            chunks_path: Path to chunks JSON file
        """

        # Loading chunks from disk
        from ..chunking.basic import BasicChunker
        chunk_texts, chunk_metadata = BasicChunker.load_chunks(chunks_path)

        self.chunks = chunk_texts
        self.chunk_metadata = chunk_metadata

    def _get_valid_ids(self, filters):
        """
        Pre-computing valid chunk IDs matching the given filters function.

        Scans metadata to find chunks that match all filter criteria,
        returning their indices for use with FAISS IDSelector.

        Args:
            filters: Dictionary of metadata filters

        Returns:
            valid_ids: Numpy array of valid chunk indices
        """

        filter_helper = BM25Retriever()

        valid_ids = []
        for idx, meta in enumerate(self.chunk_metadata):
            if filter_helper.matches_filter(meta, filters):
                valid_ids.append(idx)

        return np.array(valid_ids, dtype=np.int64)

    def search(self, query_embedding, top_k = 10, filters = None):
        """
        Searching for relevant chunks using FAISS index function.

        Uses IDSelector pre-filtering when metadata filters are provided,
        avoiding wasteful over-retrieval and post-filtering.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Dictionary of metadata filters:
                - conference: str
                - year: int
                - title: str

        Returns:
            results: List of retrieved chunks with scores and metadata
        """

        if self.index is None or len(self.chunks) == 0:
            return []

        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        # Normalizing query if IP metric
        if self.metric == "IP":
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm

        # Pre-filtering with FAISS IDSelector when filters are provided
        valid_id_set = None
        if filters:
            valid_ids = self._get_valid_ids(filters)

            if len(valid_ids) == 0:
                return []

            search_k = min(top_k, len(valid_ids))

            try:
                # FAISS IDSelector for efficient pre-filtering during search
                id_selector = faiss.IDSelectorBatch(valid_ids)

                if self._active_index_type == "hnsw":
                    params = faiss.SearchParametersHNSW(sel = id_selector)
                else:
                    params = faiss.SearchParameters(sel = id_selector)

                scores, indices = self.index.search(query_embedding, search_k, params = params)

            except (AttributeError, RuntimeError):
                # Fallback for older FAISS versions: over-retrieve and post-filter
                valid_id_set = set(valid_ids.tolist())
                scores, indices = self.index.search(query_embedding, top_k * 3)
        else:
            scores, indices = self.index.search(query_embedding, top_k)

        # Building results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            # Post-filtering fallback (only when IDSelector wasn't available)
            if valid_id_set is not None and int(idx) not in valid_id_set:
                continue

            results.append({
                "text": self.chunks[idx],
                "metadata": self.chunk_metadata[idx],
                "score": float(score),
                "rank": len(results) + 1
            })

            if len(results) >= top_k:
                break

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
        It converts a query string to embedding, searches, and returns Document objects.

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

        # Check if embedding_generator is available
        if self.embedding_generator is None:
            raise ValueError(
                "embedding_generator is required for LangChain compatibility. "
                "Pass it to __init__: SemanticRetriever(..., embedding_generator=embedding_gen)"
            )

        query_embedding = self.embedding_generator.embed_query(query)
        results = self.search(query_embedding, top_k=k, filters=filters)

        return [
            Document(
                page_content=result["text"],
                metadata={**result["metadata"], "score": result["score"], "rank": result["rank"], "retriever": "semantic"}
            )
            for result in results
        ]

    def save_index(self, path):
        """
        Saving FAISS index to disk function.

        Args:
            path: Path to save the index
        """

        faiss.write_index(self.index, path)

    def save_chunks(self, chunks_path):
        """
        Save chunks and metadata to JSON file.

        Use this after add_chunks() to persist the updated corpus to disk.

        Args:
            chunks_path: Path to save chunks JSON file
        """
        import json

        chunks_data = [
            {
                "text": chunk,
                "metadata": metadata
            }
            for chunk, metadata in zip(self.chunks, self.chunk_metadata)
        ]

        Path(chunks_path).parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    def load_index(self, path):
        """
        Loading FAISS index from disk function.

        Args:
            path: Path to load the index from
        """

        # Loading the index from disk
        self.index = faiss.read_index(path)

        # Detecting index type from loaded index
        if isinstance(self.index, faiss.IndexHNSWFlat):
            self._active_index_type = "hnsw"
        else:
            self._active_index_type = "flat"

        # Detecting metric type from loaded index
        if hasattr(self.index, 'metric_type'):
            if self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
                self.metric = "IP"
            elif self.index.metric_type == faiss.METRIC_L2:
                self.metric = "L2"
