import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
from pydantic import ConfigDict
import bm25s

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

class BM25Retriever(BaseRetriever):

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, chunks_path = None, index_path = None, top_k: int = 10, **kwargs):
        """
        Initialize BM25 retriever (LangChain-compatible).

        Uses bm25s for fast sparse retrieval with disk persistence.

        Args:
            chunks_path: Path to chunks JSON file
            index_path: Path to saved BM25 index directory
            top_k: Default number of results to return (used by LangChain)
            **kwargs: Additional arguments for BaseRetriever
        """
        lc_kwargs = {k: v for k, v in kwargs.items() if k in ("tags",)}
        super().__init__(**lc_kwargs)

        self.bm25 = None
        self.chunks = []
        self.chunk_metadata = []
        self.top_k = top_k

        # Loading from persisted BM25 index if available
        if index_path and Path(index_path).exists():
            self.load_index(index_path)

        if chunks_path and Path(chunks_path).exists():
            self.load_chunks(chunks_path)

    def add_chunks(self, chunks, metadata):
        """
        Add new chunks to the BM25 index.

        NOTE: BM25 requires rebuilding the entire index when adding chunks
        because IDF (inverse document frequency) must be recalculated across
        all documents for accurate scoring. This is fast (~seconds for 150K+ chunks)
        and necessary for correct BM25 scores.

        Args:
            chunks: New chunks to be added
            metadata: New metadata to be added along with the chunks
        """

        if not chunks:
            return  # Nothing to add

        # Extend corpus
        self.chunks.extend(chunks)
        self.chunk_metadata.extend(metadata)

        # Rebuild index with full corpus (required for accurate IDF calculation)
        # BM25 indexing is fast enough that rebuilding is acceptable
        corpus_tokens = bm25s.tokenize(self.chunks, stopwords="en", show_progress=False)
        self.bm25 = bm25s.BM25()
        self.bm25.index(corpus_tokens, show_progress=False)

    def load_chunks(self, chunks_path):
        """
        Load chunks and build BM25 index function.

        Handles both chunk formats:
          - Flat:   [{"text": ..., "chunk_id": ..., "title": ..., ...}, ...]
          - Nested: [{"text": ..., "metadata": {"chunk_id": ..., ...}}, ...]

        Args:
            chunks_path: Path to chunks JSON file
        """
        import json as _json
        with open(chunks_path, encoding="utf-8") as f:
            data = _json.load(f)

        chunk_texts, chunk_metadata = [], []
        for item in data:
            chunk_texts.append(item["text"])
            if "metadata" in item:
                chunk_metadata.append(item["metadata"])
            else:
                chunk_metadata.append({k: v for k, v in item.items() if k != "text"})

        self.chunks = chunk_texts
        self.chunk_metadata = chunk_metadata

        # Building the BM25 index (skip if already loaded from persisted index)
        if self.chunks and self.bm25 is None:
            corpus_tokens = bm25s.tokenize(self.chunks, stopwords = "en")
            self.bm25 = bm25s.BM25()
            self.bm25.index(corpus_tokens)

    def save_index(self, path):
        """
        Saving BM25 index to disk for fast loading function.

        Args:
            path: Directory path to save the BM25 index
        """

        if self.bm25 is not None:
            Path(path).mkdir(parents = True, exist_ok = True)
            self.bm25.save(path)

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
        Loading BM25 index from disk with memory mapping function.

        Args:
            path: Directory path to load the BM25 index from
        """

        self.bm25 = bm25s.BM25.load(path, mmap = True)

    def matches_filter(self, metadata, filters):
        """
        Metadata matching the filter function.

        Args:
            metadata: Chunk metadata dictionary
            filters: Filter dictionary with:
                - conference: str or list
                - year: int, list, or dict (range: {"min": 2020, "max": 2024})
                - title: str or list (partial match)
                - section_type: str or list (e.g., "methods", ["methods", "experiments"])

        Returns:
            bool: True if metadata matches filter criteria, False otherwise
        """

        if not filters:
            return True

        # Conference filter
        if "conference" in filters:
            filter_conf = filters["conference"]
            chunk_conf = metadata.get("conference", "").upper()

            if isinstance(filter_conf, str):
                if chunk_conf != filter_conf.upper():
                    return False
            else:
                if chunk_conf not in [c.upper() for c in filter_conf]:
                    return False

        # Year filter
        if "year" in filters:
            filter_year = filters["year"]
            chunk_year = metadata.get("year")

            if chunk_year is None:
                return False

            if isinstance(filter_year, int):
                if chunk_year != filter_year:
                    return False

            elif isinstance(filter_year, (list, tuple)):
                if chunk_year not in filter_year:
                    return False

            elif isinstance(filter_year, dict):

                # Range filter: {"min": 2020, "max": 2024}
                if "min" in filter_year and chunk_year < filter_year["min"]:
                    return False
                if "max" in filter_year and chunk_year > filter_year["max"]:
                    return False

        # Title filter, not a full-fledged match, just for partial match
        if "title" in filters:
            filter_title = filters["title"]
            chunk_title = metadata.get("title", "").lower()

            if isinstance(filter_title, str):
                if filter_title.lower() not in chunk_title:
                    return False
            else:
                if not any(ft.lower() in chunk_title for ft in filter_title):
                    return False

        # Section type filter (for structured chunking)
        if "section_type" in filters:
            filter_section = filters["section_type"]
            chunk_section = metadata.get("section_type", "").lower()

            # Skip chunks without section metadata (e.g., basic chunking)
            if not chunk_section or chunk_section == "unknown":
                # Allow filtering out chunks without section info
                # If filter is strict (not "unknown"), exclude these chunks
                if isinstance(filter_section, str):
                    if filter_section.lower() != "unknown":
                        return False
                elif isinstance(filter_section, list):
                    if "unknown" not in [s.lower() for s in filter_section]:
                        return False

            # Match section type
            if isinstance(filter_section, str):
                if chunk_section != filter_section.lower():
                    return False
            else:
                # List of allowed sections
                if chunk_section not in [s.lower() for s in filter_section]:
                    return False

        return True

    def search(self, query, top_k = 10, filters = None):
        """
        Searching for relevant chunks using BM25 index function.

        Args:
            query: Query string
            top_k: Number of results to return
            filters: Dictionary of metadata filters:
                - conference: str
                - year: int
                - title: str

        Returns:
            results: List of retrieved chunks with scores and metadata
        """

        if self.bm25 is None or len(self.chunks) == 0:
            return []

        # Tokenizing the query using bm25s tokenizer
        query_tokens = bm25s.tokenize([query], stopwords = "en")

        # If filters are provided, retrieve more candidates for filtering
        search_k = min(top_k * 3, len(self.chunks)) if filters else min(top_k, len(self.chunks))

        # Retrieving top candidates (returns doc indices and scores)
        doc_ids, scores = self.bm25.retrieve(query_tokens, k = search_k)

        results = []
        for i in range(len(doc_ids[0])):
            idx = int(doc_ids[0][i])
            score = float(scores[0][i])

            # Applying the metadata filter
            if filters and not self.matches_filter(self.chunk_metadata[idx], filters):
                continue

            results.append({
                "text": self.chunks[idx],
                "metadata": self.chunk_metadata[idx],
                "score": score,
                "rank": len(results) + 1
            })

            # Stop if we have enough results
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
        It searches using BM25 and returns Document objects.

        Args:
            query: Query string
            run_manager: Callback manager for logging/tracing (LangChain provides this)
            top_k: Number of results to return (overrides self.top_k if provided)
            filters: Metadata filters (conference, year, title, section_type)

        Returns:
            List of LangChain Document objects with page_content and metadata
        """
        # Use instance default if not specified
        k = top_k if top_k is not None else self.top_k

        results = self.search(query, top_k=k, filters=filters)

        return [
            Document(
                page_content=result["text"],
                metadata={**result["metadata"], "score": result["score"], "rank": result["rank"], "retriever": "bm25"}
            )
            for result in results
        ]
