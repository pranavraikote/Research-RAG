"""
Adaptive Retriever - Automatically chooses retrieval strategy based on query.

Strategy selection:
1. Detect section keywords in query → use section filtering
2. No keywords → basic retrieval
3. Optionally expand with neighbors for better context

Works with both:
- BM25Retriever (sparse only)
- HybridRetriever (sparse + dense fusion)
"""
import re
from typing import List, Dict, Optional, Union, Any
from pydantic import ConfigDict
from .bm25 import BM25Retriever
from .semantic import SemanticRetriever
from .context_expander import ContextExpander

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

try:
    from .hybrid import HybridRetriever
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False


class AdaptiveRetriever(BaseRetriever):
    """
    Smart retriever that automatically adapts strategy based on query.

    Features:
    - Keyword-based section detection
    - Automatic section filtering
    - Optional context expansion
    - Optional same-section merging
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Section keywords mapping (includes both singular and plural forms)
    SECTION_KEYWORDS = {
        "abstract": ["abstract", "summary", "overview"],
        "introduction": ["introduction", "intro", "background", "motivation"],
        "related_work": ["related work", "prior work", "previous work", "literature"],
        "methods": ["method", "methods", "methodology", "approach", "technique", "algorithm", "model", "architecture"],
        "experiments": ["experiment", "experiments", "experimental", "evaluation", "setup", "implementation"],
        "results": ["result", "results", "finding", "findings", "performance", "accuracy", "metric", "metrics", "score"],
        "discussion": ["discussion", "analysis", "interpretation"],
        "conclusion": ["conclusion", "conclusions", "summary", "future work", "future"],
        "limitations": ["limitation", "limitations", "weakness", "weaknesses"],
    }

    def __init__(
        self,
        retriever: Union[BM25Retriever, 'HybridRetriever'],
        all_chunks: List[str] = None,
        all_metadata: List[Dict] = None,
        enable_expansion: bool = True,
        enable_merging: bool = False,
        expansion_window: int = 1,
        merge_threshold: int = 3,
        embedding_model = None,  # For HybridRetriever query embedding
        top_k: int = 10,  # LangChain default
        **kwargs
    ):
        """
        Initialize adaptive retriever (LangChain-compatible).

        Works with either:
        - BM25Retriever (sparse only)
        - HybridRetriever (sparse + dense fusion)

        Args:
            retriever: BM25Retriever or HybridRetriever instance
            all_chunks: All chunk texts (for expansion)
            all_metadata: All chunk metadata (for expansion)
            enable_expansion: Whether to expand with neighbors
            enable_merging: Whether to merge same-section chunks
            expansion_window: Number of chunks to add before/after
            merge_threshold: Min chunks to trigger merging
            embedding_model: SentenceTransformer model (for HybridRetriever)
            top_k: Default number of results to return (used by LangChain)
            **kwargs: Additional arguments for BaseRetriever
        """
        lc_kwargs = {k: v for k, v in kwargs.items() if k in ("tags",)}
        super().__init__(**lc_kwargs)

        self.retriever = retriever
        self.enable_expansion = enable_expansion
        self.enable_merging = enable_merging
        self.expansion_window = expansion_window
        self.merge_threshold = merge_threshold
        self.embedding_model = embedding_model
        self.top_k = top_k

        # Detect retriever type
        self.is_hybrid = HYBRID_AVAILABLE and hasattr(retriever, 'semantic_retriever')
        self.is_semantic = isinstance(retriever, SemanticRetriever)

        # Setup context expander if enabled
        if (enable_expansion or enable_merging) and all_chunks and all_metadata:
            self.expander = ContextExpander(all_chunks, all_metadata)
        else:
            self.expander = None

    def detect_section_keywords(self, query: str) -> Optional[List[str]]:
        """
        Detect section keywords in query.

        Args:
            query: User query text

        Returns:
            List of detected section types, or None if no keywords found
        """
        query_lower = query.lower()
        detected_sections = []

        for section_type, keywords in self.SECTION_KEYWORDS.items():
            for keyword in keywords:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, query_lower):
                    detected_sections.append(section_type)
                    break  # One match per section type is enough

        return detected_sections if detected_sections else None

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        force_basic: bool = False,
        verbose: bool = False,
        query_embedding = None  # For HybridRetriever
    ) -> List[Dict]:
        """
        Adaptive search that automatically chooses strategy.

        Logic:
        1. Detect section keywords in query
        2. If keywords found → add section filter
        3. Retrieve chunks (sparse + dense if hybrid)
        4. Optionally expand with neighbors
        5. Optionally merge same-section chunks

        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            filters: Additional filters (year, conference, etc.)
            force_basic: Force basic retrieval (no section filtering)
            verbose: Print strategy decisions
            query_embedding: Pre-computed query embedding (for HybridRetriever)

        Returns:
            List of retrieved chunks with metadata
        """
        # Initialize filters
        if filters is None:
            filters = {}

        strategy_used = "basic"

        # STEP 1: Detect section keywords
        if not force_basic and "section_type" not in filters:
            detected_sections = self.detect_section_keywords(query)

            if detected_sections:
                # Add section filter
                filters["section_type"] = detected_sections
                strategy_used = f"section_filtered({', '.join(detected_sections)})"

                if verbose:
                    print(f"🔍 Detected sections: {detected_sections}")
                    print(f"   Filtering to: {detected_sections}")

        # STEP 2: Retrieval (sparse + dense if hybrid)
        if verbose:
            retriever_type = "Hybrid (sparse+dense)" if self.is_hybrid else "BM25 (sparse)"
            print(f"📊 Retriever: {retriever_type}")
            print(f"   Strategy: {strategy_used}")
            print(f"   Filters: {filters}")

        # Retrieve more chunks if we'll be merging
        retrieve_k = top_k * 2 if self.enable_merging else top_k

        # Handle different retriever types
        if self.is_hybrid:
            # HybridRetriever needs both query and query_embedding
            if query_embedding is None and self.embedding_model:
                query_embedding = self.embedding_model.encode(query)

            results = self.retriever.search(
                query=query,
                query_embedding=query_embedding,
                top_k=retrieve_k,
                filters=filters
            )
        elif self.is_semantic:
            # SemanticRetriever only needs query_embedding
            if query_embedding is None and self.embedding_model:
                query_embedding = self.embedding_model.encode(query)

            results = self.retriever.search(
                query_embedding=query_embedding,
                top_k=retrieve_k,
                filters=filters
            )
        else:
            # BM25Retriever only needs query text
            results = self.retriever.search(
                query=query,
                top_k=retrieve_k,
                filters=filters
            )

        if verbose:
            print(f"   Retrieved: {len(results)} chunks")

        # STEP 3: Context expansion
        if self.enable_expansion and self.expander and results:
            results = self.expander.expand_with_neighbors(
                results,
                window_size=self.expansion_window
            )

            if verbose:
                print(f"   After expansion: {len(results)} chunks (+{len(results) - retrieve_k})")

        # STEP 4: Merging (optional)
        if self.enable_merging and self.expander and results:
            results = self.expander.merge_section_chunks(
                results,
                merge_threshold=self.merge_threshold
            )

            if verbose:
                print(f"   After merging: {len(results)} chunks")

        # Trim to top_k if needed
        results = results[:top_k]

        if verbose:
            print(f"   Final: {len(results)} chunks")

        return results

    def search_with_explanation(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        query_embedding = None
    ) -> tuple:
        """
        Search with explanation of strategy chosen.

        Returns:
            (results, explanation_dict)
        """
        detected_sections = self.detect_section_keywords(query)

        explanation = {
            "query": query,
            "retriever_type": "hybrid (sparse+dense)" if self.is_hybrid else "sparse (BM25 only)",
            "detected_sections": detected_sections,
            "strategy": "basic" if not detected_sections else "section_filtered",
            "section_filter": detected_sections,
            "expansion_enabled": self.enable_expansion,
            "merging_enabled": self.enable_merging,
        }

        results = self.search(
            query,
            top_k=top_k,
            filters=filters,
            query_embedding=query_embedding,
            verbose=False
        )

        explanation["retrieved_count"] = len(results)

        return results, explanation

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
        It performs adaptive retrieval with automatic section detection.

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

        detected_sections = self.detect_section_keywords(query)

        query_embedding = None
        if (self.is_hybrid or self.is_semantic) and self.embedding_model:
            query_embedding = self.embedding_model.embed_query(query)

        results = self.search(
            query,
            top_k=k,
            filters=filters,
            query_embedding=query_embedding,
            verbose=False
        )

        return [
            Document(
                page_content=result["text"],
                metadata={**result["metadata"], "score": result["score"], "rank": result["rank"], "retriever": "adaptive", "detected_sections": detected_sections}
            )
            for result in results
        ]


# Example usage
if __name__ == "__main__":
    """
    Example: Using adaptive retriever
    """

    # Mock setup
    from .bm25 import BM25Retriever

    retriever = BM25Retriever()
    # ... add chunks ...

    adaptive = AdaptiveRetriever(
        retriever,
        enable_expansion=True,
        enable_merging=False,
        expansion_window=1
    )

    # Example 1: General query → basic retrieval
    results = adaptive.search(
        "What is this paper about?",
        top_k=5,
        verbose=True
    )

    # Example 2: Method query → section filtering
    results = adaptive.search(
        "How does the model architecture work?",
        top_k=5,
        verbose=True
    )

    # Example 3: Results query → section filtering
    results = adaptive.search(
        "What are the experimental results on accuracy?",
        top_k=5,
        verbose=True
    )
