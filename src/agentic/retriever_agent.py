"""
Retriever agent — fetches relevant paper chunks for a query.
"""

import logging
from typing import Any, Dict

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class RetrieverAgent(BaseAgent):

    def __init__(self, rag_chain, llm):
        """
        Args:
            rag_chain: RAGChain instance used for retrieval.
            llm: Language model (unused for retrieval; kept for interface consistency).
        """
        self.rag_chain = rag_chain
        super().__init__(
            name="RetrieverAgent",
            description="Retrieves relevant research paper chunks",
            llm=llm,
        )

    def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run retrieval for the given query.

        Args:
            task: Query string.
            context: Dict containing top_k and filters keys.

        Returns:
            Dict with keys: agent, query, chunks, count, filters.
        """
        query = task
        top_k = context.get("top_k", 5)
        filters = context.get("filters")

        logger.debug("Retrieving: %r  top_k=%d  filters=%s", query[:50], top_k, filters)

        if filters is None:
            _, filters = self.rag_chain.query_parser.parse(query)
            if filters:
                logger.debug("Auto-parsed filters: %s", filters)

        query_embedding = self.rag_chain.embedding_generator.embed_query(query)
        initial_k = top_k * 3

        from src.retrieval.hybrid import HybridRetriever
        from src.retrieval.bm25 import BM25Retriever

        if isinstance(self.rag_chain.retriever, HybridRetriever):
            chunks = self.rag_chain.retriever.search(query, query_embedding, top_k=initial_k, filters=filters)
        elif isinstance(self.rag_chain.retriever, BM25Retriever):
            chunks = self.rag_chain.retriever.search(query, top_k=initial_k, filters=filters)
        else:
            chunks = self.rag_chain.retriever.search(query_embedding, top_k=initial_k, filters=filters)

        if chunks:
            reranked = self.rag_chain.reranker.rerank(query, chunks, top_k=top_k)
            logger.info("Retrieved and reranked %d chunks", len(reranked))
        else:
            reranked = []
            logger.warning("No chunks retrieved for query: %r", query[:50])

        return {
            "agent": self.name,
            "query": query,
            "chunks": reranked,
            "count": len(reranked),
            "filters": filters,
        }
