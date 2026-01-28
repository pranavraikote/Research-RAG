"""
Retriever Agent

Handles retrieval of relevant paper chunks.
"""

import logging
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class RetrieverAgent(BaseAgent):
    
    def __init__(self, rag_chain, llm):
        """
        Initialize retriever agent function.
        
        Args:
            rag_chain: RAGChain instance for retrieval
            llm: Language model (not used for retrieval, but kept for consistency)
        """
        
        self.rag_chain = rag_chain
        
        super().__init__(
            name = "RetrieverAgent",
            description = "Retrieves relevant research paper chunks",
            llm = llm
        )
    
    def execute(self, task, context):
        """
        Execute retrieval task function.
        
        Args:
            task: Query string
            context: Context with top_k, filters, etc.
            
        Returns:
            Retrieval results with chunks
        """
        
        query = task
        top_k = context.get("top_k", 5)
        filters = context.get("filters")
        
        logger.debug(f"Retrieving with query: {query[:50]}..., top_k={top_k}, filters={filters}")
        
        # Parse query for filters if not provided
        if filters is None:
            _, parsed_filters = self.rag_chain.query_parser.parse(query)
            filters = parsed_filters
            if filters:
                logger.debug(f"Auto-parsed filters: {filters}")
        
        # Get embeddings for query
        query_embedding = self.rag_chain.embedding_generator.embed_query(query)
        
        # Retrieve chunks using retriever's search method
        initial_k = top_k * 3  # Retrieve more for reranking
        logger.debug(f"Retrieving {initial_k} chunks initially, will rerank to {top_k}")
        
        from src.retrieval.hybrid import HybridRetriever
        from src.retrieval.bm25 import BM25Retriever
        
        if isinstance(self.rag_chain.retriever, HybridRetriever):
            chunks = self.rag_chain.retriever.search(
                query, query_embedding, top_k = initial_k, filters = filters
            )
        elif isinstance(self.rag_chain.retriever, BM25Retriever):
            chunks = self.rag_chain.retriever.search(
                query, top_k = initial_k, filters = filters
            )
        else:
            # Semantic retriever
            chunks = self.rag_chain.retriever.search(
                query_embedding, top_k = initial_k, filters = filters
            )
        
        # Rerank if we have chunks
        if chunks:
            logger.debug(f"Reranking {len(chunks)} chunks to top {top_k}")
            reranked = self.rag_chain.reranker.rerank(query, chunks, top_k = top_k)
            logger.info(f"Retrieved and reranked {len(reranked)} chunks")
        else:
            reranked = []
            logger.warning("No chunks retrieved")
        
        return {
            "agent": self.name,
            "query": query,
            "chunks": reranked,
            "count": len(reranked),
            "filters": filters
        }
