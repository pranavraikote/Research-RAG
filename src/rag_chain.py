import json
import logging
import urllib.request

from langchain_ollama import ChatOllama

from .retrieval.reranker import CrossEncoderReranker
from .retrieval.query_parser import QueryParser

logger = logging.getLogger(__name__)


class RAGChain:
    """Wires together the retriever, reranker, and Ollama LLM.

    This class owns no retrieval logic itself — callers (tools, the ReAct
    agent, and the Streamlit app) access .retriever, .reranker,
    .embedding_generator, and .llm directly.
    """

    @staticmethod
    def _check_ollama_available(model_name: str) -> bool:
        """Return True if Ollama is running and model_name is loaded."""
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                return any(m == model_name or m.startswith(model_name + "-") for m in models)
        except Exception:
            return False

    def __init__(self, embedding_generator, retriever, ollama_model: str = "qwen2.5:7b"):
        """
        Args:
            embedding_generator: EmbeddingGenerator instance.
            retriever:           SemanticRetriever, BM25Retriever, or HybridRetriever.
            ollama_model:        Ollama model tag (e.g. "qwen3:14b").
        """
        self.retriever = retriever
        self.embedding_generator = embedding_generator
        self.query_parser = QueryParser()
        self.reranker = CrossEncoderReranker()

        # Propagate embedding_generator into retriever internals when needed.
        if embedding_generator:
            if hasattr(retriever, "embedding_generator") and retriever.embedding_generator is None:
                retriever.embedding_generator = embedding_generator
            if hasattr(retriever, "semantic_retriever"):
                sem = retriever.semantic_retriever
                if hasattr(sem, "embedding_generator") and sem.embedding_generator is None:
                    sem.embedding_generator = embedding_generator

        ollama_kwargs: dict = {"model": ollama_model, "temperature": 0}
        if ollama_model.startswith("qwen3"):
            ollama_kwargs["think"] = False
        self.llm = ChatOllama(**ollama_kwargs)
        logger.info("RAGChain ready — model=%s", ollama_model)
