"""Shared LangGraph state for the ResearchRAG pipeline."""

from __future__ import annotations

from typing import Annotated, Any, Dict, List

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ResearchRAGState(TypedDict):
    # Persisted across turns by MemorySaver — full conversation history.
    messages: Annotated[List[BaseMessage], add_messages]

    # Ephemeral per turn — reset in decompose_node at the start of each turn.
    query: str
    sub_queries: List[str]
    chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    tool_iterations: int
    human_approved: bool   # set by human_approval node; True means proceed to synthesis.
    retrieval_feedback: str  # user hint on rejection; fed back into decompose for retry.
    retrieval_attempts: int  # counts feedback-loop iterations; capped at _MAX_RETRIEVAL_ATTEMPTS.
