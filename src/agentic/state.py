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
