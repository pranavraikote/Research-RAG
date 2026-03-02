"""
LangChain @tool definitions for the ReAct agent.

Tools
-----
search_papers         — general hybrid retrieval + cross-encoder rerank
search_papers_in_section — section-filtered retrieval
search_papers_multi   — query decomposition → multi-sub-query retrieval → RRF merge
detect_relevant_sections — map query to relevant section types

The global _rag_chain and _decomposer references are injected once at
agent-build time via set_rag_chain(). This avoids passing rag_chain through
the LangGraph state, keeping state serialisable.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from langchain_core.tools import tool

from .safety import sanitize_query, wrap_retrieved_content
from src.retrieval.query_parser import QueryParser

logger = logging.getLogger(__name__)

_rag_chain = None
_decomposer = None
_corpus_has_sections: bool = False   # True only when chunks carry section_type metadata
_query_parser = QueryParser()        # Extracts conference/year filters from natural language

# Section keyword map — mirrors AdaptiveRetriever.SECTION_KEYWORDS so the tool
# works regardless of which retriever backend is in use (Hybrid, BM25, Adaptive).
_SECTION_KEYWORDS: dict[str, list[str]] = {
    "abstract":     ["abstract", "summary", "overview"],
    "introduction": ["introduction", "intro", "background", "motivation"],
    "related_work": ["related work", "prior work", "previous work", "literature"],
    "methods":      ["method", "methods", "methodology", "approach", "technique",
                     "algorithm", "model", "architecture"],
    "experiments":  ["experiment", "experiments", "experimental", "evaluation",
                     "setup", "implementation"],
    "results":      ["result", "results", "finding", "findings", "performance",
                     "accuracy", "metric", "metrics", "score"],
    "discussion":   ["discussion", "analysis", "interpretation"],
    "conclusion":   ["conclusion", "conclusions", "future work", "future"],
    "limitations":  ["limitation", "limitations", "weakness", "weaknesses"],
}


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def set_rag_chain(rag_chain) -> None:
    """Inject the RAGChain instance and initialise the query decomposer."""
    global _rag_chain, _decomposer, _corpus_has_sections
    _rag_chain = rag_chain

    from .decomposer import QueryDecomposer
    _decomposer = QueryDecomposer(llm=getattr(rag_chain, "llm", None), max_sub_queries=4)

    # Probe corpus for section_type metadata — used to decide whether section
    # filters are worth applying. Basic chunks lack this; adaptive chunks have it.
    try:
        import json
        chunks_path = getattr(rag_chain.retriever, "chunks_path", None)
        if chunks_path is None:
            # HybridRetriever — try the inner semantic retriever
            chunks_path = getattr(
                getattr(rag_chain.retriever, "semantic_retriever", None), "chunks_path", None
            )
        if chunks_path:
            with open(chunks_path) as f:
                sample = json.load(f)
            items = sample if isinstance(sample, list) else list(sample.values())
            _corpus_has_sections = any(
                isinstance(c, dict) and c.get("metadata", {}).get("section_type")
                for c in items[:200]          # only probe first 200 chunks
            )
            logger.info("Corpus section_type support: %s (path=%s)", _corpus_has_sections, chunks_path)
    except Exception as exc:
        logger.warning("Could not probe corpus for section_type: %s", exc)
        _corpus_has_sections = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _search(query: str, top_k: int, filters: Optional[dict] = None) -> list:
    """
    Unified search that routes to the correct retriever API.

    Each retriever has a different search() signature:
      - HybridRetriever:  search(query, query_embedding, top_k, filters)
      - SemanticRetriever: search(query_embedding, top_k, filters)
      - BM25Retriever / AdaptiveRetriever: search(query, top_k, filters)

    Metadata filters (conference, year) are automatically extracted from the query
    text via QueryParser and merged with any explicit filters passed in.
    Explicit filters take precedence on key conflicts.
    """
    # Sanitize before hitting the retriever
    clean_query, flagged = sanitize_query(query)
    if flagged:
        logger.warning("Sanitized query before retrieval: %r → %r", query[:80], clean_query[:80])

    # Auto-extract metadata filters (conference, year) from the query text.
    # e.g. "Any from EMNLP?" → {"conference": "EMNLP"}
    _, auto_filters = _query_parser.parse(clean_query)

    # Merge: auto-extracted as base, explicit caller filters override on conflict.
    # This lets section_type filters from search_papers_in_section coexist with
    # conference/year filters inferred from the query string.
    if auto_filters:
        merged_filters = {**auto_filters, **(filters or {})}
        if merged_filters != (filters or {}):
            logger.debug("Auto-extracted metadata filters: %s", auto_filters)
    else:
        merged_filters = filters

    retriever = _rag_chain.retriever
    retriever_type = type(retriever).__name__

    if retriever_type == "HybridRetriever":
        embedding = _rag_chain.embedding_generator.embed_query(clean_query)
        return retriever.search(clean_query, embedding, top_k=top_k, filters=merged_filters)

    if retriever_type == "SemanticRetriever":
        embedding = _rag_chain.embedding_generator.embed_query(clean_query)
        return retriever.search(embedding, top_k=top_k, filters=merged_filters)

    # BM25Retriever, AdaptiveRetriever — query string only
    return retriever.search(clean_query, top_k=top_k, filters=merged_filters)


def _rrf_merge(result_lists: List[List[dict]], top_k: int, k: int = 60) -> List[dict]:
    """
    Reciprocal Rank Fusion across multiple result lists.

    Each chunk is scored by summing 1/(k + rank) across all lists it appears in.
    Deduplication is by chunk_id from metadata.
    """
    scores: dict[str, float] = {}
    all_chunks: dict[str, dict] = {}

    for results in result_lists:
        for rank, result in enumerate(results, 1):
            chunk_id = result.get("metadata", {}).get("chunk_id", f"_r{rank}")
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in all_chunks:
                all_chunks[chunk_id] = result

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]
    merged = []
    for rank, cid in enumerate(sorted_ids, 1):
        chunk = all_chunks[cid].copy()
        chunk["score"] = scores[cid]
        chunk["rank"] = rank
        merged.append(chunk)

    return merged


def _detect_sections(query: str) -> List[str]:
    """Return section types whose keywords appear in the query."""
    q = query.lower()
    detected = []
    for section_type, keywords in _SECTION_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", q):
                detected.append(section_type)
                break
    return detected


def _format_chunks(chunks: list, query: str, label: str = "") -> str:
    """Format retrieved chunks as a cited, injection-safe string for the agent."""
    if not chunks:
        return "No results."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        title = meta.get("title", "Unknown")
        conf = meta.get("conference", "")
        year = meta.get("year", "")
        section = meta.get("section_type", "")
        score = chunk.get("score", 0.0)
        raw_text = chunk.get("text", "")[:400]

        header = f"[{i}] {title} ({conf} {year})"
        if section:
            header += f" — {label or section}"
        header += f" | score={score:.3f}"

        # Wrap content so LLM treats it as data, not instructions
        safe_text = wrap_retrieved_content(raw_text)
        parts.append(f"{header}\n{safe_text}\n")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def search_papers(query: str, top_k: int = 5) -> str:
    """Search ACL Anthology research papers using hybrid retrieval and cross-encoder reranking.

    Use this tool for simple, single-topic queries.
    The search combines semantic (FAISS) and keyword (BM25) retrieval then
    reranks results with a cross-encoder for high precision.
    For complex multi-aspect queries, prefer search_papers_multi instead.

    Args:
        query: The search query — be specific and use research terminology.
        top_k: Number of top results to return (default 5, max 10).

    Returns:
        Formatted string listing matching paper chunks with title, conference,
        year, section, score, and text excerpt inside <retrieved_content> tags.
    """
    top_k = min(top_k, 10)
    chunks = _search(query, top_k=top_k * 3)
    if not chunks:
        return "No results found."
    reranked = _rag_chain.reranker.rerank(query, chunks, top_k=top_k)
    return _format_chunks(reranked, query)


@tool
def search_papers_in_section(query: str, section_type: str, top_k: int = 5) -> str:
    """Search for content within a specific section type across all papers.

    Use this tool when you need information from a particular part of papers,
    e.g. 'methods' for technical details, 'results' for performance numbers,
    'limitations' for weaknesses.

    Args:
        query: The search query.
        section_type: One of: abstract, introduction, related_work, methods,
                      experiments, results, discussion, conclusion, limitations.
        top_k: Number of results to return (default 5).

    Returns:
        Formatted string of matching chunks from the specified section type.
    """
    valid_sections = {
        "abstract", "introduction", "related_work", "methods",
        "experiments", "results", "discussion", "conclusion", "limitations",
    }
    if section_type not in valid_sections:
        return f"Invalid section_type '{section_type}'. Choose from: {', '.join(sorted(valid_sections))}"

    # Section filters only work on adaptive chunks that carry section_type metadata.
    # When the corpus uses basic chunks (no metadata), fall back to unfiltered search
    # so the tool still returns useful results instead of an empty set.
    if _corpus_has_sections:
        filters = {"section_type": [section_type]}
    else:
        filters = None
        logger.info(
            "search_papers_in_section: corpus has no section_type metadata — "
            "falling back to unfiltered search for section '%s'", section_type
        )

    chunks = _search(query, top_k=top_k * 3, filters=filters)
    if not chunks:
        return f"No results found for '{query}'."
    reranked = _rag_chain.reranker.rerank(query, chunks, top_k=top_k)
    label = f"[{section_type}]" if _corpus_has_sections else f"[unfiltered — no section metadata]"
    return _format_chunks(reranked, query, label=label)


@tool
def search_papers_multi(query: str, top_k: int = 5) -> str:
    """Search using query decomposition and multi-query retrieval with RRF merge.

    Use this tool for complex multi-part queries such as:
    - Comparisons ("compare X and Y")
    - Multi-aspect questions ("what methods do papers use and what results do they report?")
    - Queries spanning different paper sections

    Internally this tool:
    1. Decomposes the query into 1-4 focused sub-queries (LLM or heuristic)
    2. Runs retrieval for each sub-query with appropriate section filters
    3. Merges all results using Reciprocal Rank Fusion (RRF)
    4. Reranks the merged set with the cross-encoder

    This directly addresses the retrieval fragmentation problem in adaptive chunking,
    where related content lives in separate section chunks.

    Args:
        query: The complex research query to decompose and search.
        top_k: Number of final results after merging (default 5, max 10).

    Returns:
        Formatted string of merged, reranked chunks with full citation metadata.
    """
    top_k = min(top_k, 10)

    # --- Decompose ---
    sub_queries = _decomposer.decompose(query)
    logger.info(
        "search_papers_multi: %d sub-queries for %r: %s",
        len(sub_queries),
        query[:60],
        [(sq.query[:40], sq.section_hint) for sq in sub_queries],
    )

    # --- Retrieve per sub-query ---
    # Only apply section_type filters when the corpus actually carries that metadata
    # (adaptive chunks do; basic chunks do not — filtering on missing field = 0 results)
    all_result_lists: List[List[dict]] = []
    for sq in sub_queries:
        use_section = _corpus_has_sections and sq.section_hint and sq.section_hint != "general"
        filters = {"section_type": [sq.section_hint]} if use_section else None
        results = _search(sq.query, top_k=top_k * 3, filters=filters)
        if results:
            all_result_lists.append(results)

    if not all_result_lists:
        return "No results found across all sub-queries."

    # --- RRF merge ---
    merged = _rrf_merge(all_result_lists, top_k=top_k * 2)

    # --- Rerank merged set ---
    reranked = _rag_chain.reranker.rerank(query, merged, top_k=top_k)

    # Prepend decomposition info so the agent knows what was searched
    sub_query_summary = "\n".join(
        f"  • {sq.query}" + (f" [section:{sq.section_hint}]" if sq.section_hint else "")
        for sq in sub_queries
    )
    header = f"[Decomposed into {len(sub_queries)} sub-queries]\n{sub_query_summary}\n\n"
    return header + _format_chunks(reranked, query)


@tool
def detect_relevant_sections(query: str) -> str:
    """Detect which paper sections are most relevant to a query.

    Use this tool to understand what section types to target before calling
    search_papers_in_section. For example, a query about 'training procedure'
    maps to 'methods', while 'benchmark comparison' maps to 'results'.

    Args:
        query: The query to analyse.

    Returns:
        Comma-separated list of relevant section types, or a suggestion to
        use search_papers for a general search.
    """
    sections = _detect_sections(query)
    if not sections:
        return "No specific section type detected — use search_papers or search_papers_multi."
    return f"Relevant sections: {', '.join(sections)}"


TOOLS = [search_papers_multi, search_papers, search_papers_in_section, detect_relevant_sections]
