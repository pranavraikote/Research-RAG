"""
LangChain @tool definitions for the ReAct agent.

Tools
-----
search_papers             — general hybrid retrieval + cross-encoder rerank
search_papers_in_section  — section-filtered retrieval
search_papers_multi       — query decomposition → multi-sub-query retrieval → RRF merge
detect_relevant_sections  — map query to relevant section types
search_pubmed             — live PubMed search via NCBI E-utilities

All tools are created per-agent via make_tools(rag_chain), which returns a list of
@tool closures that each capture their own rag_chain reference. There is no module-level
state — multiple ResearchRAGGraph instances in the same process are fully independent.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import List

from langchain_core.tools import tool

from .safety import sanitize_query, wrap_retrieved_content
from src.retrieval.query_parser import QueryParser


logger = logging.getLogger(__name__)

_query_parser = QueryParser()  # stateless; safe to share across instances

# Section keyword map — mirrors AdaptiveRetriever.SECTION_KEYWORDS so the tool
# works regardless of which retriever backend is in use (Hybrid, BM25, Adaptive).
_SECTION_KEYWORDS: dict[str, list[str]] = {
    "abstract":     ["abstract", "overview"],
    "introduction": ["introduction", "intro", "background", "motivation"],
    "related_work": ["related work", "prior work", "previous work", "literature review"],
    "methods":      ["method", "methods", "methodology", "approach", "technique",
                     "algorithm"],
    "experiments":  ["experiment", "experiments", "experimental",
                     "evaluation setup", "implementation details"],
    "results":      ["result", "results", "finding", "findings",
                     "accuracy", "metric", "metrics"],
    "discussion":   ["discussion", "interpretation"],
    "conclusion":   ["conclusion", "conclusions", "future work"],
    "limitations":  ["limitation", "limitations", "weakness", "weaknesses"],
}


# ---------------------------------------------------------------------------
# Internal helpers (pure — no global state)
# ---------------------------------------------------------------------------

def _probe_corpus(rag_chain) -> bool:
    """Return True if the corpus chunks carry section_type metadata."""
    try:
        chunks_path = getattr(rag_chain.retriever, "chunks_path", None)
        if chunks_path is None:
            chunks_path = getattr(
                getattr(rag_chain.retriever, "semantic_retriever", None), "chunks_path", None
            )
        if not chunks_path:
            return False
        with open(chunks_path) as f:
            sample = json.load(f)
        items = sample if isinstance(sample, list) else list(sample.values())
        result = any(
            isinstance(c, dict) and c.get("metadata", {}).get("section_type")
            for c in items[:200]
        )
        logger.info("Corpus section_type support: %s (path=%s)", result, chunks_path)
        return result
    except Exception as exc:
        logger.warning("Could not probe corpus for section_type: %s", exc)
        return False


def _search(query: str, top_k: int, rag_chain, filters=None) -> list:
    """
    Unified search that routes to the correct retriever API.

    Each retriever has a different search() signature:
      - HybridRetriever:   search(query, query_embedding, top_k, filters)
      - SemanticRetriever: search(query_embedding, top_k, filters)
      - BM25Retriever / AdaptiveRetriever: search(query, top_k, filters)

    Metadata filters (conference, year) are automatically extracted from the query
    text via QueryParser and merged with any explicit filters passed in.
    Explicit filters take precedence on key conflicts.
    """
    clean_query, flagged = sanitize_query(query)
    if flagged:
        logger.warning("Sanitized query before retrieval: %r → %r", query[:80], clean_query[:80])

    _, auto_filters = _query_parser.parse(clean_query)
    if auto_filters:
        merged_filters = {**auto_filters, **(filters or {})}
        if merged_filters != (filters or {}):
            logger.debug("Auto-extracted metadata filters: %s", auto_filters)
    else:
        merged_filters = filters

    retriever = rag_chain.retriever
    retriever_type = type(retriever).__name__

    if retriever_type == "HybridRetriever":
        embedding = rag_chain.embedding_generator.embed_query(clean_query)
        return retriever.search(clean_query, embedding, top_k=top_k, filters=merged_filters)

    if retriever_type == "SemanticRetriever":
        embedding = rag_chain.embedding_generator.embed_query(clean_query)
        return retriever.search(embedding, top_k=top_k, filters=merged_filters)

    return retriever.search(clean_query, top_k=top_k, filters=merged_filters)


def _rrf_merge(result_lists: List[List[dict]], top_k: int, k: int = 60) -> List[dict]:
    """
    Reciprocal Rank Fusion across multiple result lists.

    Each chunk is scored by summing 1/(k + rank) across all lists it appears in.
    Deduplication is by chunk_id from metadata.
    """
    scores: dict[str, float] = {}
    all_chunks: dict[str, dict] = {}

    for list_idx, results in enumerate(result_lists):
        for rank, result in enumerate(results, 1):
            chunk_id = result.get("metadata", {}).get("chunk_id") \
                or f"_noid_L{list_idx}_R{rank}"
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


def _detect_sections(query: str) -> list[str]:
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
        raw_text = chunk.get("text", "")

        header = f"[{i}] {title} ({conf} {year})"
        if section:
            header += f" — {label or section}"

        safe_text = wrap_retrieved_content(raw_text)
        parts.append(f"{header}\n{safe_text}\n→ Use [{i}] when citing any fact from the above.\n")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# PubMed helpers (no rag_chain dependency)
# ---------------------------------------------------------------------------

_NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def _pubmed_fetch(query: str, max_results: int) -> list[dict]:
    """
    Query NCBI E-utilities (esearch + efetch) and return article dicts.

    Each dict has: pmid, title, authors, journal, year, abstract.
    Respects NCBI_API_KEY env var for higher rate limits (10 req/s vs 3 req/s).
    """
    api_key = os.environ.get("NCBI_API_KEY", "")

    esearch_params: dict = {
        "db": "pubmed", "term": query,
        "retmax": str(max_results), "retmode": "json",
    }
    if api_key:
        esearch_params["api_key"] = api_key

    url = _NCBI_BASE + "esearch.fcgi?" + urllib.parse.urlencode(esearch_params)
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read())
    pmids = data.get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return []

    efetch_params: dict = {
        "db": "pubmed", "id": ",".join(pmids),
        "rettype": "abstract", "retmode": "xml",
    }
    if api_key:
        efetch_params["api_key"] = api_key

    url = _NCBI_BASE + "efetch.fcgi?" + urllib.parse.urlencode(efetch_params)
    with urllib.request.urlopen(url, timeout=15) as resp:
        root = ET.fromstring(resp.read())

    articles = []
    for article_el in root.findall(".//PubmedArticle"):
        citation = article_el.find("MedlineCitation")
        if citation is None:
            continue
        article = citation.find("Article")
        if article is None:
            continue

        pmid = citation.findtext("PMID", "")

        title_el = article.find("ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else "Unknown"

        abstract_els = article.findall(".//AbstractText")
        if abstract_els:
            parts = []
            for at in abstract_els:
                label = at.get("Label", "")
                text = "".join(at.itertext()).strip()
                parts.append(f"{label}: {text}" if label else text)
            abstract = " ".join(parts)
        else:
            abstract = ""

        last_names = [
            a.findtext("LastName", "") for a in article.findall(".//Author")
            if a.findtext("LastName")
        ]
        author_str = (
            f"{last_names[0]} et al." if len(last_names) > 1
            else last_names[0] if last_names
            else "Unknown"
        )

        journal = article.findtext(".//Journal/Title", "")

        pub_date = article.find(".//PubDate")
        year = ""
        if pub_date is not None:
            year = pub_date.findtext("Year", "")
            if not year:
                medline = pub_date.findtext("MedlineDate", "")
                year = medline[:4] if medline else ""

        articles.append({
            "pmid": pmid, "title": title, "authors": author_str,
            "journal": journal, "year": year, "abstract": abstract[:700],
        })

    return articles


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def make_tools(rag_chain) -> list:
    """
    Build and return a fresh list of @tool callables bound to rag_chain.

    Each call produces independent closures — no module-level state is mutated.
    Multiple ResearchRAGGraph instances can coexist safely in the same process.

    Args:
        rag_chain: Initialised RAGChain instance.

    Returns:
        List of LangChain tool callables ready for bind_tools() and ToolNode.
    """
    from .decomposer import QueryDecomposer

    decomposer = QueryDecomposer(llm=rag_chain.llm, max_sub_queries=4)
    corpus_has_sections = _probe_corpus(rag_chain)

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
            Formatted string of matching paper chunks with title, conference, year, section, and text.
        """
        top_k = min(top_k, 10)
        chunks = _search(query, top_k * 3, rag_chain)
        if not chunks:
            return "No results found."
        reranked = rag_chain.reranker.rerank(query, chunks, top_k=top_k)
        result = _format_chunks(reranked, query)
        return result + "\nRemember: cite each fact with its [N] number in your answer."

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

        if corpus_has_sections:
            filters = {"section_type": [section_type]}
        else:
            filters = None
            logger.info(
                "search_papers_in_section: corpus has no section_type metadata — "
                "falling back to unfiltered search for section '%s'", section_type
            )

        chunks = _search(query, top_k * 3, rag_chain, filters=filters)
        if not chunks:
            return f"No results found for '{query}'."
        reranked = rag_chain.reranker.rerank(query, chunks, top_k=top_k)
        label = f"[{section_type}]" if corpus_has_sections else "[unfiltered — no section metadata]"
        result = _format_chunks(reranked, query, label=label)
        return result + "\nRemember: cite each fact with its [N] number in your answer."

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

        Args:
            query: The complex research query to decompose and search.
            top_k: Number of final results after merging (default 5, max 10).

        Returns:
            Formatted string of merged, reranked chunks with full citation metadata.
        """
        top_k = min(top_k, 10)

        sub_queries = decomposer.decompose(query)
        logger.info(
            "search_papers_multi: %d sub-queries for %r: %s",
            len(sub_queries), query[:60],
            [(sq.query[:40], sq.section_hint) for sq in sub_queries],
        )

        all_result_lists: List[List[dict]] = []
        for sq in sub_queries:
            use_section = corpus_has_sections and sq.section_hint and sq.section_hint != "general"
            filters = {"section_type": [sq.section_hint]} if use_section else None
            results = _search(sq.query, top_k * 3, rag_chain, filters=filters)
            if results:
                all_result_lists.append(results)

        if not all_result_lists:
            return "No results found across all sub-queries."

        merged = _rrf_merge(all_result_lists, top_k=top_k * 2)
        reranked = rag_chain.reranker.rerank(query, merged, top_k=top_k)

        logger.info(
            "search_papers_multi: returning %d chunks from %d sub-queries",
            len(reranked), len(sub_queries),
        )
        result = _format_chunks(reranked, query)
        return result + "\nRemember: cite each fact with its [N] number in your answer."

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

    @tool
    def search_pubmed(query: str, max_results: int = 5) -> str:
        """Search PubMed for biomedical and clinical NLP literature via NCBI E-utilities.

        Use this for queries about biomedical, clinical, or health-related NLP topics
        not well-covered by the ACL corpus — e.g. clinical NLP, BioNLP, medical text
        mining, electronic health records, biomedical relation extraction.
        For core NLP/ML/AI research, prefer search_papers or search_papers_multi.

        Results are fetched live from PubMed (not pre-indexed). Only abstracts are
        returned — full text is not available via this tool.

        Supports MeSH terms and Boolean operators, e.g.:
          "clinical NLP AND electronic health records"
          "biomedical named entity recognition[MeSH]"

        Args:
            query: PubMed search query string.
            max_results: Number of results to return (default 5, max 10).

        Returns:
            Formatted list of PubMed articles with title, authors, journal, year,
            PMID, and abstract excerpt.
        """
        max_results = min(max_results, 10)
        clean_query, _ = sanitize_query(query)
        try:
            articles = _pubmed_fetch(clean_query, max_results)
        except Exception as exc:
            logger.warning("PubMed fetch failed for query %r: %s", clean_query[:60], exc)
            return "PubMed search failed. Try rephrasing your query."

        if not articles:
            return "No PubMed results found."

        parts = []
        for i, art in enumerate(articles, 1):
            header = (
                f"[{i}] {art['title']} — {art['authors']}"
                f" ({art['journal']}, {art['year']}) PMID:{art['pmid']}"
            )
            body = wrap_retrieved_content(art["abstract"]) if art["abstract"] else "[No abstract available]"
            parts.append(f"{header}\n{body}\n")

        return "\n".join(parts)

    return [search_papers_multi, search_papers, search_papers_in_section,
            detect_relevant_sections, search_pubmed]
