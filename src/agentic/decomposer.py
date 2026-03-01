"""
LLM-based query decomposition with heuristic fallback.

Breaks complex multi-aspect queries into focused sub-queries, each optionally
tagged with a target paper section type (methods, results, limitations, …).

Decision flow
-------------
1. LLM pass  — ask the model to split the query into 1-4 sub-queries with
               section hints.  Falls back if the LLM is unavailable or its
               output cannot be parsed.
2. Heuristic — lightweight keyword-based splitting for common patterns
               (comparisons, method+results, limitation queries).
3. Passthrough — return the original query unchanged as a single SubQuery.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

_VALID_SECTIONS = {
    "abstract", "introduction", "related_work", "methods",
    "experiments", "results", "discussion", "conclusion", "limitations", "general",
}

# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_DECOMPOSE_SYSTEM = """\
You are a query decomposition assistant for a research paper retrieval system.
Break the user's research query into 1-4 focused sub-queries for targeted retrieval.

Output format — numbered list, one sub-query per line:
1. <sub-query text> [section:<section_type>]

Section types: abstract, introduction, methods, results, limitations, conclusion, general

Rules:
- Simple, single-topic queries → output exactly 1 sub-query (restate it cleanly)
- Complex queries (multiple aspects, comparisons, method+results) → split by aspect
- Each sub-query must be fully self-contained and searchable on its own
- Output ONLY the numbered list — no explanation, no preamble, nothing else"""

_DECOMPOSE_HUMAN = "Query: {query}\n\nSub-queries:"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SubQuery:
    """A single focused sub-query derived from the original user query."""
    query: str
    section_hint: Optional[str] = None   # e.g. "methods", "results", None → general


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------

class QueryDecomposer:
    """
    Decomposes a research query into focused sub-queries.

    Args:
        llm: LangChain BaseChatModel instance (ChatOllama or ChatHuggingFace).
             If None, only the heuristic fallback is used.
        max_sub_queries: Upper bound on number of sub-queries returned.
    """

    def __init__(self, llm=None, max_sub_queries: int = 4):
        self.llm = llm
        self.max_sub_queries = max_sub_queries

    def decompose(self, query: str) -> List[SubQuery]:
        """
        Return a list of SubQuery objects for the given query.

        Always returns at least one SubQuery (the original query).
        """
        if self.llm is not None:
            try:
                result = self._llm_decompose(query)
                if result:
                    logger.debug(
                        "Decomposed %r into %d sub-queries via LLM", query, len(result)
                    )
                    return result[: self.max_sub_queries]
            except Exception as exc:
                logger.warning("LLM decomposition failed (%s); using heuristic fallback", exc)

        fallback = self._heuristic_decompose(query)
        logger.debug("Heuristic decomposition → %d sub-queries", len(fallback))
        return fallback[: self.max_sub_queries]

    # ------------------------------------------------------------------
    # LLM pass
    # ------------------------------------------------------------------

    def _llm_decompose(self, query: str) -> List[SubQuery]:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self.llm.invoke([
            SystemMessage(content=_DECOMPOSE_SYSTEM),
            HumanMessage(content=_DECOMPOSE_HUMAN.format(query=query)),
        ])
        raw = response.content.strip() if hasattr(response, "content") else str(response).strip()

        if not raw:
            return []

        parsed = self._parse_llm_output(raw)
        # If LLM returned garbage (empty or just repeated the original), fall through
        if not parsed or (len(parsed) == 1 and parsed[0].query.lower() == query.lower()):
            return parsed  # still valid — single clean sub-query is fine
        return parsed

    def _parse_llm_output(self, raw: str) -> List[SubQuery]:
        """
        Parse numbered-list output from the LLM.

        Accepts formats:
          1. sub-query text [section:methods]
          1. sub-query text | section:methods
          1) sub-query text
        """
        sub_queries: List[SubQuery] = []

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue

            # Match leading number
            m = re.match(
                r"^\d+[.)]\s*"          # "1. " or "1) "
                r"(.+?)"                # query text (non-greedy)
                r"(?:"                  # optional section tag
                r"\s*[\[|]\s*section:\s*(\w+)\s*\]?"
                r")?$",
                line,
                re.IGNORECASE,
            )
            if not m:
                continue

            query_text = m.group(1).strip().rstrip("|").strip()
            section = (m.group(2) or "").strip().lower() or None
            if section not in _VALID_SECTIONS:
                section = None

            if query_text and len(query_text) > 3:
                sub_queries.append(SubQuery(query=query_text, section_hint=section))

        return sub_queries

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    def _heuristic_decompose(self, query: str) -> List[SubQuery]:
        """Keyword-based decomposition for common research query patterns."""
        q = query.lower()
        sub_queries: List[SubQuery] = []

        # Comparison: "compare X and Y" / "X vs Y" / "differences between X and Y"
        if re.search(r"\bcompare\b|\bvs\.?\b|\bversus\b|\bdifference(?:s)?\s+between\b", q):
            sub_queries.append(SubQuery(query=f"{query} methods and approach", section_hint="methods"))
            sub_queries.append(SubQuery(query=f"{query} evaluation and results", section_hint="results"))
            return sub_queries

        # Method + results present in the same query → split
        has_method = bool(re.search(r"\b(method|approach|architecture|model|technique|algorithm|training)\b", q))
        has_results = bool(re.search(r"\b(result|performance|benchmark|accuracy|metric|score|evaluation)\b", q))
        has_limit = bool(re.search(r"\b(limitation|weakness|drawback|challenge|fail|problem)\b", q))

        if has_method:
            sub_queries.append(SubQuery(query=query, section_hint="methods"))
        if has_results:
            sub_queries.append(SubQuery(query=query, section_hint="results"))
        if has_limit:
            sub_queries.append(SubQuery(query=query, section_hint="limitations"))

        # Default: pass through unchanged
        if not sub_queries:
            sub_queries.append(SubQuery(query=query, section_hint=None))

        return sub_queries
