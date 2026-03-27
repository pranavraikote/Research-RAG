"""
Semantic answer cache for the ReAct agent.

Caches (query_embedding, answer) pairs. On each incoming query, embeds it and
checks cosine similarity against stored embeddings. A hit (similarity >= threshold)
returns the cached answer, skipping retrieval + reranking + LLM inference (~14-16s saved).

Context-dependent queries (those/that/they/…) are excluded from caching.
"""

import re
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


_CONTEXT_PRONOUNS = re.compile(
    r"\b(those|that paper|that model|that approach|those papers|"
    r"those results|they|their|it|its|the same|the above|the previous|"
    r"earlier|mentioned above)\b",
    re.IGNORECASE,
)

_MIN_ANSWER_LEN = 120   # skip short refusals / out-of-corpus responses


@dataclass
class _CacheEntry:
    embedding: np.ndarray   # L2-normalised
    answer: str
    query_preview: str      # first 60 chars, for stats display
    hits: int = field(default=0)


class SemanticCache:
    """
    LRU-bounded semantic cache mapping query embeddings to cached answers.

    Args:
        maxsize: Maximum number of entries (oldest evicted when full).
        threshold: Cosine similarity threshold for a cache hit (default 0.92).
    """

    def __init__(self, maxsize: int = 128, threshold: float = 0.92):
        self.maxsize = maxsize
        self.threshold = threshold
        self._entries: deque = deque()
        self._total_checks = 0
        self._total_hits = 0

    def is_cacheable(self, query: str) -> bool:
        """Return False for context-dependent queries (pronoun references to prior turns)."""
        return not bool(_CONTEXT_PRONOUNS.search(query))

    def get(self, query: str, query_embedding: np.ndarray) -> Optional[str]:
        """
        Return a cached answer if any stored entry meets the similarity threshold.

        Skips the cache for context-dependent queries (pronoun references).

        Args:
            query: Raw query text (used for cacheability check).
            query_embedding: Raw (un-normalised) embedding of the incoming query.

        Returns:
            Cached answer string, or None on a miss.
        """
        self._total_checks += 1
        if not self.is_cacheable(query):
            return None
        if not self._entries:
            return None

        emb_norm = self._normalise(query_embedding)
        best_sim, best_entry = 0.0, None
        for entry in self._entries:
            sim = float(np.dot(emb_norm, entry.embedding))
            if sim > best_sim:
                best_sim, best_entry = sim, entry

        if best_sim >= self.threshold and best_entry is not None:
            best_entry.hits += 1
            self._total_hits += 1
            return best_entry.answer
        return None

    def put(self, query: str, query_embedding: np.ndarray, answer: str) -> bool:
        """
        Store a (query_embedding, answer) pair.

        Returns True if stored, False if skipped (context-dependent or answer too short).
        """
        if not self.is_cacheable(query):
            return False
        if len(answer) < _MIN_ANSWER_LEN:
            return False
        if len(self._entries) >= self.maxsize:
            self._entries.popleft()
        self._entries.append(_CacheEntry(
            embedding=self._normalise(query_embedding),
            answer=answer,
            query_preview=query[:60],
        ))
        return True

    def info(self) -> dict:
        """Return cache statistics."""
        return {
            "size": len(self._entries),
            "maxsize": self.maxsize,
            "threshold": self.threshold,
            "total_checks": self._total_checks,
            "total_hits": self._total_hits,
            "hit_rate": (
                f"{100 * self._total_hits // self._total_checks}%"
                if self._total_checks else "n/a"
            ),
        }

    def top_entries(self, n: int = 5) -> List[dict]:
        """Return the n most-hit entries."""
        sorted_entries = sorted(self._entries, key=lambda e: e.hits, reverse=True)
        return [{"query": e.query_preview, "hits": e.hits} for e in sorted_entries[:n]]

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-10)
