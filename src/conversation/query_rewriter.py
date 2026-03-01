import re
from dataclasses import dataclass, field
from typing import Optional, List
from .history import ConversationHistory

# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------

_REWRITE_SYSTEM = (
    "You are a query rewriting assistant. "
    "Given a conversation history and a follow-up question, rewrite the follow-up "
    "into a single, self-contained question that can be understood without any prior context. "
    "Resolve all pronouns, references to 'it', 'this', 'that', 'they', 'the paper', etc. "
    "Output ONLY the rewritten question — no explanation, no preamble."
)

_REWRITE_HUMAN = (
    "Conversation so far:\n{history}\n\n"
    "Follow-up question: {query}\n\n"
    "Rewritten question:"
)

# ---------------------------------------------------------------------------
# Signal patterns
# ---------------------------------------------------------------------------

# Pronouns that, in a follow-up context, likely refer to something earlier
_PRONOUN_SIGNALS = re.compile(
    r'\b(it|this|that|they|them|their|those|these)\b',
    re.IGNORECASE,
)

# "The <domain noun>" — strong indicator the referent lives in history
_DEFINITE_DOMAIN = re.compile(
    r'\bthe\s+(paper|method|approach|model|system|technique|framework|'
    r'results?|authors?|work|algorithm|architecture|dataset|baseline|experiment)\b',
    re.IGNORECASE,
)

# Very short queries when history exists — likely a bare follow-up ("Why?", "How?")
_SHORT_QUERY_WORDS = 5

# Continuation markers at the start of the query
_CONTINUATION_START = re.compile(
    r'^(and|also|but|so|then|additionally|furthermore|what about|how about|'
    r'tell me more|more on|can you elaborate|explain)\b',
    re.IGNORECASE,
)

# Explicit citation back-references
_CITATION_REF = re.compile(
    r'\bpaper\s+\d+\b|\[\d+\]',
    re.IGNORECASE,
)


@dataclass
class RewriteDecision:
    """
    Records whether rewriting was attempted and which signals triggered it.

    Attributes:
        needed: True if any signal suggested the query is a follow-up.
        signals: List of human-readable signal names that fired.
        heuristic_changed: True if heuristic rewriting changed the query.
        llm_used: True if LLM rewriting was invoked.
        final_query: The query after all rewriting.
    """
    needed: bool
    signals: List[str]
    heuristic_changed: bool = False
    llm_used: bool = False
    final_query: str = ""


class QueryRewriter:
    """
    Rewrites follow-up queries using conversation history context.

    Decision flow
    -------------
    1. Gate: ``_needs_rewriting()`` inspects 5 signals to decide whether the
       query is a self-contained question (skip) or a follow-up (rewrite).
    2. Heuristic pass: fast regex-based pronoun/reference resolution and
       context keyword expansion. Always the first rewriter; runs even when
       LLM rewriting is enabled.
    3. LLM pass (optional): only fires when ``use_llm_rewriting=True``, an
       ``llm`` instance is provided, AND the heuristic pass left unresolved
       pronouns / definite-domain references in the output. Falls back to the
       heuristic result on any LLM error.

    The separation means:
    - Self-contained queries (no signals) → zero rewriting cost.
    - Simple follow-ups (pronoun only) → heuristic resolves them cheaply.
    - Complex coreference → LLM invoked only when heuristic is insufficient.
    """

    def __init__(self, use_llm_rewriting: bool = False, llm=None):
        """
        Args:
            use_llm_rewriting: Enable LLM-based rewriting as a second pass.
                               Default False keeps the original heuristic-only
                               behaviour for backward compatibility.
            llm: LangChain LLM instance (ChatOllama or HuggingFacePipeline).
                 Only used when use_llm_rewriting=True.
        """
        self.use_llm_rewriting = use_llm_rewriting
        self.llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rewrite(self, query: str, history: ConversationHistory) -> str:
        """Return a rewritten query; returns the original if no rewriting is needed."""
        decision = self.rewrite_with_decision(query, history)
        return decision.final_query

    def rewrite_with_decision(self, query: str, history: ConversationHistory) -> RewriteDecision:
        """
        Rewrite the query and return a ``RewriteDecision`` explaining what happened.

        Useful for debugging, logging, and evaluation.
        """
        # No history → nothing to resolve
        if not history.messages:
            return RewriteDecision(needed=False, signals=[], final_query=query)

        # --- Gate: check rewriting signals ---
        needed, signals = self._needs_rewriting(query, history)
        if not needed:
            return RewriteDecision(needed=False, signals=signals, final_query=query)

        # --- Heuristic pass ---
        entities = history.get_recent_entities()
        heuristic_result = self._resolve_references(query, history, entities)
        heuristic_result = self._expand_with_context(heuristic_result, history)
        heuristic_changed = heuristic_result != query

        decision = RewriteDecision(
            needed=True,
            signals=signals,
            heuristic_changed=heuristic_changed,
            llm_used=False,
            final_query=heuristic_result,
        )

        # --- LLM pass (only if enabled, llm wired in, and residual references remain) ---
        if (
            self.use_llm_rewriting
            and self.llm is not None
            and self._has_residual_references(heuristic_result)
        ):
            llm_result = self._llm_rewrite(query, history)
            if llm_result:
                decision.final_query = llm_result
                decision.llm_used = True

        return decision

    # ------------------------------------------------------------------
    # Signal detection — "does this query need rewriting?"
    # ------------------------------------------------------------------

    def _needs_rewriting(self, query: str, history: ConversationHistory) -> tuple[bool, List[str]]:
        """
        Inspect the query against 5 signals to decide if rewriting is warranted.

        Signals
        -------
        pronoun         — query contains 'it', 'this', 'that', 'they', etc.
        definite_domain — query contains 'the paper/method/model/...'
        short_followup  — query is very short (< 5 words) and history exists
        continuation    — query starts with a continuation marker ('and', 'also', …)
        citation_ref    — query contains 'paper N' or '[N]' style back-references

        Returns (needed: bool, signals: list[str])
        """
        signals = []

        if _PRONOUN_SIGNALS.search(query):
            signals.append("pronoun")

        if _DEFINITE_DOMAIN.search(query):
            signals.append("definite_domain")

        if len(query.split()) < _SHORT_QUERY_WORDS:
            signals.append("short_followup")

        if _CONTINUATION_START.search(query):
            signals.append("continuation")

        if _CITATION_REF.search(query):
            signals.append("citation_ref")

        return bool(signals), signals

    def _has_residual_references(self, query: str) -> bool:
        """True if unresolved pronouns or definite-domain phrases remain after heuristic pass."""
        return bool(_PRONOUN_SIGNALS.search(query) or _DEFINITE_DOMAIN.search(query))

    # ------------------------------------------------------------------
    # LLM rewriting
    # ------------------------------------------------------------------

    def _format_history_for_prompt(self, history: ConversationHistory) -> str:
        """Compact representation of the last 6 messages for the LLM prompt."""
        lines = []
        for msg in history.messages[-6:]:
            role_label = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role_label}: {msg.content[:300]}")
        return "\n".join(lines)

    def _llm_rewrite(self, query: str, history: ConversationHistory) -> Optional[str]:
        """
        Ask the LLM to produce a self-contained rewrite.

        Returns the rewritten string, or None if the call fails or the output
        looks malformed.
        """
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            history_text = self._format_history_for_prompt(history)
            human_text = _REWRITE_HUMAN.format(history=history_text, query=query)

            response = self.llm.invoke(
                [SystemMessage(content=_REWRITE_SYSTEM), HumanMessage(content=human_text)]
            )

            result = response.content.strip() if hasattr(response, "content") else str(response).strip()

            # Sanity checks — reject obviously bad outputs
            if not result or len(result) > 500 or "\n" in result[:50]:
                return None

            return result

        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # Heuristic rewriting (unchanged logic, refactored name)
    # ------------------------------------------------------------------

    def _resolve_references(self, query: str, history: ConversationHistory, entities: List[str]) -> str:
        rewritten = query

        # "the last N papers" → "paper 1, paper 2, ..."
        last_papers_match = re.search(r'the\s+last\s+(\d+)\s+papers?', query, re.IGNORECASE)
        if last_papers_match:
            n = int(last_papers_match.group(1))
            paper_entities = [e for e in entities if "paper" in e.lower() or any(c.isdigit() for c in e)]
            if len(paper_entities) >= n:
                papers_str = ", ".join(paper_entities[-n:])
                rewritten = rewritten.replace(last_papers_match.group(0), papers_str)

        # "previous paper" → last mentioned paper
        if re.search(r'previous\s+paper', query, re.IGNORECASE):
            paper_entities = [e for e in entities if "paper" in e.lower() or any(c.isdigit() for c in e)]
            if paper_entities:
                rewritten = re.sub(r'previous\s+paper', paper_entities[-1], rewritten, flags=re.IGNORECASE)

        # "that method" → last mentioned method entity
        if re.search(r'that\s+method', query, re.IGNORECASE):
            method_entities = [e for e in entities if e[0].isupper() and len(e.split()) <= 3]
            if method_entities:
                rewritten = re.sub(r'that\s+method', method_entities[-1], rewritten, flags=re.IGNORECASE)

        # "this/that approach/method/..." → most recent named entity
        if re.search(r'\b(this|that)\s+(approach|method|technique|framework|model|system)\b', rewritten, re.IGNORECASE):
            recent_entities = history.get_recent_entities()
            method_entities = [e for e in recent_entities if e[0].isupper() and len(e.split()) <= 3]
            replacement = None
            if method_entities:
                replacement = method_entities[-1]
            else:
                for msg in reversed(history.messages):
                    if msg.role == "assistant":
                        key_terms = self._extract_key_terms(msg.content)
                        if key_terms:
                            replacement = key_terms[0]
                        break
            if replacement:
                rewritten = re.sub(
                    r'\b(this|that)\s+(approach|method|technique|framework|model|system)\b',
                    replacement, rewritten, flags=re.IGNORECASE,
                )

        # "what/how/... about it/this/that" at start → inject last query's topic
        if re.match(r'^(what|how|why|tell|explain|can|could|would)\s+(about\s+)?(it|this|that)', rewritten, re.IGNORECASE):
            for msg in reversed(history.messages):
                if msg.role == "user":
                    key_terms = self._extract_key_terms(msg.content)
                    if key_terms:
                        rewritten = re.sub(
                            r'^(what|how|why|tell|explain|can|could|would)\s+(about\s+)?(it|this|that)',
                            f"{rewritten.split()[0]} about {key_terms[0]}",
                            rewritten, flags=re.IGNORECASE,
                        )
                    break

        return rewritten

    def _expand_with_context(self, query: str, history: ConversationHistory) -> str:
        if not history.messages:
            return query

        context_keywords: List[str] = []
        for msg in reversed(history.messages):
            if msg.role == "assistant":
                context_keywords.extend(self._extract_key_terms(msg.content))
                break
        for msg in reversed(history.messages):
            if msg.role == "user":
                context_keywords.extend(self._extract_key_terms(msg.content))
                break

        # Only expand genuinely short bare follow-ups
        if len(query.split()) < _SHORT_QUERY_WORDS and context_keywords:
            query_lower = query.lower()
            new_keywords = [kw for kw in context_keywords if kw.lower() not in query_lower]
            if new_keywords:
                query = f"{query} {' '.join(new_keywords[:2])}"

        return query

    def _extract_key_terms(self, text: str, max_terms: int = 5) -> List[str]:
        words = text.split()
        key_terms = []
        for word in words:
            clean_word = re.sub(r'[^\w\s]', '', word)
            if not clean_word:
                continue
            if clean_word[0].isupper() and len(clean_word) > 2:
                key_terms.append(clean_word)
            elif clean_word.isdigit():
                key_terms.append(clean_word)
            elif clean_word.isupper() and len(clean_word) > 1:
                key_terms.append(clean_word)

        seen: set = set()
        unique_terms = []
        for term in key_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)
        return unique_terms[:max_terms]

    def extract_entities_from_query(self, query: str) -> List[str]:
        entities = []
        for match in re.findall(r'paper\s+(\d+)', query, re.IGNORECASE):
            entities.append(f"paper {match}")
        entities.extend(re.findall(r'\b([A-Z][A-Z0-9]{2,})\b', query))
        entities.extend(re.findall(r'"([^"]+)"', query))
        return entities
