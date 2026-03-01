"""
Input sanitization and prompt injection protection for the ReAct agent.

Two surfaces are protected:
  1. User input — queries are sanitized before reaching the LLM.
  2. Retrieved content — chunk text is wrapped in explicit delimiters so the
     LLM treats it as data to summarise, not instructions to follow.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_MAX_QUERY_LENGTH = 500

# Patterns that are characteristic of prompt injection attempts.
# We log a warning and strip them rather than blocking outright.
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions?",
        r"disregard\s+(?:all\s+)?(?:previous|prior|above|earlier|the\s+)\w*",
        r"forget\s+(?:everything|all\s+(?:previous|prior|above))",
        r"you\s+are\s+now\s+(?:a|an|the)\s+",
        r"new\s+(?:system\s+)?(?:instructions?|prompt|role|persona)\s*:",
        r"act\s+as\s+(?:a|an|the|if\s+you\s+were)",
        r"pretend\s+(?:you\s+are|to\s+be)",
        r"jailbreak",
        r"<\s*/?(?:system|instruction|prompt)\s*>",
        r"\[INST\]|\[\/INST\]",
        r"###\s*(?:System|Instruction)\b",
        r"(?:print|output|reveal|show|leak)\s+(?:your\s+)?(?:system\s+)?prompt",
        r"override\s+(?:your\s+)?(?:instructions?|rules?|constraints?|guidelines?)",
        r"do\s+anything\s+now",        # DAN-style
        r"developer\s+mode",
    ]
]


def sanitize_query(query: str) -> tuple[str, bool]:
    """
    Sanitize a user query and detect injection attempts.

    Returns:
        (sanitized_query, was_flagged)

    The query is always returned (possibly cleaned/truncated).
    was_flagged=True means an injection pattern was detected and logged.
    The suspicious portion is stripped before passing the query forward.
    """
    # Remove null bytes and non-printable control chars (keep \\t and \\n)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", query)

    # Normalise whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Truncate
    if len(cleaned) > _MAX_QUERY_LENGTH:
        cleaned = cleaned[:_MAX_QUERY_LENGTH].rstrip() + "…"
        logger.warning("Query truncated to %d characters.", _MAX_QUERY_LENGTH)

    # Detect and strip injection patterns
    flagged = False
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            flagged = True
            logger.warning(
                "Potential prompt injection detected and stripped: %r",
                cleaned[:120],
            )
            cleaned = pattern.sub("", cleaned).strip()

    return cleaned, flagged


def wrap_retrieved_content(text: str) -> str:
    """
    Wrap a retrieved chunk's text in explicit markers.

    This signals to the LLM that the enclosed content is untrusted external
    data to be summarised, not system instructions to be followed.
    """
    return f"<retrieved_content>\n{text}\n</retrieved_content>"
