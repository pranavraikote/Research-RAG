"""
Streamlit chat interface for ResearchRAG.

Run with:
    streamlit run app.py

The agent and retriever are loaded once at startup via @st.cache_resource.
Changing the model or retrieval method in the sidebar triggers a reload.
"""

import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.agentic.react_agent import build_react_agent, run_react_agent, reflect_on_answer
from src.agentic.safety import sanitize_query
from src.embeddings import EmbeddingGenerator
from src.rag_chain import RAGChain
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.semantic import SemanticRetriever

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent
_INDEX = str(_ROOT / "artifacts/adaptive_faiss_index")
_CHUNKS = str(_ROOT / "artifacts/adaptive_chunks.json")
_BM25 = str(_ROOT / "artifacts/adaptive_bm25")
_LOG_DIR = _ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Background model loading
# ---------------------------------------------------------------------------

@st.cache_resource
def _bg_state() -> dict:
    """Single dict shared across all reruns: {threads: {}, results: {}}."""
    return {"threads": {}, "results": {}}


def _load_agent_bg(key: tuple, ollama_model: str, retrieval: str) -> None:
    """Load agent in a background thread, storing result in _bg_state()."""
    state = _bg_state()
    try:
        result = load_agent(ollama_model, retrieval)
        state["results"][key] = {"ok": True, "result": result}
    except Exception as exc:
        state["results"][key] = {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

# Human-readable labels for agent tool calls.
_TOOL_LABELS = {
    "search_papers_multi": "Searching across multiple queries",
    "search_papers": "Searching papers",
    "search_papers_in_section": "Searching section",
    "detect_relevant_sections": "Detecting relevant sections",
    "search_pubmed": "Searching PubMed",
}

_EXAMPLE_QUERIES = [
    "What are the main approaches to efficient attention in transformers?",
    "Compare linear attention and standard attention — methods and results",
    "What limitations do ACL 2025 papers identify in retrieval-augmented generation?",
    "How do recent papers address hallucination in LLMs?",
]


def _tool_label(tool: str, args: dict) -> str:
    """Return a clean, human-readable description of a tool call."""
    label = _TOOL_LABELS.get(tool, tool)
    section = args.get("section", "")
    query = args.get("query", args.get("queries", ""))
    if isinstance(query, list):
        query = " · ".join(query)
    query = str(query)[:80]
    if section:
        return f"{label} ({section}) — {query}"
    return f"{label} — {query}" if query else label


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_turn(
    thread_id: str,
    query: str,
    answer: str,
    model: str,
    retrieval: str,
    tool_calls: list,
    latency_s: float,
    reflected: bool = False,
    flagged: bool = False,
) -> None:
    """Append one conversation turn to a JSONL log file (one file per day)."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "model": model,
        "retrieval": retrieval,
        "query": query,
        "answer": answer,
        "tool_calls": tool_calls,
        "latency_s": round(latency_s, 2),
        "reflected": reflected,
        "flagged": flagged,
    }
    log_file = _LOG_DIR / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Cached resource — models load once, keyed by (model, retrieval)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_agent(ollama_model: str, retrieval: str):
    """Build and cache the RAGChain + ReAct agent."""
    embedding_gen = EmbeddingGenerator()

    if retrieval == "hybrid":
        semantic_ret = SemanticRetriever(index_path=_INDEX, chunks_path=_CHUNKS)
        bm25_ret = BM25Retriever(chunks_path=_CHUNKS, index_path=_BM25)
        retriever = HybridRetriever(semantic_ret, bm25_ret)
    elif retrieval == "semantic":
        retriever = SemanticRetriever(index_path=_INDEX, chunks_path=_CHUNKS)
    else:
        retriever = BM25Retriever(chunks_path=_CHUNKS, index_path=_BM25)

    rag_chain = RAGChain(
        embedding_generator=embedding_gen,
        retriever=retriever,
        llm_model=ollama_model,
        llm_provider="ollama",
        ollama_model=ollama_model,
        enable_prompt_cache=True,
    )
    return build_react_agent(rag_chain), rag_chain


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="ResearchRAG", page_icon="📚", layout="wide")

with st.sidebar:
    st.markdown("## ResearchRAG")
    st.caption("ACL Anthology · 842 papers · 152K chunks")
    st.divider()

    ollama_model = st.selectbox(
        "Model",
        [
            "qwen3:14b",
            "qwen3.5:35b",
            "qwen3.5:9b",
            "glm-4.7",
            "qwen3:8b",
            "qwen3:30b-a3b",
        ],
        index=0,
        help="Ollama model used for reasoning and generation.",
    )
    retrieval = st.selectbox(
        "Retrieval",
        ["hybrid", "semantic", "bm25"],
        index=0,
        help="hybrid: FAISS + BM25 fused via RRF (recommended)\nsemantic: dense vectors only\nbm25: keyword match only",
    )
    thinking = st.toggle(
        "Thinking mode",
        value=False,
        help="Qwen3 internal reasoning pass before answering. Better quality, slower. Turn off for faster responses.",
    )
    reflect = st.toggle(
        "Reflection pass",
        value=False,
        help="Run a second LLM pass after generation to catch missing citations. Adds ~5s.",
    )

    st.divider()
    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid4())
        st.rerun()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

# ---------------------------------------------------------------------------
# Load agent — background thread, non-blocking
# ---------------------------------------------------------------------------

model_key = (ollama_model, retrieval)
_state = _bg_state()

if model_key not in _state["results"]:
    # Kick off background load if not already running.
    if model_key not in _state["threads"] or not _state["threads"][model_key].is_alive():
        t = threading.Thread(
            target=_load_agent_bg, args=(model_key, ollama_model, retrieval), daemon=True
        )
        _state["threads"][model_key] = t
        t.start()

    # Render the page layout but keep input disabled while loading.
    if not st.session_state.messages:
        st.markdown("### What would you like to know?")
        st.caption("Ask anything about ACL, EMNLP, NAACL, EACL, or COLING 2025 papers.")

    st.chat_input(f"Loading {ollama_model}…", disabled=True)
    with st.spinner(f"Loading **{ollama_model}**…"):
        time.sleep(0.4)
    st.rerun()
    st.stop()

bg = _state["results"][model_key]
if not bg["ok"]:
    st.error(
        f"Failed to load **{ollama_model}**. Check that Ollama is running and the model is pulled.\n\n"
        f"```\n{bg['error']}\n```"
    )
    st.stop()

agent, rag_chain = bg["result"]

# ---------------------------------------------------------------------------
# Welcome screen (empty state)
# ---------------------------------------------------------------------------

# Hide welcome screen if a prefill is already queued (button was just clicked).
if not st.session_state.messages and "_prefill" not in st.session_state:
    st.markdown("### What would you like to know?")
    st.caption("Ask anything about ACL, EMNLP, NAACL, EACL, or COLING 2025 papers.")
    cols = st.columns(2)
    for i, example in enumerate(_EXAMPLE_QUERIES):
        if cols[i % 2].button(example, use_container_width=True, key=f"example_{i}"):
            st.session_state["_prefill"] = example
            st.rerun()

# Inject a pre-filled query from the welcome screen button click.
if "_prefill" in st.session_state:
    prefill = st.session_state.pop("_prefill")
    st.session_state.messages.append({"role": "user", "content": prefill})
    st.session_state["_run_prefill"] = prefill
    st.rerun()

# ---------------------------------------------------------------------------
# Replay conversation history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input + streaming response
# ---------------------------------------------------------------------------

_pending_query = st.session_state.pop("_run_prefill", None)
prompt = st.chat_input("Ask about ACL Anthology research papers…") or _pending_query

if prompt:
    clean_prompt, flagged = sanitize_query(prompt)

    # Qwen3 /no_think prefix disables the internal reasoning pass for faster responses.
    agent_prompt = clean_prompt if thinking else f"/no_think {clean_prompt}"

    if flagged:
        st.toast("Injection pattern detected and stripped from query.", icon="⚠️")

    if not _pending_query:
        # Prefill queries are already in history; only add typed ones.
        with st.chat_message("user"):
            st.markdown(clean_prompt)
        st.session_state.messages.append({"role": "user", "content": clean_prompt})

    with st.chat_message("assistant"):
        # Status box starts expanded so the user can follow tool calls live.
        agent_status = st.status("Thinking…", expanded=True)
        answer_area = st.empty()

        accumulated = ""
        generating = False
        n_searches = 0
        tool_calls_log = []
        t_start = time.perf_counter()

        for event in run_react_agent(agent, agent_prompt, thread_id=st.session_state.thread_id):
            etype = event["type"]

            if etype == "tool_call":
                n_searches += 1
                tool = event["tool"]
                line = _tool_label(tool, event["args"])
                tool_calls_log.append({"tool": tool, "query": event["args"].get("query", "")[:70]})
                agent_status.update(label=f"Searching… ({n_searches})")
                with agent_status:
                    st.caption(line)

            elif etype == "token":
                if not generating:
                    # First token — collapse the status box.
                    agent_status.update(
                        label=f"Read {n_searches} source{'s' if n_searches != 1 else ''}",
                        state="complete",
                        expanded=False,
                    )
                    generating = True
                accumulated += event["content"]
                answer_area.markdown(accumulated + "▌")

            elif etype == "answer":
                if not accumulated:
                    accumulated = event["content"]

            elif etype == "error":
                agent_status.update(label="Error", state="error", expanded=True)
                with agent_status:
                    st.error(event["error"])
                # Clear the MemorySaver checkpoint so the next turn isn't
                # poisoned by the incomplete LangGraph state left by the crash.
                agent.checkpointer.delete_thread(st.session_state.thread_id)
                break

        latency = time.perf_counter() - t_start

        # Finalise status label if we never got tokens (e.g. direct answer).
        if not generating:
            label = (
                f"Read {n_searches} source{'s' if n_searches != 1 else ''}"
                if n_searches else "Done"
            )
            agent_status.update(label=label, state="complete", expanded=False)

        # Update status label to include latency now that we know the total time.
        if generating or n_searches:
            sources = f"{n_searches} source{'s' if n_searches != 1 else ''}"
            agent_status.update(
                label=f"Read {sources} · {latency:.1f}s",
                state="complete",
                expanded=False,
            )

        # ── Optional reflection pass ────────────────────────────────────────
        if reflect and accumulated:
            with st.spinner("Reflecting on answer…"):
                critique, needs_revision = reflect_on_answer(
                    rag_chain.llm, clean_prompt, accumulated
                )

            if needs_revision:
                st.caption(f"Revising: {critique[:150]}")
                revised = ""
                for event in run_react_agent(
                    agent,
                    f"Revise your previous answer to fix: {critique}\n\nOriginal question: {clean_prompt}",
                    thread_id=st.session_state.thread_id,
                ):
                    if event["type"] == "token":
                        revised += event["content"]
                        answer_area.markdown(revised + "▌")
                    elif event["type"] == "answer" and not revised:
                        revised = event["content"]
                if revised:
                    accumulated = revised

        answer_area.markdown(accumulated)

        _log_turn(
            thread_id=st.session_state.thread_id,
            query=clean_prompt,
            answer=accumulated,
            model=ollama_model,
            retrieval=retrieval,
            tool_calls=tool_calls_log,
            latency_s=latency,
            reflected=reflect and accumulated != "",
            flagged=flagged,
        )

    st.session_state.messages.append({"role": "assistant", "content": accumulated})
