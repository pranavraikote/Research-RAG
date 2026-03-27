"""
Streamlit chat interface for ResearchRAG.

Run with:
    streamlit run app.py

The agent and retriever are loaded once at startup via @st.cache_resource.
Changing the model or retrieval method in the sidebar triggers a reload.
"""

import json
import sys
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

@st.cache_resource(show_spinner="Loading models and index…")
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
st.title("ResearchRAG")
st.caption("ACL Anthology · 842 papers · 152K adaptive chunks")

with st.sidebar:
    st.header("Settings")
    ollama_model = st.selectbox(
        "Model",
        ["qwen3:8b", "qwen3:14b", "qwen3:30b-a3b", "qwen2.5:14b", "qwen2.5:7b"],
        index=0,
    )
    retrieval = st.selectbox(
        "Retrieval",
        ["hybrid", "semantic", "bm25"],
        index=0,
    )
    reflect = st.toggle("Reflection pass", value=False, help="Run a second LLM pass to catch citation gaps.")
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
# Replay conversation history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Load agent at startup (cached — runs once per unique model+retrieval combo)
# ---------------------------------------------------------------------------

try:
    agent, rag_chain = load_agent(ollama_model, retrieval)
except Exception as exc:
    st.error(
        f"Failed to load agent. Check that Ollama is running and "
        f"the model **{ollama_model}** is pulled.\n\n"
        f"```\n{exc}\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Chat input + streaming response
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask about ACL Anthology research papers…"):
    clean_prompt, flagged = sanitize_query(prompt)

    with st.chat_message("user"):
        st.markdown(clean_prompt)
        if flagged:
            st.caption("Injection pattern detected and stripped from query.")

    st.session_state.messages.append({"role": "user", "content": clean_prompt})

    with st.chat_message("assistant"):
        # Agent steps shown in a live status box; collapses when done.
        agent_status = st.status("Thinking…", expanded=False)
        answer_area = st.empty()

        accumulated = ""
        generating = False
        n_searches = 0
        tool_calls_log = []
        t_start = time.perf_counter()

        for event in run_react_agent(agent, clean_prompt, thread_id=st.session_state.thread_id):
            etype = event["type"]

            if etype == "tool_call":
                n_searches += 1
                tool = event["tool"]
                q = event["args"].get("query", "")[:70]
                tool_calls_log.append({"tool": tool, "query": q})
                with agent_status:
                    st.markdown(f"**{tool}** — {q}")

            elif etype == "token":
                if not generating:
                    # First token — collapse the status box
                    agent_status.update(
                        label=f"Searched {n_searches} time(s)",
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
                break

        # Collapse status in case we never got tokens (e.g. cached answer)
        if not generating:
            agent_status.update(
                label=f"Searched {n_searches} time(s)",
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

        # Log the turn to JSONL.
        _log_turn(
            thread_id=st.session_state.thread_id,
            query=clean_prompt,
            answer=accumulated,
            model=ollama_model,
            retrieval=retrieval,
            tool_calls=tool_calls_log,
            latency_s=time.perf_counter() - t_start,
            reflected=reflect and accumulated != "",
            flagged=flagged,
        )

    st.session_state.messages.append({"role": "assistant", "content": accumulated})
