"""
ReAct agent CLI for ResearchRAG.

Usage
-----
    # Interactive conversational mode
    python -m src.agentic_main

    # Single-shot query
    python -m src.agentic_main -q "What are the main approaches to efficient attention?"

    # With reflection pass
    python -m src.agentic_main -q "..." --reflect
"""

import argparse
import logging
import re as _re
import signal
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage

from src.agentic.graph import build_graph, checkpoint_reflection, reflect_on_answer, run_graph
from src.agentic.safety import sanitize_query
from src.embeddings import EmbeddingGenerator
from src.rag_chain import RAGChain
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.semantic import SemanticRetriever

_SESSION_COUNTER = 0
_CITATION_RE = _re.compile(r"\[\d+\]")
_QUERY_TIMEOUT = 90


def _resolve_bm25_path(index_path: str) -> str:
    """Return the BM25 index path that lives alongside the given FAISS index."""
    parent = Path(index_path).parent
    name = "adaptive_bm25" if "adaptive" in Path(index_path).name else "bm25_index"
    return str(parent / name)


def _warn_if_no_citations(answer: str) -> None:
    if not _CITATION_RE.search(answer):
        print(
            "\n  [Citation warning] No citations detected — verify claims against source papers."
        )


def _cli_approve(data: dict):
    """Prompt the user to approve, abort, or annotate the retrieved context.

    Returns:
        True        — approved; proceed to synthesis.
        False       — aborted; end the turn.
        str (non-empty) — feedback hint; retry retrieval with the hint.
    """
    n = data.get("n_chunks", 0)
    preview = data.get("context_preview", "")
    attempt = data.get("attempt", 1)
    max_att = data.get("max_attempts", 3)
    label = f"attempt {attempt}/{max_att}" if attempt > 1 else ""
    print(f"\n  [Retrieved {n} chunk{'s' if n != 1 else ''}{' · ' + label if label else ''}]")
    if preview:
        lines = preview.splitlines()[:3]
        for line in lines:
            print(f"    {line}")
        if len(preview.splitlines()) > 3:
            print("    …")
    try:
        answer = input("\n  Proceed to generate answer? [Y/n]: ").strip().lower()
        if answer in ("", "y", "yes"):
            return True
        feedback = input(
            "  What's wrong with the retrieved context? (Enter to abort, or type a hint to retry): "
        ).strip()
        return feedback if feedback else False
    except (EOFError, KeyboardInterrupt):
        return False


def _run_turn(agent, embedding_gen, question: str, thread_id: str, llm, reflect: bool) -> None:
    """Execute one agent turn, print streamed output, optionally reflect."""
    clean_q, flagged = sanitize_query(question)
    if flagged:
        print("  [!] Injection pattern detected and stripped.")
    question = clean_q

    if hasattr(signal, "SIGALRM"):
        signal.alarm(_QUERY_TIMEOUT)

    try:
        streaming = False
        answer_buf = ""

        for event in run_graph(agent, question, thread_id=thread_id, on_approve=_cli_approve):
            etype = event["type"]

            if etype == "tool_call":
                q_text = event["args"].get("query", "")
                section = event["args"].get("section_type", "")
                display = f"  [Searching] \"{q_text}\"" if q_text else f"  [{event['tool']}]"
                if section:
                    display += f"  [section: {section}]"
                print(display)

            elif etype == "tool_result":
                preview = event["result_preview"].replace("\n", " ")[:100]
                print(f"  [Retrieved] {preview}…")

            elif etype == "token":
                if not streaming:
                    print("\nAssistant: ", end="", flush=True)
                    streaming = True
                print(event["content"], end="", flush=True)
                answer_buf += event["content"]

            elif etype == "answer":
                answer_buf = event["content"]
                if not streaming:
                    print(f"\nAssistant: {answer_buf}")
                else:
                    print()

            elif etype == "approval_feedback":
                print(f"  [Retrying] Searching with hint: \"{event['feedback']}\"")

            elif etype == "approval_rejected":
                print("\n  [Aborted] Synthesis rejected — no answer generated.")
                return

            elif etype == "error":
                print(f"\n[Error] {event['error']}")

        _warn_if_no_citations(answer_buf)

        if reflect and answer_buf:
            print("\n  [Reflecting…]")
            critique, needs_revision = reflect_on_answer(llm, question, answer_buf)
            checkpoint_reflection(agent, thread_id, critique)
            if needs_revision:
                print(f"  [Revision needed] {critique[:200]}")
                revised_question = (
                    f"{question}\n\n"
                    f"[REFLECTION] Your previous answer had issues:\n{critique}\n"
                    f"Please search again and provide an improved answer."
                )
                streaming = False
                answer_buf = ""
                for event in run_graph(agent, revised_question, thread_id=thread_id):
                    etype = event["type"]
                    if etype == "tool_call":
                        q_text = event["args"].get("query", "")
                        print(f"  [Re-searching] \"{q_text}\"" if q_text else f"  [{event['tool']}]")
                    elif etype == "token":
                        if not streaming:
                            print("\nAssistant (revised): ", end="", flush=True)
                            streaming = True
                        print(event["content"], end="", flush=True)
                        answer_buf += event["content"]
                    elif etype == "answer":
                        answer_buf = event["content"]
                        if not streaming:
                            print(f"\nAssistant (revised): {answer_buf}")
                        else:
                            print()
                    elif etype == "error":
                        print(f"\n[Error during revision] {event['error']}")
                _warn_if_no_citations(answer_buf)
            else:
                print("  [Reflection] Answer looks good — no revision needed.")

    except TimeoutError as te:
        print(f"\n[Timeout] {te}")
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)


def main():
    project_root = Path(__file__).parent.parent
    _adaptive_index  = project_root / "artifacts/adaptive_faiss_index"
    _adaptive_chunks = project_root / "artifacts/adaptive_chunks.json"
    _basic_index     = project_root / "artifacts/faiss_index"
    _basic_chunks    = project_root / "artifacts/chunks.json"
    default_index  = str(_adaptive_index  if _adaptive_index.exists()  else _basic_index)
    default_chunks = str(_adaptive_chunks if _adaptive_chunks.exists() else _basic_chunks)

    parser = argparse.ArgumentParser(description="ResearchRAG — ReAct agent CLI")
    parser.add_argument("-q", "--query", default=None,
                        help="Query to ask (omit for interactive conversation mode)")
    parser.add_argument("-r", "--retrieval", default="hybrid",
                        choices=["semantic", "bm25", "hybrid"],
                        help="Retrieval strategy (default: hybrid)")
    parser.add_argument("--metric", default="IP", choices=["IP", "L2"],
                        help="Distance metric for semantic search")
    parser.add_argument("--ollama-model", default="qwen2.5:7b",
                        help="Ollama model tag (default: qwen2.5:7b)")
    parser.add_argument("--reflect", action="store_true",
                        help="Run a reflection pass after generation to catch missing citations")
    parser.add_argument("--session-id",
                        help="Conversation thread ID (auto-generated if omitted)")
    parser.add_argument("--index-path", default=default_index,
                        help="Path to FAISS index")
    parser.add_argument("--chunks-path", default=default_chunks,
                        help="Path to chunks JSON")
    parser.add_argument("--conference",
                        help="Filter by conference (e.g. ACL or ACL,EMNLP)")
    parser.add_argument("--year",
                        help="Filter by year: single (2024), list (2023,2024), or range (2020-2024)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    # ---------------------------------------------------------------------------
    # Build retriever
    # ---------------------------------------------------------------------------
    print("Loading embeddings…")
    embedding_gen = EmbeddingGenerator()

    print(f"Loading {args.retrieval} retriever…")
    if args.retrieval == "semantic":
        retriever = SemanticRetriever(
            index_path=args.index_path, chunks_path=args.chunks_path, metric=args.metric
        )
    elif args.retrieval == "bm25":
        retriever = BM25Retriever(
            chunks_path=args.chunks_path, index_path=_resolve_bm25_path(args.index_path)
        )
    else:
        semantic_ret = SemanticRetriever(
            index_path=args.index_path, chunks_path=args.chunks_path, metric=args.metric
        )
        bm25_ret = BM25Retriever(
            chunks_path=args.chunks_path, index_path=_resolve_bm25_path(args.index_path)
        )
        retriever = HybridRetriever(semantic_ret, bm25_ret)

    # ---------------------------------------------------------------------------
    # Build RAGChain + agent
    # ---------------------------------------------------------------------------
    print(f"Loading {args.ollama_model}…")
    rag_chain = RAGChain(
        embedding_generator=embedding_gen,
        retriever=retriever,
        ollama_model=args.ollama_model,
    )

    print("Building ReAct agent…")
    agent = build_graph(rag_chain)

    print("Warming up LLM…", end=" ", flush=True)
    try:
        rag_chain.llm.invoke([HumanMessage(content="hi")])
        print("ready.")
    except Exception:
        print("(skipped)")

    # ---------------------------------------------------------------------------
    # SIGALRM timeout handler (Unix/macOS)
    # ---------------------------------------------------------------------------
    def _timeout_handler(signum, frame):
        raise TimeoutError(f"No response after {_QUERY_TIMEOUT}s — Ollama may be overloaded.")

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout_handler)

    # ---------------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------------
    global _SESSION_COUNTER
    _SESSION_COUNTER += 1
    thread_id = args.session_id or str(uuid4())
    session_label = f"session-{_SESSION_COUNTER}"

    try:
        if args.query:
            print(f"\n{session_label}  |  retrieval: {args.retrieval}\n")
            _run_turn(agent, embedding_gen, args.query, thread_id, rag_chain.llm, args.reflect)
        else:
            print(f"\nReAct Agent ready  |  {session_label}  |  retrieval: {args.retrieval}")
            print("Commands: 'quit'/'exit' to stop, 'new' for a fresh session.\n")
            while True:
                try:
                    user_input = input("You: ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\nGoodbye.")
                    break
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q"):
                    print("Goodbye.")
                    break
                if user_input.lower() == "new":
                    _SESSION_COUNTER += 1
                    thread_id = str(uuid4())
                    session_label = f"session-{_SESSION_COUNTER}"
                    print(f"New session: {session_label}\n")
                    continue
                _run_turn(agent, embedding_gen, user_input, thread_id, rag_chain.llm, args.reflect)
                print()

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(0)
    except Exception as exc:
        logging.getLogger(__name__).error("Fatal error: %s", exc, exc_info=True)
        print(f"\nError: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
