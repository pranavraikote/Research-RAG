import argparse
import logging
import os
import re as _re
import signal
import sys
from pathlib import Path
from uuid import uuid4

# Adding parent directory to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage

from src.embeddings import EmbeddingGenerator
from src.rag_chain import RAGChain
from src.retrieval.semantic import SemanticRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.agentic.orchestrator import AgenticRAGOrchestrator
from src.agentic.react_agent import build_react_agent, run_react_agent, reflect_on_answer
from src.agentic.safety import sanitize_query
from src.agentic.semantic_cache import SemanticCache

_SESSION_COUNTER = 0   # increments with each "new" command


def _resolve_bm25_path(index_path: str) -> str:
    """Return the correct BM25 index path sibling to the given FAISS index path."""
    parent = Path(index_path).parent
    name = "adaptive_bm25" if "adaptive" in Path(index_path).name else "bm25_index"
    return str(parent / name)


def format_output(result: dict):
    """Format and display agentic RAG results."""
    if result["type"] == "workflow":
        print(f"\nWorkflow: {result['workflow']} - {result['message']}")
    
    elif result["type"] == "iteration":
        print(f"\nIteration {result['iteration']}: {result['message']}")
    
    elif result["type"] == "agent":
        print(f"{result['agent']}: {result['message']}")
    
    elif result["type"] == "retrieval":
        count = result.get("chunks_count", 0)
        print(f"   Retrieved {count} relevant chunks")
    
    elif result["type"] == "reasoning":
        task_type = result.get("task_type", "general")
        print(f"   Reasoning completed ({task_type})")
    
    elif result["type"] == "refinement":
        print(f"\n{result['message']}")
        print(f"New query: {result['new_query']}")
    
    elif result["type"] == "complete":
        print("\n## Answer\n")
        
        answer = result.get("answer", "")
        print(answer)
        
        # Display gaps if any
        gaps = result.get("gaps", [])
        if gaps:
            print("\n## Research Gaps Identified\n")
            for i, gap in enumerate(gaps, 1):
                # Gaps are strings, not dicts
                if isinstance(gap, str):
                    print(f"{i}. {gap}")
                else:
                    print(f"{i}. **{gap.get('category', 'General')}**: {gap.get('gap_description', '')}")
                    if gap.get('importance'):
                        print(f"   Importance: {gap['importance']}")
        
        # Display follow-up questions
        followup = result.get("followup_questions", [])
        if followup:
            print("\n## Suggested Follow-up Questions\n")
            for i, question in enumerate(followup, 1):
                print(f"{i}. {question}")
        
        # Performance metrics
        print("\n### Performance Metrics\n")
        print(f"- **Workflow**: {result.get('workflow', 'N/A')}")
        print(f"- **Iterations**: {result.get('iterations', 0)}")
        print(f"- **Total Time**: {result.get('total_time', 0):.2f} s")
        
        # Task type
        task_type = result.get("task_type", "general")
        print(f"- **Task Type**: {task_type}")


_CITATION_RE = _re.compile(
    r"\((?!e\.g\.|i\.e\.|cf\.|see )[^)]{3,80},\s*(?:ACL|EMNLP|NAACL|EACL|COLING|NeurIPS|ICML|ICLR|\d{4})[^)]*\)",
    _re.IGNORECASE,
)

def _warn_if_no_citations(answer: str) -> None:
    """Print a warning if the answer contains no detectable citations."""
    if not _CITATION_RE.search(answer):
        print(
            "\n  [Citation warning] No citations detected in this answer. "
            "Verify claims against source papers."
        )


def main():
    """
    Main function for agentic RAG CLI.
    """
    
    parser = argparse.ArgumentParser(description='Research Paper Navigator - Agentic Mode')
    
    parser.add_argument('-q', '--query', default=None,
                       help='Query to ask (omit with --react to enter interactive conversation mode)')
    parser.add_argument('-r', '--retrieval', default='hybrid',
                       choices=['semantic', 'bm25', 'hybrid'],
                       help='Retrieval strategy')
    parser.add_argument('-k', '--top-k', type=int, default=5, 
                       help='Number of papers to retrieve per iteration')
    parser.add_argument('--metric', default='IP', choices=['IP', 'L2'], 
                       help='Distance metric for semantic search')
    parser.add_argument('--workflow', choices=['comparison', 'synthesis', 'general'],
                       help='Workflow pattern (auto-determined if not specified)')
    parser.add_argument('--max-iterations', type=int, default=3,
                       help='Maximum iterations for iterative refinement')
    
    # LLM options
    parser.add_argument('--llm-model', default='Qwen/Qwen2-1.5B-Instruct',
                       help='LLM model name (HuggingFace format)')
    parser.add_argument('--llm-provider', default='auto', choices=['auto', 'ollama', 'huggingface'],
                       help='LLM provider (auto = try Ollama first, fall back to HuggingFace)')
    parser.add_argument('--ollama-model', default='qwen2.5:7b',
                       help='Ollama model name (used when provider is auto or ollama)')
    parser.add_argument('--no-quantization', action='store_true',
                       help='Disable quantization (use full precision)')
    parser.add_argument('--quantization-bits', type=int, default=4, choices=[4, 8],
                       help='Quantization bits (4 or 8)')
    parser.add_argument('--no-prompt-cache', action='store_true',
                       help='Disable prompt caching')
    parser.add_argument('--react', action='store_true',
                       help='Use ReAct agent (LangGraph create_react_agent + bind_tools) '
                            'instead of the default v1 orchestrator')
    parser.add_argument('--reflect', action='store_true',
                       help='Enable reflection: critique the draft answer and revise if issues found')
    parser.add_argument('--session-id',
                       help='Conversation session ID for --react mode (auto-generated if omitted)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging (DEBUG level)')
    parser.add_argument('--log-file', help='Log to file instead of console')
    parser.add_argument('--show-prompts', action='store_true',
                       help='Show the actual prompts sent to LLM (very verbose)')
    
    project_root = Path(__file__).parent.parent
    # Prefer adaptive indices (152K section-aware chunks) when available
    _adaptive_index  = project_root / "artifacts/adaptive_faiss_index"
    _adaptive_chunks = project_root / "artifacts/adaptive_chunks.json"
    _basic_index     = project_root / "artifacts/faiss_index"
    _basic_chunks    = project_root / "artifacts/chunks.json"
    default_index  = str(_adaptive_index  if _adaptive_index.exists()  else _basic_index)
    default_chunks = str(_adaptive_chunks if _adaptive_chunks.exists() else _basic_chunks)

    parser.add_argument('--index-path', default=default_index,
                       help='Path to FAISS index (default: adaptive if available)')
    parser.add_argument('--chunks-path', default=default_chunks,
                       help='Path to chunks JSON file (default: adaptive if available)')
    
    # Metadata filtering options
    parser.add_argument('--conference', 
                       help='Filter by conference (e.g., "ACL" or "ACL,EMNLP")')
    parser.add_argument('--year', 
                       help='Filter by year: single (e.g., "2024"), multiple (e.g., "2023,2024"), or range (e.g., "2020-2024")')
    parser.add_argument('--title', 
                       help='Filter by title (partial match, e.g., "transformer")')
    
    args = parser.parse_args()

    # -q is required unless using --react (which supports interactive mode)
    if not args.react and not args.query:
        parser.error("-q/--query is required when not using --react mode")

    # Building filters dictionary
    filters = {}
    if args.conference:
        conferences = [c.strip() for c in args.conference.split(',')]
        filters['conference'] = conferences[0] if len(conferences) == 1 else conferences
    
    if args.year:
        year_str = args.year.strip()
        if '-' in year_str:
            try:
                min_year, max_year = map(int, year_str.split('-'))
                filters['year'] = {'min': min_year, 'max': max_year}
            except ValueError:
                print(f"Invalid year range format '{year_str}'. Use format 'YYYY-YYYY'")
                sys.exit(1)
        else:
            try:
                years = [int(y.strip()) for y in year_str.split(',')]
                filters['year'] = years[0] if len(years) == 1 else years
            except ValueError:
                print(f"Invalid year format '{year_str}'. Use format 'YYYY' or 'YYYY,YYYY'")
                sys.exit(1)
    
    if args.title:
        titles = [t.strip() for t in args.title.split(',')]
        filters['title'] = titles[0] if len(titles) == 1 else titles
    
    filters = filters if filters else None
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Store show_prompts flag for agents to use.
    if args.show_prompts:
        os.environ['AGENTIC_SHOW_PROMPTS'] = '1'
    
    if args.log_file:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            filename=args.log_file,
            filemode='a'
        )
        print(f"Logging to file: {args.log_file}")
    else:
        logging.basicConfig(
            level=log_level,
            format=log_format
        )
    
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("Starting Agentic RAG System")
    logger.info(f"Query: {args.query or '(interactive)'}")
    logger.info(f"Retrieval: {args.retrieval}, Model: {args.llm_model}")
    logger.info("="*60)
    
    # Initialize embedding generator
    print("Initializing embedding generator...")
    logger.debug("Initializing embedding generator")
    embedding_gen = EmbeddingGenerator()
    
    # Initialize retriever
    print(f"Initializing {args.retrieval} retriever...")
    if args.retrieval == 'semantic':
        retriever = SemanticRetriever(
            index_path=args.index_path,
            chunks_path=args.chunks_path,
            metric=args.metric
        )
    elif args.retrieval == 'bm25':
        bm25_index_path = _resolve_bm25_path(args.index_path)
        retriever = BM25Retriever(chunks_path=args.chunks_path, index_path=bm25_index_path)
    elif args.retrieval == 'hybrid':
        bm25_index_path = _resolve_bm25_path(args.index_path)
        semantic_ret = SemanticRetriever(
            index_path=args.index_path,
            chunks_path=args.chunks_path,
            metric=args.metric
        )
        bm25_ret = BM25Retriever(chunks_path=args.chunks_path, index_path=bm25_index_path)
        retriever = HybridRetriever(semantic_ret, bm25_ret)
    
    # Initialize RAG chain
    print(f"Initializing RAG chain with {args.llm_provider} provider...")
    rag_chain = RAGChain(
        embedding_generator=embedding_gen,
        retriever=retriever,
        llm_model=args.llm_model,
        llm_provider=args.llm_provider,
        ollama_model=args.ollama_model,
        use_quantization=not args.no_quantization,
        quantization_bits=args.quantization_bits,
        enable_prompt_cache=not args.no_prompt_cache
    )
    
    # Get LLM from RAG chain for agents
    llm = rag_chain.llm

    logger.info("Starting query execution")

    try:
        if args.react:
            # ── ReAct agent (Phase 5) — conversational by default ──────────
            global _SESSION_COUNTER

            print("Building ReAct agent (LangGraph create_react_agent)...")
            agent = build_react_agent(rag_chain)

            # Warm up Ollama: load the model into memory before the first real query
            print("Warming up LLM...", end=" ", flush=True)
            try:
                rag_chain.llm.invoke([HumanMessage(content="hi")])
                print("ready.")
            except Exception:
                print("(skipped)")

            _SESSION_COUNTER += 1
            thread_id = args.session_id or str(uuid4())
            session_label = f"session-{_SESSION_COUNTER}"

            # Semantic cache: skip the full agent round-trip (~14-16s) when a
            # semantically similar question was already answered this session.
            _sem_cache = SemanticCache(maxsize=128, threshold=0.92)

            # ── Per-query timeout (90 s) using SIGALRM (Unix/macOS) ─────────
            _QUERY_TIMEOUT = 90

            def _timeout_handler(signum, frame):
                raise TimeoutError(f"No response after {_QUERY_TIMEOUT}s — Ollama may be overloaded.")

            if hasattr(signal, "SIGALRM"):
                signal.signal(signal.SIGALRM, _timeout_handler)

            def _run_turn(question: str, reflect: bool = False) -> None:
                # Sanitize before the query reaches the LLM
                clean_q, flagged = sanitize_query(question)
                if flagged:
                    print("  [!] Injection pattern detected and stripped.")
                question = clean_q

                # ── Semantic cache check ─────────────────────────────────────
                # Embed the question to probe the cache.  Context-dependent
                # questions (those/that/they/…) are excluded — their answers
                # rely on prior turns and cannot be safely reused.
                _query_emb = None
                if _sem_cache.is_cacheable(question):
                    _query_emb = embedding_gen.embed_query(question)
                    cached = _sem_cache.get(_query_emb)
                    if cached is not None:
                        print(f"\nAssistant: {cached}")
                        print("  [Semantic cache hit]")
                        _warn_if_no_citations(cached)
                        return

                # Arm timeout
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(_QUERY_TIMEOUT)

                try:
                    _streaming = False   # tracks whether we've started printing tokens
                    _answer_buf = ""

                    for event in run_react_agent(agent, question, thread_id=thread_id):
                        etype = event["type"]

                        if etype == "tool_call":
                            # Friendly one-liner instead of raw dict dump
                            tool = event["tool"]
                            args_dict = event["args"]
                            q_text = args_dict.get("query", "")
                            section = args_dict.get("section_type", "")
                            if q_text:
                                display = f"  [Searching] \"{q_text}\""
                                if section:
                                    display += f"  [section: {section}]"
                            else:
                                display = f"  [{tool}]"
                            print(display)

                        elif etype == "tool_result":
                            preview = event["result_preview"].replace("\n", " ")[:100]
                            print(f"  [Retrieved] {preview}…")

                        elif etype == "token":
                            if not _streaming:
                                print("\nAssistant: ", end="", flush=True)
                                _streaming = True
                            print(event["content"], end="", flush=True)
                            _answer_buf += event["content"]

                        elif etype == "answer":
                            # Full answer for citation check; tokens already printed
                            _answer_buf = event["content"]
                            if not _streaming:
                                print(f"\nAssistant: {_answer_buf}")
                            else:
                                print()   # newline after last token

                        elif etype == "error":
                            print(f"\n[Error] {event['error']}")

                    _warn_if_no_citations(_answer_buf)

                    # ── Reflection pass ──────────────────────────────────────
                    # One extra LLM call to critique the draft answer.
                    # If issues are found, re-run the agent with the critique
                    # embedded so it searches again and revises.
                    if reflect and _answer_buf:
                        print("\n  [Reflecting on answer...]")
                        critique, needs_revision = reflect_on_answer(
                            rag_chain.llm, question, _answer_buf
                        )
                        if needs_revision:
                            print(f"  [Revision needed] {critique[:200]}")
                            revised_question = (
                                f"{question}\n\n"
                                f"[REFLECTION] Your previous answer had issues:\n{critique}\n"
                                f"Please search again and provide an improved answer that addresses these issues."
                            )
                            # Reset streaming state for the revised answer
                            _streaming = False
                            _answer_buf = ""
                            for event in run_react_agent(agent, revised_question, thread_id=thread_id):
                                etype = event["type"]
                                if etype == "tool_call":
                                    args_dict = event["args"]
                                    q_text = args_dict.get("query", "")
                                    section = args_dict.get("section_type", "")
                                    display = f"  [Re-searching] \"{q_text}\"" if q_text else f"  [{event['tool']}]"
                                    if section:
                                        display += f"  [section: {section}]"
                                    print(display)
                                elif etype == "token":
                                    if not _streaming:
                                        print("\nAssistant (revised): ", end="", flush=True)
                                        _streaming = True
                                    print(event["content"], end="", flush=True)
                                    _answer_buf += event["content"]
                                elif etype == "answer":
                                    _answer_buf = event["content"]
                                    if not _streaming:
                                        print(f"\nAssistant (revised): {_answer_buf}")
                                    else:
                                        print()
                                elif etype == "error":
                                    print(f"\n[Error during revision] {event['error']}")
                            _warn_if_no_citations(_answer_buf)
                        else:
                            print("  [Reflection] Answer looks good — no revision needed.")

                    # Store in semantic cache if the question was cacheable
                    if _query_emb is not None and _answer_buf:
                        _sem_cache.put(question, _query_emb, _answer_buf)

                except TimeoutError as te:
                    print(f"\n[Timeout] {te}")
                finally:
                    if hasattr(signal, "SIGALRM"):
                        signal.alarm(0)   # disarm

            if args.query:
                # Single-shot mode
                print(f"\n{session_label}  |  retrieval: {args.retrieval}\n")
                _run_turn(args.query, reflect=args.reflect)
                info = embedding_gen.cache_info()
                if info:
                    print(f"\n[Embed cache] hits={info.hits}  misses={info.misses}  cached={info.currsize}/{info.maxsize}")
                sc = _sem_cache.info()
                print(f"[Semantic cache] checks={sc['total_checks']}  hits={sc['total_hits']}  hit_rate={sc['hit_rate']}  size={sc['size']}/{sc['maxsize']}")
            else:
                # Interactive conversational loop
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
                        info = embedding_gen.cache_info()
                        if info:
                            print(f"[Embed cache] hits={info.hits}  misses={info.misses}  cached={info.currsize}/{info.maxsize}")
                        sc = _sem_cache.info()
                        print(f"[Semantic cache] checks={sc['total_checks']}  hits={sc['total_hits']}  hit_rate={sc['hit_rate']}  size={sc['size']}/{sc['maxsize']}")
                        if _sem_cache.top_entries():
                            for e in _sem_cache.top_entries(3):
                                print(f"  {e['hits']}x  \"{e['query']}\"")
                        print("Goodbye.")
                        break
                    if user_input.lower() == "new":
                        _SESSION_COUNTER += 1
                        thread_id = str(uuid4())
                        session_label = f"session-{_SESSION_COUNTER}"
                        print(f"New session started: {session_label}\n")
                        continue
                    _run_turn(user_input, reflect=args.reflect)
                    print()

        else:
            # ── V1 orchestrator (original) ─────────────────────────────────
            print("Initializing agentic orchestrator...")
            orchestrator = AgenticRAGOrchestrator(rag_chain, llm)
            print(f"\nQuerying in agentic mode with {args.retrieval} retrieval strategy...")

            for result in orchestrator.query(
                question=args.query,
                top_k=args.top_k,
                filters=filters,
                workflow=args.workflow,
                max_iterations=args.max_iterations
            ):
                format_output(result)

    except KeyboardInterrupt:
        logger.warning("Query interrupted by user")
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during query execution: {e}", exc_info=True)
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        logger.info("Query execution completed")


if __name__ == "__main__":
    main()

