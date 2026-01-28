import argparse
import sys
import logging
from pathlib import Path

# Adding parent directory to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import EmbeddingGenerator
from src.rag_chain import RAGChain
from src.retrieval.semantic import SemanticRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.agentic.orchestrator import AgenticRAGOrchestrator


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


def main():
    """
    Main function for agentic RAG CLI.
    """
    
    parser = argparse.ArgumentParser(description='Research Paper Navigator - Agentic Mode')
    
    parser.add_argument('-q', '--query', required=True, help='Query to ask about research papers')
    parser.add_argument('-r', '--retrieval', default='semantic', 
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
                       help='LLM model name (HuggingFace) - Use 1.5B for MPS stability')
    parser.add_argument('--llm-provider', default='huggingface', choices=['huggingface', 'ollama'],
                       help='LLM provider')
    parser.add_argument('--no-quantization', action='store_true',
                       help='Disable quantization (use full precision)')
    parser.add_argument('--quantization-bits', type=int, default=4, choices=[4, 8],
                       help='Quantization bits (4 or 8)')
    parser.add_argument('--no-prompt-cache', action='store_true',
                       help='Disable prompt caching')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging (DEBUG level)')
    parser.add_argument('--log-file', help='Log to file instead of console')
    parser.add_argument('--show-prompts', action='store_true',
                       help='Show the actual prompts sent to LLM (very verbose)')
    
    project_root = Path(__file__).parent.parent
    default_index = str(project_root / "artifacts/faiss_index")
    default_chunks = str(project_root / "artifacts/chunks.json")
    
    parser.add_argument('--index-path', default=default_index, 
                       help='Path to FAISS index')
    parser.add_argument('--chunks-path', default=default_chunks, 
                       help='Path to chunks JSON file')
    
    # Metadata filtering options
    parser.add_argument('--conference', 
                       help='Filter by conference (e.g., "ACL" or "ACL,EMNLP")')
    parser.add_argument('--year', 
                       help='Filter by year: single (e.g., "2024"), multiple (e.g., "2023,2024"), or range (e.g., "2020-2024")')
    parser.add_argument('--title', 
                       help='Filter by title (partial match, e.g., "transformer")')
    
    args = parser.parse_args()
    
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
    
    # Store show_prompts flag for agents to use
    import os
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
    logger.info(f"Query: {args.query}")
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
        retriever = BM25Retriever(chunks_path=args.chunks_path)
    elif args.retrieval == 'hybrid':
        semantic_ret = SemanticRetriever(
            index_path=args.index_path,
            chunks_path=args.chunks_path,
            metric=args.metric
        )
        bm25_ret = BM25Retriever(chunks_path=args.chunks_path)
        retriever = HybridRetriever(semantic_ret, bm25_ret)
    
    # Initialize RAG chain
    print(f"Initializing RAG chain with {args.llm_provider}...")
    rag_chain = RAGChain(
        embedding_generator=embedding_gen,
        retriever=retriever,
        llm_model=args.llm_model,
        llm_provider=args.llm_provider,
        use_quantization=not args.no_quantization,
        quantization_bits=args.quantization_bits,
        enable_prompt_cache=not args.no_prompt_cache
    )
    
    # Get LLM from RAG chain for agents
    llm = rag_chain.llm
    
    # Initialize orchestrator
    print("Initializing agentic orchestrator...")
    logger.debug("Initializing agentic orchestrator")
    orchestrator = AgenticRAGOrchestrator(rag_chain, llm)
    
    print(f"\nQuerying in agentic mode with {args.retrieval} retrieval strategy...")
    
    logger.info("Starting query execution")
    
    # Execute query
    try:
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

