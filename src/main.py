import argparse
import re
import sys
from pathlib import Path

# Adding parent directory to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import EmbeddingGenerator
from src.rag_chain import RAGChain
from src.retrieval.semantic import SemanticRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever

def main():
    """
    Main function.
    """

    parser = argparse.ArgumentParser(description='Research Paper Navigator')
    
    parser.add_argument('-q', '--query', required = True, help = 'Query to ask about research papers')
    parser.add_argument('-r', '--retrieval', default = 'semantic', choices = ['semantic', 'bm25', 'hybrid'], help = 'Retrieval strategy')
    parser.add_argument('-k', '--top-k', type = int, default = 5, help = 'Number of chunks to return in sources')
    parser.add_argument('--initial-retrieval-k', type = int, default = 20, help = 'Number of chunks to retrieve initially')
    parser.add_argument('--rerank-k', type = int, default = 3, help = 'Number of top chunks to be used by LLM')
    parser.add_argument('--metric', default = 'IP', choices = ['IP', 'L2'], help = 'Distance metric for semantic search')
    parser.add_argument('--fusion', default = 'rrf', choices = ['rrf', 'weighted'], help = 'Fusion method for hybrid retrieval (rrf or weighted)')
    
    project_root = Path(__file__).parent.parent
    default_index = str(project_root / "artifacts/faiss_index")
    default_chunks = str(project_root / "artifacts/chunks.json")
    
    # Chunk and FAISS path
    parser.add_argument('--index-path', default = default_index, help = 'Path to FAISS index')
    parser.add_argument('--chunks-path', default = default_chunks, help = 'Path to chunks JSON file')
    
    # Metadata filtering options
    parser.add_argument('--conference', help = 'Filter by conference (e.g., "ACL" or "ACL,EMNLP" for multiple)')
    parser.add_argument('--year', help = 'Filter by year: single year (e.g., "2024"), multiple (e.g., "2023,2024"), or range (e.g., "2020-2024")')
    parser.add_argument('--title', help = 'Filter by title (partial match, e.g., "transformer" or "transformer,bert" for multiple)')
    
    args = parser.parse_args()
    
    # Building filters dictionary
    filters = {}
    if args.conference:
        conferences = [c.strip() for c in args.conference.split(',')]
        filters['conference'] = conferences[0] if len(conferences) == 1 else conferences
    
    if args.year:
        year_str = args.year.strip()
        if '-' in year_str:
            # Range format: "2020-2024"
            try:
                min_year, max_year = map(int, year_str.split('-'))
                filters['year'] = {'min': min_year, 'max': max_year}
            except ValueError:
                print(f"Invalid year range format '{year_str}'. Use format 'YYYY-YYYY'")
        else:
            # Single or multiple years: "2024" or "2023,2024"
            try:
                years = [int(y.strip()) for y in year_str.split(',')]
                filters['year'] = years[0] if len(years) == 1 else years
            except ValueError:
                print(f"Invalid year format '{year_str}'. Use format 'YYYY' or 'YYYY,YYYY'")
    
    if args.title:
        titles = [t.strip() for t in args.title.split(',')]
        filters['title'] = titles[0] if len(titles) == 1 else titles
    
    # Converting empty filters to None
    filters = filters if filters else None
    
    # Initializing the EmbeddingGenerator
    embedding_gen = EmbeddingGenerator()
    
    # Initializing retriever based on strategy
    if args.retrieval == 'semantic':
        retriever = SemanticRetriever(index_path = args.index_path, 
                                    chunks_path = args.chunks_path, 
                                    metric = args.metric
                                    )

    elif args.retrieval == 'bm25':
        bm25_index_path = str(Path(args.index_path).parent / "bm25_index")
        retriever = BM25Retriever(chunks_path = args.chunks_path, index_path = bm25_index_path)

    elif args.retrieval == 'hybrid':
        bm25_index_path = str(Path(args.index_path).parent / "bm25_index")

        semantic_ret = SemanticRetriever(index_path = args.index_path,
                                        chunks_path = args.chunks_path,
                                        metric = args.metric
                                        )
        bm25_ret = BM25Retriever(chunks_path = args.chunks_path, index_path = bm25_index_path)

        retriever = HybridRetriever(semantic_ret, bm25_ret, fusion_method = args.fusion)
    
    # Initializing the RAG chain
    rag = RAGChain(
        embedding_generator = embedding_gen,
        retriever = retriever,
        llm_model = "Qwen/Qwen2-1.5B-Instruct",
        llm_provider = "huggingface",
        enable_prompt_cache = True
    )
    
    # Building filter info string for display
    filter_info = ""
    if filters:
        filter_parts = []
        if 'conference' in filters:
            conf = filters['conference']
            filter_parts.append(f"conference={conf if isinstance(conf, str) else ','.join(conf)}")
        if 'year' in filters:
            yr = filters['year']
            if isinstance(yr, dict):
                filter_parts.append(f"year={yr['min']}-{yr['max']}")
            elif isinstance(yr, list):
                filter_parts.append(f"year={','.join(map(str, yr))}")
            else:
                filter_parts.append(f"year={yr}")
        if 'title' in filters:
            tit = filters['title']
            filter_parts.append(f"title={tit if isinstance(tit, str) else ','.join(tit)}")
        filter_info = f" with filters: {', '.join(filter_parts)}"
    
    print(f"Querying with {args.retrieval} retrieval strategy{filter_info}...")
    
    # Streaming output
    print("\n## Answer\n")
    print("---\n")
    
    result_metadata = None
    for token, metadata in rag.query(
        args.query, 
        top_k=args.top_k,
        initial_retrieval_k=args.initial_retrieval_k,
        rerank_k=args.rerank_k,
        filters=filters,
        auto_parse_filters=True
    ):
        
        # Streaming tokens
        if token:
            sys.stdout.write(token)
            sys.stdout.flush()
        
        result_metadata = metadata
    
    print("\n\n---\n")
    
    # Checking for citations in the answer
    if result_metadata and result_metadata.get('answer'):
        answer = result_metadata['answer']
        has_citations = bool(re.search(r'\[\d+\]', answer))
        if not has_citations and result_metadata.get('citation_map'):
            print("The answer does not contain citation numbers\n")
    
    # Displaying metrics
    if result_metadata:
        tfft = result_metadata.get('tfft')
        total_time = result_metadata.get('total_time')
        
        print("### Performance Metrics\n")
        if tfft:
            print(f"- **Time to First Token (TFFT)**: {tfft*1000:.2f} ms")
        if total_time:
            print(f"- **Total Generation Time**: {total_time:.2f} s")
        print()
        
        # Displaying auto-detected filters if any
        if result_metadata.get('filters_applied') and not filters:
            auto_filters = result_metadata['filters_applied']
            filter_parts = []
            if 'conference' in auto_filters:
                conf = auto_filters['conference']
                filter_parts.append(f"conference={conf if isinstance(conf, str) else ','.join(conf)}")
            if 'year' in auto_filters:
                yr = auto_filters['year']
                if isinstance(yr, dict):
                    filter_parts.append(f"year={yr['min']}-{yr['max']}")
                elif isinstance(yr, list):
                    filter_parts.append(f"year={','.join(map(str, yr))}")
                else:
                    filter_parts.append(f"year={yr}")
            if 'title' in auto_filters:
                tit = auto_filters['title']
                filter_parts.append(f"title={tit if isinstance(tit, str) else ','.join(tit)}")
            if filter_parts:
                print(f"**Auto-detected filters**: {', '.join(filter_parts)}\n")
        
        # Displaying sources in markdown format
        if 'sources' in result_metadata:
            print("### Sources\n")
            for source in result_metadata['sources']:
                citation_num = source.get('citation_number', '?')
                reranked_score = source.get('score', 0)  # Reranked score
                original_score = source.get('original_score', None)  # Original retrieval score
                metadata = source.get('metadata', {})
                text = source.get('text', '')
                
                # Display both scores if original_score is available
                if original_score is not None:
                    print(f"### [{citation_num}] Retrieval Score: {original_score:.4f} | Reranked Score: {reranked_score:.4f}\n")
                else:
                    print(f"### [{citation_num}] Score: {reranked_score:.4f}\n")
                
                print(f"**Title**: {metadata.get('title', 'Unknown')}  \n")
                print(f"**Conference**: {metadata.get('conference', 'Unknown')} {metadata.get('year', 'Unknown')}  \n")
                print(f"**Preview**: {text}\n")

if __name__ == '__main__':
    main()