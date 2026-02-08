import argparse
import re
import sys
from pathlib import Path

# Adding parent directory to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import EmbeddingGenerator
from src.retrieval.semantic import SemanticRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.conversation.conversation_rag import ConversationalRAGChain
from src.conversation.history import ConversationHistory

def main():
    """
    Main function.
    """
    
    parser = argparse.ArgumentParser(description='Research Paper Navigator')
    
    parser.add_argument('-r', '--retrieval', default = 'semantic', choices = ['semantic', 'bm25', 'hybrid'], 
        help = 'Retrieval strategy')
    parser.add_argument('-k', '--top-k', type = int, default = 5, 
        help = 'Number of chunks to return in sources')
    parser.add_argument('--initial-retrieval-k', type = int, default = 20, 
        help = 'Number of chunks to retrieve initially')
    parser.add_argument('--rerank-k', type = int, default = 3, 
        help = 'Number of top chunks to be used by LLM')
    parser.add_argument('--metric', default = 'IP', choices = ['IP', 'L2'], 
        help = 'Distance metric for semantic search')
    parser.add_argument('--max-tokens', type = int, default = 2000, 
        help = 'Maximum tokens to keep in conversation history')
    parser.add_argument('--max-turns', type = int, default = None, 
        help = 'Maximum number of Q&A pairs to keep (None = unlimited)')
    
    project_root = Path(__file__).parent.parent
    default_index = str(project_root / "artifacts/faiss_index")
    default_chunks = str(project_root / "artifacts/chunks.json")
    
    # Chunk and FAISS path
    parser.add_argument('--index-path', default = default_index, help = 'Path to FAISS index')
    parser.add_argument('--chunks-path', default = default_chunks, help = 'Path to chunks JSON file')
    
    # LLM options
    parser.add_argument('--llm-model', default = 'Qwen/Qwen2-1.5B-Instruct',
        help = 'LLM model name (HuggingFace format)')
    parser.add_argument('--llm-provider', default = 'auto', choices = ['auto', 'ollama', 'huggingface'],
        help = 'LLM provider (auto = try Ollama first, fall back to HuggingFace)')
    parser.add_argument('--ollama-model', default = 'qwen2:1.5b',
        help = 'Ollama model name (used when provider is auto or ollama)')
    
    args = parser.parse_args()
    
    # Initializing the EmbeddingGenerator
    embedding_gen = EmbeddingGenerator()
    
    # Initializing retriever based on strategy
    if args.retrieval == 'semantic':
        retriever = SemanticRetriever(
            index_path = args.index_path, 
            chunks_path = args.chunks_path, 
            metric = args.metric
        )
    elif args.retrieval == 'bm25':
        bm25_index_path = str(Path(args.index_path).parent / "bm25_index")
        retriever = BM25Retriever(chunks_path = args.chunks_path, index_path = bm25_index_path)
    elif args.retrieval == 'hybrid':
        bm25_index_path = str(Path(args.index_path).parent / "bm25_index")
        semantic_ret = SemanticRetriever(
            index_path = args.index_path,
            chunks_path = args.chunks_path,
            metric = args.metric
        )
        bm25_ret = BM25Retriever(chunks_path = args.chunks_path, index_path = bm25_index_path)
        retriever = HybridRetriever(semantic_ret, bm25_ret)
    
    # Initializing the conversational RAG chain
    conversation_rag = ConversationalRAGChain(
        embedding_generator = embedding_gen,
        retriever = retriever,
        llm_model = args.llm_model,
        llm_provider = args.llm_provider,
        ollama_model = args.ollama_model,
        enable_prompt_cache = True
    )
    
    # Creating conversation history
    conversation = ConversationHistory(
        max_tokens = args.max_tokens,
        max_turns = args.max_turns
    )
    
    print(f"Querying in conversational mode with {args.retrieval} retrieval strategy...")
    print(f"Model: {args.llm_model}")
    print("\nCommands:")
    print("  - Type your question and press Enter")
    print("  - 'clear' or 'reset' - Clear conversation history")
    print("  - 'history' - Show conversation history")
    print("  - 'exit' or 'quit' - Exit the program")
    print()
    
    # Interactive loop
    while True:
        try:
            # Getting user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handling commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
            
            elif user_input.lower() in ['clear', 'reset']:
                conversation.clear()
                conversation_rag.clear_paper_context()
                print("Conversation history cleared.\n")
                continue
            
            elif user_input.lower() == 'history':
                if conversation.messages:
                    print("\n### Conversation History\n")
                    print(conversation.get_history_text(include_system=True))
                    print(f"\nTotal turns: {conversation.get_turn_count()}")
                    print(f"Total tokens: {conversation.get_total_tokens()}\n")
                else:
                    print("No conversation history yet.\n")
                continue
            
            # Processing query
            print("\n## Answer\n")
            print("---\n")
            
            result_metadata = None
            for token, metadata in conversation_rag.query(
                user_input,
                conversation = conversation,
                top_k = args.top_k,
                initial_retrieval_k = args.initial_retrieval_k,
                rerank_k = args.rerank_k,
                filters = None,
                auto_parse_filters = True
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
                
                # Displaying sources in markdown format
                if 'sources' in result_metadata:
                    print("### Sources\n")
                    for source in result_metadata['sources']:
                        citation_num = source.get('citation_number', '?')
                        reranked_score = source.get('score', 0)
                        original_score = source.get('original_score', None)
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
            
            # Showing conversation stats
            print(f"Conversation: {conversation.get_turn_count()} turns | {conversation.get_total_tokens()} tokens\n")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()

