import sys
from pathlib import Path

# Adding parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag_chain import RAGChain
from .history import ConversationHistory
from .query_rewriter import QueryRewriter

class ConversationalRAGChain:
    """
    Conversational wrapper around RAGChain.
    
    Maintains conversation history, rewrites queries with context,
    and integrates history into prompts for multi-turn conversations.
    """
    
    def __init__(self, embedding_generator, retriever, llm_model, llm_provider,
        use_quantization = True, quantization_bits = 4, enable_prompt_cache = True):
        """
        Initialize conversational RAG chain.
        
        Args:
            embedding_generator: Embedding generator instance
            retriever: Retriever instance (semantic, BM25, or hybrid)
            llm_model: LLM model name
            llm_provider: LLM provider
            use_quantization: Quantization for memory efficiency
            quantization_bits: Quantization bits (4 or 8)
            enable_prompt_cache: Enable prompt caching for faster generation
        """
        
        # Wrapping the existing RAGChain (composition, not modification)
        self.rag_chain = RAGChain(
            embedding_generator=embedding_generator,
            retriever=retriever,
            llm_model=llm_model,
            llm_provider=llm_provider,
            use_quantization=use_quantization,
            quantization_bits=quantization_bits,
            enable_prompt_cache=enable_prompt_cache
        )
        
        self.query_rewriter = QueryRewriter()
        self.conversation_history: ConversationHistory = None
    
    def query(self, question, conversation: ConversationHistory = None, 
        top_k = 5, initial_retrieval_k = 20, rerank_k = 3, filters = None, 
        auto_parse_filters = True):
        """
        Query function with conversation history support.
        
        Args:
            question: User query
            conversation: ConversationHistory instance (optional, creates new if None)
            top_k: Number of chunks to return in sources (after re-ranking)
            initial_retrieval_k: Number of initial chunks
            rerank_k: Number of top chunks delivered to model
            filters: Dictionary of metadata filters
            auto_parse_filters: If True, automatically extract filters from the query
        
        Yields:
            Tuples of (token, metadata_dict) with timing metrics, answer, sources, and citations
        """
        
        # Creating new conversation if not provided
        if conversation is None:
            conversation = ConversationHistory()
        
        self.conversation_history = conversation
        
        # Detecting acknowledgments/greetings (don't process as queries)
        acknowledgment_patterns = [
            r'^(thanks?|thank you|thx|ty)$',
            r'^(ok|okay|okay thanks|ok thanks)$',
            r'^(good job|well done|nice|great)$',
            r'^(thanks|thank you).*(good|job|nice|great)',
            r'^(ok|okay).*(thanks?|thank you)',
        ]
        
        import re
        question_lower = question.strip().lower()
        is_acknowledgment = any(re.match(pattern, question_lower) for pattern in acknowledgment_patterns)
        
        if is_acknowledgment:
            # Short acknowledgment response, don't process as query
            acknowledgment_response = "You're welcome! Feel free to ask if you have any other questions."
            conversation.add_message("user", question, entities=[])
            conversation.add_message("assistant", acknowledgment_response, entities=[])
            
            # Yield empty response
            yield ("", {
                "tfft": None,
                "total_time": 0.0,
                "complete": True,
                "answer": acknowledgment_response,
                "sources": [],
                "citation_map": {},
                "question": question,
                "filters_applied": None
            })
            return
        
        # Getting conversation history text (before adding current message)
        # Aggressively summarizing to prevent long prompts and repetition
        history_text = conversation.get_history_text(include_system=False, max_chars_per_message=150)
        
        # Extracting entities from query
        entities = self.query_rewriter.extract_entities_from_query(question)
        
        # Rewriting query with conversation context
        rewritten_query = self.query_rewriter.rewrite(question, conversation)
        
        # Adding user message to history (after rewriting, before processing)
        conversation.add_message("user", question, entities=entities)
        
        # Injecting history into system prompt (better design than prepending to question)
        # Temporarily modify system prompt to include conversation context
        original_system_prompt = self.rag_chain.system_prompt
        
        if history_text:
            # Adding conversation context to system prompt with better structure
            enhanced_system_prompt = f"""{original_system_prompt}

PREVIOUS CONVERSATION:
{history_text}

CONVERSATION INSTRUCTIONS:
- The conversation history above shows previous questions and answers.
- If the current question references previous topics (e.g., "the last 2 papers", "both papers", "that method"), use the conversation history to understand what was discussed.
- However, you MUST still cite sources from the CURRENT context provided below, not from conversation history.
- Use conversation history ONLY to understand references and provide coherent answers.
- If asked to compare papers mentioned in history, retrieve information from the current context about those papers."""
            
            # Temporarily replace system prompt
            self.rag_chain.system_prompt = enhanced_system_prompt
            
            # Disable prompt cache when history is present (prompt structure changes)
            # Save original cache state
            original_cache_enabled = self.rag_chain.prompt_cache_enabled
            if self.rag_chain.prompt_cache_enabled:
                self.rag_chain.prompt_cache_enabled = False
                self.rag_chain.cached_prompt_kv_cache = None
        else:
            original_cache_enabled = None
        
        # Using clean rewritten query (no history prepended)
        enhanced_question = rewritten_query
        
        # Querying the underlying RAG chain with enhanced question
        answer = ""
        answer_entities = []
        
        # Repetition detection for conversational mode
        recent_text = ""
        repetition_threshold = 100  # Characters to check for repetition
        
        for token, metadata in self.rag_chain.query(
            enhanced_question,
            top_k=top_k,
            initial_retrieval_k=initial_retrieval_k,
            rerank_k=rerank_k,
            filters=filters,
            auto_parse_filters=auto_parse_filters
        ):
            # Accumulating answer
            if token:
                answer += token
                recent_text += token
                
                # Keeping only last N characters for repetition check
                if len(recent_text) > repetition_threshold * 2:
                    recent_text = recent_text[-repetition_threshold:]
                
                # Checking for repetition pattern
                if len(answer) > repetition_threshold * 2:
                    last_chunk = answer[-repetition_threshold:]
                    prev_chunk = answer[-repetition_threshold*2:-repetition_threshold]
                    if last_chunk == prev_chunk:
                        # Repetition detected, stop yielding tokens
                        break
            
            # Yielding tokens as they come
            yield token, metadata
        
        # Extracting entities from answer
        answer_entities = self.query_rewriter.extract_entities_from_query(answer)
        
        # Adding assistant response to history
        conversation.add_message("assistant", answer, entities=answer_entities)
        
        # Restoring original system prompt and cache state
        if history_text:
            self.rag_chain.system_prompt = original_system_prompt
            if original_cache_enabled:
                self.rag_chain.prompt_cache_enabled = True
                # Reinitialize cache for next query (if no history)
                self.rag_chain._initialize_prompt_cache()
    

