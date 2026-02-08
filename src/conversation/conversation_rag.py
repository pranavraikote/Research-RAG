import re
import sys
from pathlib import Path
from typing import List

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
        ollama_model = "qwen2:1.5b", use_quantization = True, quantization_bits = 4,
        enable_prompt_cache = True):
        """
        Initialize conversational RAG chain.

        Args:
            embedding_generator: Embedding generator instance
            retriever: Retriever instance (semantic, BM25, or hybrid)
            llm_model: LLM model name (HuggingFace format)
            llm_provider: LLM provider ("auto", "ollama", or "huggingface")
            ollama_model: Ollama model name (used when provider is "auto" or "ollama")
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
            ollama_model=ollama_model,
            use_quantization=use_quantization,
            quantization_bits=quantization_bits,
            enable_prompt_cache=enable_prompt_cache
        )
        
        self.query_rewriter = QueryRewriter()
        self.conversation_history: ConversationHistory = None
        self._discussed_papers: List[str] = []
        self._last_sources: List[dict] = []
    
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

        # Resetting paper tracking if conversation was cleared
        if conversation and not conversation.messages:
            self._discussed_papers = []
            self._last_sources = []

        # Detecting acknowledgments/greetings (don't process as queries)
        acknowledgment_patterns = [
            r'^(thanks?|thank you|thx|ty)$',
            r'^(ok|okay|okay thanks|ok thanks)$',
            r'^(good job|well done|nice|great)$',
            r'^(thanks|thank you).*(good|job|nice|great)',
            r'^(ok|okay).*(thanks?|thank you)',
        ]
        
        question_lower = question.strip().lower()
        is_acknowledgment = any(re.match(pattern, question_lower) for pattern in acknowledgment_patterns)
        
        if is_acknowledgment:
            # Short acknowledgment response, don't process as query
            acknowledgment_response = "You're welcome! Feel free to ask if you have any other questions."
            conversation.add_message("user", question, entities=[])
            conversation.add_message("assistant", acknowledgment_response, entities=[])

            # Yielding the actual response text so the CLI can display it
            yield (acknowledgment_response, {
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
        history_text = conversation.get_history_text(include_system=False, max_chars_per_message=300)

        # Extracting entities from query
        entities = self.query_rewriter.extract_entities_from_query(question)

        # Resolving citation-based references ("paper 1", "[1]") to actual titles
        resolved_question = self._resolve_citation_reference(question)

        # Rewriting query with conversation context (uses resolved question)
        rewritten_query = self.query_rewriter.rewrite(resolved_question, conversation)

        # Boosting query with discussed paper titles for follow-up queries
        if self._is_followup_query(question):
            rewritten_query = self._boost_query_with_discussed_papers(rewritten_query)

        # Adding user message to history (original question, not modified)
        conversation.add_message("user", question, entities=entities)

        # Injecting history into user message (not system prompt) so the KV cache stays valid
        if history_text:
            enhanced_question = f"""CONVERSATION HISTORY:
{history_text}

INSTRUCTIONS: The history above shows previous questions and answers. Use it to understand references (e.g., "the last 2 papers", "that method"). You MUST still cite sources from the CURRENT context provided, not from conversation history.

CURRENT QUESTION: {rewritten_query}"""
        else:
            enhanced_question = rewritten_query
        
        # Querying the underlying RAG chain with enhanced question
        answer = ""
        answer_entities = []
        
        # Repetition detection for conversational mode
        recent_text = ""
        repetition_threshold = 100  # Characters to check for repetition
        last_metadata = None

        for token, metadata in self.rag_chain.query(
            enhanced_question,
            top_k=top_k,
            initial_retrieval_k=initial_retrieval_k,
            rerank_k=rerank_k,
            filters=filters,
            auto_parse_filters=auto_parse_filters
        ):
            last_metadata = metadata

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
        
        # Tracking discussed papers from retrieval results
        if last_metadata and last_metadata.get("sources"):
            self._last_sources = last_metadata["sources"]
            new_titles = self._extract_paper_titles_from_sources(last_metadata["sources"])
            seen_lower = {t.lower() for t in self._discussed_papers}
            for title in new_titles:
                if title.lower() not in seen_lower:
                    self._discussed_papers.append(title)
                    seen_lower.add(title.lower())

        # Extracting entities from answer
        answer_entities = self.query_rewriter.extract_entities_from_query(answer)
        
        # Adding assistant response to history
        conversation.add_message("assistant", answer, entities=answer_entities)

    def clear_paper_context(self):
        """Clear discussed paper tracking state."""
        self._discussed_papers = []
        self._last_sources = []

    def _extract_paper_titles_from_sources(self, sources: list) -> list:
        """Extract unique paper titles from retrieval sources."""
        titles = []
        seen = set()
        for source in sources:
            title = source.get("metadata", {}).get("title", "")
            if title and title != "Unknown" and title.lower() not in seen:
                seen.add(title.lower())
                titles.append(title)
        return titles

    def _is_followup_query(self, question: str) -> bool:
        """
        Detect if a query is a follow-up about previously discussed papers.

        Uses heuristics: reference pronouns, follow-up indicators + short query,
        and explicit citation references.
        """
        if not self._discussed_papers:
            return False

        q_lower = question.strip().lower()
        words = q_lower.split()

        # Reference pronouns/determiners pointing to previous context
        reference_patterns = [
            r'\b(it|its|they|their|them)\b',
            r'\b(this|that|these|those)\s+(paper|method|approach|technique|model|system|framework|study|work|result)',
            r'\bthe\s+(paper|method|approach|technique|model|system|framework|study|work)\b',
            r'\b(same|above|mentioned|discussed|previous)\s+(paper|method|approach)',
        ]
        has_reference = any(re.search(p, q_lower) for p in reference_patterns)

        # Follow-up indicator words
        followup_patterns = [
            r'\b(more|also|additionally|furthermore|besides)\b',
            r'\b(what|how|tell)\s+(about|else)',
            r'\b(explain|elaborate|expand|detail)',
            r'\b(compare|contrast|differ)',
            r'\b(and\s+what|but\s+what|so\s+what)',
        ]
        has_followup_word = any(re.search(p, q_lower) for p in followup_patterns)
        is_short = len(words) < 8

        # Explicit citation references ("paper 1", "[1]", "first paper")
        has_citation_ref = bool(re.search(r'paper\s+\d+|\[\d+\]|(first|second|third|last)\s+paper', q_lower))

        return has_citation_ref or has_reference or (is_short and has_followup_word)

    def _resolve_citation_reference(self, question: str) -> str:
        """
        Resolve citation-based references ("paper 1", "[1]") to actual paper titles
        from the most recent sources.
        """
        if not self._last_sources:
            return question

        result = question

        # Matching "paper N" patterns
        for match in re.finditer(r'paper\s+(\d+)', question, re.IGNORECASE):
            citation_num = int(match.group(1))
            for source in self._last_sources:
                if source.get("citation_number") == citation_num:
                    title = source.get("metadata", {}).get("title", "")
                    if title and title != "Unknown":
                        result = result.replace(match.group(0), f'"{title}"')
                        break

        # Matching "[N]" patterns
        for match in re.finditer(r'\[(\d+)\]', question):
            citation_num = int(match.group(1))
            for source in self._last_sources:
                if source.get("citation_number") == citation_num:
                    title = source.get("metadata", {}).get("title", "")
                    if title and title != "Unknown":
                        result = result.replace(match.group(0), f'"{title}"')
                        break

        return result

    def _boost_query_with_discussed_papers(self, rewritten_query: str, max_titles: int = 2) -> str:
        """
        Boost a follow-up query by appending recently discussed paper titles.

        Works as natural query expansion - the paper title tokens increase
        relevance scores for chunks from that paper in both semantic and BM25 retrieval.
        """
        if not self._discussed_papers:
            return rewritten_query

        # Taking the most recent N titles (most likely to be relevant)
        recent_titles = self._discussed_papers[-max_titles:]

        # Filtering out titles already substantially present in the query
        query_lower = rewritten_query.lower()
        stopwords = {'a', 'an', 'the', 'of', 'for', 'in', 'on', 'to', 'and', 'with', 'by', 'from', 'is', 'are', 'at'}
        titles_to_add = []
        for title in recent_titles:
            significant_words = set(title.lower().split()) - stopwords
            if significant_words:
                overlap = sum(1 for w in significant_words if w in query_lower)
                overlap_ratio = overlap / len(significant_words)
                if overlap_ratio < 0.5:
                    titles_to_add.append(title)

        if not titles_to_add:
            return rewritten_query

        titles_str = "; ".join(titles_to_add)
        return f"{rewritten_query} [context: {titles_str}]"
