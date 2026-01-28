import re
from typing import Optional, List
from .history import ConversationHistory

class QueryRewriter:
    """
    Rewrites queries using conversation history context.
    
    Handles reference resolution, pronoun replacement, and query expansion
    based on previous conversation turns.
    """
    
    def __init__(self, use_llm_rewriting: bool = False):
        """
        Initialize query rewriter.
        
        Args:
            use_llm_rewriting: Whether to use LLM for complex rewriting (not implemented yet)
        """
        
        self.use_llm_rewriting = use_llm_rewriting
    
    def rewrite(self, query: str, history: ConversationHistory) -> str:
        """
        Rewriting query with conversation context function.
        
        Args:
            query: Original query string
            history: Conversation history instance
        
        Returns:
            result: Rewritten query with context injected
        """
        
        if not history.messages:
            return query
        
        # Extracting entities from history
        entities = history.get_recent_entities()
        
        # Resolving references
        rewritten = self._resolve_references(query, history, entities)
        
        # Expanding with context keywords
        rewritten = self._expand_with_context(rewritten, history)
        
        return rewritten
    
    def _resolve_references(self, query: str, history: ConversationHistory, entities: List[str]) -> str:
        """
        Resolving references in query function.
        
        Args:
            query: Query string
            history: Conversation history
            entities: List of entities from history
        
        Returns:
            result: Query with references resolved
        """
        
        rewritten = query
        
        # Resolving "the last N papers" → "paper 1, paper 2, ..."
        last_papers_match = re.search(r'the\s+last\s+(\d+)\s+papers?', query, re.IGNORECASE)
        if last_papers_match:
            n = int(last_papers_match.group(1))
            paper_entities = [e for e in entities if "paper" in e.lower() or any(c.isdigit() for c in e)]
            
            if len(paper_entities) >= n:
                # Taking the last N papers mentioned
                recent_papers = paper_entities[-n:]
                papers_str = ", ".join(recent_papers)
                rewritten = rewritten.replace(last_papers_match.group(0), papers_str)
        
        # Resolving "previous paper" → last mentioned paper
        if re.search(r'previous\s+paper', query, re.IGNORECASE):
            paper_entities = [e for e in entities if "paper" in e.lower() or any(c.isdigit() for c in e)]
            if paper_entities:
                last_paper = paper_entities[-1]
                rewritten = re.sub(r'previous\s+paper', last_paper, rewritten, flags=re.IGNORECASE)
        
        # Resolving "that method" → last mentioned method
        if re.search(r'that\s+method', query, re.IGNORECASE):
            method_entities = [e for e in entities if e[0].isupper() and len(e.split()) <= 3]
            if method_entities:
                last_method = method_entities[-1]
                rewritten = re.sub(r'that\s+method', last_method, rewritten, flags=re.IGNORECASE)
        
        # Resolving "this approach", "that method", "this method" anywhere in query
        if re.search(r'\b(this|that)\s+(approach|method|technique|framework|model|system)\b', rewritten, re.IGNORECASE):
            # First try to get from entities (most reliable)
            recent_entities = history.get_recent_entities()
            method_entities = [e for e in recent_entities if e[0].isupper() and len(e.split()) <= 3]
            
            replacement = None
            if method_entities:
                # Use the most recent method entity
                replacement = method_entities[-1]
            else:
                # Fallback: extract from last assistant message
                last_assistant_msg = None
                for msg in reversed(history.messages):
                    if msg.role == "assistant":
                        last_assistant_msg = msg.content
                        break
                
                if last_assistant_msg:
                    # Extracting key terms (likely the main topic)
                    key_terms = self._extract_key_terms(last_assistant_msg)
                    if key_terms:
                        replacement = key_terms[0]
            
            if replacement:
                # Replace "this/that approach/method" with the actual term
                rewritten = re.sub(
                    r'\b(this|that)\s+(approach|method|technique|framework|model|system)\b',
                    replacement,
                    rewritten,
                    flags=re.IGNORECASE
                )
        
        # Resolving "it", "this", "that" at start of query
        if re.match(r'^(what|how|why|tell|explain|can|could|would)\s+(about\s+)?(it|this|that)', rewritten, re.IGNORECASE):
            # Using the last user query as context
            last_user_msg = None
            for msg in reversed(history.messages):
                if msg.role == "user":
                    last_user_msg = msg.content
                    break
            
            if last_user_msg:
                # Extracting key terms from last query
                key_terms = self._extract_key_terms(last_user_msg)
                if key_terms:
                    rewritten = re.sub(
                        r'^(what|how|why|tell|explain|can|could|would)\s+(about\s+)?(it|this|that)',
                        f"{rewritten.split()[0]} about {key_terms[0] if key_terms else 'GAPO'}",
                        rewritten,
                        flags=re.IGNORECASE
                    )
        
        return rewritten
    
    def _expand_with_context(self, query: str, history: ConversationHistory) -> str:
        """
        Expanding query with context keywords function.
        
        Args:
            query: Query string
            history: Conversation history
        
        Returns:
            result: Expanded query
        """
        
        if not history.messages:
            return query
        
        # Extracting keywords from recent messages
        context_keywords = []
        
        # Getting keywords from last assistant message (answer)
        for msg in reversed(history.messages):
            if msg.role == "assistant":
                keywords = self._extract_key_terms(msg.content)
                context_keywords.extend(keywords)
                break
        
        # Getting keywords from last user message (question)
        for msg in reversed(history.messages):
            if msg.role == "user":
                keywords = self._extract_key_terms(msg.content)
                context_keywords.extend(keywords)
                break
        
        # If query is very short, add context
        if len(query.split()) < 5 and context_keywords:
            # Adding relevant keywords that aren't already in query
            query_lower = query.lower()
            new_keywords = [kw for kw in context_keywords if kw.lower() not in query_lower]
            if new_keywords:
                query = f"{query} {' '.join(new_keywords[:3])}"
        
        return query
    
    def _extract_key_terms(self, text: str, max_terms: int = 5) -> List[str]:
        """
        Extracting key terms from text function.
        
        Args:
            text: Text to extract terms from
            max_terms: Maximum number of terms to extract
        
        Returns:
            result: List of key terms
        """
        
        # Simple extraction: capitalized words, technical terms, numbers
        words = text.split()
        key_terms = []
        
        for word in words:
            # Removing punctuation
            clean_word = re.sub(r'[^\w\s]', '', word)
            
            if not clean_word:
                continue
            
            # Capitalized words (likely proper nouns, methods, etc.)
            if clean_word[0].isupper() and len(clean_word) > 2:
                key_terms.append(clean_word)
            
            # Numbers (paper numbers, years, etc.)
            elif clean_word.isdigit():
                key_terms.append(clean_word)
            
            # Technical terms (all caps or mixed case)
            elif clean_word.isupper() and len(clean_word) > 1:
                key_terms.append(clean_word)
        
        # Removing duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)
        
        return unique_terms[:max_terms]
    
    def extract_entities_from_query(self, query: str) -> List[str]:
        """
        Extracting entities from query function.
        
        Args:
            query: Query string
        
        Returns:
            result: List of extracted entities
        """
        
        entities = []
        
        # Paper references: "paper 1", "paper 2", etc.
        paper_matches = re.findall(r'paper\s+(\d+)', query, re.IGNORECASE)
        for match in paper_matches:
            entities.append(f"paper {match}")
        
        # Method names (capitalized words, likely acronyms)
        method_matches = re.findall(r'\b([A-Z][A-Z0-9]{2,})\b', query)
        entities.extend(method_matches)
        
        # Quoted strings (likely paper titles or method names)
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)
        
        return entities

