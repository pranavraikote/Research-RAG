import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

@dataclass
class Message:
    """
    Single message in conversation history.
    
    Attributes:
        role: Message role ("user" or "assistant")
        content: Message content
        timestamp: Unix timestamp when message was created
        entities: Extracted entities (papers, methods, etc.)
    """
    
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    entities: List[str] = field(default_factory=list)

class ConversationHistory:
    """
    Manages conversation history with token counting and truncation.
    
    Stores Q&A pairs, tracks entities mentioned, and handles context
    window management to prevent overflow.
    """
    
    def __init__(self, max_tokens: int = 2000, max_turns: Optional[int] = None):
        """
        Initialize conversation history.
        
        Args:
            max_tokens: Maximum tokens to keep in history (approximate)
            max_turns: Maximum number of Q&A pairs to keep (None = unlimited)
        """
        
        self.messages: List[Message] = []
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.entity_tracker: Dict[str, List[str]] = {}  # entity_type -> [entity_names]
    
    def add_message(self, role: str, content: str, entities: Optional[List[str]] = None):
        """
        Adding message to history function.
        
        Args:
            role: Message role ("user" or "assistant")
            content: Message content
            entities: Optional list of entities extracted from message
        """
        
        message = Message(
            role=role,
            content=content,
            entities=entities or []
        )
        
        self.messages.append(message)
        
        # Tracking entities
        if entities:
            for entity in entities:
                entity_type = self._classify_entity(entity)
                if entity_type not in self.entity_tracker:
                    self.entity_tracker[entity_type] = []
                if entity not in self.entity_tracker[entity_type]:
                    self.entity_tracker[entity_type].append(entity)
        
        # Truncating if needed
        self._truncate_if_needed()
    
    def get_history_text(self, include_system: bool = False, max_chars_per_message: int = 200) -> str:
        """
        Getting formatted history text function.
        
        Args:
            include_system: Whether to include system message prefix
            max_chars_per_message: Maximum characters per message (for summarization)
        
        Returns:
            result: Formatted history as text
        """
        
        if not self.messages:
            return ""
        
        lines = []
        if include_system:
            lines.append("Previous conversation:")
        
        for msg in self.messages:
            role_label = "Q" if msg.role == "user" else "A"
            content = msg.content
            
            # Aggressive summarization for assistant messages (answers)
            if msg.role == "assistant" and len(content) > max_chars_per_message:
                # Taking first sentence and last sentence, with truncation indicator
                sentences = content.split('. ')
                if len(sentences) > 1:
                    # First sentence + "..." + last sentence
                    first = sentences[0]
                    last = sentences[-1] if sentences[-1] else sentences[-2] if len(sentences) > 1 else ""
                    content = f"{first}. ... {last}"[:max_chars_per_message]
                else:
                    # If no sentence breaks, just truncate
                    content = content[:max_chars_per_message] + "..."
            
            # Truncating user messages too if too long
            elif msg.role == "user" and len(content) > max_chars_per_message:
                content = content[:max_chars_per_message] + "..."
            
            lines.append(f"- {role_label}: {content}")
        
        return "\n".join(lines)
    
    def get_recent_entities(self, entity_type: Optional[str] = None) -> List[str]:
        """
        Getting recently mentioned entities function.
        
        Args:
            entity_type: Optional filter for entity type ("paper", "method", etc.)
        
        Returns:
            result: List of entity names
        """
        
        if entity_type:
            return self.entity_tracker.get(entity_type, [])
        
        # Returning all entities
        all_entities = []
        for entities in self.entity_tracker.values():
            all_entities.extend(entities)
        return all_entities
    
    def count_tokens(self, text: str) -> int:
        """
        Approximate token counting function (simple word-based).
        
        Args:
            text: Text to count tokens for
        
        Returns:
            result: Approximate token count
        """
        
        # Simple approximation: ~4 characters per token
        return len(text) // 4
    
    def get_total_tokens(self) -> int:
        """
        Getting total tokens in history function.
        
        Returns:
            result: Total approximate tokens
        """
        
        total = 0
        for msg in self.messages:
            total += self.count_tokens(msg.content)
        return total
    
    def clear(self):
        """
        Clearing conversation history function.
        """
        
        self.messages = []
        self.entity_tracker = {}
    
    def get_turn_count(self) -> int:
        """
        Getting number of Q&A pairs function.
        
        Returns:
            result: Number of turns (Q&A pairs)
        """
        
        user_messages = sum(1 for msg in self.messages if msg.role == "user")
        return user_messages
    
    def _truncate_if_needed(self):
        """
        Truncating history if it exceeds limits function.
        """
        
        # Turn-based truncation
        if self.max_turns and self.get_turn_count() > self.max_turns:
            # Keeping only the most recent turns
            user_indices = [i for i, msg in enumerate(self.messages) if msg.role == "user"]
            if len(user_indices) > self.max_turns:
                keep_from = user_indices[-self.max_turns]
                self.messages = self.messages[keep_from:]
                # Rebuilding entity tracker
                self._rebuild_entity_tracker()
        
        # Token-based truncation
        while self.get_total_tokens() > self.max_tokens and len(self.messages) > 2:
            # Removing oldest messages (but keep at least one Q&A pair)
            self.messages.pop(0)
            self._rebuild_entity_tracker()
    
    def _rebuild_entity_tracker(self):
        """
        Rebuilding entity tracker from current messages function.
        """
        
        self.entity_tracker = {}
        for msg in self.messages:
            for entity in msg.entities:
                entity_type = self._classify_entity(entity)
                if entity_type not in self.entity_tracker:
                    self.entity_tracker[entity_type] = []
                if entity not in self.entity_tracker[entity_type]:
                    self.entity_tracker[entity_type].append(entity)
    
    def _classify_entity(self, entity: str) -> str:
        """
        Classifying entity type function (simple heuristic).
        
        Args:
            entity: Entity name
        
        Returns:
            result: Entity type ("paper", "method", "concept", "unknown")
        """
        
        entity_lower = entity.lower()
        
        # Paper references
        if "paper" in entity_lower or any(char.isdigit() for char in entity):
            return "paper"
        
        # Method names (often capitalized or have specific patterns)
        if entity[0].isupper() and len(entity.split()) <= 3:
            return "method"
        
        return "concept"
    
    def export(self) -> Dict[str, Any]:
        """
        Exporting history to dictionary function.
        
        Returns:
            result: Dictionary representation of history
        """
        
        return {
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "entities": msg.entities
                }
                for msg in self.messages
            ],
            "entity_tracker": self.entity_tracker,
            "total_tokens": self.get_total_tokens(),
            "turn_count": self.get_turn_count()
        }
    
    def load(self, data: Dict[str, Any]):
        """
        Loading history from dictionary function.
        
        Args:
            data: Dictionary representation of history
        """
        
        self.messages = [
            Message(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=msg_data.get("timestamp", time.time()),
                entities=msg_data.get("entities", [])
            )
            for msg_data in data.get("messages", [])
        ]
        self.entity_tracker = data.get("entity_tracker", {})

