"""
Base Agent Framework

Simplified base class for agents - no complex tool system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAgent(ABC):
    
    def __init__(self, name, description, llm):
        """
        Initialize base agent function.
        
        Args:
            name: Agent name
            description: Agent description
            llm: Language model for reasoning
        """
        
        self.name = name
        self.description = description
        self.llm = llm
        self.state = {}
    
    def update_state(self, key, value):
        """
        Update agent state function.
        
        Args:
            key: State key
            value: State value
        """
        
        self.state[key] = value
    
    def get_state(self, key, default = None):
        """
        Get value from agent state function.
        
        Args:
            key: State key
            default: Default value if key not found
            
        Returns:
            State value or default
        """
        
        return self.state.get(key, default)
    
    @abstractmethod
    def execute(self, task, context):
        """
        Execute the agent's main task function.
        
        Args:
            task: Task description
            context: Context from previous agents or user
            
        Returns:
            Agent execution result
        """
        
        pass
