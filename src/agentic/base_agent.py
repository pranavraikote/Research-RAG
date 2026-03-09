"""
Base agent class for the v1 two-agent orchestrator (RetrieverAgent + ReasonerAgent).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):

    def __init__(self, name: str, description: str, llm):
        """
        Args:
            name: Agent name.
            description: Short description of the agent's role.
            llm: Language model for reasoning.
        """
        self.name = name
        self.description = description
        self.llm = llm
        self.state: Dict[str, Any] = {}

    def update_state(self, key: str, value: Any) -> None:
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    @abstractmethod
    def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's main task and return a result dict."""
