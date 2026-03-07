"""
BaseAssistant — abstract contract that all AI assistant implementations must satisfy.

Adding a new provider (e.g. Anthropic Claude, OpenAI) means subclassing
BaseAssistant and implementing the two abstract methods; no other file needs
to change (Open/Closed Principle).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseAssistant(ABC):
    """Abstract base class for all AI assistant implementations."""

    #: Human-readable name shown in the UI.
    name: str = "Abstract Assistant"

    @abstractmethod
    def analyze_alert(
        self,
        alert_data: Dict[str, Any],
        prediction: Dict[str, Any],
        explanation: Dict[str, Any],
        playbook: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a structured analysis of the alert.

        The returned dict must contain at least:
        - ``timestamp`` (ISO-8601 str)
        - ``assistant_name`` (str)
        - ``alert_assessment`` (str — the predicted label)
        - ``risk_score`` (float — malicious probability)
        """

    @abstractmethod
    def answer_question(self, question: str, context: Dict[str, Any]) -> str:
        """Return a plain-text answer to a user question given alert context."""
