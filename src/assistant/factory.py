"""
Factory function for AI assistant creation.
"""

import logging
from typing import Optional

from src.assistant.base import BaseAssistant
from src.assistant.simple import SimpleAIAssistant


def create_assistant(
    use_llm: bool = False,
    llm_provider: str = "gemini",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseAssistant:
    """Create and return an AI assistant instance.

    Parameters
    ----------
    use_llm:
        When True, attempt to initialise the requested LLM provider.
        Falls back to SimpleAIAssistant only on ImportError or ValueError
        (missing/invalid credentials), not on unrecognised provider names.
    llm_provider:
        LLM backend to use.  Currently only ``"gemini"`` is supported.
        Passing any other value raises ``ValueError`` so misconfiguration is
        visible rather than silently degraded.
    api_key:
        Provider API key.  Can also be supplied via environment variables.
    model:
        Model identifier.  Defaults to the value in config.GEMINI_MODEL.

    Returns
    -------
    BaseAssistant instance (GeminiAssistant or SimpleAIAssistant).

    Raises
    ------
    ValueError
        When ``use_llm=True`` and ``llm_provider`` is not a supported value.
    """
    if not use_llm:
        return SimpleAIAssistant()

    supported = {"gemini"}
    if llm_provider.lower() not in supported:
        raise ValueError(
            f"Unknown LLM provider '{llm_provider}'. "
            f"Supported providers: {', '.join(sorted(supported))}."
        )

    if llm_provider.lower() == "gemini":
        from src.assistant.gemini import GeminiAssistant  # local import avoids circular deps

        try:
            return GeminiAssistant(api_key=api_key, model=model)
        except (ImportError, ValueError) as exc:
            logging.warning(
                "Could not initialise GeminiAssistant: %s. Using SimpleAIAssistant.", exc
            )
            return SimpleAIAssistant()

    # Unreachable given the guard above, but satisfies the type-checker.
    return SimpleAIAssistant()  # pragma: no cover
