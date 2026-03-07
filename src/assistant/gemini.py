"""
Gemini-powered AI assistant.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from src import config
from src.assistant.base import BaseAssistant
from src.assistant.simple import SimpleAIAssistant

try:
    import google.generativeai as genai  # type: ignore[import]
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None  # type: ignore[assignment]

_SYSTEM_PROMPT = """You are an expert cybersecurity analyst assistant specializing in alert analysis and incident response.
Your role is to:
1. Analyze security alerts with deep understanding of threat patterns
2. Provide clear, actionable insights about alerts
3. Explain technical findings in accessible language
4. Recommend appropriate response actions
5. Answer questions about alerts, classifications, and recommended responses

You have access to:
- Alert classification results (Benign/Malignant)
- XAI explanations showing top contributing features
- Generated response playbooks for malicious alerts
- Complete alert metadata

Always be:
- Precise and technical when needed
- Clear and accessible for non-technical users
- Proactive in identifying risks
- Practical in recommendations"""


class GeminiAssistant(BaseAssistant):
    """LLM-powered assistant using Google Gemini."""

    name: str = "Gemini Security Analyst Assistant"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai is not installed. Run: pip install google-generativeai"
            )

        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Gemini API key not provided. "
                "Set GEMINI_API_KEY environment variable or pass api_key parameter."
            )

        resolved_model = model if model is not None else config.GEMINI_MODEL
        genai.configure(api_key=resolved_key)
        self.model_name: str = resolved_model
        self._model = genai.GenerativeModel(resolved_model)

    def analyze_alert(
        self,
        alert_data: Dict[str, Any],
        prediction: Dict[str, Any],
        explanation: Dict[str, Any],
        playbook: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        features_text = "\n".join(
            f"- {f.get('feature', 'Unknown')}: {f.get('value', 'N/A')} "
            f"(contribution: {f.get('contribution', 0):+.4f})"
            for f in explanation.get("top_features", [])[:5]
        )
        playbook_info = (
            f"PLAYBOOK AVAILABLE: Yes - Threat Level: {playbook.get('threat_level', 'Unknown')}"
            if playbook and playbook.get("playbook_required")
            else "PLAYBOOK: Not required (benign alert)"
        )

        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            "Analyze this security alert and provide your expert assessment:\n\n"
            f"ALERT DATA:\n{json.dumps(alert_data, indent=2)}\n\n"
            f"CLASSIFICATION:\n"
            f"- Label: {prediction.get('label', 'Unknown')}\n"
            f"- Malicious Probability: {prediction.get('probability', 0.0):.2%}\n\n"
            f"EXPLANATION:\n{explanation.get('explanation_text', 'No explanation available')}\n\n"
            f"Top Contributing Features:\n{features_text}\n\n"
            f"{playbook_info}\n\n"
            "Please provide:\n"
            "1. Your assessment of the alert\n"
            "2. Key findings and concerns\n"
            "3. Immediate recommendations\n"
            "4. Risk level assessment\n"
            "5. Any additional insights\n\n"
            "Format your response as a structured analysis."
        )

        try:
            response = self._model.generate_content(
                prompt,
                generation_config={
                    "temperature": config.GEMINI_TEMPERATURE_ANALYSIS,
                    "max_output_tokens": config.GEMINI_MAX_TOKENS_ANALYSIS,
                },
            )
            return {
                "timestamp": datetime.now().isoformat(),
                "assistant_name": self.name,
                "analysis": response.text,
                "alert_assessment": prediction.get("label", "Unknown"),
                "risk_score": prediction.get("probability", 0.0),
                "model_used": self.model_name,
            }
        except Exception as exc:
            logging.warning("Gemini analysis failed: %s. Using fallback.", exc)
            result = SimpleAIAssistant().analyze_alert(alert_data, prediction, explanation, playbook)
            result["error"] = f"Gemini analysis failed: {exc}. Using fallback analysis."
            return result

    def answer_question(self, question: str, context: Dict[str, Any]) -> str:
        context_summary = (
            f"ALERT CONTEXT:\n"
            f"- Classification: {context.get('prediction', {}).get('label', 'Unknown')}\n"
            f"- Probability: {context.get('prediction', {}).get('probability', 0.0):.2%}\n"
            f"- Explanation: {context.get('explanation', {}).get('explanation_text', 'N/A')}\n"
            f"- Has Playbook: {'Yes' if context.get('playbook') else 'No'}"
        )

        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"User Question: {question}\n\n"
            f"{context_summary}\n\n"
            "Please provide a clear, helpful answer to the user's question based on the alert context."
        )

        try:
            response = self._model.generate_content(
                prompt,
                generation_config={
                    "temperature": config.GEMINI_TEMPERATURE_CHAT,
                    "max_output_tokens": config.GEMINI_MAX_TOKENS_CHAT,
                },
            )
            return response.text
        except Exception as exc:
            fallback_answer = SimpleAIAssistant().answer_question(question, context)
            return f"{fallback_answer}\n\n(Note: Gemini unavailable, using fallback. Error: {exc})"
