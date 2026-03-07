"""
Rule-based fallback AI assistant (no LLM dependency).
"""

from datetime import datetime
from typing import Any, Dict, Optional

from src import config
from src.assistant.base import BaseAssistant


class SimpleAIAssistant(BaseAssistant):
    """Deterministic rule-based assistant used when no LLM API key is configured."""

    name: str = "Security Analyst Assistant"

    def analyze_alert(
        self,
        alert_data: Dict[str, Any],
        prediction: Dict[str, Any],
        explanation: Dict[str, Any],
        playbook: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        label = prediction.get("label", "Unknown")
        probability: float = prediction.get("probability", 0.0)

        analysis: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "assistant_name": self.name,
            "alert_assessment": label,
            "confidence_level": (
                "HIGH" if abs(probability - 0.5) > config.CONFIDENCE_HIGH_DELTA else "MEDIUM"
            ),
            "key_findings": [],
            "recommendations": [],
            "risk_score": probability,
        }

        top_features = explanation.get("top_features", [])

        if label == "Malicious":
            analysis["key_findings"].append(
                f"Alert classified as malicious with {probability:.1%} confidence"
            )
            for feat in top_features[:3]:
                feat_name = feat.get("feature", "").lower()
                contribution = feat.get("contribution", 0)
                if abs(contribution) > 0.1:
                    if "packet" in feat_name:
                        analysis["key_findings"].append("Unusual packet patterns detected")
                        analysis["recommendations"].append("Review network traffic for similar patterns")
                    elif "port" in feat_name:
                        analysis["key_findings"].append("Suspicious port activity identified")
                        analysis["recommendations"].append("Verify if port usage is authorized")
                    elif "bytes" in feat_name:
                        analysis["key_findings"].append("Anomalous data transfer detected")
                        analysis["recommendations"].append("Investigate data exfiltration possibilities")

            if probability > config.IMMEDIATE_CONTAINMENT_THRESHOLD:
                analysis["recommendations"].append("Immediate containment recommended")
                analysis["recommendations"].append("Notify incident response team immediately")
        else:
            analysis["key_findings"].append(
                f"Alert appears benign ({probability:.1%} malicious probability)"
            )
            analysis["recommendations"].append("Continue monitoring for similar patterns")
            analysis["recommendations"].append("No immediate action required")

        return analysis

    def answer_question(self, question: str, context: Dict[str, Any]) -> str:
        q = question.lower()
        prediction = context.get("prediction", {})

        if "malicious" in q or "threat" in q:
            label = prediction.get("label", "Unknown")
            if label == "Malicious":
                return (
                    "Yes, this alert has been classified as malicious. "
                    "The system detected suspicious patterns that indicate a potential security threat. "
                    "Immediate action is recommended."
                )
            return "No, this alert appears to be benign. However, continue monitoring for similar patterns."

        if "probability" in q or "confidence" in q:
            prob: float = prediction.get("probability", 0.0)
            level = "high" if prob > 0.7 else ("medium" if prob > 0.4 else "low")
            return f"The malicious probability is {prob:.1%}. This indicates a {level} level of concern."

        if "playbook" in q or "response" in q:
            if context.get("playbook"):
                return (
                    "A response playbook has been generated for this alert. "
                    "It includes step-by-step instructions for containment, investigation, and remediation."
                )
            return "No playbook is available for this alert. Playbooks are only generated for malicious alerts."

        if "explain" in q or "why" in q:
            text = context.get("explanation", {}).get("explanation_text", "No explanation available")
            return f"Based on the analysis: {text}"

        return (
            "I can help you understand the alert classification, probability, explanations, "
            "and response recommendations. Please ask a specific question about the alert."
        )
