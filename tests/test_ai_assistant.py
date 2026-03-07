"""
Unit tests for src/assistant/* modules.
"""

import pytest

from src.assistant.base import BaseAssistant
from src.assistant.simple import SimpleAIAssistant
from src.assistant.factory import create_assistant


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def malicious_context():
    return {
        "prediction": {"label": "Malicious", "probability": 0.92},
        "explanation": {"explanation_text": "High packet rate detected."},
        "playbook": {"playbook_required": True, "threat_level": "CRITICAL"},
    }


@pytest.fixture
def benign_context():
    return {
        "prediction": {"label": "Normal", "probability": 0.15},
        "explanation": {"explanation_text": "Traffic appears normal."},
        "playbook": None,
    }


@pytest.fixture
def alert_data():
    return {"Source_IP": "10.0.0.1", "Destination_IP": "192.168.1.5", "Protocol": "TCP"}


# ---------------------------------------------------------------------------
# BaseAssistant (abstract contract)
# ---------------------------------------------------------------------------


class TestBaseAssistant:
    def test_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            BaseAssistant()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_both_methods(self):
        class Incomplete(BaseAssistant):
            def analyze_alert(self, *args, **kwargs):
                return {}
            # answer_question deliberately missing

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_fully_implemented_subclass_instantiates(self):
        class Complete(BaseAssistant):
            def analyze_alert(self, alert_data, prediction, explanation, playbook=None):
                return {}

            def answer_question(self, question, context):
                return ""

        obj = Complete()
        assert isinstance(obj, BaseAssistant)


# ---------------------------------------------------------------------------
# SimpleAIAssistant
# ---------------------------------------------------------------------------


class TestSimpleAIAssistant:
    @pytest.fixture(autouse=True)
    def assistant(self):
        self.assistant = SimpleAIAssistant()

    def test_is_base_assistant_subtype(self):
        assert isinstance(self.assistant, BaseAssistant)

    def test_analyze_malicious_returns_correct_assessment(self, alert_data, malicious_context):
        result = self.assistant.analyze_alert(
            alert_data,
            malicious_context["prediction"],
            malicious_context["explanation"],
        )
        assert result["alert_assessment"] == "Malicious"

    def test_analyze_benign_returns_correct_assessment(self, alert_data, benign_context):
        result = self.assistant.analyze_alert(
            alert_data,
            benign_context["prediction"],
            benign_context["explanation"],
        )
        assert result["alert_assessment"] == "Normal"

    def test_analyze_result_has_required_keys(self, alert_data, malicious_context):
        result = self.assistant.analyze_alert(
            alert_data,
            malicious_context["prediction"],
            malicious_context["explanation"],
        )
        assert {"timestamp", "assistant_name", "alert_assessment", "risk_score"} <= result.keys()

    def test_high_confidence_classification(self, alert_data, malicious_context):
        result = self.assistant.analyze_alert(
            alert_data,
            malicious_context["prediction"],
            malicious_context["explanation"],
        )
        assert result["confidence_level"] == "HIGH"

    def test_answer_question_malicious_keyword(self, malicious_context):
        answer = self.assistant.answer_question("Is this malicious?", malicious_context)
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_answer_question_probability_keyword(self, malicious_context):
        answer = self.assistant.answer_question("What is the probability?", malicious_context)
        assert "92" in answer or "%" in answer

    def test_answer_question_playbook_present(self, malicious_context):
        answer = self.assistant.answer_question("What does the playbook say?", malicious_context)
        assert "playbook" in answer.lower()

    def test_answer_question_no_playbook(self, benign_context):
        answer = self.assistant.answer_question("Is there a playbook?", benign_context)
        assert "no playbook" in answer.lower() or "not" in answer.lower()

    def test_answer_question_why_keyword(self, malicious_context):
        answer = self.assistant.answer_question("Why was this flagged?", malicious_context)
        assert "High packet rate" in answer or "analysis" in answer.lower()

    def test_answer_question_fallback(self, malicious_context):
        answer = self.assistant.answer_question("Random unrelated question", malicious_context)
        assert isinstance(answer, str)
        assert len(answer) > 0


# ---------------------------------------------------------------------------
# create_assistant factory
# ---------------------------------------------------------------------------


class TestCreateAssistant:
    def test_no_llm_returns_simple_assistant(self):
        assistant = create_assistant(use_llm=False)
        assert isinstance(assistant, SimpleAIAssistant)

    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="anthropic"):
            create_assistant(use_llm=True, llm_provider="anthropic")

    def test_unknown_provider_error_message_lists_supported(self):
        with pytest.raises(ValueError, match="gemini"):
            create_assistant(use_llm=True, llm_provider="openai")

    def test_gemini_without_key_falls_back_to_simple(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assistant = create_assistant(use_llm=True, llm_provider="gemini", api_key=None)
        assert isinstance(assistant, SimpleAIAssistant)

    def test_provider_check_is_case_insensitive(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        # GEMINI (uppercase) should not raise, should fall back gracefully
        assistant = create_assistant(use_llm=True, llm_provider="GEMINI", api_key=None)
        assert isinstance(assistant, SimpleAIAssistant)
