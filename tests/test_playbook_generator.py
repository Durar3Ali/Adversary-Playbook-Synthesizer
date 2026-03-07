"""
Unit tests for src/analysis/playbook_generator.py.
"""

import pytest

from src.analysis.playbook_generator import (
    generate_playbook,
    format_playbook_for_display,
    _detect_indicators,
    _build_steps,
    _build_recommendations,
    _BASE_STEPS,
    _BASE_RECOMMENDATIONS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def benign_prediction():
    return {"label": "Normal", "probability": 0.2}


@pytest.fixture
def critical_prediction():
    return {"label": "Malicious", "probability": 0.95}


@pytest.fixture
def high_prediction():
    return {"label": "Malicious", "probability": 0.7}


@pytest.fixture
def medium_prediction():
    return {"label": "Malicious", "probability": 0.5}


@pytest.fixture
def alert_data():
    return {
        "Source_IP": "10.0.0.1",
        "Destination_IP": "192.168.1.5",
        "Protocol": "TCP",
        "Packet_Length": 1500,
    }


@pytest.fixture
def high_packet_explanation():
    return {
        "top_features": [
            {"feature": "Flow_Packets_s", "value": 10000.0, "contribution": 0.8},
        ]
    }


@pytest.fixture
def empty_explanation():
    return {"top_features": []}


# ---------------------------------------------------------------------------
# generate_playbook
# ---------------------------------------------------------------------------


class TestGeneratePlaybook:
    def test_benign_returns_no_playbook(self, alert_data, benign_prediction, empty_explanation):
        result = generate_playbook(alert_data, benign_prediction, empty_explanation)
        assert result["playbook_required"] is False

    def test_critical_threat_level(self, alert_data, critical_prediction, empty_explanation):
        result = generate_playbook(alert_data, critical_prediction, empty_explanation)
        assert result["threat_level"] == "CRITICAL"
        assert result["priority"] == "IMMEDIATE"

    def test_high_threat_level(self, alert_data, high_prediction, empty_explanation):
        result = generate_playbook(alert_data, high_prediction, empty_explanation)
        assert result["threat_level"] == "HIGH"
        assert result["priority"] == "URGENT"

    def test_medium_threat_level(self, alert_data, medium_prediction, empty_explanation):
        result = generate_playbook(alert_data, medium_prediction, empty_explanation)
        assert result["threat_level"] == "MEDIUM"

    def test_steps_count_matches_base_definition(self, alert_data, critical_prediction, empty_explanation):
        result = generate_playbook(alert_data, critical_prediction, empty_explanation)
        assert len(result["steps"]) == len(_BASE_STEPS)

    def test_source_ip_substituted_in_steps(self, alert_data, critical_prediction, empty_explanation):
        result = generate_playbook(alert_data, critical_prediction, empty_explanation)
        step1_actions = result["steps"][0]["actions"]
        assert any("10.0.0.1" in action for action in step1_actions)

    def test_alert_summary_fields(self, alert_data, critical_prediction, empty_explanation):
        result = generate_playbook(alert_data, critical_prediction, empty_explanation)
        summary = result["alert_summary"]
        assert summary["source_ip"] == "10.0.0.1"
        assert summary["protocol"] == "TCP"

    def test_indicator_triggers_extra_recommendation(
        self, alert_data, critical_prediction, high_packet_explanation
    ):
        result = generate_playbook(alert_data, critical_prediction, high_packet_explanation)
        assert "High network activity detected" in result["attack_indicators"]
        assert any("rate limiting" in r.lower() for r in result["recommendations"])

    def test_required_keys_present(self, alert_data, critical_prediction, empty_explanation):
        result = generate_playbook(alert_data, critical_prediction, empty_explanation)
        required = {
            "playbook_required", "threat_level", "priority", "generated_at",
            "alert_summary", "attack_indicators", "steps", "recommendations",
            "total_estimated_time", "contact_info",
        }
        assert required <= result.keys()


# ---------------------------------------------------------------------------
# _detect_indicators
# ---------------------------------------------------------------------------


class TestDetectIndicators:
    def test_packet_feature_detected(self):
        features = [{"feature": "Flow_Packets_s", "value": 10000, "contribution": 0.5}]
        indicators = _detect_indicators(features)
        assert "High network activity detected" in indicators

    def test_low_contribution_not_detected(self):
        features = [{"feature": "Flow_Packets_s", "value": 10000, "contribution": 0.01}]
        indicators = _detect_indicators(features)
        assert indicators == []

    def test_no_duplicate_indicators(self):
        features = [
            {"feature": "Packet_Length", "value": 500, "contribution": 0.5},
            {"feature": "Flow_Packets_s", "value": 9000, "contribution": 0.6},
        ]
        indicators = _detect_indicators(features)
        assert len(indicators) == len(set(indicators))


# ---------------------------------------------------------------------------
# _build_steps
# ---------------------------------------------------------------------------


class TestBuildSteps:
    def test_all_base_steps_included(self):
        steps = _build_steps("1.2.3.4", "5.6.7.8")
        assert len(steps) == len(_BASE_STEPS)

    def test_ip_substitution_in_step_one(self):
        steps = _build_steps("1.2.3.4", "5.6.7.8")
        assert any("1.2.3.4" in a for a in steps[0]["actions"])

    def test_step_numbers_sequential(self):
        steps = _build_steps("a", "b")
        for i, step in enumerate(steps, 1):
            assert step["step_number"] == i


# ---------------------------------------------------------------------------
# _build_recommendations
# ---------------------------------------------------------------------------


class TestBuildRecommendations:
    def test_base_recommendations_always_present(self):
        recs = _build_recommendations([])
        for base in _BASE_RECOMMENDATIONS:
            assert base in recs

    def test_indicator_recommendation_appended(self):
        recs = _build_recommendations(["High network activity detected"])
        assert any("rate limiting" in r.lower() for r in recs)

    def test_no_duplicate_recommendations(self):
        recs = _build_recommendations(["High network activity detected"])
        assert len(recs) == len(set(recs))


# ---------------------------------------------------------------------------
# format_playbook_for_display
# ---------------------------------------------------------------------------


class TestFormatPlaybookForDisplay:
    def test_no_playbook_returns_message(self):
        result = format_playbook_for_display({"playbook_required": False, "message": "OK"})
        assert result == "OK"

    def test_formatted_output_contains_threat_level(self, alert_data, critical_prediction, empty_explanation):
        playbook = generate_playbook(alert_data, critical_prediction, empty_explanation)
        text = format_playbook_for_display(playbook)
        assert "CRITICAL" in text

    def test_formatted_output_contains_all_step_titles(self, alert_data, critical_prediction, empty_explanation):
        playbook = generate_playbook(alert_data, critical_prediction, empty_explanation)
        text = format_playbook_for_display(playbook)
        for step in _BASE_STEPS:
            assert step["title"] in text
