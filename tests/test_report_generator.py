"""
Unit tests for src/analysis/report_generator.py.
"""

import json
import os

import numpy as np
import pytest

from src.analysis.report_generator import (
    convert_to_serializable,
    generate_report,
    generate_report_summary,
    save_report,
    format_report_for_display,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def alert_data():
    return {"Source_IP": "10.0.0.1", "Destination_IP": "192.168.1.5", "Protocol": "TCP"}


@pytest.fixture
def prediction():
    return {"label": "Malicious", "probability": 0.92, "prediction": 1}


@pytest.fixture
def explanation():
    return {
        "explanation_text": "The alert was classified as Malicious mainly because...",
        "top_features": [
            {"feature": "Packet_Length", "value": 1500.0, "contribution": 0.45},
        ],
    }


@pytest.fixture
def playbook():
    return {
        "playbook_required": True,
        "threat_level": "CRITICAL",
        "priority": "IMMEDIATE",
        "steps": [],
        "recommendations": [],
    }


@pytest.fixture
def report(alert_data, prediction, explanation, playbook):
    return generate_report(alert_data, prediction, explanation, playbook)


# ---------------------------------------------------------------------------
# convert_to_serializable
# ---------------------------------------------------------------------------


class TestConvertToSerializable:
    def test_numpy_int_converted(self):
        assert convert_to_serializable(np.int64(42)) == 42
        assert isinstance(convert_to_serializable(np.int64(42)), int)

    def test_numpy_float_converted(self):
        result = convert_to_serializable(np.float32(3.14))
        assert isinstance(result, float)

    def test_numpy_bool_converted(self):
        assert convert_to_serializable(np.bool_(True)) is True

    def test_numpy_array_converted_to_list(self):
        arr = np.array([1, 2, 3])
        result = convert_to_serializable(arr)
        assert result == [1, 2, 3]

    def test_nested_dict_converted(self):
        data = {"a": np.int64(1), "b": {"c": np.float64(2.5)}}
        result = convert_to_serializable(data)
        assert result == {"a": 1, "b": {"c": 2.5}}

    def test_list_converted(self):
        data = [np.int32(1), np.float32(2.0)]
        result = convert_to_serializable(data)
        assert result == [1, 2.0]

    def test_native_types_unchanged(self):
        assert convert_to_serializable("hello") == "hello"
        assert convert_to_serializable(3.14) == 3.14
        assert convert_to_serializable(None) is None


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_report_id_starts_with_prefix(self, report):
        assert report["report_id"].startswith("RPT-")

    def test_alert_classification_present(self, report, prediction):
        cls = report["alert_classification"]
        assert cls["label"] == prediction["label"]
        assert cls["probability"] == prediction["probability"]

    def test_explanation_embedded(self, report, explanation):
        assert report["explanation"]["summary"] == explanation["explanation_text"]

    def test_playbook_embedded_when_provided(self, report):
        assert report["playbook"] is not None
        assert report["playbook"]["playbook_required"] is True

    def test_playbook_none_when_not_provided(self, alert_data, prediction, explanation):
        report = generate_report(alert_data, prediction, explanation, playbook=None)
        assert report["playbook"] is None

    def test_metadata_keys_present(self, report):
        assert {"report_version", "system", "model_version"} <= report["metadata"].keys()

    def test_confidence_high_for_extreme_probability(self, alert_data, explanation):
        pred = {"label": "Malicious", "probability": 0.95, "prediction": 1}
        report = generate_report(alert_data, pred, explanation)
        assert report["alert_classification"]["confidence"] == "HIGH"

    def test_confidence_medium_for_borderline_probability(self, alert_data, explanation):
        pred = {"label": "Malicious", "probability": 0.55, "prediction": 1}
        report = generate_report(alert_data, pred, explanation)
        assert report["alert_classification"]["confidence"] == "MEDIUM"


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------


class TestSaveReport:
    def test_file_created(self, report, tmp_path):
        path = save_report(report, output_dir=str(tmp_path))
        assert os.path.exists(path)

    def test_saved_json_is_valid(self, report, tmp_path):
        path = save_report(report, output_dir=str(tmp_path))
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["report_id"] == report["report_id"]

    def test_filename_matches_report_id(self, report, tmp_path):
        path = save_report(report, output_dir=str(tmp_path))
        assert report["report_id"] in os.path.basename(path)

    def test_directory_created_if_absent(self, report, tmp_path):
        nested = tmp_path / "deep" / "reports"
        save_report(report, output_dir=str(nested))
        assert nested.exists()


# ---------------------------------------------------------------------------
# generate_report_summary
# ---------------------------------------------------------------------------


class TestGenerateReportSummary:
    def test_required_keys_present(self, report):
        summary = generate_report_summary(report)
        assert {"report_id", "classification", "probability", "confidence", "has_playbook", "threat_level"} <= summary.keys()

    def test_has_playbook_true_when_playbook_present(self, report):
        summary = generate_report_summary(report)
        assert summary["has_playbook"] is True

    def test_has_playbook_false_when_absent(self, alert_data, prediction, explanation):
        report = generate_report(alert_data, prediction, explanation, playbook=None)
        summary = generate_report_summary(report)
        assert summary["has_playbook"] is False


# ---------------------------------------------------------------------------
# format_report_for_display
# ---------------------------------------------------------------------------


class TestFormatReportForDisplay:
    def test_returns_string(self, report):
        assert isinstance(format_report_for_display(report), str)

    def test_contains_report_id(self, report):
        text = format_report_for_display(report)
        assert report["report_id"] in text

    def test_contains_classification_label(self, report):
        text = format_report_for_display(report)
        assert "Malicious" in text
