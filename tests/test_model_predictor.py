"""
Unit tests for src/ml/model_predictor.py.

All tests use a lightweight mock model_data dict so that no joblib file or
GPU/CPU-heavy training is required.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.ml.model_predictor import (
    _preprocess_alert,
    _build_explanation_text,
    _is_private_ip,
    _subnet_24,
    predict_alert,
    explain_alert,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_label_encoder(classes):
    """Return a minimal LabelEncoder-like object."""
    le = MagicMock()
    le.classes_ = classes
    le.transform = lambda x: [list(classes).index(v) for v in x]
    return le


@pytest.fixture
def model_data():
    """Minimal model_data dict with a mocked RandomForest."""
    rf = MagicMock()
    rf.predict_proba.return_value = np.array([[0.3, 0.7]])
    rf.feature_importances_ = np.array([0.5, 0.3, 0.2])

    return {
        "model": rf,
        "label_encoders": {
            "Protocol": _make_label_encoder(["TCP", "UDP", "ICMP"]),
        },
        "feature_columns": ["Packet_Length", "Duration", "Protocol"],
        "categorical_columns": ["Protocol"],
        "optimal_threshold": 0.5,
        "feature_stds": np.array([100.0, 1.0, 1.0]),
        "training_medians": {"Packet_Length": 500.0, "Duration": 1.0, "Protocol": 0.0},
    }


@pytest.fixture
def alert_data():
    return {"Packet_Length": 1500, "Duration": 2.0, "Protocol": "TCP"}


# ---------------------------------------------------------------------------
# IP helper functions
# ---------------------------------------------------------------------------


class TestIPHelpers:
    def test_private_ip_returns_1(self):
        assert _is_private_ip("192.168.1.1") == 1

    def test_public_ip_returns_0(self):
        assert _is_private_ip("8.8.8.8") == 0

    def test_loopback_is_private(self):
        assert _is_private_ip("127.0.0.1") == 1

    def test_invalid_ip_returns_0(self):
        assert _is_private_ip("not_an_ip") == 0

    def test_subnet_24_extracts_prefix(self):
        assert _subnet_24("192.168.1.100") == "192.168.1"

    def test_subnet_24_same_subnet(self):
        assert _subnet_24("10.0.0.1") == _subnet_24("10.0.0.254")

    def test_subnet_24_different_subnet(self):
        assert _subnet_24("10.0.0.1") != _subnet_24("10.0.1.1")


# ---------------------------------------------------------------------------
# _preprocess_alert
# ---------------------------------------------------------------------------


class TestPreprocessAlert:
    def test_returns_dataframe_with_correct_columns(self, alert_data, model_data):
        X_pred, alert_df = _preprocess_alert(alert_data, model_data)
        assert list(X_pred.columns) == model_data["feature_columns"]

    def test_missing_feature_filled_with_training_median(self, model_data):
        # Duration is in training_medians as 1.0; missing from alert should use that
        partial_alert = {"Packet_Length": 100, "Protocol": "TCP"}
        X_pred, _ = _preprocess_alert(partial_alert, model_data)
        assert X_pred["Duration"].iloc[0] == 1.0

    def test_missing_feature_falls_back_to_zero_when_no_median(self, model_data):
        # Remove Duration from training_medians to test the fallback
        model_data_no_median = {
            **model_data,
            "training_medians": {"Packet_Length": 500.0, "Protocol": 0.0},
        }
        partial_alert = {"Packet_Length": 100, "Protocol": "TCP"}
        X_pred, _ = _preprocess_alert(partial_alert, model_data_no_median)
        assert X_pred["Duration"].iloc[0] == 0

    def test_unseen_categorical_encoded_as_minus_one(self, model_data):
        alert = {"Packet_Length": 100, "Duration": 1.0, "Protocol": "UNKNOWN"}
        X_pred, _ = _preprocess_alert(alert, model_data)
        assert X_pred["Protocol"].iloc[0] == -1

    def test_known_categorical_encoded_correctly(self, model_data):
        alert = {"Packet_Length": 100, "Duration": 1.0, "Protocol": "UDP"}
        X_pred, _ = _preprocess_alert(alert, model_data)
        assert X_pred["Protocol"].iloc[0] == 1  # index of "UDP" in ["TCP","UDP","ICMP"]

    def test_dict_input_converted_to_dataframe(self, alert_data, model_data):
        X_pred, alert_df = _preprocess_alert(alert_data, model_data)
        assert len(alert_df) == 1

    def test_ip_columns_derive_features_not_label_encoded(self):
        """Source_IP and Destination_IP must produce derived features, not be label-encoded."""
        model_data_with_ip_features = {
            "model": MagicMock(),
            "label_encoders": {},
            "feature_columns": [
                "Packet_Length", "src_is_private", "dst_is_private",
                "same_subnet_24", "same_source_dest_ip",
            ],
            "categorical_columns": [],
            "optimal_threshold": 0.5,
            "feature_stds": np.array([100.0, 1.0, 1.0, 1.0, 1.0]),
            "training_medians": {},
        }
        alert = {
            "Packet_Length": 500,
            "Source_IP": "192.168.1.10",
            "Destination_IP": "192.168.1.20",
        }
        X_pred, _ = _preprocess_alert(alert, model_data_with_ip_features)

        assert X_pred["src_is_private"].iloc[0] == 1   # 192.168.x.x is private
        assert X_pred["dst_is_private"].iloc[0] == 1   # 192.168.x.x is private
        assert X_pred["same_subnet_24"].iloc[0] == 1   # same /24 subnet
        assert X_pred["same_source_dest_ip"].iloc[0] == 0  # different IPs

    def test_ip_same_address_sets_same_source_dest(self):
        model_data_with_ip_features = {
            "model": MagicMock(),
            "label_encoders": {},
            "feature_columns": ["same_source_dest_ip", "same_subnet_24"],
            "categorical_columns": [],
            "optimal_threshold": 0.5,
            "feature_stds": np.array([1.0, 1.0]),
            "training_medians": {},
        }
        alert = {
            "Source_IP": "10.0.0.1",
            "Destination_IP": "10.0.0.1",
        }
        X_pred, _ = _preprocess_alert(alert, model_data_with_ip_features)
        assert X_pred["same_source_dest_ip"].iloc[0] == 1
        assert X_pred["same_subnet_24"].iloc[0] == 1

    def test_raises_when_too_many_features_missing(self, model_data):
        """ValueError must be raised when > MAX_MISSING_FEATURE_FRACTION features are missing."""
        # model_data has 3 feature_columns; providing none of them = 100% missing
        empty_alert = {}
        with pytest.raises(ValueError, match="required features are missing"):
            _preprocess_alert(empty_alert, model_data)

    def test_does_not_raise_when_few_features_missing(self, model_data):
        """Only a warning should be issued when missing fraction is at or below the threshold."""
        import warnings as _warnings
        # Provide 2 of 3 features (33% missing — below 50% threshold)
        partial_alert = {"Packet_Length": 100, "Duration": 1.0}
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            X_pred, _ = _preprocess_alert(partial_alert, model_data)
        assert any("Missing features" in str(w.message) for w in caught)
        assert X_pred is not None


# ---------------------------------------------------------------------------
# predict_alert
# ---------------------------------------------------------------------------


class TestPredictAlert:
    def test_malicious_prediction_above_threshold(self, alert_data, model_data):
        result = predict_alert(alert_data, model_data=model_data)
        assert result["prediction"] == 1
        assert result["label"] == "Malicious"
        assert 0.0 <= result["probability"] <= 1.0

    def test_benign_prediction_below_threshold(self, alert_data, model_data):
        model_data["model"].predict_proba.return_value = np.array([[0.8, 0.2]])
        result = predict_alert(alert_data, model_data=model_data)
        assert result["prediction"] == 0
        assert result["label"] == "Normal"

    def test_custom_threshold_respected(self, alert_data, model_data):
        # probability = 0.7; with threshold=0.8 it should classify as Normal
        result = predict_alert(alert_data, model_data=model_data, threshold=0.8)
        assert result["prediction"] == 0

    def test_result_keys_present(self, alert_data, model_data):
        result = predict_alert(alert_data, model_data=model_data)
        assert {"prediction", "probability", "label"} <= result.keys()

    def test_loads_from_path_when_model_data_none(self, alert_data, model_data):
        with patch("src.ml.model_predictor.load_model", return_value=model_data) as mock_load:
            result = predict_alert(alert_data, model_path="/fake/path/model.joblib")
        mock_load.assert_called_once_with("/fake/path/model.joblib")
        assert result["label"] in ("Normal", "Malicious")


# ---------------------------------------------------------------------------
# explain_alert
# ---------------------------------------------------------------------------


class TestExplainAlert:
    def test_returns_expected_keys(self, alert_data, model_data):
        result = explain_alert(alert_data, model_data=model_data)
        assert {"prediction", "probability", "label", "top_features", "explanation_text"} <= result.keys()

    def test_top_features_length_capped_by_top_k(self, alert_data, model_data):
        result = explain_alert(alert_data, model_data=model_data, top_k=2)
        assert len(result["top_features"]) <= 2

    def test_fallback_used_when_shap_unavailable(self, alert_data, model_data):
        with patch("src.ml.model_predictor.SHAP_AVAILABLE", False):
            result = explain_alert(alert_data, model_data=model_data)
        assert result["top_features"]

    def test_explanation_text_mentions_label(self, alert_data, model_data):
        result = explain_alert(alert_data, model_data=model_data)
        assert "Malicious" in result["explanation_text"] or "Normal" in result["explanation_text"]


# ---------------------------------------------------------------------------
# _build_explanation_text
# ---------------------------------------------------------------------------


class TestBuildExplanationText:
    def test_malicious_label_in_output(self):
        text = _build_explanation_text("Malicious", [])
        assert "Malicious" in text

    def test_top_features_mentioned(self):
        features = [{"feature": "Packet_Length", "value": 9000.0, "contribution": 0.5}]
        text = _build_explanation_text("Malicious", features)
        assert "packet length" in text.lower()

    def test_empty_features_graceful(self):
        text = _build_explanation_text("Normal", [])
        assert isinstance(text, str)
        assert len(text) > 0
