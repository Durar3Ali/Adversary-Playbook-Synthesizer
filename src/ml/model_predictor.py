"""
Inference module for Adversary-Playbook Synthesizer.

Provides predict_alert and explain_alert functions that operate on a
pre-loaded model_data dict.  Callers are responsible for loading (and
optionally caching) the model; see load_model() in this module.
"""

import ipaddress
import warnings
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src import config

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model serialization
# ---------------------------------------------------------------------------


def load_model(model_path: Optional[str] = None) -> Dict[str, Any]:
    """Load and return the joblib model_data dict.

    Parameters
    ----------
    model_path:
        Path to the joblib file.  Defaults to config.MODEL_PATH.
    """
    path = model_path or config.MODEL_PATH
    return joblib.load(path)


# ---------------------------------------------------------------------------
# IP feature helpers (must mirror model_trainer._is_private_ip / _subnet_24)
# ---------------------------------------------------------------------------


def _is_private_ip(ip_str: str) -> int:
    """Return 1 if the IP string is a private/loopback/link-local address, else 0."""
    try:
        return int(ipaddress.ip_address(str(ip_str)).is_private)
    except ValueError:
        return 0


def _subnet_24(ip_str: str) -> str:
    """Return the /24 subnet prefix (first three octets) of an IP string."""
    try:
        parts = str(ip_str).split(".")
        if len(parts) >= 3:
            return ".".join(parts[:3])
    except Exception:
        pass
    return ip_str


def _derive_ip_features(alert_df: pd.DataFrame) -> pd.DataFrame:
    """Add derived IP features to alert_df in-place and return it."""
    if "Source_IP" in alert_df.columns:
        alert_df["src_is_private"] = alert_df["Source_IP"].apply(_is_private_ip).astype(int)
    if "Destination_IP" in alert_df.columns:
        alert_df["dst_is_private"] = alert_df["Destination_IP"].apply(_is_private_ip).astype(int)
    if "Source_IP" in alert_df.columns and "Destination_IP" in alert_df.columns:
        alert_df["same_subnet_24"] = (
            alert_df["Source_IP"].apply(_subnet_24) == alert_df["Destination_IP"].apply(_subnet_24)
        ).astype(int)
        alert_df["same_source_dest_ip"] = (
            alert_df["Source_IP"] == alert_df["Destination_IP"]
        ).astype(int)
    return alert_df


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------


def _preprocess_alert(
    alert_data: Any,
    model_data: Dict[str, Any],
) -> tuple:
    """Preprocess alert_data into a model-ready DataFrame.

    Returns (X_pred, alert_df).

    Raises
    ------
    ValueError
        When the fraction of missing required features exceeds
        config.MAX_MISSING_FEATURE_FRACTION.
    """
    label_encoders: dict = model_data["label_encoders"]
    feature_columns: List[str] = model_data["feature_columns"]
    training_medians: dict = model_data.get("training_medians", {})

    alert_df: pd.DataFrame = (
        pd.DataFrame([alert_data]) if isinstance(alert_data, dict) else alert_data.copy()
    )

    # Derive IP-based features before column alignment.
    alert_df = _derive_ip_features(alert_df)

    missing = [f for f in feature_columns if f not in alert_df.columns]
    if missing:
        missing_fraction = len(missing) / len(feature_columns)
        if missing_fraction > config.MAX_MISSING_FEATURE_FRACTION:
            raise ValueError(
                f"{len(missing)}/{len(feature_columns)} required features are missing "
                f"({missing_fraction:.0%}). Prediction would be unreliable. "
                f"Missing: {missing}"
            )
        warnings.warn(
            f"Missing features {missing}; filling with training-set medians (or 0).",
            stacklevel=3,
        )

    X_pred = pd.DataFrame(index=alert_df.index)
    for feature in feature_columns:
        if feature in alert_df.columns:
            X_pred[feature] = alert_df[feature]
        else:
            X_pred[feature] = training_medians.get(feature, 0)

    for col, le in label_encoders.items():
        if col in X_pred.columns:
            X_pred[col] = X_pred[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    # Fill any remaining NaNs with training-set medians (falls back to 0 if not stored).
    for col in X_pred.columns:
        if X_pred[col].isnull().any():
            X_pred[col] = X_pred[col].fillna(training_medians.get(col, 0))

    return X_pred, alert_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_alert(
    alert_data: Any,
    model_data: Optional[Dict[str, Any]] = None,
    model_path: Optional[str] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Predict whether an alert is Normal or Malicious.

    Parameters
    ----------
    alert_data:
        Dict or DataFrame with alert features.
    model_data:
        Pre-loaded model dict (preferred — avoids repeated disk I/O).
        If omitted, the model is loaded from model_path / config.MODEL_PATH.
    model_path:
        Used only when model_data is None.
    threshold:
        Classification threshold.  Defaults to the optimal threshold stored
        in the model dict, then 0.5.

    Returns
    -------
    dict with keys ``prediction`` (int), ``probability`` (float), ``label`` (str).
    """
    if model_data is None:
        model_data = load_model(model_path)

    model = model_data["model"]
    if threshold is None:
        threshold = model_data.get("optimal_threshold", 0.5)

    X_pred, _ = _preprocess_alert(alert_data, model_data)
    proba = model.predict_proba(X_pred)[:, 1]
    prediction = (proba >= threshold).astype(int)

    return {
        "prediction": int(prediction[0]),
        "probability": float(proba[0]),
        "label": "Malicious" if prediction[0] == 1 else "Normal",
    }


def explain_alert(
    alert_data: Any,
    model_data: Optional[Dict[str, Any]] = None,
    model_path: Optional[str] = None,
    threshold: Optional[float] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Explain an alert classification using SHAP (or feature importances as fallback).

    Parameters
    ----------
    alert_data:
        Dict or DataFrame with alert features.
    model_data:
        Pre-loaded model dict (preferred).
    model_path:
        Used only when model_data is None.
    threshold:
        Classification threshold.
    top_k:
        Number of top features to return.  Defaults to config.XAI_TOP_K_FEATURES.

    Returns
    -------
    dict with keys ``prediction``, ``probability``, ``label``,
    ``top_features`` (list of dicts), ``explanation_text`` (str).
    """
    if model_data is None:
        model_data = load_model(model_path)
    if top_k is None:
        top_k = config.XAI_TOP_K_FEATURES

    model = model_data["model"]
    if threshold is None:
        threshold = model_data.get("optimal_threshold", 0.5)

    X_pred, alert_df = _preprocess_alert(alert_data, model_data)
    proba = model.predict_proba(X_pred)[:, 1]
    prediction = (proba >= threshold).astype(int)
    pred_label = "Malicious" if prediction[0] == 1 else "Normal"

    feature_names: List[str] = X_pred.columns.tolist()
    top_features: List[dict] = []
    shap_succeeded = False

    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_pred)
            shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
            instance_shap = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals

            contributions = []
            for i, feat_name in enumerate(feature_names):
                original = (
                    alert_df.iloc[0][feat_name]
                    if feat_name in alert_df.columns
                    else (bool(float(X_pred.iloc[0, i])) if feat_name == "same_source_dest_ip" else float(X_pred.iloc[0, i]))
                )
                contributions.append({
                    "feature": feat_name,
                    "value": original,
                    "contribution": float(instance_shap[i]),
                })

            contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            top_features = contributions[:top_k]
            shap_succeeded = True
        except Exception as exc:
            warnings.warn(f"SHAP explanation failed: {exc}. Falling back to feature importances.")

    if not shap_succeeded or not top_features:
        importances = model.feature_importances_
        instance_values = X_pred.iloc[0].values
        feature_stds: Optional[np.ndarray] = model_data.get("feature_stds")

        contributions = []
        for i, feat_name in enumerate(feature_names):
            imp = float(importances[i])
            val = float(instance_values[i])
            if feature_stds is not None and i < len(feature_stds) and feature_stds[i] > 0:
                contribution = imp * val / feature_stds[i]
            else:
                contribution = imp

            original = (
                alert_df.iloc[0][feat_name]
                if feat_name in alert_df.columns
                else (bool(val) if feat_name == "same_source_dest_ip" else val)
            )
            contributions.append({
                "feature": feat_name,
                "value": original,
                "contribution": contribution,
            })

        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        top_features = contributions[:top_k]

    explanation_text = _build_explanation_text(pred_label, top_features)

    return {
        "prediction": int(prediction[0]),
        "probability": float(proba[0]),
        "label": pred_label,
        "top_features": top_features,
        "explanation_text": explanation_text,
    }


def _build_explanation_text(label: str, top_features: List[dict]) -> str:
    """Compose a human-readable explanation string from top feature contributions."""
    parts = [f"The alert was classified as {label}"]

    if top_features:
        parts.append("mainly because:")
        mentions = []
        for feat in top_features[:3]:
            name = feat["feature"].replace("_", " ")
            contribution = feat["contribution"]
            value = feat["value"]
            if abs(contribution) > config.XAI_MIN_CONTRIBUTION_FOR_TEXT:
                direction = "high" if contribution > 0 else "low"
                if isinstance(value, (int, float)):
                    mentions.append(f"{name} is {direction} ({value:.2f})")
                elif isinstance(value, bool):
                    mentions.append(f"{name} is {value}")
                else:
                    mentions.append(f"{name} is {direction}")
        if mentions:
            parts.append(", ".join(mentions) + ".")
        else:
            parts.append("the feature values align with the predicted class.")
    else:
        parts.append("based on the overall pattern of features.")

    return " ".join(parts)
