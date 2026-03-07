"""
Standalone training script for the Adversary-Playbook Synthesizer ML model.

Run directly to (re)train the Random Forest classifier and save it as a joblib file:

    python -m src.ml.model_trainer

The trained model is saved to the path specified by config.MODEL_PATH
(default: data/cyber_alert_model.joblib).
"""

import ipaddress
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder

from src import config


# ---------------------------------------------------------------------------
# IP feature helpers
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


def _derive_ip_features(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Add derived IP features to X from the raw df columns.

    Replaces raw IP label-encoding with three meaningful boolean features:
    - src_is_private: whether the source IP is RFC-1918 / private
    - dst_is_private: whether the destination IP is RFC-1918 / private
    - same_subnet_24: whether source and destination share the same /24 subnet
    - same_source_dest_ip: exact IP equality (already present, kept for compatibility)
    """
    if "Source_IP" in df.columns:
        X["src_is_private"] = df["Source_IP"].apply(_is_private_ip).astype(int)
    if "Destination_IP" in df.columns:
        X["dst_is_private"] = df["Destination_IP"].apply(_is_private_ip).astype(int)
    if "Source_IP" in df.columns and "Destination_IP" in df.columns:
        X["same_subnet_24"] = (
            df["Source_IP"].apply(_subnet_24) == df["Destination_IP"].apply(_subnet_24)
        ).astype(int)
        X["same_source_dest_ip"] = (df["Source_IP"] == df["Destination_IP"]).astype(int)
    return X


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train(dataset_path: str = None, model_out_path: str = None) -> None:
    """Train the Random Forest classifier and persist the model dict.

    Parameters
    ----------
    dataset_path:
        CSV file to train on.  Defaults to config.DATASET_PATH.
    model_out_path:
        Destination joblib file.  Defaults to config.MODEL_PATH.
    """
    dataset_path = dataset_path or config.DATASET_PATH
    model_out_path = model_out_path or config.MODEL_PATH

    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    print(f"\nDataset columns: {df.columns.tolist()}")

    print("\nAttack Type distribution:")
    print(df["Attack_Type"].value_counts())

    df["is_malicious"] = (df["Attack_Type"] != "Normal").astype(int)
    print("\nTarget distribution:")
    print(df["is_malicious"].value_counts())

    print("\nPreprocessing features...")
    X = df.drop(columns=config.FEATURES_TO_DROP + ["is_malicious"], errors="ignore")

    # Label-encode true ordinal categoricals (Protocol, Flags) — NOT raw IPs.
    categorical_cols: list = config.CATEGORICAL_COLUMNS
    label_encoders: dict = {}
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    # Derive meaningful IP features instead of encoding raw addresses.
    X = _derive_ip_features(df, X)

    y = df["is_malicious"]

    print("\nMissing values per column:")
    missing_counts = X.isnull().sum()
    print(missing_counts[missing_counts > 0] if missing_counts.any() else "None")
    X = X.fillna(X.median(numeric_only=True))

    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TRAIN_TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size:     {X_test.shape}")

    # Compute per-feature statistics on the training split only to avoid leakage.
    training_medians: dict = X_train.median(numeric_only=True).to_dict()
    feature_stds: np.ndarray = X_train.std(numeric_only=True).values

    print("\nTraining Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        min_samples_split=config.MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.MIN_SAMPLES_LEAF,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
    )
    rf_model.fit(X_train, y_train)
    print(f"Out-of-bag score: {rf_model.oob_score_:.4f}")

    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    y_test_proba = rf_model.predict_proba(X_test)[:, 1]

    print("\n--- ROC Analysis ---")
    auc_score = roc_auc_score(y_test, y_test_proba)
    print(f"ROC AUC Score: {auc_score:.4f}")

    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    optimal_idx = int(np.argmax(tpr - fpr))
    optimal_threshold = float(thresholds[optimal_idx])
    print(f"Optimal threshold (Youden J): {optimal_threshold:.4f}")

    y_test_pred_optimal = (y_test_proba >= optimal_threshold).astype(int)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("\n--- Training Set Performance ---")
    print(f"Accuracy: {train_acc:.4f}")
    print(classification_report(y_train, y_train_pred, target_names=["Normal", "Malicious"]))

    print("\n--- Test Set Performance (default threshold 0.5) ---")
    print(f"Accuracy: {test_acc:.4f}")
    print(classification_report(y_test, y_test_pred, target_names=["Normal", "Malicious"]))

    cm = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:")
    print(cm)

    print("\n--- Test Set Performance (optimal threshold) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred_optimal):.4f}")
    print(classification_report(y_test, y_test_pred_optimal, target_names=["Normal", "Malicious"]))

    cm_opt = confusion_matrix(y_test, y_test_pred_optimal)
    print("Confusion Matrix (optimal threshold):")
    print(cm_opt)

    gap = train_acc - test_acc
    if gap > 0.05:
        warnings.warn(
            f"Possible overfitting detected: train_acc={train_acc:.4f}, "
            f"test_acc={test_acc:.4f}, gap={gap:.4f}. "
            "Consider reducing MAX_DEPTH or increasing MIN_SAMPLES_LEAF.",
            stacklevel=2,
        )

    # ---------------------------------------------------------------------------
    # 5-fold stratified cross-validation
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 60)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    cv_results = cross_validate(
        rf_model,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "f1", "roc_auc"],
        return_train_score=True,
        n_jobs=-1,
    )
    for metric in ("accuracy", "f1", "roc_auc"):
        test_scores = cv_results[f"test_{metric}"]
        train_scores = cv_results[f"train_{metric}"]
        print(
            f"{metric:>12s}  "
            f"cv_test={test_scores.mean():.4f} ± {test_scores.std():.4f}  "
            f"cv_train={train_scores.mean():.4f} ± {train_scores.std():.4f}"
        )

    print("\n" + "=" * 60)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("=" * 60)
    fi = (
        pd.DataFrame({"feature": X.columns, "importance": rf_model.feature_importances_})
        .sort_values("importance", ascending=False)
    )
    print(fi.head(15).to_string(index=False))

    print(f"\nSaving model to '{model_out_path}'...")
    joblib.dump(
        {
            "model": rf_model,
            "label_encoders": label_encoders,
            "feature_columns": X.columns.tolist(),
            "categorical_columns": categorical_cols,
            "optimal_threshold": optimal_threshold,
            "feature_stds": feature_stds,
            "training_medians": training_medians,
        },
        model_out_path,
    )
    print("Model saved successfully!")
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train()
