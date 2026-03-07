"""
Central configuration module.
All hardcoded values that may vary per environment or deployment live here.
Environment variables (from .env or the shell) take precedence over the defaults below.
"""

import os

# ---------------------------------------------------------------------------
# File / directory paths
# ---------------------------------------------------------------------------

MODEL_PATH: str = os.getenv("MODEL_PATH", "data/cyber_alert_model.joblib")
DATASET_PATH: str = os.getenv("DATASET_PATH", "data/cyberfeddefender_dataset.csv")
REPORTS_DIR: str = os.getenv("REPORTS_DIR", "reports")
SCENARIOS_FILE: str = os.getenv("SCENARIOS_FILE", "scenarios.json")

# ---------------------------------------------------------------------------
# Application branding
# ---------------------------------------------------------------------------

APP_NAME: str = os.getenv("APP_NAME", "Adversary-Playbook Synthesizer")

# ---------------------------------------------------------------------------
# Google Gemini LLM settings
# ---------------------------------------------------------------------------

GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_TEMPERATURE_ANALYSIS: float = float(os.getenv("GEMINI_TEMPERATURE_ANALYSIS", "0.3"))
GEMINI_TEMPERATURE_CHAT: float = float(os.getenv("GEMINI_TEMPERATURE_CHAT", "0.5"))
GEMINI_MAX_TOKENS_ANALYSIS: int = int(os.getenv("GEMINI_MAX_TOKENS_ANALYSIS", "1000"))
GEMINI_MAX_TOKENS_CHAT: int = int(os.getenv("GEMINI_MAX_TOKENS_CHAT", "500"))

# ---------------------------------------------------------------------------
# Contact / operational info (set via env for each deployment)
# ---------------------------------------------------------------------------

INCIDENT_RESPONSE_EMAIL: str = os.getenv("INCIDENT_RESPONSE_EMAIL", "security@company.com")
EMERGENCY_HOTLINE: str = os.getenv("EMERGENCY_HOTLINE", "+1-XXX-XXX-XXXX")

# ---------------------------------------------------------------------------
# Report metadata
# ---------------------------------------------------------------------------

REPORT_VERSION: str = "1.0"
MODEL_VERSION: str = "cyber_alert_model_v2"
REPORT_ID_PREFIX: str = "RPT"

# ---------------------------------------------------------------------------
# Classification / confidence thresholds
# ---------------------------------------------------------------------------

# abs(probability - 0.5) > CONFIDENCE_HIGH_DELTA  →  confidence = "HIGH"
CONFIDENCE_HIGH_DELTA: float = 0.3

# Probability threshold above which "immediate containment" is recommended
IMMEDIATE_CONTAINMENT_THRESHOLD: float = 0.8

# ---------------------------------------------------------------------------
# Playbook threat-level thresholds
# ---------------------------------------------------------------------------

THREAT_CRITICAL_MIN_PROB: float = 0.8   # probability >= this  →  CRITICAL / IMMEDIATE
THREAT_HIGH_MIN_PROB: float = 0.6       # probability >= this  →  HIGH / URGENT
# below THREAT_HIGH_MIN_PROB             →  MEDIUM / HIGH

# ---------------------------------------------------------------------------
# XAI settings
# ---------------------------------------------------------------------------

XAI_TOP_K_FEATURES: int = 5

# Minimum absolute SHAP / feature-importance contribution to mention in text
XAI_MIN_CONTRIBUTION_FOR_TEXT: float = 0.01

# Feature-importance contribution threshold for attack-indicator detection in playbook
ATTACK_INDICATOR_CONTRIBUTION_THRESHOLD: float = 0.1

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

# Fraction of required features that may be missing before raising ValueError.
# At or below this fraction, missing values are filled and a warning is issued.
# Above this fraction, a ValueError is raised to prevent unreliable predictions.
MAX_MISSING_FEATURE_FRACTION: float = 0.5

# ---------------------------------------------------------------------------
# ML training constants
# ---------------------------------------------------------------------------

TRAIN_TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
N_ESTIMATORS: int = 200
MAX_DEPTH: int = 25
MIN_SAMPLES_SPLIT: int = 10
MIN_SAMPLES_LEAF: int = 4

# Columns dropped before feature matrix construction (not features)
FEATURES_TO_DROP: list[str] = [
    "Timestamp", "Attack_Type", "Label",
    "Source_IP", "Destination_IP",  # replaced by derived IP features
]

# Columns encoded with LabelEncoder (ordinal categoricals only — NOT raw IPs)
CATEGORICAL_COLUMNS: list[str] = ["Protocol", "Flags"]
