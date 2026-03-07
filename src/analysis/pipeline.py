"""
Analysis pipeline for Adversary-Playbook Synthesizer.

Contains the pure business-logic orchestration for a single alert analysis:
prediction -> explanation -> playbook -> report -> AI analysis.

This module has no Streamlit dependency; all UI / session-state concerns live
in app.py.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from src import config
from src.ml.model_predictor import predict_alert, explain_alert
from src.analysis.playbook_generator import generate_playbook
from src.analysis.report_generator import generate_report, save_report


def run_analysis(
    alert_data: Dict[str, Any],
    alert_name: str,
    model_data: Dict[str, Any],
    ai_assistant: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute the full analysis pipeline for a single alert.

    Parameters
    ----------
    alert_data:
        Raw alert feature dict.
    alert_name:
        Human-readable label for this alert.
    model_data:
        Pre-loaded model dict.  Callers are responsible for caching.
    ai_assistant:
        Optional BaseAssistant instance.

    Returns
    -------
    alert_entry dict ready for storage in session state.

    Raises
    ------
    Propagates any exception to the caller; no UI calls are made here.
    """
    prediction = predict_alert(alert_data, model_data=model_data)
    explanation = explain_alert(
        alert_data, model_data=model_data, top_k=config.XAI_TOP_K_FEATURES
    )

    playbook = (
        generate_playbook(alert_data, prediction, explanation)
        if prediction["label"] == "Malicious"
        else None
    )

    report = generate_report(alert_data, prediction, explanation, playbook)
    report_path = save_report(report)

    ai_analysis: Optional[Dict] = None
    if ai_assistant is not None:
        try:
            ai_analysis = ai_assistant.analyze_alert(alert_data, prediction, explanation, playbook)
        except Exception as exc:
            ai_analysis = {"error": f"AI analysis failed: {exc}"}

    alert_id = f"{alert_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return {
        "id": alert_id,
        "name": alert_name,
        "payload": alert_data,
        "prediction": {**prediction, "label": _map_label(prediction["label"])},
        "prediction_raw": prediction,
        "explanation": explanation,
        "playbook": playbook,
        "report": report,
        "report_path": report_path,
        "ai_analysis": ai_analysis,
        "timestamp": datetime.now().isoformat(),
    }


def _map_label(label: str) -> str:
    return {"Normal": "Benign", "Malicious": "Malignant"}.get(label, label)
