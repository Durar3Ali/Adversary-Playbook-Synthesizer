"""
Automated Report Generator
Generates comprehensive reports for security alerts including predictions, explanations, and playbooks
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional
import os
import numpy as np

from src import config



def convert_to_serializable(obj):
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def generate_report(
    alert_data: Dict,
    prediction_result: Dict,
    explanation: Dict,
    playbook: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive automated report for a security alert.
    
    Parameters:
    -----------
    alert_data : dict
        Original alert data
    prediction_result : dict
        Prediction result (label, probability)
    explanation : dict
        XAI explanation with top features
    playbook : dict, optional
        Generated playbook (if malignant)
    
    Returns:
    --------
    dict : Complete report dictionary
    """
    
    report_id = f"{config.REPORT_ID_PREFIX}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    prob = prediction_result.get('probability', 0.5)
    confidence = 'HIGH' if abs(prob - 0.5) > config.CONFIDENCE_HIGH_DELTA else 'MEDIUM'

    report = {
        'report_id': report_id,
        'generated_at': datetime.now().isoformat(),
        'alert_classification': {
            'label': prediction_result.get('label', 'Unknown'),
            'probability': prob,
            'prediction_code': prediction_result.get('prediction', 0),
            'confidence': confidence,
        },
        'alert_data': alert_data,
        'explanation': {
            'summary': explanation.get('explanation_text', 'No explanation available'),
            'top_features': explanation.get('top_features', [])
        },
        'playbook': playbook if playbook else None,
        'metadata': {
            'report_version': config.REPORT_VERSION,
            'system': config.APP_NAME,
            'model_version': config.MODEL_VERSION,
        }
    }
    
    return report


def save_report(report: Dict, output_dir: str = None) -> str:
    """
    Save report to JSON file.
    
    Parameters:
    -----------
    report : dict
        Report dictionary
    output_dir : str
        Directory to save reports
    
    Returns:
    --------
    str : Path to saved report file
    """
    if output_dir is None:
        output_dir = config.REPORTS_DIR
    # Create reports directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    report_id = report['report_id']
    filename = f"{report_id}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Convert numpy types to native Python types before saving
    report_serializable = convert_to_serializable(report)
    
    # Save report
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report_serializable, f, indent=2, ensure_ascii=False)
    
    return filepath


def format_report_for_display(report: Dict) -> str:
    """
    Format report as a readable string for display.
    
    Parameters:
    -----------
    report : dict
        Report dictionary
    
    Returns:
    --------
    str : Formatted report text
    """
    lines = []
    lines.append("=" * 80)
    lines.append("SECURITY ALERT REPORT")
    lines.append("=" * 80)
    lines.append(f"\nReport ID: {report['report_id']}")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append(f"System: {report['metadata']['system']}")
    
    # Classification Section
    lines.append("\n" + "=" * 80)
    lines.append("ALERT CLASSIFICATION")
    lines.append("=" * 80)
    classification = report['alert_classification']
    lines.append(f"Label: {classification['label']}")
    lines.append(f"Probability: {classification['probability']:.2%}")
    lines.append(f"Confidence: {classification['confidence']}")
    
    # Alert Data Section
    lines.append("\n" + "=" * 80)
    lines.append("ALERT DATA")
    lines.append("=" * 80)
    alert_data = report['alert_data']
    for key, value in alert_data.items():
        lines.append(f"{key}: {value}")
    
    # Explanation Section
    lines.append("\n" + "=" * 80)
    lines.append("EXPLANATION (XAI)")
    lines.append("=" * 80)
    explanation = report['explanation']
    lines.append(f"\nSummary: {explanation['summary']}")
    
    if explanation.get('top_features'):
        lines.append("\nTop Contributing Features:")
        for i, feat in enumerate(explanation['top_features'], 1):
            feat_name = feat.get('feature', 'Unknown')
            value = feat.get('value', 'N/A')
            contribution = feat.get('contribution', 0)
            lines.append(f"  {i}. {feat_name}: {value} (contribution: {contribution:+.4f})")
    
    # Playbook Section (if malignant)
    if report.get('playbook') and report['playbook'].get('playbook_required'):
        lines.append("\n" + "=" * 80)
        lines.append("INCIDENT RESPONSE PLAYBOOK")
        lines.append("=" * 80)
        playbook = report['playbook']
        lines.append(f"Threat Level: {playbook['threat_level']}")
        lines.append(f"Priority: {playbook['priority']}")
        
        if playbook.get('steps'):
            lines.append("\nResponse Steps:")
            for step in playbook['steps']:
                lines.append(f"\n  [STEP {step['step_number']}] {step['title']}")
                lines.append(f"    Priority: {step['priority']} | Time: {step['estimated_time']}")
                for action in step['actions']:
                    lines.append(f"    • {action}")
        
        if playbook.get('recommendations'):
            lines.append("\nRecommendations:")
            for rec in playbook['recommendations']:
                lines.append(f"  • {rec}")
    
    lines.append("\n" + "=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def generate_report_summary(report: Dict) -> Dict[str, Any]:
    """
    Generate a brief summary of the report.
    
    Parameters:
    -----------
    report : dict
        Report dictionary
    
    Returns:
    --------
    dict : Summary dictionary
    """
    classification = report['alert_classification']
    
    summary = {
        'report_id': report['report_id'],
        'timestamp': report['generated_at'],
        'classification': classification['label'],
        'probability': classification['probability'],
        'confidence': classification['confidence'],
        'has_playbook': report.get('playbook') is not None and report.get('playbook', {}).get('playbook_required', False),
        'threat_level': report.get('playbook', {}).get('threat_level', 'N/A') if report.get('playbook') else 'N/A'
    }
    
    return summary

