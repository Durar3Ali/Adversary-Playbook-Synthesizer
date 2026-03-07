"""
Playbook generator for malignant security alerts.

Response steps are defined as module-level data structures (_BASE_STEPS,
_INDICATOR_RECOMMENDATIONS) so that new steps or attack-type-specific
recommendations can be added without modifying generate_playbook() itself
(Open/Closed Principle).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from src import config

# ---------------------------------------------------------------------------
# Data definitions — edit these to change or extend playbook content.
# ---------------------------------------------------------------------------

_BASE_STEPS: List[Dict[str, Any]] = [
    {
        "step_number": 1,
        "title": "Immediate Containment",
        "actions": [
            "Block source IP: {source_ip}",
            "Block destination IP: {dest_ip} if compromised",
            "Isolate affected systems from network",
            "Preserve logs and evidence",
        ],
        "estimated_time": "5-10 minutes",
        "priority": "CRITICAL",
    },
    {
        "step_number": 2,
        "title": "Threat Investigation",
        "actions": [
            "Review security logs for related activities",
            "Check for lateral movement indicators",
            "Analyse network traffic patterns",
            "Review authentication logs",
            "Check for data exfiltration attempts",
        ],
        "estimated_time": "15-30 minutes",
        "priority": "HIGH",
    },
    {
        "step_number": 3,
        "title": "Impact Assessment",
        "actions": [
            "Identify affected systems and data",
            "Assess potential data breach scope",
            "Determine if sensitive information was accessed",
            "Evaluate business impact",
            "Document findings",
        ],
        "estimated_time": "30-60 minutes",
        "priority": "HIGH",
    },
    {
        "step_number": 4,
        "title": "Remediation Actions",
        "actions": [
            "Reset compromised credentials",
            "Apply security patches if applicable",
            "Update firewall rules",
            "Review and update security policies",
            "Implement additional monitoring",
        ],
        "estimated_time": "1-2 hours",
        "priority": "MEDIUM",
    },
    {
        "step_number": 5,
        "title": "Recovery & Hardening",
        "actions": [
            "Restore systems from clean backups if needed",
            "Implement additional security controls",
            "Conduct security awareness training",
            "Review incident response procedures",
            "Update threat intelligence",
        ],
        "estimated_time": "2-4 hours",
        "priority": "MEDIUM",
    },
    {
        "step_number": 6,
        "title": "Documentation & Reporting",
        "actions": [
            "Document incident timeline",
            "Create incident report",
            "Notify relevant stakeholders",
            "File regulatory reports if required",
            "Conduct post-incident review",
        ],
        "estimated_time": "1-2 hours",
        "priority": "LOW",
    },
]

_BASE_RECOMMENDATIONS: List[str] = [
    "Enable real-time monitoring for similar attack patterns",
    "Review and update intrusion detection rules",
    "Consider implementing network segmentation",
    "Enhance endpoint detection and response capabilities",
]

# Maps attack indicator strings to extra recommendations that fire when
# that indicator is detected.  Add new entries here to extend behaviour
# without touching generate_playbook().
_INDICATOR_RECOMMENDATIONS: Dict[str, str] = {
    "High network activity detected": "Implement rate limiting and DDoS protection",
    "Suspicious port activity": "Review and restrict unnecessary open ports",
    "Unusual connection duration": "Investigate long-lived connections and session policies",
    "Anomalous data transfer": "Review DLP policies and egress filtering rules",
}

# Maps feature-name substrings to indicator labels.
_FEATURE_INDICATOR_MAP: Dict[str, str] = {
    "packet": "High network activity detected",
    "flow": "High network activity detected",
    "port": "Suspicious port activity",
    "duration": "Unusual connection duration",
    "bytes": "Anomalous data transfer",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_playbook(
    alert_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
    explanation: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a response playbook for a malignant alert.

    Parameters
    ----------
    alert_data:
        Original alert feature dict.
    prediction_result:
        Prediction result containing ``label`` and ``probability``.
    explanation:
        XAI explanation with ``top_features`` list.

    Returns
    -------
    Playbook dict.  When the alert is not malicious returns a minimal dict
    with ``playbook_required=False``.
    """
    is_malicious = prediction_result.get("label", "").lower() == "malicious"
    if not is_malicious:
        return {"playbook_required": False, "message": "No playbook needed for benign alerts."}

    probability: float = prediction_result.get("probability", 0.0)

    if probability >= config.THREAT_CRITICAL_MIN_PROB:
        threat_level, priority = "CRITICAL", "IMMEDIATE"
    elif probability >= config.THREAT_HIGH_MIN_PROB:
        threat_level, priority = "HIGH", "URGENT"
    else:
        threat_level, priority = "MEDIUM", "HIGH"

    attack_indicators = _detect_indicators(explanation.get("top_features", []))
    steps = _build_steps(
        source_ip=alert_data.get("Source_IP", "Unknown"),
        dest_ip=alert_data.get("Destination_IP", "Unknown"),
    )
    recommendations = _build_recommendations(attack_indicators)

    return {
        "playbook_required": True,
        "threat_level": threat_level,
        "priority": priority,
        "generated_at": datetime.now().isoformat(),
        "alert_summary": {
            "source_ip": alert_data.get("Source_IP", "Unknown"),
            "destination_ip": alert_data.get("Destination_IP", "Unknown"),
            "protocol": alert_data.get("Protocol", "Unknown"),
            "malicious_probability": f"{probability:.2%}",
        },
        "attack_indicators": attack_indicators,
        "steps": steps,
        "recommendations": recommendations,
        "total_estimated_time": "4-8 hours",
        "contact_info": {
            "incident_response_team": config.INCIDENT_RESPONSE_EMAIL,
            "emergency_hotline": config.EMERGENCY_HOTLINE,
        },
    }


def format_playbook_for_display(playbook: Dict[str, Any]) -> str:
    """Format a playbook dict as a human-readable string."""
    if not playbook.get("playbook_required", False):
        return playbook.get("message", "No playbook needed.")

    lines: List[str] = []
    lines += ["=" * 60, "INCIDENT RESPONSE PLAYBOOK", "=" * 60]
    lines.append(f"\nThreat Level: {playbook['threat_level']}")
    lines.append(f"Priority: {playbook['priority']}")
    lines.append(f"Generated: {playbook['generated_at']}")
    lines += ["\n" + "-" * 60, "ALERT SUMMARY", "-" * 60]

    summary = playbook["alert_summary"]
    lines.append(f"Source IP: {summary['source_ip']}")
    lines.append(f"Destination IP: {summary['destination_ip']}")
    lines.append(f"Protocol: {summary['protocol']}")
    lines.append(f"Malicious Probability: {summary['malicious_probability']}")

    if playbook.get("attack_indicators"):
        lines += ["\n" + "-" * 60, "ATTACK INDICATORS", "-" * 60]
        lines.extend(f"* {ind}" for ind in playbook["attack_indicators"])

    lines += ["\n" + "=" * 60, "RESPONSE STEPS", "=" * 60]
    for step in playbook["steps"]:
        lines.append(f"\n[STEP {step['step_number']}] {step['title']}")
        lines.append(f"Priority: {step['priority']} | Est. Time: {step['estimated_time']}")
        lines.append("-" * 60)
        lines.extend(f"  {i}. {action}" for i, action in enumerate(step["actions"], 1))

    lines += ["\n" + "=" * 60, "RECOMMENDATIONS", "=" * 60]
    lines.extend(f"{i}. {rec}" for i, rec in enumerate(playbook["recommendations"], 1))
    lines.append(f"\n\nTotal Estimated Response Time: {playbook['total_estimated_time']}")

    lines += ["\n" + "=" * 60, "CONTACT INFORMATION", "=" * 60]
    contact = playbook["contact_info"]
    lines.append(f"Incident Response Team: {contact['incident_response_team']}")
    lines.append(f"Emergency Hotline: {contact['emergency_hotline']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _detect_indicators(top_features: List[Dict[str, Any]]) -> List[str]:
    """Return a deduplicated list of attack indicator labels for the given features."""
    found: List[str] = []
    for feat in top_features[:3]:
        feat_name = feat.get("feature", "").lower()
        contribution = feat.get("contribution", 0)
        if contribution <= config.ATTACK_INDICATOR_CONTRIBUTION_THRESHOLD:
            continue
        for keyword, indicator in _FEATURE_INDICATOR_MAP.items():
            if keyword in feat_name and indicator not in found:
                found.append(indicator)
    return found


def _build_steps(source_ip: str, dest_ip: str) -> List[Dict[str, Any]]:
    """Materialise _BASE_STEPS, substituting IP placeholders."""
    steps = []
    for step_def in _BASE_STEPS:
        actions = [
            a.format(source_ip=source_ip, dest_ip=dest_ip)
            for a in step_def["actions"]
        ]
        steps.append({**step_def, "actions": actions})
    return steps


def _build_recommendations(attack_indicators: List[str]) -> List[str]:
    """Compose base recommendations plus any indicator-specific additions."""
    recs = list(_BASE_RECOMMENDATIONS)
    for indicator in attack_indicators:
        extra = _INDICATOR_RECOMMENDATIONS.get(indicator)
        if extra and extra not in recs:
            recs.append(extra)
    return recs
