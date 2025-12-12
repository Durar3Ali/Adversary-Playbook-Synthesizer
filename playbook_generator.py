"""
Playbook Generator for Malignant Security Alerts
Generates step-by-step response playbooks for malicious alerts
"""

import json
from datetime import datetime
from typing import Dict, List, Any


def generate_playbook(alert_data: Dict, prediction_result: Dict, explanation: Dict) -> Dict[str, Any]:
    """
    Generate a response playbook for a malignant alert.
    
    Parameters:
    -----------
    alert_data : dict
        Original alert data
    prediction_result : dict
        Prediction result with label, probability, etc.
    explanation : dict
        XAI explanation with top features
    
    Returns:
    --------
    dict : Playbook with steps, priority, and recommendations
    """
    
    # Extract key information
    is_malicious = prediction_result.get('label', '').lower() == 'malicious'
    probability = prediction_result.get('probability', 0.0)
    
    if not is_malicious:
        return {
            'playbook_required': False,
            'message': 'No playbook needed for benign alerts.'
        }
    
    # Determine threat level based on probability
    if probability >= 0.8:
        threat_level = "CRITICAL"
        priority = "IMMEDIATE"
    elif probability >= 0.6:
        threat_level = "HIGH"
        priority = "URGENT"
    else:
        threat_level = "MEDIUM"
        priority = "HIGH"
    
    # Analyze top contributing features to determine attack type
    top_features = explanation.get('top_features', [])
    attack_indicators = []
    
    for feat in top_features[:3]:
        feat_name = feat.get('feature', '').lower()
        contribution = feat.get('contribution', 0)
        
        if 'packet' in feat_name or 'flow' in feat_name:
            if contribution > 0.1:
                attack_indicators.append("High network activity detected")
        if 'port' in feat_name:
            if contribution > 0.1:
                attack_indicators.append("Suspicious port activity")
        if 'duration' in feat_name:
            if contribution > 0.1:
                attack_indicators.append("Unusual connection duration")
        if 'bytes' in feat_name:
            if contribution > 0.1:
                attack_indicators.append("Anomalous data transfer")
    
    # Generate playbook steps
    playbook_steps = []
    
    # Step 1: Immediate containment
    playbook_steps.append({
        'step_number': 1,
        'title': 'Immediate Containment',
        'actions': [
            f"Block source IP: {alert_data.get('Source_IP', 'Unknown')}",
            f"Block destination IP: {alert_data.get('Destination_IP', 'Unknown')} if compromised",
            "Isolate affected systems from network",
            "Preserve logs and evidence"
        ],
        'estimated_time': '5-10 minutes',
        'priority': 'CRITICAL'
    })
    
    # Step 2: Investigation
    playbook_steps.append({
        'step_number': 2,
        'title': 'Threat Investigation',
        'actions': [
            "Review security logs for related activities",
            "Check for lateral movement indicators",
            "Analyze network traffic patterns",
            "Review authentication logs",
            "Check for data exfiltration attempts"
        ],
        'estimated_time': '15-30 minutes',
        'priority': 'HIGH'
    })
    
    # Step 3: Assessment
    playbook_steps.append({
        'step_number': 3,
        'title': 'Impact Assessment',
        'actions': [
            "Identify affected systems and data",
            "Assess potential data breach scope",
            "Determine if sensitive information was accessed",
            "Evaluate business impact",
            "Document findings"
        ],
        'estimated_time': '30-60 minutes',
        'priority': 'HIGH'
    })
    
    # Step 4: Remediation
    playbook_steps.append({
        'step_number': 4,
        'title': 'Remediation Actions',
        'actions': [
            "Reset compromised credentials",
            "Apply security patches if applicable",
            "Update firewall rules",
            "Review and update security policies",
            "Implement additional monitoring"
        ],
        'estimated_time': '1-2 hours',
        'priority': 'MEDIUM'
    })
    
    # Step 5: Recovery
    playbook_steps.append({
        'step_number': 5,
        'title': 'Recovery & Hardening',
        'actions': [
            "Restore systems from clean backups if needed",
            "Implement additional security controls",
            "Conduct security awareness training",
            "Review incident response procedures",
            "Update threat intelligence"
        ],
        'estimated_time': '2-4 hours',
        'priority': 'MEDIUM'
    })
    
    # Step 6: Reporting
    playbook_steps.append({
        'step_number': 6,
        'title': 'Documentation & Reporting',
        'actions': [
            "Document incident timeline",
            "Create incident report",
            "Notify relevant stakeholders",
            "File regulatory reports if required",
            "Conduct post-incident review"
        ],
        'estimated_time': '1-2 hours',
        'priority': 'LOW'
    })
    
    # Generate recommendations
    recommendations = [
        "Enable real-time monitoring for similar attack patterns",
        "Review and update intrusion detection rules",
        "Consider implementing network segmentation",
        "Enhance endpoint detection and response capabilities"
    ]
    
    # Add specific recommendations based on indicators
    if "High network activity" in attack_indicators:
        recommendations.append("Implement rate limiting and DDoS protection")
    if "Suspicious port activity" in attack_indicators:
        recommendations.append("Review and restrict unnecessary open ports")
    
    playbook = {
        'playbook_required': True,
        'threat_level': threat_level,
        'priority': priority,
        'generated_at': datetime.now().isoformat(),
        'alert_summary': {
            'source_ip': alert_data.get('Source_IP', 'Unknown'),
            'destination_ip': alert_data.get('Destination_IP', 'Unknown'),
            'protocol': alert_data.get('Protocol', 'Unknown'),
            'malicious_probability': f"{probability:.2%}"
        },
        'attack_indicators': attack_indicators,
        'steps': playbook_steps,
        'recommendations': recommendations,
        'total_estimated_time': '4-8 hours',
        'contact_info': {
            'incident_response_team': 'security@company.com',
            'emergency_hotline': '+1-XXX-XXX-XXXX'
        }
    }
    
    return playbook


def format_playbook_for_display(playbook: Dict) -> str:
    """
    Format playbook as a readable string for display.
    
    Parameters:
    -----------
    playbook : dict
        Playbook dictionary
    
    Returns:
    --------
    str : Formatted playbook text
    """
    if not playbook.get('playbook_required', False):
        return playbook.get('message', 'No playbook needed.')
    
    lines = []
    lines.append("=" * 60)
    lines.append("INCIDENT RESPONSE PLAYBOOK")
    lines.append("=" * 60)
    lines.append(f"\nThreat Level: {playbook['threat_level']}")
    lines.append(f"Priority: {playbook['priority']}")
    lines.append(f"Generated: {playbook['generated_at']}")
    lines.append("\n" + "-" * 60)
    lines.append("ALERT SUMMARY")
    lines.append("-" * 60)
    summary = playbook['alert_summary']
    lines.append(f"Source IP: {summary['source_ip']}")
    lines.append(f"Destination IP: {summary['destination_ip']}")
    lines.append(f"Protocol: {summary['protocol']}")
    lines.append(f"Malicious Probability: {summary['malicious_probability']}")
    
    if playbook.get('attack_indicators'):
        lines.append("\n" + "-" * 60)
        lines.append("ATTACK INDICATORS")
        lines.append("-" * 60)
        for indicator in playbook['attack_indicators']:
            lines.append(f"• {indicator}")
    
    lines.append("\n" + "=" * 60)
    lines.append("RESPONSE STEPS")
    lines.append("=" * 60)
    
    for step in playbook['steps']:
        lines.append(f"\n[STEP {step['step_number']}] {step['title']}")
        lines.append(f"Priority: {step['priority']} | Est. Time: {step['estimated_time']}")
        lines.append("-" * 60)
        for i, action in enumerate(step['actions'], 1):
            lines.append(f"  {i}. {action}")
    
    lines.append("\n" + "=" * 60)
    lines.append("RECOMMENDATIONS")
    lines.append("=" * 60)
    for i, rec in enumerate(playbook['recommendations'], 1):
        lines.append(f"{i}. {rec}")
    
    lines.append(f"\n\nTotal Estimated Response Time: {playbook['total_estimated_time']}")
    
    lines.append("\n" + "=" * 60)
    lines.append("CONTACT INFORMATION")
    lines.append("=" * 60)
    contact = playbook['contact_info']
    lines.append(f"Incident Response Team: {contact['incident_response_team']}")
    lines.append(f"Emergency Hotline: {contact['emergency_hotline']}")
    
    return "\n".join(lines)

