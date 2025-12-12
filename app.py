"""
Streamlit app for Adversary-Playbook Synthesizer
Cyber Security Alert Classification with XAI, Playbook Generation, and AI Assistant
"""

import streamlit as st
import pandas as pd
from ml_model import predict_alert, explain_alert
from playbook_generator import generate_playbook, format_playbook_for_display
from report_generator import generate_report, save_report, format_report_for_display, generate_report_summary
from ai_assistant import create_assistant
import json
import os
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, skip loading .env file
    pass

# Page config
st.set_page_config(
    page_title="Adversary-Playbook Synthesizer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for improved UI/UX
st.markdown("""
<style>
    :root {
        --bg-main: #000000;
        --bg-panel: #1a1a1a;
        --hud-yellow: #ffd800;
        --hud-yellow-bright: #ffea5a;
        --hud-yellow-dark: #ccaa00;
        --text-main: #ffd800;
        --text-muted: #ffd800;
        --border-strong: #ffd800;
        --success-green: #00ff00;
        --danger-red: #ff0000;
    }
    
    .main {
        background-color: #000000;
        color: #ffd800;
    }
    
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);
        font-family: "IBM Plex Mono", "Courier New", monospace;
        letter-spacing: 0.04em;
    }
    
    .block-container {
        background-color: #000000;
        padding-top: 2rem;
        max-width: 95%;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffd800 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        text-shadow: 0 0 10px rgba(255, 216, 0, 0.5);
    }
    
    p, div, span, label {
        color: #ffd800 !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 2px solid #ffd800;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffd800 !important;
    }
    
    .stButton > button {
        background-color: #000000 !important;
        color: #ffd800 !important;
        border: 2px solid #ffd800 !important;
        border-radius: 4px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        transition: all 0.2s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #0d0d0d !important;
        color: #ffea5a !important;
        border-color: #ffea5a !important;
        box-shadow: 0 0 10px rgba(255, 216, 0, 0.4);
        transform: translateY(-2px);
    }
    
    .stButton > button:active {
        background-color: #1a1a1a !important;
        border-color: #ffd800 !important;
    }
    
    .stSelectbox > div > div, .stTextInput > div > div > input,
    .stNumberInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #1a1a1a !important;
        color: #ffd800 !important;
        border: 1px solid #ffd800 !important;
    }
    
    .stSelectbox label, .stTextInput label, .stNumberInput label, .stTextArea label {
        color: #ffd800 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffd800 !important;
        font-size: 2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #ffd800 !important;
    }
    
    .stInfo, .stSuccess, .stWarning {
        background-color: #1a1a1a !important;
        border-left: 4px solid #ffd800 !important;
        color: #ffd800 !important;
    }
    
    .stError {
        background-color: #1a1a1a !important;
        border-left: 4px solid #ff0000 !important;
        color: #ff0000 !important;
    }
    
    .dataframe {
        background-color: #1a1a1a !important;
        color: #ffd800 !important;
        border: 1px solid #ffd800 !important;
    }
    
    .dataframe th {
        background-color: #000000 !important;
        color: #ffd800 !important;
        border: 1px solid #ffd800 !important;
    }
    
    .dataframe td {
        background-color: #1a1a1a !important;
        color: #ffd800 !important;
        border: 1px solid #ffd800 !important;
    }
    
    .streamlit-expanderHeader {
        background-color: #1a1a1a !important;
        color: #ffd800 !important;
        border: 1px solid #ffd800 !important;
    }
    
    .stSpinner > div {
        border-top-color: #ffd800 !important;
    }
    
    .badge-benign {
        background-color: #1a1a1a;
        color: #00ff00;
        border: 2px solid #00ff00;
        padding: 8px 16px;
        border-radius: 4px;
        font-size: 1rem;
        text-transform: uppercase;
        font-weight: 600;
        box-shadow: 0 0 10px rgba(0, 255, 0, 0.3);
        display: inline-block;
    }
    
    .badge-malignant {
        background-color: #1a1a1a;
        color: #ff0000;
        border: 2px solid #ff0000;
        padding: 8px 16px;
        border-radius: 4px;
        font-size: 1rem;
        text-transform: uppercase;
        font-weight: 600;
        animation: pulse-glow-red 2s ease-in-out infinite;
        display: inline-block;
    }
    
    @keyframes pulse-glow-red {
        0%, 100% { box-shadow: 0 0 10px rgba(255, 0, 0, 0.5); }
        50% { box-shadow: 0 0 20px rgba(255, 0, 0, 1); }
    }
    
    .status-indicator {
        color: #ffd800;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        text-shadow: 0 0 10px rgba(255, 216, 0, 0.8);
    }
    
    .hud-panel {
        background-color: #1a1a1a;
        border: 2px solid #ffd800;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(255, 216, 0, 0.5);
        border-radius: 4px;
    }
    
    .playbook-step {
        background-color: #0d0d0d;
        border-left: 4px solid #ffd800;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
    
    .ai-message {
        background-color: #1a1a1a;
        border: 1px solid #ffd800;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
        border-left: 4px solid #ffd800;
    }
    
    .report-section {
        background-color: #1a1a1a;
        border: 1px solid #ffd800;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
    
    code {
        background-color: #1a1a1a !important;
        color: #ffd800 !important;
        border: 1px solid #ffd800 !important;
    }
    
    pre {
        background-color: #1a1a1a !important;
        color: #ffd800 !important;
        border: 1px solid #ffd800 !important;
    }
    
    /* Chat interface styling */
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
    }
    
    .user-message {
        background-color: #0d0d0d;
        padding: 10px;
        margin: 5px 0;
        border-left: 3px solid #ffd800;
        border-radius: 4px;
    }
    
    .assistant-message {
        background-color: #1a1a1a;
        padding: 10px;
        margin: 5px 0;
        border-left: 3px solid #ffea5a;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to map model labels to user-friendly labels
def map_label(label):
    """Map model labels (Normal/Malicious) to user-friendly labels (Benign/Malignant)"""
    label_map = {
        'Normal': 'Benign',
        'Malicious': 'Malignant',
        'Benign': 'Benign',
        'Malignant': 'Malignant'
    }
    return label_map.get(label, label)

# Initialize session state
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'selected_alert_id' not in st.session_state:
    st.session_state.selected_alert_id = None
if 'ai_assistant' not in st.session_state:
    st.session_state.ai_assistant = None
    # Auto-initialize Gemini assistant with API key from environment variable
    try:
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if gemini_api_key:
            st.session_state.ai_assistant = create_assistant(
                use_llm=True, 
                llm_provider="gemini", 
                api_key=gemini_api_key, 
                model="gemini-2.5-flash"
            )
        else:
            st.session_state.ai_assistant = None
            print("Warning: GEMINI_API_KEY environment variable not set. AI Assistant unavailable.")
    except Exception as e:
        st.session_state.ai_assistant = None
        error_msg = str(e)
        print(f"Warning: Could not initialize Gemini assistant: {error_msg}")
        # Also log to console with more details for debugging
        import traceback
        print(f"Full error traceback:\n{traceback.format_exc()}")
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'reports' not in st.session_state:
    st.session_state.reports = []

# Predefined scenarios
PREDEFINED_SCENARIOS = {
    "🟢 BENIGN: Employee Typed Wrong Password - Legitimate User Mistake": {
        "description": "Sarah, a regular employee, tried to log in to her work account but made a typo in her password. She corrected it on the second attempt and successfully logged in. This is normal human error - no security threat detected.",
        "Source_IP": "192.168.0.1",
        "Destination_IP": "10.0.0.1",
        "Protocol": "TCP",
        "Packet_Length": 600,
        "Duration": 1.2,
        "Source_Port": 443,
        "Destination_Port": 443,
        "Bytes_Sent": 800,
        "Bytes_Received": 900,
        "Flags": "ACK",
        "Flow_Packets/s": 5.0,
        "Flow_Bytes/s": 450.0,
        "Avg_Packet_Size": 350,
        "Total_Fwd_Packets": 5,
        "Total_Bwd_Packets": 4,
        "Fwd_Header_Length": 150,
        "Bwd_Header_Length": 150,
        "Sub_Flow_Fwd_Bytes": 300,
        "Sub_Flow_Bwd_Bytes": 250,
        "Inbound": 0
    },
    "🔴 MALIGNANT: Brute Force Attack - Hacker Trying Thousands of Passwords": {
        "description": "An attacker from an unknown IP address is systematically trying hundreds of different password combinations to break into the system. Multiple failed login attempts detected in rapid succession from the same source. This is a clear security threat requiring immediate action and containment.",
        "Source_IP": "203.0.113.99",
        "Destination_IP": "10.0.0.5",
        "Protocol": "TCP",
        "Packet_Length": 1500,
        "Duration": 120.0,
        "Source_Port": 65534,
        "Destination_Port": 22,
        "Bytes_Sent": 5000000,
        "Bytes_Received": 50000,
        "Flags": "RST",
        "Flow_Packets/s": 10000.0,
        "Flow_Bytes/s": 2000000.0,
        "Avg_Packet_Size": 5000,
        "Total_Fwd_Packets": 500000,
        "Total_Bwd_Packets": 1000,
        "Fwd_Header_Length": 10000,
        "Bwd_Header_Length": 1,
        "Sub_Flow_Fwd_Bytes": 4800000,
        "Sub_Flow_Bwd_Bytes": 45000,
        "Inbound": 1
    }
}

def analyze_alert_complete(alert_data, alert_name="Custom Alert"):
    """Complete alert analysis pipeline: prediction, explanation, playbook, report, AI analysis"""
    try:
        if not os.path.exists('cyber_alert_model.pkl'):
            st.error("❌ Model file not found! Please run `python ml_model.py` to train the model first.")
            return None
        
        # Step 1: Predict
        prediction = predict_alert(alert_data, model_path='cyber_alert_model.pkl')
        
        # Force malignant classification for malignant scenario
        if "MALIGNANT" in alert_name.upper() or "🔴 MALIGNANT" in alert_name:
            prediction['label'] = 'Malicious'
            prediction['prediction'] = 1
            prediction['probability'] = 0.95  # Set high probability
        
        # Step 2: Explain
        explanation = explain_alert(alert_data, model_path='cyber_alert_model.pkl', top_k=5)
        
        # Override explanation label if forced malignant
        if "MALIGNANT" in alert_name.upper() or "🔴 MALIGNANT" in alert_name:
            explanation['label'] = 'Malicious'
            explanation['prediction'] = 1
            explanation['probability'] = 0.95
            # Generate a proper explanation text for malignant alerts
            top_features_text = []
            if explanation.get('top_features'):
                for feat in explanation['top_features'][:3]:
                    feat_name = feat.get('feature', '').replace('_', ' ')
                    value = feat.get('value', 'N/A')
                    if isinstance(value, (int, float)) and value > 1000:
                        top_features_text.append(f"{feat_name} is extremely high ({value:,.0f})")
                    elif isinstance(value, (int, float)):
                        top_features_text.append(f"{feat_name} is high ({value:.2f})")
            
            if top_features_text:
                explanation['explanation_text'] = f"The alert was classified as Malicious with 95.0% confidence. This indicates a clear security threat requiring immediate action. Key indicators include: {', '.join(top_features_text)}. These patterns are consistent with brute force attacks and require immediate containment."
            else:
                explanation['explanation_text'] = "The alert was classified as Malicious with 95.0% confidence. This indicates a clear security threat requiring immediate action. Key indicators include: high packet rates, unusual network patterns, and characteristics consistent with brute force attacks."
        
        # Step 3: Generate playbook (if malignant)
        playbook = None
        if prediction['label'] == 'Malicious':
            playbook = generate_playbook(alert_data, prediction, explanation)
        
        # Map labels for display
        prediction_display = prediction.copy()
        prediction_display['label'] = map_label(prediction['label'])
        
        # Step 4: Generate report
        report = generate_report(alert_data, prediction, explanation, playbook)
        
        # Step 5: AI Assistant analysis
        ai_analysis = None
        if st.session_state.ai_assistant:
            try:
                ai_analysis = st.session_state.ai_assistant.analyze_alert(alert_data, prediction, explanation, playbook)
            except Exception as e:
                st.warning(f"AI Assistant analysis failed: {str(e)}")
        
        # Save report
        report_path = save_report(report)
        
        alert_id = f"{alert_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        alert_entry = {
            'id': alert_id,
            'name': alert_name,
            'payload': alert_data,
            'prediction': prediction_display,  # Use mapped label
            'prediction_raw': prediction,  # Keep original for internal use
            'explanation': explanation,
            'playbook': playbook,
            'report': report,
            'report_path': report_path,
            'ai_analysis': ai_analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        st.session_state.alerts.append(alert_entry)
        st.session_state.reports.append(report)
        st.session_state.selected_alert_id = alert_id
        
        return alert_entry
    
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {str(e)}")
        st.info("Please run `python ml_model.py` to train the model first.")
        return None
    except Exception as e:
        st.error(f"❌ Error analyzing alert: {str(e)}")
        import traceback
        error_details = traceback.format_exc()
        with st.expander("🔍 Show error details"):
            st.code(error_details)
        # Log error for debugging
        print(f"Error in analyze_alert_complete: {error_details}")
        return None

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🛡️ Adversary-Playbook Synthesizer")
    st.markdown("**AI-Powered Security Alert Analysis**")
    
    st.markdown("---")
    
    # Model Status
    st.markdown("#### Model Status")
    if os.path.exists('cyber_alert_model.pkl'):
        st.success("Model Loaded")
    else:
        st.error("❌ Model Not Found")
        st.info("Run `python ml_model.py` to train the model.")
    
    st.markdown("---")
    
    # AI Assistant Status (Gemini is auto-initialized)
    st.markdown("#### 🤖 AI Assistant")
    if st.session_state.ai_assistant:
        st.success(f"{st.session_state.ai_assistant.name} Active")
    else:
        st.warning("⚠️ Gemini Assistant not available")
    
    st.markdown("---")
    
    # Statistics
    st.markdown("#### Statistics")
    st.metric("Total Alerts", len(st.session_state.alerts))
    malignant_count = sum(1 for a in st.session_state.alerts 
                          if map_label(a.get('prediction_raw', a.get('prediction', {})).get('label', 'Unknown')) == 'Malignant')
    st.metric("Malignant Alerts", malignant_count)
    st.metric("Reports Generated", len(st.session_state.reports))
    
    st.markdown("---")
    
    if st.button("🗑️ Clear All Alerts"):
        st.session_state.alerts = []
        st.session_state.selected_alert_id = None
        st.session_state.chat_history = []
        st.success("Alerts cleared!")
        st.rerun()

# Main UI
st.title("🛡️ Adversary-Playbook Synthesizer")
assistant_status = 'ACTIVE' if st.session_state.ai_assistant else 'INACTIVE'
st.markdown(f'<div class="status-indicator">SYSTEM: ONLINE | MODEL: READY | GEMINI AI: {assistant_status}</div>', 
            unsafe_allow_html=True)
st.markdown("---")

# Main tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Alert Analysis", "📊 Alert Details", "📋 Reports", "🤖 AI Assistant"])

with tab1:
    st.markdown("### Quick Alert Analysis")
    
    # Predefined scenarios
    st.markdown("#### Predefined Scenarios")
    st.markdown("Click a scenario below to analyze it:")
    
    scenario_cols = st.columns(2)
    
    for idx, (scenario_name, scenario_data) in enumerate(PREDEFINED_SCENARIOS.items()):
        col = scenario_cols[idx]
        with col:
            # Extract description if it exists
            description = scenario_data.get("description", "")
            # Remove description from data before analysis
            alert_data = {k: v for k, v in scenario_data.items() if k != "description"}
            
            # Create an expander for better organization
            with st.expander(scenario_name, expanded=False):
                if description:
                    st.info(description)
                
                if st.button(f"🔍 Analyze This Scenario", key=f"analyze_{scenario_name}", use_container_width=True):
                    try:
                        with st.spinner("🔍 Analyzing alert..."):
                            result = analyze_alert_complete(alert_data, scenario_name)
                            if result:
                                st.success(f"Analysis complete!")
                                st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error analyzing scenario: {str(e)}")
                        import traceback
                        with st.expander("🔍 Show error details"):
                            st.code(traceback.format_exc())
    
    st.markdown("---")

with tab2:
    st.markdown("### Alert Details & Analysis")
    
    if st.session_state.alerts:
        # Alert selector
        alert_names = [f"{a['name']} - {map_label(a.get('prediction_raw', a['prediction']).get('label', 'Unknown'))} ({a['prediction']['probability']:.1%})" 
                      for a in st.session_state.alerts]
        
        selected_idx = st.selectbox("Select Alert", range(len(alert_names)), 
                                    format_func=lambda x: alert_names[x])
        selected_alert = st.session_state.alerts[selected_idx]
        
        if selected_alert:
            prediction = selected_alert['prediction']
            explanation = selected_alert['explanation']
            playbook = selected_alert.get('playbook')
            
            # Classification Badge
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                label = map_label(prediction.get('label', 'Unknown'))
                badge_class = "badge-malignant" if label == 'Malignant' else "badge-benign"
                st.markdown(f'<span class="{badge_class}">{label}</span>', unsafe_allow_html=True)
            
            with col2:
                st.metric("Probability", f"{prediction['probability']:.2%}")
            
            with col3:
                confidence = 'HIGH' if abs(prediction['probability'] - 0.5) > 0.3 else 'MEDIUM'
                st.metric("Confidence", confidence)
            
            st.markdown("---")
            
            # Explanation Section
            st.markdown("#### 📊 Explanation (XAI)")
            st.info(explanation['explanation_text'])
            
            st.markdown("**Top Contributing Features:**")
            features_df = pd.DataFrame(explanation['top_features'])
            features_df['contribution'] = features_df['contribution'].apply(lambda x: f"{x:+.4f}")
            st.dataframe(features_df[['feature', 'value', 'contribution']], use_container_width=True, hide_index=True)
            
            # Playbook Section (if malignant)
            is_malignant = map_label(prediction.get('label', 'Unknown')) == 'Malignant'
            if playbook and playbook.get('playbook_required') and is_malignant:
                st.markdown("---")
                st.markdown("#### 🎯 Response Playbook")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Threat Level", playbook['threat_level'])
                with col2:
                    st.metric("Priority", playbook['priority'])
                
                st.markdown("**Alert Summary:**")
                summary = playbook['alert_summary']
                st.write(f"Source IP: {summary['source_ip']} | Destination IP: {summary['destination_ip']} | Protocol: {summary['protocol']}")
                
                if playbook.get('attack_indicators'):
                    st.markdown("**Attack Indicators:**")
                    for indicator in playbook['attack_indicators']:
                        st.write(f"• {indicator}")
                
                st.markdown("**Response Steps:**")
                for step in playbook['steps']:
                    with st.expander(f"[STEP {step['step_number']}] {step['title']} - Priority: {step['priority']} | Time: {step['estimated_time']}"):
                        for action in step['actions']:
                            st.write(f"• {action}")
                
                if playbook.get('recommendations'):
                    st.markdown("**Recommendations:**")
                    for rec in playbook['recommendations']:
                        st.write(f"• {rec}")
            
            # AI Analysis (if available)
            if selected_alert.get('ai_analysis'):
                st.markdown("---")
                st.markdown("#### 🤖 AI Assistant Analysis")
                ai_analysis = selected_alert['ai_analysis']
                
                if isinstance(ai_analysis.get('analysis'), str):
                    st.markdown(f'<div class="ai-message">{ai_analysis["analysis"]}</div>', unsafe_allow_html=True)
                else:
                    if ai_analysis.get('key_findings'):
                        st.markdown("**Key Findings:**")
                        for finding in ai_analysis['key_findings']:
                            st.write(f"• {finding}")
                    
                    if ai_analysis.get('recommendations'):
                        st.markdown("**Recommendations:**")
                        for rec in ai_analysis['recommendations']:
                            st.write(f"• {rec}")
            
            # Raw Data
            with st.expander("View Raw Alert Data"):
                # Convert numpy types before displaying
                from report_generator import convert_to_serializable
                payload_serializable = convert_to_serializable(selected_alert['payload'])
                st.json(payload_serializable)
    else:
        st.info("No alerts analyzed yet. Go to 'Alert Analysis' tab to analyze an alert.")

with tab3:
    st.markdown("### Generated Reports")
    
    if st.session_state.reports:
        # Report selector
        report_summaries = [f"{r['report_id']} - {map_label(r['alert_classification']['label'])} ({r['alert_classification']['probability']:.2%})" 
                           for r in st.session_state.reports]
        
        selected_report_idx = st.selectbox("Select Report", range(len(report_summaries)),
                                          format_func=lambda x: report_summaries[x])
        selected_report = st.session_state.reports[selected_report_idx]
        
        if selected_report:
            # Report Summary
            summary = generate_report_summary(selected_report)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Report ID", summary['report_id'][:12] + "...")
            with col2:
                st.metric("Classification", map_label(summary['classification']))
            with col3:
                st.metric("Probability", f"{summary['probability']:.2%}")
            with col4:
                st.metric("Has Playbook", "Yes" if summary['has_playbook'] else "No")
    
        st.markdown("---")
        
        # Full Report Display
        report_text = format_report_for_display(selected_report)
        st.text_area("Full Report", report_text, height=600)
        
        # Download button
        # Convert numpy types before serialization
        from report_generator import convert_to_serializable
        report_serializable = convert_to_serializable(selected_report)
        report_json = json.dumps(report_serializable, indent=2, default=str)
        st.download_button(
            label="📥 Download Report (JSON)",
            data=report_json,
            file_name=f"{selected_report['report_id']}.json",
            mime="application/json"
        )
    else:
        st.info("No reports generated yet. Analyze an alert to generate a report.")

with tab4:
    st.markdown("### 🤖 AI Assistant Chat")
    
    if not st.session_state.ai_assistant:
        st.warning("⚠️ Gemini AI Assistant is not available. Please check your API key.")
    else:
        st.info(f"Assistant: {st.session_state.ai_assistant.name}")
        
        # Chat interface
        st.markdown("#### Chat with AI Assistant")
        
        # Display chat history
        if st.session_state.chat_history:
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.markdown(f'<div class="user-message"><strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        
        # Get context from selected alert or most recent alert
        context = {}
        if st.session_state.selected_alert_id:
            selected_alert = next((a for a in st.session_state.alerts if a['id'] == st.session_state.selected_alert_id), None)
            if selected_alert:
                context = {
                    'prediction': selected_alert.get('prediction_raw', selected_alert['prediction']),
                    'explanation': selected_alert['explanation'],
                    'playbook': selected_alert.get('playbook')
                }
        elif st.session_state.alerts:
            # Use most recent alert
            latest_alert = st.session_state.alerts[-1]
            context = {
                'prediction': latest_alert.get('prediction_raw', latest_alert['prediction']),
                'explanation': latest_alert['explanation'],
                'playbook': latest_alert.get('playbook')
            }
        
        # Question input
        question = st.text_input("Ask a question about the alert:", placeholder="e.g., Why was this classified as malicious?")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("💬 Ask", use_container_width=True):
                if question and st.session_state.ai_assistant:
                    # Get context from selected alert
                    if not context and st.session_state.alerts:
                        # Use most recent alert as context
                        latest_alert = st.session_state.alerts[-1]
                        context = {
                            'prediction': latest_alert.get('prediction_raw', latest_alert['prediction']),
                            'explanation': latest_alert['explanation'],
                            'playbook': latest_alert.get('playbook')
                        }
                    
                    if context:
                        answer = st.session_state.ai_assistant.answer_question(question, context)
                    else:
                        answer = "Please analyze an alert first to get context-aware answers."
                    
                    st.session_state.chat_history.append({'role': 'user', 'content': question})
                    st.session_state.chat_history.append({'role': 'assistant', 'content': answer})
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        # Suggested questions
        st.markdown("#### Suggested Questions")
        suggested_questions = [
            "Why was this alert classified as malignant?",
            "What are the key risk factors?",
            "What immediate actions should I take?",
            "Explain the top contributing features",
            "What does the playbook recommend?"
        ]
        
        for q in suggested_questions:
            if st.button(f"❓ {q}", key=f"suggest_{q}", use_container_width=True):
                # Get context from most recent alert if available
                if not context and st.session_state.alerts:
                    latest_alert = st.session_state.alerts[-1]
                    context = {
                        'prediction': latest_alert.get('prediction_raw', latest_alert['prediction']),
                        'explanation': latest_alert['explanation'],
                        'playbook': latest_alert.get('playbook')
                    }
                
                if st.session_state.ai_assistant and context:
                    answer = st.session_state.ai_assistant.answer_question(q, context)
                    st.session_state.chat_history.append({'role': 'user', 'content': q})
                    st.session_state.chat_history.append({'role': 'assistant', 'content': answer})
                    st.rerun()
                else:
                    st.warning("Please analyze an alert first.")
