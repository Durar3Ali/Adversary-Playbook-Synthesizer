"""
Streamlit app for Adversary-Playbook Synthesizer
Cyber Security Alert Classification with XAI, Playbook Generation, and AI Assistant
"""

import json
import logging
import os
import traceback

import streamlit as st
import pandas as pd

# Load environment variables from .env file before importing config.
# Warn visibly if python-dotenv is not installed so the user knows why
# GEMINI_API_KEY may not be picked up from .env.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning(
        "python-dotenv is not installed; .env file will NOT be loaded. "
        "Run: pip install python-dotenv"
    )

from src import config
from src.ml.model_predictor import load_model
from src.analysis.pipeline import run_analysis
from src.analysis.playbook_generator import format_playbook_for_display
from src.analysis.report_generator import (
    format_report_for_display,
    generate_report_summary,
    convert_to_serializable,
)
from src.assistant.factory import create_assistant

@st.cache_resource
def _cached_load_model(model_path: str) -> dict:
    """Load the pickled model once per process and cache it across Streamlit reruns."""
    return load_model(model_path)


# Page config
st.set_page_config(
    page_title=config.APP_NAME,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base colors (primaryColor, backgroundColor, textColor, font) are set in
# .streamlit/config.toml. Only structural overrides and custom component
# classes that Streamlit's theme API cannot express are kept here.
st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 95%; }

    h1, h2, h3, h4, h5, h6 {
        text-transform: uppercase;
        letter-spacing: 0.1em;
        text-shadow: 0 0 10px rgba(255, 216, 0, 0.5);
    }

    [data-testid="stSidebar"] { border-right: 2px solid #ffd800; }

    .stButton > button {
        background-color: #000000 !important;
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
        border-color: #ffea5a !important;
        box-shadow: 0 0 10px rgba(255, 216, 0, 0.4);
        transform: translateY(-2px);
    }
    .stButton > button:active {
        background-color: #1a1a1a !important;
        border-color: #ffd800 !important;
    }

    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        border: 1px solid #ffd800 !important;
    }

    .stInfo, .stSuccess, .stWarning { border-left: 4px solid #ffd800 !important; }
    .stError { border-left: 4px solid #ff0000 !important; color: #ff0000 !important; }

    .dataframe, .dataframe th, .dataframe td { border: 1px solid #ffd800 !important; }
    .streamlit-expanderHeader { border: 1px solid #ffd800 !important; }
    .stSpinner > div { border-top-color: #ffd800 !important; }
    code, pre { border: 1px solid #ffd800 !important; }

    /* Classification badges */
    .badge-benign {
        color: #00ff00; border: 2px solid #00ff00;
        padding: 8px 16px; border-radius: 4px;
        font-size: 1rem; text-transform: uppercase; font-weight: 600;
        box-shadow: 0 0 10px rgba(0, 255, 0, 0.3); display: inline-block;
    }
    .badge-malignant {
        color: #ff0000; border: 2px solid #ff0000;
        padding: 8px 16px; border-radius: 4px;
        font-size: 1rem; text-transform: uppercase; font-weight: 600;
        animation: pulse-glow-red 2s ease-in-out infinite; display: inline-block;
    }
    @keyframes pulse-glow-red {
        0%, 100% { box-shadow: 0 0 10px rgba(255, 0, 0, 0.5); }
        50%       { box-shadow: 0 0 20px rgba(255, 0, 0, 1); }
    }

    .status-indicator {
        font-size: 0.75rem; text-transform: uppercase;
        letter-spacing: 0.1em; text-shadow: 0 0 10px rgba(255, 216, 0, 0.8);
    }
    .hud-panel {
        border: 2px solid #ffd800; padding: 20px; margin: 10px 0;
        box-shadow: 0 0 20px rgba(255, 216, 0, 0.5); border-radius: 4px;
    }
    .playbook-step {
        border-left: 4px solid #ffd800; padding: 15px; margin: 10px 0; border-radius: 4px;
    }
    .ai-message {
        border: 1px solid #ffd800; border-left: 4px solid #ffd800;
        padding: 15px; margin: 10px 0; border-radius: 4px;
    }
    .report-section {
        border: 1px solid #ffd800; padding: 15px; margin: 10px 0; border-radius: 4px;
    }
    .chat-container { max-height: 400px; overflow-y: auto; padding: 10px; }
    .user-message {
        padding: 10px; margin: 5px 0;
        border-left: 3px solid #ffd800; border-radius: 4px;
    }
    .assistant-message {
        padding: 10px; margin: 5px 0;
        border-left: 3px solid #ffea5a; border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

def map_label(label: str) -> str:
    """Map model labels (Normal/Malicious) to user-facing labels (Benign/Malignant)."""
    return {"Normal": "Benign", "Malicious": "Malignant"}.get(label, label)

# Initialize session state
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'selected_alert_id' not in st.session_state:
    st.session_state.selected_alert_id = None
if 'ai_assistant' not in st.session_state:
    st.session_state.ai_assistant = None
    try:
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if gemini_api_key:
            st.session_state.ai_assistant = create_assistant(
                use_llm=True,
                llm_provider="gemini",
                api_key=gemini_api_key,
                model=config.GEMINI_MODEL,
            )
        else:
            import logging
            logging.warning("GEMINI_API_KEY environment variable not set. AI Assistant unavailable.")
    except Exception as e:
        import logging
        logging.warning("Could not initialize Gemini assistant: %s", e)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'reports' not in st.session_state:
    st.session_state.reports = []

def _get_current_context() -> dict:
    """Return prediction/explanation/playbook context for the active alert.

    Prefers the explicitly selected alert; falls back to the most recent one.
    Returns an empty dict when no alerts have been analysed yet.
    """
    alert = None
    if st.session_state.selected_alert_id:
        alert = next(
            (a for a in st.session_state.alerts if a['id'] == st.session_state.selected_alert_id),
            None,
        )
    if alert is None and st.session_state.alerts:
        alert = st.session_state.alerts[-1]
    if alert is None:
        return {}
    return {
        'prediction': alert.get('prediction_raw', alert['prediction']),
        'explanation': alert['explanation'],
        'playbook': alert.get('playbook'),
    }


def _load_predefined_scenarios() -> dict:
    """Load predefined scenarios from scenarios.json (path from config). Falls back to empty dict."""
    scenarios_file = config.SCENARIOS_FILE
    if os.path.exists(scenarios_file):
        try:
            with open(scenarios_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("scenarios", {})
        except Exception as e:
            import logging
            logging.warning("Could not load scenarios file '%s': %s", scenarios_file, e)
    return {}


PREDEFINED_SCENARIOS: dict = _load_predefined_scenarios()

def analyze_alert_complete(
    alert_data: dict,
    alert_name: str = "Custom Alert",
):
    """UI wrapper: load model (cached), delegate to the pure pipeline, update session state.

    Returns the alert_entry dict on success, or None on failure (errors are
    displayed via st.error/st.info).
    """
    model_path = config.MODEL_PATH
    if not os.path.exists(model_path):
        st.error("Model file not found! Please run `python -m src.ml.model_trainer` to train the model first.")
        return None

    try:
        model_data = _cached_load_model(model_path)
        alert_entry = run_analysis(
            alert_data=alert_data,
            alert_name=alert_name,
            model_data=model_data,
            ai_assistant=st.session_state.ai_assistant,
        )
    except FileNotFoundError as exc:
        st.error(f"Model file not found: {exc}")
        st.info("Please run `python -m src.ml.model_trainer` to train the model first.")
        return None
    except Exception as exc:
        st.error(f"Error analyzing alert: {exc}")
        error_details = traceback.format_exc()
        with st.expander("Show error details"):
            st.code(error_details)
        logging.error("Error in analyze_alert_complete: %s", error_details)
        return None

    if alert_entry.get("ai_analysis", {}).get("error"):
        st.warning(alert_entry["ai_analysis"]["error"])

    st.session_state.alerts.append(alert_entry)
    st.session_state.reports.append(alert_entry["report"])
    st.session_state.selected_alert_id = alert_entry["id"]
    return alert_entry

# Sidebar Configuration
with st.sidebar:
    st.markdown(f"### {config.APP_NAME}")
    st.markdown("**AI-Powered Security Alert Analysis**")
    
    st.markdown("---")
    
    # Model Status
    st.markdown("#### Model Status")
    if os.path.exists(config.MODEL_PATH):
        st.success("Model Loaded")
    else:
        st.error("Model Not Found")
        st.info("Run `python -m src.ml.model_trainer` to train the model.")
    
    st.markdown("---")
    
    # AI Assistant Status (Gemini is auto-initialized)
    st.markdown("#### AI Assistant")
    if st.session_state.ai_assistant:
        st.success(f"{st.session_state.ai_assistant.name} Active")
    else:
        st.warning("Gemini Assistant not available")
    
    st.markdown("---")
    
    # Statistics
    st.markdown("#### Statistics")
    st.metric("Total Alerts", len(st.session_state.alerts))
    malignant_count = sum(1 for a in st.session_state.alerts 
                          if map_label(a.get('prediction_raw', a.get('prediction', {})).get('label', 'Unknown')) == 'Malignant')
    st.metric("Malignant Alerts", malignant_count)
    st.metric("Reports Generated", len(st.session_state.reports))
    
    st.markdown("---")
    
    if st.button("Clear All Alerts"):
        st.session_state.alerts = []
        st.session_state.selected_alert_id = None
        st.session_state.chat_history = []
        st.success("Alerts cleared!")
        st.rerun()

# Main UI
st.title(f"{config.APP_NAME}")
assistant_status = 'ACTIVE' if st.session_state.ai_assistant else 'INACTIVE'
st.markdown(f'<div class="status-indicator">SYSTEM: ONLINE | MODEL: READY | GEMINI AI: {assistant_status}</div>', 
            unsafe_allow_html=True)
st.markdown("---")

# Main tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs(["Alert Analysis", "Alert Details", "Reports", "AI Assistant"])

with tab1:
    st.markdown("### Quick Alert Analysis")
    
    # Predefined scenarios
    st.markdown("#### Predefined Scenarios")
    st.markdown("Click a scenario below to analyze it:")
    
    scenario_cols = st.columns(2)
    
    _NON_FEATURE_KEYS = {"description", "force_malignant"}
    for idx, (scenario_name, scenario_data) in enumerate(PREDEFINED_SCENARIOS.items()):
        col = scenario_cols[idx % len(scenario_cols)]
        with col:
            description = scenario_data.get("description", "")
            alert_data = {k: v for k, v in scenario_data.items() if k not in _NON_FEATURE_KEYS}

            with st.expander(scenario_name, expanded=False):
                if description:
                    st.info(description)

                if st.button("Analyze This Scenario", key=f"analyze_{scenario_name}", use_container_width=True):
                    try:
                        with st.spinner("Analyzing alert..."):
                            result = analyze_alert_complete(
                                alert_data, scenario_name
                            )
                            if result:
                                st.success("Analysis complete!")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error analyzing scenario: {str(e)}")
                        with st.expander("Show error details"):
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
                confidence = 'HIGH' if abs(prediction['probability'] - 0.5) > config.CONFIDENCE_HIGH_DELTA else 'MEDIUM'
                st.metric("Confidence", confidence)
            
            st.markdown("---")
            
            # Explanation Section
            st.markdown("#### Explanation (XAI)")
            st.info(explanation['explanation_text'])
            
            st.markdown("**Top Contributing Features:**")
            features_df = pd.DataFrame(explanation['top_features'])
            features_df['contribution'] = features_df['contribution'].apply(lambda x: f"{x:+.4f}")
            st.dataframe(features_df[['feature', 'value', 'contribution']], use_container_width=True, hide_index=True)
            
            # Playbook Section (if malignant)
            is_malignant = map_label(prediction.get('label', 'Unknown')) == 'Malignant'
            if playbook and playbook.get('playbook_required') and is_malignant:
                st.markdown("---")
                st.markdown("#### Response Playbook")
                
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
                st.markdown("#### AI Assistant Analysis")
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
        report_serializable = convert_to_serializable(selected_report)
        report_json = json.dumps(report_serializable, indent=2, default=str)
        st.download_button(
            label="Download Report (JSON)",
            data=report_json,
            file_name=f"{selected_report['report_id']}.json",
            mime="application/json"
        )
    else:
        st.info("No reports generated yet. Analyze an alert to generate a report.")

with tab4:
    st.markdown("### AI Assistant Chat")

    if not st.session_state.ai_assistant:
        st.warning("Gemini AI Assistant is not available. Please check your API key.")
    else:
        st.info(f"Assistant: {st.session_state.ai_assistant.name}")

        st.markdown("#### Chat with AI Assistant")

        if st.session_state.chat_history:
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.markdown(f'<div class="user-message"><strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong> {msg["content"]}</div>', unsafe_allow_html=True)

        # Single authoritative context lookup for the whole tab.
        context = _get_current_context()

        question = st.text_input("Ask a question about the alert:", placeholder="e.g., Why was this classified as malicious?")

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Ask", use_container_width=True):
                if question and st.session_state.ai_assistant:
                    if context:
                        answer = st.session_state.ai_assistant.answer_question(question, context)
                    else:
                        answer = "Please analyze an alert first to get context-aware answers."
                    st.session_state.chat_history.append({'role': 'user', 'content': question})
                    st.session_state.chat_history.append({'role': 'assistant', 'content': answer})
                    st.rerun()

        with col2:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        st.markdown("#### Suggested Questions")
        suggested_questions = [
            "Why was this alert classified as malignant?",
            "What are the key risk factors?",
            "What immediate actions should I take?",
            "Explain the top contributing features",
            "What does the playbook recommend?",
        ]

        for q in suggested_questions:
            if st.button(q, key=f"suggest_{q}", use_container_width=True):
                if st.session_state.ai_assistant and context:
                    answer = st.session_state.ai_assistant.answer_question(q, context)
                    st.session_state.chat_history.append({'role': 'user', 'content': q})
                    st.session_state.chat_history.append({'role': 'assistant', 'content': answer})
                    st.rerun()
                else:
                    st.warning("Please analyze an alert first.")
