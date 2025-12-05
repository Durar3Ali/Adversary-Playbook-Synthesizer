"""
Streamlit app for Adversary-Playbook Synthesizer
Cyber Security Alert Classification with XAI
"""

import streamlit as st
import pandas as pd
from ml_model import predict_alert, explain_alert
import json
import os
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Adversary-Playbook Synthesizer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for yellow and black theme
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
    
    /* Main content area */
    .block-container {
        background-color: #000000;
        padding-top: 2rem;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffd800 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        text-shadow: 0 0 10px rgba(255, 216, 0, 0.5);
    }
    
    /* Text colors */
    p, div, span, label {
        color: #ffd800 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 2px solid #ffd800;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffd800 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #000000 !important;
        color: #ffd800 !important;
        border: 2px solid #ffd800 !important;
        border-radius: 4px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #0d0d0d !important;
        color: #ffea5a !important;
        border-color: #ffea5a !important;
        box-shadow: 0 0 10px rgba(255, 216, 0, 0.4);
    }
    
    .stButton > button:active {
        background-color: #1a1a1a !important;
        border-color: #ffd800 !important;
    }
    
    /* Selected/Active button */
    .stButton > button:focus {
        background-color: #1a1a1a !important;
        color: #ffd800 !important;
        border-color: #ffea5a !important;
        outline: none;
    }
    
    /* Prevent the bright yellow fill on hover */
    .stButton > button:hover:not(:active) {
        background-color: #0d0d0d !important;
    }
    
    /* Form inputs */
    .stSelectbox > div > div, .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background-color: #1a1a1a !important;
        color: #ffd800 !important;
        border: 1px solid #ffd800 !important;
    }
    
    .stSelectbox label, .stTextInput label, .stNumberInput label {
        color: #ffd800 !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #ffd800 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #ffd800 !important;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #1a1a1a !important;
        border-left: 4px solid #ffd800 !important;
        color: #ffd800 !important;
    }
    
    .stSuccess {
        background-color: #1a1a1a !important;
        border-left: 4px solid #ffd800 !important;
        color: #ffd800 !important;
    }
    
    .stWarning {
        background-color: #1a1a1a !important;
        border-left: 4px solid #ffd800 !important;
        color: #ffd800 !important;
    }
    
    .stError {
        background-color: #1a1a1a !important;
        border-left: 4px solid #ff0000 !important;
        color: #ff0000 !important;
    }
    
    /* Dataframes */
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
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a1a !important;
        color: #ffd800 !important;
        border: 1px solid #ffd800 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #ffd800 !important;
    }
    
    .hud-panel {
        background-color: #1a1a1a;
        border: 2px solid #ffd800;
        padding: 16px;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(255, 216, 0, 0.5);
    }
    
    .badge-normal {
        background-color: #1a1a1a;
        color: #ffd800;
        border: 2px solid #ffd800;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 600;
        box-shadow: 0 0 10px rgba(255, 216, 0, 0.3);
    }
    
    .badge-malicious {
        background-color: #1a1a1a;
        color: #ff0000;
        border: 2px solid #ff0000;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 600;
        animation: pulse-glow-red 2s ease-in-out infinite;
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
    
    .hud-label {
        color: #ffd800;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        opacity: 0.9;
        text-shadow: 0 0 5px rgba(255, 216, 0, 0.5);
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #ffd800 !important;
    }
    
    /* Code blocks */
    code {
        background-color: #1a1a1a !important;
        color: #ffd800 !important;
        border: 1px solid #ffd800 !important;
    }
    
    /* JSON viewer */
    pre {
        background-color: #1a1a1a !important;
        color: #ffd800 !important;
        border: 1px solid #ffd800 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'selected_alert_id' not in st.session_state:
    st.session_state.selected_alert_id = None

# Predefined scenarios
PREDEFINED_SCENARIOS = {
    "CEO Traveling – Suspicious Login": {
        "Source_IP": "192.168.0.7",
        "Destination_IP": "10.0.0.3",
        "Protocol": "TCP",
        "Packet_Length": 1776,
        "Duration": 3.8,
        "Source_Port": 22,
        "Destination_Port": 443,
        "Bytes_Sent": 2000,
        "Bytes_Received": 1500,
        "Flags": "SYN",
        "Flow_Packets/s": 37.9,
        "Flow_Bytes/s": 1800.0,
        "Avg_Packet_Size": 600,
        "Total_Fwd_Packets": 30,
        "Total_Bwd_Packets": 25,
        "Fwd_Header_Length": 256,
        "Bwd_Header_Length": 256,
        "Sub_Flow_Fwd_Bytes": 1200,
        "Sub_Flow_Bwd_Bytes": 900,
        "Inbound": 1
    },
    "Credential Stuffing – Multiple Failed Logins": {
        "Source_IP": "172.16.0.4",
        "Destination_IP": "172.16.0.5",
        "Protocol": "TCP",
        "Packet_Length": 1275,
        "Duration": 4.9,
        "Source_Port": 8080,
        "Destination_Port": 22,
        "Bytes_Sent": 2600,
        "Bytes_Received": 1400,
        "Flags": "RST",
        "Flow_Packets/s": 45.2,
        "Flow_Bytes/s": 3500.0,
        "Avg_Packet_Size": 400,
        "Total_Fwd_Packets": 50,
        "Total_Bwd_Packets": 10,
        "Fwd_Header_Length": 200,
        "Bwd_Header_Length": 100,
        "Sub_Flow_Fwd_Bytes": 2000,
        "Sub_Flow_Bwd_Bytes": 500,
        "Inbound": 1
    },
    "Benign User Typo – Password Mistake": {
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
    "Phishing Follow-up – New Device & IP": {
        "Source_IP": "10.0.0.9",
        "Destination_IP": "172.16.0.4",
        "Protocol": "UDP",
        "Packet_Length": 1459,
        "Duration": 2.5,
        "Source_Port": 53,
        "Destination_Port": 8080,
        "Bytes_Sent": 1500,
        "Bytes_Received": 1600,
        "Flags": "PSH",
        "Flow_Packets/s": 30.0,
        "Flow_Bytes/s": 2000.0,
        "Avg_Packet_Size": 550,
        "Total_Fwd_Packets": 20,
        "Total_Bwd_Packets": 18,
        "Fwd_Header_Length": 300,
        "Bwd_Header_Length": 300,
        "Sub_Flow_Fwd_Bytes": 1500,
        "Sub_Flow_Bwd_Bytes": 1800,
        "Inbound": 1
    }
}

def analyze_alert(alert_data, alert_name="Custom Alert"):
    """Analyze an alert using the real model."""
    try:
        # Check if model exists
        if not os.path.exists('cyber_alert_model.pkl'):
            st.error("❌ Model file not found! Please run `python ml_model.py` to train the model first.")
            return None
        
        result = explain_alert(alert_data, model_path='cyber_alert_model.pkl', top_k=5)
        alert_id = f"{alert_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        alert_entry = {
            'id': alert_id,
            'name': alert_name,
            'payload': alert_data,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        
        st.session_state.alerts.append(alert_entry)
        st.session_state.selected_alert_id = alert_id
        return result
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {str(e)}")
        st.info("Please run `python ml_model.py` to train the model first.")
        return None
    except Exception as e:
        st.error(f"❌ Error analyzing alert: {str(e)}")
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())
        return None

# Main UI
st.markdown("### ADVERSARY-PLAYBOOK SYNTHESIZER – SECURITY CONSOLE")
st.markdown('<div class="status-indicator">MODEL: ONLINE</div>', unsafe_allow_html=True)
st.markdown("---")

# Two column layout
col1, col2 = st.columns([0.4, 0.6])

with col1:
    st.markdown("#### ALERTS STREAM")
    
    # Predefined scenarios
    st.markdown("**Predefined Scenarios:**")
    for scenario_name, scenario_data in PREDEFINED_SCENARIOS.items():
        button_key = f"scenario_{scenario_name}"
        if st.button(f"👾 {scenario_name}", key=button_key, use_container_width=True):
            with st.spinner("Analyzing alert..."):
                result = analyze_alert(scenario_data, scenario_name)
                if result:
                    st.success(f"✅ Analyzed: {scenario_name}")
    
    st.markdown("---")
    
    # Alert list
    if st.session_state.alerts:
        st.markdown("**Recent Alerts:**")
        for alert in reversed(st.session_state.alerts[-10:]):  # Show last 10
            is_selected = st.session_state.selected_alert_id == alert['id']
            badge_class = "badge-malicious" if alert['result']['label'] == 'Malicious' else "badge-normal"
            
            if st.button(
                f"{alert['name']} - {alert['result']['label']} ({(alert['result']['probability']*100):.1f}%)",
                key=f"alert_{alert['id']}",
                use_container_width=True
            ):
                st.session_state.selected_alert_id = alert['id']
    else:
        st.info("No alerts analyzed yet. Click a predefined scenario above.")

with col2:
    st.markdown("#### ALERT DETAILS")
    
    if st.session_state.selected_alert_id:
        selected_alert = next(
            (a for a in st.session_state.alerts if a['id'] == st.session_state.selected_alert_id),
            None
        )
        
        if selected_alert:
            result = selected_alert['result']
            
            # Prediction
            badge_class = "badge-malicious" if result['label'] == 'Malicious' else "badge-normal"
            st.markdown(f'<span class="{badge_class}">{result["label"]}</span>', unsafe_allow_html=True)
            st.metric("Probability", f"{(result['probability']*100):.2f}%")
            
            # Explanation
            st.markdown("**Explanation:**")
            st.info(result['explanation_text'])
            
            # Top Features
            st.markdown("**Top Contributing Features:**")
            features_df = pd.DataFrame(result['top_features'])
            features_df['contribution'] = features_df['contribution'].apply(lambda x: f"{x:+.4f}")
            features_df['color'] = features_df['contribution'].apply(
                lambda x: '🔴' if float(x.replace('+', '')) > 0 else '🔵'
            )
            st.dataframe(features_df[['feature', 'value', 'contribution', 'color']], use_container_width=True, hide_index=True)
            
            # Raw payload
            with st.expander("View Raw Alert Data"):
                st.json(selected_alert['payload'])
        else:
            st.warning("Selected alert not found.")
    else:
        st.info("Select an alert from the left panel to view details.")

# Custom Alert Form
st.markdown("---")
st.markdown("#### // CUSTOM ALERT INPUT")

with st.form("custom_alert_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        protocol = st.selectbox("Protocol", ["TCP", "UDP", "ICMP"])
        packet_length = st.number_input("Packet Length", min_value=0, value=1500)
        duration = st.number_input("Duration", min_value=0.0, value=2.5, step=0.1)
    
    with col2:
        source_port = st.number_input("Source Port", min_value=0, max_value=65535, value=80)
        dest_port = st.number_input("Destination Port", min_value=0, max_value=65535, value=443)
        flow_packets_per_sec = st.number_input("Flow Packets/s", min_value=0.0, value=30.0, step=0.1)
    
    with col3:
        source_ip = st.text_input("Source IP", value="192.168.0.1")
        dest_ip = st.text_input("Destination IP", value="10.0.0.3")
        inbound = st.selectbox("Inbound", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    
    submitted = st.form_submit_button("🔍 ANALYZE ALERT", use_container_width=True)
    
    if submitted:
        alert_data = {
            "Source_IP": source_ip,
            "Destination_IP": dest_ip,
            "Protocol": protocol,
            "Packet_Length": packet_length,
            "Duration": duration,
            "Source_Port": source_port,
            "Destination_Port": dest_port,
            "Flow_Packets/s": flow_packets_per_sec,
            "Inbound": inbound,
            "Bytes_Sent": packet_length * 2,
            "Bytes_Received": packet_length * 2,
            "Flags": "SYN",
            "Flow_Bytes/s": flow_packets_per_sec * packet_length,
            "Avg_Packet_Size": packet_length,
            "Total_Fwd_Packets": int(flow_packets_per_sec * duration),
            "Total_Bwd_Packets": int(flow_packets_per_sec * duration * 0.8),
            "Fwd_Header_Length": 256,
            "Bwd_Header_Length": 256,
            "Sub_Flow_Fwd_Bytes": packet_length,
            "Sub_Flow_Bwd_Bytes": packet_length
        }
        
        with st.spinner("Analyzing alert with real model..."):
            result = analyze_alert(alert_data, "Custom Alert")
            if result:
                st.success("✅ Alert analyzed successfully!")

# Sidebar
with st.sidebar:
    st.markdown("### 🛡️ Adversary-Playbook Synthesizer")
    st.markdown("AI-powered security console for classifying cyber alerts as Normal or Malicious with Explainable AI.")
    
    st.markdown("---")
    st.markdown("#### How It Works")
    st.markdown("""
    1. **Receive Alert** - Receive a cyber security alert
    2. **AI Classification** - Random Forest model predicts Normal or Malicious
    3. **Explainable AI (XAI)** - SHAP values explain top contributing features
    4. **Playbook Generation** - Future: Auto-generate response playbooks
    """)
    
    st.markdown("---")
    st.markdown("#### Model Status")
    if os.path.exists('cyber_alert_model.pkl'):
        st.success("✅ Model loaded")
    else:
        st.error("❌ Model not found")
        st.info("Run `python ml_model.py` to train the model first.")
    
    st.markdown("---")
    if st.button("🗑️ Clear All Alerts"):
        st.session_state.alerts = []
        st.session_state.selected_alert_id = None
        st.success("Alerts cleared!")

