# Adversary-Playbook Synthesizer

An intelligent cybersecurity alert analysis system that combines machine learning classification, explainable AI (XAI), automated playbook generation, comprehensive reporting, and AI-powered assistant capabilities.

## Overview

This system provides end-to-end security alert analysis with:
- **Machine Learning Classification**: Random Forest model to classify alerts as Normal or Malicious
- **Explainable AI (XAI)**: SHAP-based explanations for predictions
- **Automated Playbook Generation**: Step-by-step response playbooks for malicious alerts
- **Comprehensive Reporting**: Detailed reports with all analysis results
- **AI Assistant**: LLM-powered security analyst assistant (Google Gemini)

## System Architecture

```
Alert Input → ML Prediction → XAI Explanation → Playbook Generation → Report Generation → AI Analysis
```

## Features

### 1. Alert Classification
- Binary classification: Normal vs. Malicious alerts
- Probability scores for confidence assessment
- Threshold-based classification with optimal threshold tuning

### 2. Explainable AI (XAI)
- SHAP values for feature importance
- Top contributing features identification
- Human-readable explanations
- Fallback to feature importances if SHAP unavailable

### 3. Automated Playbook Generation
- Automatic generation for malicious alerts
- Step-by-step incident response procedures
- Threat level assessment (CRITICAL, HIGH, MEDIUM)
- Priority-based action items
- Estimated response times

### 4. Report Generation
- Comprehensive JSON reports
- All analysis results in structured format
- Timestamp tracking
- Exportable for documentation

### 5. AI Assistant
- LLM-powered security analyst using Google Gemini
- Fallback to rule-based assistant
- Context-aware alert analysis
- Interactive Q&A capabilities

## Dataset

The model is trained on `cyberfeddefender_dataset.csv` containing 1,430 network alerts:
- **DDoS**: 377 samples
- **Ransomware**: 361 samples
- **Brute Force**: 352 samples
- **Normal**: 340 samples

## Features Used

The model uses 21 features including:
- Network flow characteristics (Packet_Length, Duration, Bytes_Sent, Bytes_Received)
- Protocol and port information (Protocol, Source_Port, Destination_Port)
- Traffic statistics (Flow_Packets/s, Flow_Bytes/s, Avg_Packet_Size)
- Packet counts and header lengths
- IP addresses (encoded)
- Derived feature: `same_source_dest_ip` (indicates if source and destination IPs are the same)

## Model Performance

- **Test Accuracy**: ~74% (with default threshold)
- **Training Accuracy**: ~100% (potential overfitting)
- **ROC AUC Score**: 0.5285
- **Optimal Threshold**: Automatically calculated based on ROC curve

**Note**: The model shows bias toward predicting malicious alerts due to class imbalance (1090 malicious vs 340 normal).

## Project Structure

```
Adversary-Playbook Synthesizer/
├── ml_model.py              # ML model training, prediction, and XAI explanations
├── app.py                   # Streamlit web application (main interface)
├── ai_assistant.py          # AI assistant with Gemini LLM support
├── playbook_generator.py    # Automated playbook generation for malicious alerts
├── report_generator.py      # Comprehensive report generation
├── cyber_alert_model.pkl    # Trained model (generated after training)
├── cyberfeddefender_dataset.csv  # Training dataset
├── requirements.txt         # Python dependencies
├── reports/                 # Generated report files (JSON)
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn shap streamlit google-generativeai
```

## Quick Start

### 1. Train the Model

First, train the machine learning model:

```bash
python ml_model.py
```

This will:
- Load and preprocess the dataset
- Train a Random Forest classifier
- Evaluate model performance
- Save the trained model to `cyber_alert_model.pkl`

### 2. Run the Application

Start the Streamlit web application:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### 3. Configure AI Assistant (Optional)

The application automatically attempts to initialize a Gemini assistant using the `GEMINI_API_KEY` environment variable. You can set it in two ways:

#### Option 1: Using `.env` file (Recommended)

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your Gemini API key:
   ```
   GEMINI_API_KEY=your-api-key-here
   ```

3. The application will automatically load the `.env` file when it starts.

**Get your API key from:** https://makersuite.google.com/app/apikey

#### Option 2: Using Environment Variables

**On Linux/macOS:**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

**On Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

**On Windows (Command Prompt):**
```cmd
set GEMINI_API_KEY=your-api-key-here
```

**Note:** 
- The `.env` file is automatically ignored by git (see `.gitignore`)
- If the API key is not set, the application will run without the AI Assistant feature (using the simple rule-based fallback)

## Usage

### Web Application Features

#### Alert Analysis Tab
- **Predefined Scenarios**: Analyze pre-configured benign and malignant scenarios
- **Custom Alert Input**: Create and analyze custom alerts
- **Real-time Analysis**: All predictions use the trained model with SHAP explanations

#### Alert Details Tab
- View detailed analysis of any analyzed alert
- See classification results, explanations, and top contributing features
- View generated playbooks for malicious alerts
- Review AI assistant analysis

#### Reports Tab
- Browse all generated reports
- View comprehensive report details
- Download reports as JSON files

#### AI Assistant Tab
- Interactive chat interface
- Ask questions about alerts
- Get context-aware answers
- Suggested questions provided

### Python API Usage

#### Single Alert Prediction

```python
from ml_model import predict_alert

alert = {
    'Source_IP': '192.168.0.1',
    'Destination_IP': '10.0.0.3',
    'Protocol': 'TCP',
    'Packet_Length': 1500,
    'Duration': 2.5,
    'Source_Port': 80,
    'Destination_Port': 443,
    'Bytes_Sent': 1000,
    'Bytes_Received': 2000,
    'Flags': 'SYN',
    'Flow_Packets/s': 30.5,
    'Flow_Bytes/s': 1500.0,
    'Avg_Packet_Size': 512,
    'Total_Fwd_Packets': 25,
    'Total_Bwd_Packets': 30,
    'Fwd_Header_Length': 256,
    'Bwd_Header_Length': 256,
    'Sub_Flow_Fwd_Bytes': 800,
    'Sub_Flow_Bwd_Bytes': 1200,
    'Inbound': 1
}

result = predict_alert(alert)
print(f"Prediction: {result['label']}")  # 'Normal' or 'Malicious'
print(f"Probability: {result['probability']:.4f}")
```

#### Alert Explanation (XAI)

```python
from ml_model import explain_alert

result = explain_alert(alert, top_k=5)
print(f"Prediction: {result['label']}")
print(f"Probability: {result['probability']:.4f}")
print(f"\nExplanation: {result['explanation_text']}")
print("\nTop Contributing Features:")
for feat in result['top_features']:
    print(f"  - {feat['feature']}: {feat['value']}, contribution={feat['contribution']:.4f}")
```

#### Generate Playbook

```python
from playbook_generator import generate_playbook

playbook = generate_playbook(alert_data, prediction_result, explanation)
if playbook.get('playbook_required'):
    print(f"Threat Level: {playbook['threat_level']}")
    print(f"Priority: {playbook['priority']}")
    for step in playbook['steps']:
        print(f"\nStep {step['step_number']}: {step['title']}")
        for action in step['actions']:
            print(f"  - {action}")
```

#### Generate Report

```python
from report_generator import generate_report, save_report

report = generate_report(alert_data, prediction_result, explanation, playbook)
report_path = save_report(report)
print(f"Report saved to: {report_path}")
```

#### Use AI Assistant

```python
from ai_assistant import create_assistant

# Create Gemini assistant
assistant = create_assistant(
    use_llm=True,
    llm_provider="gemini",
    api_key="your-api-key",
    model="gemini-1.5-flash"
)

# Use simple rule-based assistant (no API key needed)
assistant = create_assistant(use_llm=False)

# Analyze alert
analysis = assistant.analyze_alert(alert_data, prediction, explanation, playbook)

# Answer questions
context = {
    'prediction': prediction,
    'explanation': explanation,
    'playbook': playbook
}
answer = assistant.answer_question("Why was this classified as malicious?", context)
```

## Output Formats

### Prediction Result

```python
{
    'prediction': 0 or 1,          # 0=Normal, 1=Malicious
    'probability': 0.0 to 1.0,     # Probability of being malicious
    'label': 'Normal' or 'Malicious'
}
```

### Explanation Result

```python
{
    'prediction': 0 or 1,
    'probability': 0.0 to 1.0,
    'label': 'Normal' or 'Malicious',
    'top_features': [
        {
            'feature': str,
            'value': Any,
            'contribution': float
        },
        ...
    ],
    'explanation_text': str
}
```

### Playbook Structure

```python
{
    'playbook_required': bool,
    'threat_level': 'CRITICAL' | 'HIGH' | 'MEDIUM',
    'priority': 'IMMEDIATE' | 'URGENT' | 'HIGH' | 'MEDIUM' | 'LOW',
    'steps': [
        {
            'step_number': int,
            'title': str,
            'actions': [str, ...],
            'estimated_time': str,
            'priority': str
        },
        ...
    ],
    'recommendations': [str, ...],
    'attack_indicators': [str, ...]
}
```

## Model Details

### Algorithm
- **Random Forest Classifier** with balanced class weights

### Parameters
- `n_estimators`: 200
- `max_depth`: 25
- `min_samples_split`: 10
- `min_samples_leaf`: 4
- `class_weight`: 'balanced' (handles class imbalance)
- `max_features`: 'sqrt'
- `bootstrap`: True
- `oob_score`: True

### Feature Engineering
- Label encoding for categorical features (Protocol, Flags, Source_IP, Destination_IP)
- Derived feature: `same_source_dest_ip`
- Missing values filled with median values

## Explainable AI (XAI)

The system uses **SHAP (SHapley Additive exPlanations)** values for explainability:

1. **SHAP TreeExplainer**: Optimized for Random Forest models
2. **Feature Contributions**: Positive values push toward "Malicious", negative toward "Normal"
3. **Top Features**: Ranked by absolute SHAP value
4. **Fallback**: Uses feature importances if SHAP unavailable

## AI Assistant

The AI Assistant module uses Google Gemini:

### Google Gemini
- Models: `gemini-1.5-flash` (default), `gemini-1.5-pro`, or other Gemini models
- Requires: `GEMINI_API_KEY` environment variable or API key parameter

### Simple Rule-Based Assistant
- No API key required
- Fallback when LLM is unavailable
- Basic rule-based analysis

## Limitations

1. **Class Imbalance**: Dataset has more malicious samples, causing bias toward malicious predictions
2. **Overfitting**: High training accuracy suggests possible overfitting
3. **Feature Engineering**: Limited domain-specific feature engineering
4. **Model Selection**: Only Random Forest implemented (no hyperparameter tuning)

## Future Improvements

- Implement SMOTE or other oversampling techniques
- Add cross-validation for robust evaluation
- Try alternative algorithms (XGBoost, Gradient Boosting, Neural Networks)
- Feature selection to remove redundant features
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Enhanced playbook templates
- Multi-language support for reports
- Integration with SIEM systems

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions, issues, and feature requests are welcome!

## Contact

For questions or support, please open an issue in the repository.
