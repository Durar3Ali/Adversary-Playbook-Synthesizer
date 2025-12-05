# Cyber Security Alert Classification Model

This machine learning model predicts whether a cyber security alert is **Normal** (0) or **Malicious** (1) based on network traffic features. The model includes **Explainable AI (XAI)** capabilities to provide insights into why each prediction was made.

## Overview

The model uses a Random Forest classifier trained on the `cyberfeddefender_dataset.csv` dataset, which contains network flow features such as packet lengths, protocols, IP addresses, ports, and traffic statistics. The system provides both predictions and explanations using SHAP (SHapley Additive exPlanations) values.

## Dataset

The dataset contains 1,430 network alerts with the following attack types:
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

**Note**: The model shows bias toward predicting malicious alerts due to class imbalance (1090 malicious vs 340 normal). Further tuning may be needed for better Normal class detection.

## Files

- `ml_model.py` - Main training script, prediction function (`predict_alert`), and explanation function (`explain_alert`)
- `app.py` - Streamlit web application with cyberpunk HUD theme
- `cyber_alert_model.pkl` - Trained model (created after running ml_model.py)
- `cyberfeddefender_dataset.csv` - Training dataset
- `requirements.txt` - Python dependencies

## Requirements

Install required packages:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn shap streamlit
```

## Usage

### 1. Train the Model

Run the training script:

```bash
python ml_model.py
```

This will:
- Load and preprocess the dataset
- Train a Random Forest classifier
- Evaluate the model performance
- Save the trained model to `cyber_alert_model.pkl`

### 2. Run the Streamlit Web Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

#### Features

- **Predefined Scenarios**: Click buttons to analyze pre-configured alert scenarios using the real model
- **Custom Alert Input**: Fill out a form to analyze custom alerts with the real model
- **Real-time Analysis**: All predictions use the actual trained Random Forest model with SHAP explanations
- **Alert Stream**: View history of analyzed alerts
- **Detailed Explanations**: See top contributing features and explanation text for each prediction

### 3. Make Predictions (Python API)

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
print(f"Probability: {result['probability']:.4f}")  # Probability of being malicious
print(f"Prediction Code: {result['prediction']}")  # 0=Normal, 1=Malicious
```

#### Single Alert Prediction with Explanation (XAI)

```python
from ml_model import explain_alert

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

result = explain_alert(alert, top_k=5)
print(f"Prediction: {result['label']}")
print(f"Probability: {result['probability']:.4f}")
print(f"\nExplanation: {result['explanation_text']}")
print("\nTop Contributing Features:")
for feat in result['top_features']:
    print(f"  - {feat['feature']}: value={feat['value']}, contribution={feat['contribution']:.4f}")
```

**Output Format for `explain_alert()`:**

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

## Explainable AI (XAI)

The model includes explainability features using **SHAP (SHapley Additive exPlanations)** values. For each prediction, the system:

1. **Computes SHAP values** using TreeExplainer (optimized for Random Forest models)
2. **Identifies top contributing features** that most influence the prediction
3. **Generates human-readable explanations** describing why the alert was classified as Normal or Malicious

### How it works

- **SHAP values** represent the contribution of each feature to the prediction
  - Positive values push toward "Malicious" classification
  - Negative values push toward "Normal" classification
  - Larger absolute values indicate stronger influence

- **Top features** are ranked by absolute SHAP value, showing which features had the most impact

- **Explanation text** is automatically generated from the top contributing features and their values

### Fallback behavior

If SHAP is not available or fails, the system falls back to using feature importances from the Random Forest model combined with the instance's feature values to compute contributions.

## Model Details

- **Algorithm**: Random Forest Classifier
- **Parameters**:
  - n_estimators: 200
  - max_depth: 25
  - min_samples_split: 10
  - min_samples_leaf: 4
  - class_weight: 'balanced' (handles class imbalance)
- **Feature Engineering**: 
  - Label encoding for categorical features (Protocol, Flags, Source_IP, Destination_IP)
  - Derived feature: `same_source_dest_ip`
  - Missing values filled with median values

## Output Format

### `predict_alert()` returns:

```python
{
    'prediction': 0 or 1,          # 0=Normal, 1=Malicious
    'probability': 0.0 to 1.0,      # Probability of being malicious
    'label': 'Normal' or 'Malicious'
}
```

### `explain_alert()` returns:

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

## Limitations

1. **Class Imbalance**: The dataset has more malicious samples than normal, causing the model to be biased toward predicting malicious alerts.

2. **Overfitting**: The model achieves 100% training accuracy, suggesting possible overfitting. Consider:
   - Using cross-validation
   - Adding more regularization
   - Collecting more training data

3. **Feature Engineering**: Limited feature engineering was performed. Consider:
   - More domain-specific features
   - Feature selection
   - Different encoding schemes for IP addresses

## Future Improvements

- Implement SMOTE or other oversampling techniques for better class balance
- Add cross-validation for more robust evaluation
- Try different algorithms (XGBoost, Gradient Boosting, Neural Networks)
- Feature selection to remove redundant features
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
