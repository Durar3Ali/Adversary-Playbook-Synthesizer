# Cyber Security Alert Classification Model

This machine learning model predicts whether a cyber security alert is **Normal** (0) or **Malicious** (1) based on network traffic features.

## Overview

The model uses a Random Forest classifier trained on the `cyberfeddefender_dataset.csv` dataset, which contains network flow features such as packet lengths, protocols, IP addresses, ports, and traffic statistics.

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

## Model Performance

- **Test Accuracy**: ~74% (with default threshold)
- **Training Accuracy**: ~100% (potential overfitting)
- **ROC AUC Score**: 0.5285

**Note**: The model shows bias toward predicting malicious alerts due to class imbalance (1090 malicious vs 340 normal). Further tuning may be needed for better Normal class detection.

## Files

- `ml_model.py` - Main training script and prediction function
- `predict_example.py` - Example usage script
- `cyber_alert_model.pkl` - Trained model (created after running ml_model.py)
- `predictions.csv` - Test set predictions (created after running ml_model.py)

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
- Save test predictions to `predictions.csv`

### 2. Make Predictions

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

#### Batch Predictions from CSV

See `predict_example.py` for a complete example of batch predictions.

```bash
python predict_example.py
```

## Requirements

Install required packages:

```bash
pip install pandas numpy scikit-learn
```

## Model Details

- **Algorithm**: Random Forest Classifier
- **Parameters**:
  - n_estimators: 200
  - max_depth: 25
  - min_samples_split: 10
  - min_samples_leaf: 4
  - class_weight: 'balanced' (handles class imbalance)
- **Feature Engineering**: 
  - Label encoding for categorical features
  - Derived feature: `same_source_dest_ip`

## Output Format

The `predict_alert()` function returns a dictionary:

```python
{
    'prediction': 0 or 1,          # 0=Normal, 1=Malicious
    'probability': 0.0 to 1.0,      # Probability of being malicious
    'label': 'Normal' or 'Malicious'
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

