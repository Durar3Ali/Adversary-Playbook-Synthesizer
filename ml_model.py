"""
Machine Learning Model for Cyber Security Alert Classification
Predicts if an alert is Normal (0) or Malicious (1)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')


def predict_alert(alert_data, model_path='cyber_alert_model.pkl', threshold=None):
    """
    Predict if an alert is normal or malicious.
    
    Parameters:
    -----------
    alert_data : dict or pandas.DataFrame
        Alert data with the same features as training data
    model_path : str
        Path to the saved model file
    threshold : float, optional
        Probability threshold for classification. If None, uses optimal threshold.
    
    Returns:
    --------
    dict : Prediction result with 'prediction' (0=Normal, 1=Malicious) and 'probability'
    """
    # Load model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    label_encoders = model_data['label_encoders']
    feature_columns = model_data['feature_columns']
    
    if threshold is None:
        threshold = model_data.get('optimal_threshold', 0.5)
    
    # Convert to DataFrame if dict
    if isinstance(alert_data, dict):
        alert_df = pd.DataFrame([alert_data])
    else:
        alert_df = alert_data.copy()
    
    # Create derived features (same as training)
    if 'Source_IP' in alert_df.columns and 'Destination_IP' in alert_df.columns:
        alert_df['same_source_dest_ip'] = (alert_df['Source_IP'] == alert_df['Destination_IP']).astype(int)
    
    # Preprocess - only select features that exist
    available_features = [f for f in feature_columns if f in alert_df.columns]
    missing_features = [f for f in feature_columns if f not in alert_df.columns]
    
    if missing_features:
        print(f"Warning: Missing features {missing_features}. They will be filled with 0.")
    
    X_pred = pd.DataFrame(index=alert_df.index)
    for feature in feature_columns:
        if feature in alert_df.columns:
            X_pred[feature] = alert_df[feature]
        else:
            X_pred[feature] = 0  # Fill missing features with 0
    
    # Encode categorical features
    for col in label_encoders:
        if col in X_pred.columns:
            # Handle unseen categories
            le = label_encoders[col]
            X_pred[col] = X_pred[col].astype(str)
            X_pred[col] = X_pred[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # Fill missing values
    X_pred = X_pred.fillna(X_pred.median(numeric_only=True))
    
    # Predict
    proba = model.predict_proba(X_pred)[:, 1]
    prediction = (proba >= threshold).astype(int)
    
    return {
        'prediction': prediction[0] if len(prediction) == 1 else prediction,
        'probability': proba[0] if len(proba) == 1 else proba,
        'label': 'Malicious' if (prediction[0] if len(prediction) == 1 else prediction[0]) == 1 else 'Normal'
    }


if __name__ == '__main__':
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('cyberfeddefender_dataset.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"\nDataset columns: {df.columns.tolist()}")

    # Display initial statistics
    print(f"\nAttack Type distribution:")
    print(df['Attack_Type'].value_counts())

    # Create binary target: 1 for malicious, 0 for normal
    df['is_malicious'] = (df['Attack_Type'] != 'Normal').astype(int)
    print(f"\nTarget distribution:")
    print(df['is_malicious'].value_counts())

    # Feature engineering
    print("\nPreprocessing features...")

    # Drop non-predictive columns
    features_to_drop = ['Timestamp', 'Attack_Type', 'Label']
    X = df.drop(columns=features_to_drop + ['is_malicious'], errors='ignore')

    # Handle categorical features
    categorical_cols = ['Protocol', 'Flags', 'Source_IP', 'Destination_IP']

    # Create label encoders for categorical features
    label_encoders = {}
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    # Handle IP addresses - extract numerical features
    # Convert IP addresses to numerical representation
    if 'Source_IP' in X.columns and 'Destination_IP' in X.columns:
        # IP addresses are already encoded, but we can create additional features
        # Check if same source/dest IP
        X['same_source_dest_ip'] = (df['Source_IP'] == df['Destination_IP']).astype(int)

    # Target variable
    y = df['is_malicious']

    # Check for missing values
    print(f"\nMissing values per column:")
    print(X.isnull().sum()[X.isnull().sum() > 0])

    # Fill any missing values
    X = X.fillna(X.median(numeric_only=True))

    # Split the data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # Train Random Forest Classifier with improved parameters for class imbalance
    print("\nTraining Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',  # Handle class imbalance
        max_features='sqrt',  # Better generalization
        bootstrap=True,
        oob_score=True
    )

    rf_model.fit(X_train, y_train)
    print(f"Out-of-bag score: {rf_model.oob_score_:.4f}")

    # Predictions
    print("\nMaking predictions...")
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)

    # Prediction probabilities for threshold tuning
    y_test_proba = rf_model.predict_proba(X_test)[:, 1]

    # Try different thresholds to improve Normal class detection
    print("\n--- ROC Analysis ---")
    auc_score = roc_auc_score(y_test, y_test_proba)
    print(f"ROC AUC Score: {auc_score:.4f}")

    # Find optimal threshold (balance precision and recall)
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.4f}")

    # Use optimal threshold for predictions
    y_test_pred_optimal = (y_test_proba >= optimal_threshold).astype(int)

    # Evaluate the model
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)

    print("\n--- Training Set Performance ---")
    print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_train, y_train_pred, 
                              target_names=['Normal', 'Malicious']))

    print("\n--- Test Set Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, 
                              target_names=['Normal', 'Malicious']))

    print("\nConfusion Matrix (Test Set - Default Threshold):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    print(f"\nTrue Negatives (Normal correctly predicted): {cm[0][0]}")
    print(f"False Positives (Normal predicted as Malicious): {cm[0][1]}")
    print(f"False Negatives (Malicious predicted as Normal): {cm[1][0]}")
    print(f"True Positives (Malicious correctly predicted): {cm[1][1]}")

    print("\n--- Test Set Performance (Optimal Threshold) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred_optimal):.4f}")
    print("\nClassification Report (Optimal Threshold):")
    print(classification_report(y_test, y_test_pred_optimal, 
                              target_names=['Normal', 'Malicious']))

    print("\nConfusion Matrix (Test Set - Optimal Threshold):")
    cm_optimal = confusion_matrix(y_test, y_test_pred_optimal)
    print(cm_optimal)
    print(f"\nTrue Negatives (Normal correctly predicted): {cm_optimal[0][0]}")
    print(f"False Positives (Normal predicted as Malicious): {cm_optimal[0][1]}")
    print(f"False Negatives (Malicious predicted as Normal): {cm_optimal[1][0]}")
    print(f"True Positives (Malicious correctly predicted): {cm_optimal[1][1]}")

    # Feature importance
    print("\n" + "="*60)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("="*60)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(feature_importance.head(15).to_string(index=False))

    # Save the model
    print("\nSaving model to 'cyber_alert_model.pkl'...")
    with open('cyber_alert_model.pkl', 'wb') as f:
        pickle.dump({
            'model': rf_model,
            'label_encoders': label_encoders,
            'feature_columns': X.columns.tolist(),
            'categorical_columns': categorical_cols,
            'optimal_threshold': optimal_threshold
        }, f)

    print("Model saved successfully!")

    # Save predictions for analysis
    predictions_df = X_test.copy()
    predictions_df['actual'] = y_test.values
    predictions_df['predicted'] = y_test_pred
    predictions_df.to_csv('predictions.csv', index=False)
    print("Predictions saved to 'predictions.csv'")

    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE!")
    print("="*60)
    
    print("\n" + "="*60)
    print("USAGE EXAMPLE")
    print("="*60)
    print("""
To predict a new alert, use:

from ml_model import predict_alert

alert = {
    'Source_IP': '192.168.0.1',
    'Destination_IP': '10.0.0.1',
    'Protocol': 'TCP',
    'Packet_Length': 1000,
    'Duration': 2.5,
    # ... other features
}

result = predict_alert(alert)
print(f"Prediction: {result['label']}")
print(f"Probability: {result['probability']:.4f}")
""")
