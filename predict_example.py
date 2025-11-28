"""
Example script showing how to use the trained model to predict alerts
"""

from ml_model import predict_alert
import pandas as pd

# Test Case 1: Normal Traffic
print("="*60)
print("TEST CASE 1: Normal Traffic")
print("="*60)

normal_alert = {
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

result = predict_alert(normal_alert)
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['probability']:.2%}")
print(f"Code: {result['prediction']} (0=Normal, 1=Malicious)\n")

# Test Case 2: Suspicious Traffic (DDoS-like)
print("="*60)
print("TEST CASE 2: Suspicious Traffic (Possible DDoS)")
print("="*60)

suspicious_alert = {
    'Source_IP': '192.168.0.1',
    'Destination_IP': '10.0.0.1',
    'Protocol': 'UDP',
    'Packet_Length': 100,
    'Duration': 0.5,
    'Source_Port': 53,
    'Destination_Port': 53,
    'Bytes_Sent': 50,
    'Bytes_Received': 0,
    'Flags': 'SYN',
    'Flow_Packets/s': 100.0,
    'Flow_Bytes/s': 5000.0,
    'Avg_Packet_Size': 100,
    'Total_Fwd_Packets': 100,
    'Total_Bwd_Packets': 0,
    'Fwd_Header_Length': 128,
    'Bwd_Header_Length': 0,
    'Sub_Flow_Fwd_Bytes': 10000,
    'Sub_Flow_Bwd_Bytes': 0,
    'Inbound': 1
}

result = predict_alert(suspicious_alert)
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['probability']:.2%}")
print(f"Code: {result['prediction']} (0=Normal, 1=Malicious)\n")

# Test Case 3: Batch Prediction from CSV
# This shows how to predict multiple alerts at once from a CSV file
print("="*60)
print("TEST CASE 3: Batch Prediction from CSV")
print("="*60)
print("This example reads real alerts from the dataset CSV file,")
print("makes predictions on them, and compares predictions to actual labels.\n")

try:
    # Load the first 5 rows from the dataset CSV file
    test_df = pd.read_csv('cyberfeddefender_dataset.csv').head(5)
    
    # Remove columns that are labels/answers (we want to predict these, not use them)
    # We keep only the feature columns needed for prediction
    test_df_for_prediction = test_df.drop(columns=['Timestamp', 'Attack_Type', 'Label'], errors='ignore')
    
    print(f"Loaded {len(test_df_for_prediction)} alerts from CSV file")
    print("Making predictions on each alert...\n")
    
    # Predict each alert and store results
    results = []
    for idx, row in test_df_for_prediction.iterrows():
        # Convert the row to a dictionary and make a prediction
        result = predict_alert(row.to_dict())
        
        # Convert actual Attack_Type to Normal/Malicious (to match model output)
        actual_attack_type = test_df.iloc[idx]['Attack_Type']
        actual_label = 'Malicious' if actual_attack_type != 'Normal' else 'Normal'
        
        # Store both the prediction and the actual label (for comparison)
        results.append({
            'index': idx,
            'actual': actual_label,                          # Normal or Malicious
            'predicted': result['label'],                     # What the model predicted
            'probability': result['probability']              # Confidence level
        })
    
    # Display the results in a table
    results_df = pd.DataFrame(results)
    print("Comparison of Predictions vs Actual Labels:")
    print(results_df.to_string(index=False))
    
    # Calculate how many predictions were correct
    # Both actual and predicted are now "Normal" or "Malicious", so we can compare directly
    accuracy = (results_df['actual'] == results_df['predicted']).mean()
    print(f"\nModel Accuracy on these 5 alerts: {accuracy:.2%}")
    print("(This shows how well the model performs on real data)")
    
except Exception as e:
    print(f"Error: {e}")
    print("Make sure 'cyberfeddefender_dataset.csv' exists in the current directory")

print("\n" + "="*60)
print("Testing Complete!")
print("="*60)

