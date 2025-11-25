"""
Example script showing how to use the trained model to predict alerts
"""

from ml_model import predict_alert
import pandas as pd

# Example 1: Predict a single alert
print("="*60)
print("EXAMPLE 1: Single Alert Prediction")
print("="*60)

# Create an example alert (you would get this from your security system)
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
print(f"\nAlert Details:")
print(f"  Source IP: {alert['Source_IP']}")
print(f"  Destination IP: {alert['Destination_IP']}")
print(f"  Protocol: {alert['Protocol']}")
print(f"\nPrediction Result:")
print(f"  Label: {result['label']}")
print(f"  Probability: {result['probability']:.4f}")
print(f"  Prediction Code: {result['prediction']} (0=Normal, 1=Malicious)")

# Example 2: Predict multiple alerts from CSV
print("\n" + "="*60)
print("EXAMPLE 2: Batch Prediction from CSV")
print("="*60)

try:
    # Load test data (using the original dataset as an example)
    test_df = pd.read_csv('cyberfeddefender_dataset.csv').head(5)
    
    # Drop target columns for prediction
    test_df_for_prediction = test_df.drop(columns=['Timestamp', 'Attack_Type', 'Label'], errors='ignore')
    
    print(f"\nPredicting {len(test_df_for_prediction)} alerts...")
    
    results = []
    for idx, row in test_df_for_prediction.iterrows():
        result = predict_alert(row.to_dict())
        results.append({
            'index': idx,
            'actual': test_df.iloc[idx]['Attack_Type'],
            'predicted': result['label'],
            'probability': result['probability']
        })
    
    results_df = pd.DataFrame(results)
    print("\nPrediction Results:")
    print(results_df.to_string(index=False))
    
    # Calculate accuracy if we know the actual labels
    results_df['actual_is_malicious'] = (results_df['actual'] != 'Normal').astype(int)
    results_df['predicted_is_malicious'] = (results_df['predicted'] == 'Malicious').astype(int)
    accuracy = (results_df['actual_is_malicious'] == results_df['predicted_is_malicious']).mean()
    print(f"\nAccuracy on sample: {accuracy:.2%}")
    
except Exception as e:
    print(f"Could not load test data: {e}")
    print("Make sure 'cyberfeddefender_dataset.csv' exists in the current directory")

print("\n" + "="*60)
print("Prediction Examples Complete!")
print("="*60)

