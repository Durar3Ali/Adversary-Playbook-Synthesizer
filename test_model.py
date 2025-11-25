"""
Simple test script for the cyber security alert classification model
You can modify this script to test your own alerts
"""

from ml_model import predict_alert
import json

# Test Case 1: Normal Traffic Alert
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
print(f"Prediction Code: {result['prediction']} (0=Normal, 1=Malicious)")
print()

# Test Case 2: Suspicious Traffic Alert (DDoS-like)
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
    'Flow_Packets/s': 100.0,  # High packet rate
    'Flow_Bytes/s': 5000.0,   # High byte rate
    'Avg_Packet_Size': 100,
    'Total_Fwd_Packets': 100,  # Many forward packets
    'Total_Bwd_Packets': 0,    # No backward packets (suspicious)
    'Fwd_Header_Length': 128,
    'Bwd_Header_Length': 0,
    'Sub_Flow_Fwd_Bytes': 10000,
    'Sub_Flow_Bwd_Bytes': 0,
    'Inbound': 1
}

result = predict_alert(suspicious_alert)
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['probability']:.2%}")
print(f"Prediction Code: {result['prediction']} (0=Normal, 1=Malicious)")
print()

# Test Case 3: Ransomware-like Traffic
print("="*60)
print("TEST CASE 3: Possible Ransomware Traffic")
print("="*60)

ransomware_alert = {
    'Source_IP': '172.16.0.5',
    'Destination_IP': '10.0.0.3',
    'Protocol': 'TCP',
    'Packet_Length': 1800,
    'Duration': 3.5,
    'Source_Port': 443,
    'Destination_Port': 443,
    'Bytes_Sent': 2000,
    'Bytes_Received': 1500,
    'Flags': 'FIN',
    'Flow_Packets/s': 20.0,
    'Flow_Bytes/s': 2000.0,
    'Avg_Packet_Size': 1024,
    'Total_Fwd_Packets': 40,
    'Total_Bwd_Packets': 35,
    'Fwd_Header_Length': 256,
    'Bwd_Header_Length': 256,
    'Sub_Flow_Fwd_Bytes': 1500,
    'Sub_Flow_Bwd_Bytes': 1200,
    'Inbound': 0
}

result = predict_alert(ransomware_alert)
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['probability']:.2%}")
print(f"Prediction Code: {result['prediction']} (0=Normal, 1=Malicious)")
print()

print("="*60)
print("Testing Complete!")
print("="*60)
print("\nTo test your own alerts, modify the alert dictionaries above")
print("or create new ones with the same structure.")

