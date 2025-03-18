import time
import joblib
import numpy as np
import scapy.all as scapy
import xgboost as xgb
import pandas as pd
import self_heal
import sys
import os

# Ensure Python finds the 'security' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from security.encryption import encrypt_alert  # Now it should work!

# Load trained model
model = xgb.XGBClassifier()
model.load_model("models/xgboost_intrusion_detection.json")
print("✅ Model Loaded Successfully!")

# Feature extraction from network packets
def extract_features(packet):
    return {
        "duration": 0,  # Placeholder (packet timing can be added)
        "protocol_type": packet.proto,
        "service": 1 if packet.haslayer(scapy.TCP) else 2 if packet.haslayer(scapy.UDP) else 3,
        "flag": 1 if packet.haslayer(scapy.IP) else 0,
        "src_bytes": len(packet),
        "dst_bytes": len(packet.payload),
        "land": 0,  # Placeholder (requires TCP session tracking)
        "wrong_fragment": 0,  # Placeholder (requires fragmentation analysis)
        "urgent": 0,  # Placeholder (check TCP urgent flag)
        "hot": 0,  # Placeholder
        "num_failed_logins": 0,
        "logged_in": 0,  
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "count": 0,  # Placeholder (requires windowed packet counting)
        "srv_count": 0,  
        "serror_rate": 0,  
        "srv_serror_rate": 0,  
        "rerror_rate": 0,  
        "srv_rerror_rate": 0,  
        "same_srv_rate": 0,  
        "diff_srv_rate": 0,  
        "srv_diff_host_rate": 0,  
        "dst_host_count": 0,  
        "dst_host_srv_count": 0,  
        "dst_host_same_srv_rate": 0,  
        "dst_host_diff_srv_rate": 0,  
        "dst_host_same_src_port_rate": 0,  
        "dst_host_srv_diff_host_rate": 0,  
        "dst_host_serror_rate": 0,  
        "dst_host_srv_serror_rate": 0,  
        "dst_host_rerror_rate": 0,  
        "dst_host_srv_rerror_rate": 0
    }


# Packet processing
def process_packet(packet):
    print(f"📡 Captured Packet: {packet.summary()}")  # Print every packet sniffed

    features = extract_features(packet)
 
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]

    if prediction == 1:
        print("🚨 Malicious Packet Detected!")

        alert_msg = {
            "attack_id": np.random.randint(1000, 9999),
            "type": "Malicious Network Activity",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "confidence": np.random.uniform(0.9, 1.0),
        }
        encrypted_alert = encrypt_alert(str(alert_msg))
        print(f"🚨 Attack Detected! Encrypted Alert Sent: {encrypted_alert[:50]}...")
        
        self_heal.isolate_threat(packet)

    
# Start monitoring
print("🔍 Monitoring Network Traffic in Real Time...")
scapy.sniff(iface="Wi-Fi", filter="ip", prn=process_packet, store=0)

