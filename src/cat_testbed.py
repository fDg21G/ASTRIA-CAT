import time
import random
import math
import sys

# ==========================================
# FILE: cat_testbed.py
# SYSTEM: ASTRIA-CAT Edge Processor
# MISSION: Real-time Turbulence Prediction
# ==========================================

print("\n" + "="*50)
print("   ASTRIA-CAT | SMART SKIN SENSOR ARRAY   ")
print("   STATUS: ONLINE | MODE: FLIGHT OPS      ")
print("="*50 + "\n")

# Simulation Parameters
SAMPLING_RATE = 20  # Hz
CALIBRATION_TIME = 2

def visualize_bar(level, threshold=0.6):
    """Draws a visual bar chart for turbulence intensity"""
    bar_len = int(level * 20)
    color = "\033[92m" # Green
    if level > 0.4: color = "\033[93m" # Yellow
    if level > threshold: color = "\033[91m" # Red
    
    bar = "█" * bar_len
    return f"{color}[{bar:<20}]\033[0m"

print("[INIT] Calibrating MEMS Pressure Sensors...", end="")
time.sleep(1.5)
print(" DONE.")
print("[INIT] Loading 1D-CNN Model (Quantized)...", end="")
time.sleep(1.0)
print(" DONE.\n")
print("-" * 50)
print(f"{'TIMESTAMP':<10} | {'ALT (ft)':<10} | {'G-FORCE':<10} | {'CAT PROB':<10} | {'STATUS'}")
print("-" * 50)

# Start Simulation Loop
altitude = 35000
current_time = 0.0
turbulence_active = False

try:
    while current_time < 30: # Run for 30 seconds
        # 1. Simulate Sensor Data (Normal Flight)
        g_force = 1.0 + random.uniform(-0.05, 0.05)
        cat_probability = random.uniform(0.01, 0.15)
        status = "\033[92mSMOOTH\033[0m"
        
        # 2. INJECT TURBULENCE EVENT (At T+12 seconds)
        if 12.0 < current_time < 20.0:
            turbulence_active = True
            # Simulate aerodynamic instability (Precursor)
            g_force += random.uniform(-0.8, 0.8) * math.sin(current_time)
            # AI Model Confidence spikes
            cat_probability = 0.85 + random.uniform(-0.05, 0.10)
            status = "\033[91m!!! WARNING !!!\033[0m"
        
        # 3. Display Telemetry
        vis = visualize_bar(cat_probability)
        print(f"T+{current_time:04.1f}s    | {altitude:<10} | {g_force:.3f} G    | {cat_probability:.2f} {vis} | {status}")
        
        # 4. Trigger Alert
        if cat_probability > 0.80:
             print(f"       >>> \033[91mALERT: CLEAR AIR TURBULENCE PREDICTED!\033[0m")
             print(f"       >>> ACTION: ENGAGE AUTOPILOT STABILIZATION")
        
        current_time += 0.5
        time.sleep(0.3) # Fast update for realism

except KeyboardInterrupt:
    print("\n[STOP] Flight Simulation Halted.")
