"""
ASTRIA-CAT: Synthetic Turbulence Generator & Feature Extractor
This module simulates the core data pipeline for clear-air turbulence prediction.
"""

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

def generate_turbulence_signal(duration_sec=10, sampling_hz=200):
    """
    Generates a synthetic pressure signal mimicking aircraft skin sensors.
    Combines low-frequency Kelvin-Helmholtz waves with sensor noise.
    """
    print("[ASTRIA-CAT] Generating synthetic turbulence dataset...")
    
    t = np.linspace(0, duration_sec, duration_sec * sampling_hz)
    
    # 1. Low-frequency KHI wave (precursor signal, 2-5 Hz)
    khi_wave = 0.5 * np.sin(2 * np.pi * 3.5 * t) * np.exp(-0.1 * t)
    
    # 2. High-frequency turbulent bursts (actual turbulence event)
    turbulence_event = np.zeros_like(t)
    event_start = int(5 * sampling_hz)  # Event at 5 seconds
    event_duration = int(1.5 * sampling_hz)
    turbulence_event[event_start:event_start+event_duration] = (
        1.2 * np.random.randn(event_duration) * np.hanning(event_duration)
    )
    
    # 3. Sensor and boundary layer noise
    noise = 0.15 * np.random.randn(len(t))
    
    # Combined signal
    raw_signal = khi_wave + turbulence_event + noise
    
    # Simulate sensor calibration (high-pass filter)
    b, a = signal.butter(3, 1.0, 'highpass', fs=sampling_hz)
    calibrated_signal = signal.filtfilt(b, a, raw_signal)
    
    print(f"   • Generated {len(t)} samples at {sampling_hz} Hz")
    print(f"   • Turbulence event injected at t = 5.0 s")
    
    return t, calibrated_signal, (event_start / sampling_hz)

def extract_features(time_vector, signal_vector, window_size=100):
    """
    Extracts time-domain features used for 1D-CNN training.
    """
    features = []
    labels = []
    
    for i in range(0, len(signal_vector) - window_size, window_size//2):
        window = signal_vector[i:i+window_size]
        
        # Feature engineering (as described in the dissertation)
        f_mean = np.mean(window)
        f_std = np.std(window)
        f_rms = np.sqrt(np.mean(window**2))
        f_peak = np.max(np.abs(window))
        
        # Spectral feature: dominant frequency in 2-5 Hz band
        freqs, psd = signal.welch(window, fs=200, nperseg=64)
        mask = (freqs >= 2) & (freqs <= 5)
        if np.any(mask):
            f_dominant = freqs[mask][np.argmax(psd[mask])]
        else:
            f_dominant = 0.0
        
        features.append([f_mean, f_std, f_rms, f_peak, f_dominant])
        
        # Simple label: 1 if window contains the main event
        labels.append(1 if (5.0 <= time_vector[i] <= 6.5) else 0)
    
    print(f"   • Extracted {len(features)} feature windows")
    return np.array(features), np.array(labels)

def visualize_pipeline(t, sig, event_time):
    """
    Creates a publication-ready figure of the signal processing pipeline.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    # Raw signal plot
    axes[0].plot(t, sig, 'b-', linewidth=0.8, alpha=0.7, label='Simulated Sensor Signal')
    axes[0].axvspan(event_time, event_time+1.5, color='red', alpha=0.2, label='Turbulence Event')
    axes[0].set_ylabel('Pressure (Pa)')
    axes[0].set_title('ASTRIA-CAT: Simulated Aircraft Skin Pressure Signal')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Spectrogram plot
    axes[1].specgram(sig, Fs=200, NFFT=256, noverlap=128, cmap='viridis')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title('Spectrogram - Energy concentration in 2-5 Hz band (KHI precursor)')
    axes[1].axhline(y=3.5, color='white', linestyle='--', linewidth=1, label='KHI Frequency (3.5 Hz)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('results/turbulence_simulation.png', dpi=150, bbox_inches='tight')
    print("   • Visualization saved to 'results/turbulence_simulation.png'")
    return fig

# ===== Execute the pipeline =====
if __name__ == "__main__":
    # 1. Generate data
    time, signal_data, event_start = generate_turbulence_signal()
    
    # 2. Extract features (for ML model)
    X_features, y_labels = extract_features(time, signal_data)
    
    # 3. Create visualization
    fig = visualize_pipeline(time, signal_data, event_start)
    
    # 4. Save a sample of the features
    sample_df = pd.DataFrame(
        X_features[:50],  # Save first 50 samples
        columns=['Mean', 'Std_Dev', 'RMS', 'Peak_Amplitude', 'Dominant_Freq_2_5Hz']
    )
    sample_df['Label'] = y_labels[:50]
    sample_df.to_csv('data/sample_features.csv', index=False)
    
    print("\n" + "="*60)
    print("[ASTRIA-CAT] Synthetic data pipeline executed successfully!")
    print("="*60)
    print("Outputs generated:")
    print("   • 'src/synthetic_turbulence.py' - Core simulation module")
    print("   • 'data/sample_features.csv'    - Feature dataset sample")
    print("   • 'results/turbulence_simulation.png' - Analysis figure")
    print("\nNext step: Train the 1D-CNN model (see 'train_model.py')")
