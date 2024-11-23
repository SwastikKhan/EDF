import streamlit as st
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import tempfile
import io
from pathlib import Path

def bandpass_filter(data, fs, lowcut=0.5, highcut=40, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def pan_tompkins_r_detection(ecg_signal, fs):
    # 1. Bandpass filter
    filtered_ecg = bandpass_filter(ecg_signal, fs)
    
    # 2. Differentiation
    diff_signal = np.diff(filtered_ecg)
    
    # 3. Squaring
    squared_signal = diff_signal ** 2
    
    # 4. Moving average
    window_size = int(0.150 * fs)  # 150 ms window
    moving_avg = np.convolve(squared_signal, np.ones(window_size) / window_size, mode='same')
    
    # 5. Peak detection
    peaks, _ = find_peaks(moving_avg, distance=0.6 * fs)
    return peaks, moving_avg

def get_plot_as_image():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

def preprocess_eeg_data(uploaded_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Load the EEG data
    raw = mne.io.read_raw_edf(tmp_path, preload=True)
    
    # Apply bandpass filter
    raw.filter(l_freq=0.5, h_freq=40, picks='eeg')
    
    # Generate and store plots
    plots = {}
    
    # Original plots
    raw.plot(duration=30, n_channels=30, scalings='auto', show=False)
    plots['original_graph'] = get_plot_as_image()
    
    raw.plot(title="Raw Data", show=False)
    plots['simplified_original_graph'] = get_plot_as_image()
    
    # Perform ICA
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(raw)
    
    # Automatically exclude components
    ica.exclude = [0, 1]
    raw_clean = ica.apply(raw)
    
    # Cleaned plots
    raw_clean.plot(duration=30, n_channels=30, scalings='auto', show=False)
    plots['cleaned_graph'] = get_plot_as_image()
    
    raw_clean.plot(title="Cleaned Data", show=False)
    plots['simplified_cleaned_graph'] = get_plot_as_image()
    
    # Clean up
    Path(tmp_path).unlink()
    
    return raw_clean, ica.exclude, plots

def process_ecg_data(uploaded_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Load the dataset
    raw = mne.io.read_raw_edf(tmp_path, preload=True)
    fs = raw.info['sfreq']
    channel_names = raw.ch_names
    
    results = {}
    
    for channel_name in channel_names:
        data = raw.get_data(picks=channel_name)[0]
        r_peaks, processed_signal = pan_tompkins_r_detection(data, fs)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(data, label=f"{channel_name} - Original ECG")
        plt.scatter(r_peaks, data[r_peaks], color='red', label="R-peaks", zorder=5)
        plt.title(f"{channel_name} - ECG Signal with R-peaks")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(processed_signal, label="Processed Signal (Pan-Tompkins)")
        plt.title(f"{channel_name} - Processed ECG Signal")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.legend()
        
        plt.tight_layout()
        
        results[channel_name] = {
            'plot': get_plot_as_image(),
            'r_peaks': r_peaks.tolist()
        }
        
    # Clean up
    Path(tmp_path).unlink()
    
    return results

def display_ica_results():
    st.title("EEG Data Preprocessing")
    
    uploaded_file = st.file_uploader("Choose an EEG file (EDF format)", type="edf")
    
    if uploaded_file is not None:
        st.info("Processing EEG data...")
        
        raw_clean, excluded_components, plots = preprocess_eeg_data(uploaded_file)
        
        st.header("ICA Artifact Removal Summary")
        st.write(f"Number of ICA components removed: {len(excluded_components)}")
        st.write(f"Removed ICA components: {excluded_components}")
        
        st.subheader("Original EEG Data")
        st.image(plots['original_graph'])
        
        st.subheader("Simplified Original EEG Data")
        st.image(plots['simplified_original_graph'])
        
        st.subheader("Cleaned EEG Data (Artifacts Removed)")
        st.image(plots['cleaned_graph'])
        
        st.subheader("Simplified Cleaned EEG Data")
        st.image(plots['simplified_cleaned_graph'])

def display_ecg_analysis():
    st.title("ECG Analysis and R-Peak Detection")
    
    uploaded_file = st.file_uploader("Choose an EEG file (EDF format)", type="edf")
    
    if uploaded_file is not None:
        st.info("Processing ECG data and detecting R-peaks...")
        
        results = process_ecg_data(uploaded_file)
        
        st.header("R-Peak Detection Results")
        
        for channel_name, data in results.items():
            st.subheader(f"Channel: {channel_name}")
            st.image(data['plot'])
            st.text(f"R-peaks: {data['r_peaks']}")

def main():
    st.set_page_config(page_title="EEG/ECG Analysis", layout="wide")
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["EEG Data Preprocessing", "ECG Analysis and R-Peak Detection"])
    
    if app_mode == "EEG Data Preprocessing":
        display_ica_results()
    elif app_mode == "ECG Analysis and R-Peak Detection":
        display_ecg_analysis()

if __name__ == "__main__":
    main()
