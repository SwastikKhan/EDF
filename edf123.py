# -*- coding: utf-8 -*-
"""edf123.py
Original file is located at
    https://colab.research.google.com/drive/1YHjnQX0mCW9mKDkglSUiLwBfS4n0GzLg
"""

from google.colab import drive
drive.mount('/content/drive')

!pip install pyedflib
!pip install mne
!pip install neurokit2
!pip install streamlit pyngrok

from pyngrok import ngrok
import os

# Write the Streamlit app (paste the entire code from the artifact above)
with open("streamlit_app.py", "w") as f:
    f.write('''
import streamlit as st
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import os

def bandpass_filter(data, fs, lowcut=0.5, highcut=40, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def pan_tompkins_r_detection(ecg_signal, fs):
    # 1. Bandpass filter (already done above)
    filtered_ecg = bandpass_filter(ecg_signal, fs)

    # 2. Differentiation
    diff_signal = np.diff(filtered_ecg)

    # 3. Squaring
    squared_signal = diff_signal ** 2

    # 4. Moving average
    window_size = int(0.150 * fs)  # 150 ms window
    moving_avg = np.convolve(squared_signal, np.ones(window_size) / window_size, mode='same')

    # 5. Peak detection
    peaks, _ = find_peaks(moving_avg, distance=0.6 * fs)  # Min distance = 600 ms
    return peaks, moving_avg

def preprocess_eeg_data(file_path):
    # Load the EEG data
    raw = mne.io.read_raw_edf(file_path, preload=True)

    # Apply bandpass filter (0.5 Hz - 40 Hz)
    raw.filter(l_freq=0.5, h_freq=40, picks='eeg')

    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)

    # Save Original Graph
    raw.plot(duration=30, n_channels=30, scalings='auto', show=False)
    plt.savefig("output/original_graph.png")
    plt.close()

    raw.plot(title="Raw Data", show=False)
    plt.savefig("output/simplified_original_graph.png")
    plt.close()

    # Perform ICA to remove artifacts
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(raw)

    # Automatically exclude ICA components for artifacts
    ica.exclude = [0, 1]
    raw_clean = ica.apply(raw)

    # Save excluded components
    with open("output/excluded_components.txt", "w") as f:
        f.write(",".join(map(str, ica.exclude)))

    # Save Cleaned Graph
    raw_clean.plot(duration=30, n_channels=30, scalings='auto', show=False)
    plt.savefig("output/cleaned_graph.png")
    plt.close()

    raw_clean.plot(title="Cleaned Data", show=False)
    plt.savefig("output/simplified_cleaned_graph.png")
    plt.close()

    return raw_clean

def process_ecg_data(file_path):
    # Load the dataset
    raw = mne.io.read_raw_edf(file_path, preload=True)
    fs = raw.info['sfreq']  # Sampling frequency
    channel_names = raw.ch_names

    # Ensure output directory exists
    os.makedirs('output/ecg_plots', exist_ok=True)

    # Process each channel
    for channel_name in channel_names:
        # Get data for the current channel
        data = raw.get_data(picks=channel_name)[0]

        # Apply R-peak detection
        r_peaks, processed_signal = pan_tompkins_r_detection(data, fs)

        # Save results
        results_path = os.path.join("output/ecg_plots", f"{channel_name}_results.txt")
        with open(results_path, "w") as f:
            f.write(f"R-peaks: {r_peaks.tolist()}")  # Added closing quote

        # Save plots
        plt.figure(figsize=(12, 6))
        # Original ECG signal
        plt.subplot(2, 1, 1)
        plt.plot(data, label=f"{channel_name} - Original ECG")
        plt.scatter(r_peaks, data[r_peaks], color='red', label="R-peaks", zorder=5)
        plt.title(f"{channel_name} - ECG Signal with R-peaks")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.legend()

        # Processed signal
        plt.subplot(2, 1, 2)
        plt.plot(processed_signal, label="Processed Signal (Pan-Tompkins)")
        plt.title(f"{channel_name} - Processed ECG Signal (R-Peak Detection)")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.legend()

        plot_path = os.path.join("output/ecg_plots", f"{channel_name}_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

def display_ica_results():
    st.title("EEG Data Preprocessing")

    # File uploader
    uploaded_file = st.file_uploader("Choose an EEG file (EDF format)", type="edf")

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp.edf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Preprocess the data
        st.info("Processing EEG data...")
        preprocess_eeg_data("temp.edf")

        try:
            with open("output/excluded_components.txt", "r") as f:
                excluded_components = list(map(int, f.read().strip().split(",")))
        except FileNotFoundError:
            excluded_components = []

        st.header("ICA Artifact Removal Summary")
        st.write(f"Number of ICA components removed: {len(excluded_components)}")
        st.write(f"Removed ICA components: {excluded_components}")

        # Display graphs
        st.subheader("Original EEG Data")
        st.image("output/original_graph.png", caption="Original EEG Data")

        st.subheader("Simplified Original EEG Data")
        st.image("output/simplified_original_graph.png", caption="Simplified Original EEG Data")

        st.subheader("Cleaned EEG Data (Artifacts Removed)")
        st.image("output/cleaned_graph.png", caption="Cleaned EEG Data")

        st.subheader("Simplified Cleaned EEG Data")
        st.image("output/simplified_cleaned_graph.png", caption="Simplified Cleaned EEG Data")

def display_ecg_analysis():
    st.title("ECG Analysis and R-Peak Detection")

    # File uploader
    uploaded_file = st.file_uploader("Choose an EEG file (EDF format)", type="edf")

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp.edf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process ECG data
        st.info("Processing ECG data and detecting R-peaks...")
        process_ecg_data("temp.edf")

        # Read and display ECG plots
        ecg_plots_dir = "output/ecg_plots"

        st.header("R-Peak Detection Results")

        # Get list of plot files
        plot_files = [f for f in os.listdir(ecg_plots_dir) if f.endswith("_plot.png")]

        for plot_file in plot_files:
            channel_name = plot_file.replace("_plot.png", "")

            st.subheader(f"Channel: {channel_name}")

            # Display the plot
            plot_path = os.path.join(ecg_plots_dir, plot_file)
            st.image(plot_path, caption=f"{channel_name} - R-Peak Detection")

            # Display R-peak results
            results_file = plot_file.replace("_plot.png", "_results.txt")
            results_path = os.path.join(ecg_plots_dir, results_file)

            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    st.text(f.read())

# Main App
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["EEG Data Preprocessing", "ECG Analysis and R-Peak Detection"])

    if app_mode == "EEG Data Preprocessing":
        display_ica_results()
    elif app_mode == "ECG Analysis and R-Peak Detection":
        display_ecg_analysis()

if __name__ == "__main__":
    main()
    ''')

os.system("ngrok authtoken 2m6ibJzBrgq35SdeD3HUR5Bx7tk_2DgqTd3UFQqayShQCgu4e")
url = ngrok.connect("http://localhost:8501")
print(f"Access your Streamlit app here: {url}")
# Expose the Streamlit app via ngrok
!streamlit run streamlit_app.py