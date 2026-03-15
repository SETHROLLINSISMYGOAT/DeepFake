# advanced_features_page.py
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import parselmouth
from parselmouth.praat import call
from utils.forensics import run_forensic_checks

st.set_page_config(page_title="Advanced Audio Forensics", page_icon="🧠")

def extract_formants(y, sr):
    """Extract mean formant values using Praat via Parselmouth."""
    sound = parselmouth.Sound(y, sr)
    formant = call(sound, "To Formant (burg)", 0.02, 5, 5500, 0.025, 50)
    times = np.linspace(0, len(y) / sr, num=formant.get_number_of_frames())
    f1_vals, f2_vals, f3_vals = [], [], []
    for t in times:
        try:
            f1_vals.append(formant.get_value_at_time(1, t))
            f2_vals.append(formant.get_value_at_time(2, t))
            f3_vals.append(formant.get_value_at_time(3, t))
        except Exception:
            pass
    f1_vals, f2_vals, f3_vals = np.array(f1_vals), np.array(f2_vals), np.array(f3_vals)
    return {
        "F1": np.nanmean(f1_vals),
        "F2": np.nanmean(f2_vals),
        "F3": np.nanmean(f3_vals),
        "tracks": {"F1": f1_vals, "F2": f2_vals, "F3": f3_vals, "times": times}
    }

def display_formant_plot(formant_data):
    """Plot F1–F3 trajectories over time."""
    times = formant_data["tracks"]["times"]
    plt.figure(figsize=(8, 4))
    plt.plot(times, formant_data["tracks"]["F1"], label="F1")
    plt.plot(times, formant_data["tracks"]["F2"], label="F2")
    plt.plot(times, formant_data["tracks"]["F3"], label="F3")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Formant Trajectories")
    plt.legend()
    st.pyplot(plt)

def advanced_features_page():
    st.title("🧠 Advanced Audio Forensics & Analysis")
    st.write("Upload an audio file to inspect **formant structure**, **pitch stability**, and **synthetic artifacts**.")

    uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "flac"])

    if uploaded_file:
        y, sr = librosa.load(uploaded_file, sr=None)
        st.audio(uploaded_file)

        st.subheader("🎵 Basic Information")
        st.write(f"**Sample Rate:** {sr} Hz")
        st.write(f"**Duration:** {len(y)/sr:.2f} seconds")

        # Plot waveform
        st.subheader("📈 Waveform")
        plt.figure(figsize=(8, 2))
        librosa.display.waveshow(y, sr=sr)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Waveform")
        st.pyplot(plt)

        # Extract formants
        with st.spinner("Extracting formant data..."):
            formants = extract_formants(y, sr)

        st.subheader("🔬 Mean Formant Values")
        st.json({k: round(v, 2) for k, v in formants.items() if k != "tracks"})

        display_formant_plot(formants)

        # Run forensic checks
        st.subheader("🕵️ Forensic Analysis")
        with st.spinner("Running forensic checks..."):
            forensic_results = run_forensic_checks(y, sr, formant_tracks=formants["tracks"])

        for k, (score, reason) in forensic_results.items():
            if score > 0.5:
                color = "🔴 Suspicious"
            elif score > 0.2:
                color = "🟠 Watch"
            else:
                color = "🟢 OK"
            st.markdown(f"**{k}**: {score:.2f} — {color}<br>_{reason}_", unsafe_allow_html=True)

        st.info("These forensic checks highlight possible synthetic traits or irregularities in the uploaded audio.")

if __name__ == "__main__":
    advanced_features_page()
