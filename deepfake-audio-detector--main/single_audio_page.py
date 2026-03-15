# single_audio_page.py
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import os

from utils.preprocessing import preprocess_audio
from utils.plotting import plot_waveform, plot_energy, plot_pitch, plot_band_energy
from utils.model_utils import load_trained_model, predict_audio
from utils.explainability import explain_prediction, generate_text_reason

def single_audio_page():
    st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎧", layout="wide")

    st.title("🎵 Deepfake Audio Detection App")
    st.markdown("Upload an audio file (.wav, .mp3) and let the model predict whether it's **Real** or **Fake**.")

    model = load_trained_model()

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        # Save to a temp file for Librosa
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Display audio player
        st.audio(uploaded_file, format="audio/wav")

        with st.spinner("🔄 Processing audio..."):
            features, sr = preprocess_audio(tmp_path)
            y, _ = librosa.load(tmp_path, sr=sr)

        st.success("✅ Audio processed successfully!")
        
        # Visualization Section (Vertical Layout)
        with st.expander("🔊 Waveform (Amplitude over Time)"):
            st.pyplot(plot_waveform(y, sr))

        with st.expander("⚡ Energy Over Time"):
            st.pyplot(plot_energy(y, sr))

        with st.expander("🎵 Pitch Contour (Fundamental Frequency)"):
            st.pyplot(plot_pitch(y, sr))

        with st.expander("🎚️ Frequency Band Energy Distribution"):
            st.pyplot(plot_band_energy(features, sr))

        with st.expander("🎼 Mel Spectrogram"):
            fig, ax = plt.subplots(figsize=(8, 3))
            librosa.display.specshow(features.T, sr=sr, x_axis='time', y_axis='mel', cmap='magma', ax=ax)
            ax.set_title("Mel Spectrogram")
            st.pyplot(fig)


        # Prediction
        with st.spinner("🧠 Running Deepfake detection model..."):
            label, confidence = predict_audio(model, features)

        st.subheader("📊 Prediction Result")
        col1, col2 = st.columns(2)
        col1.metric("Prediction", label)
        col2.metric("Confidence", f"{np.max(confidence)*100:.2f}%")

        # Confidence bar
        st.progress(float(np.max(confidence)))

        # Detailed output
        st.markdown(f"**Real:** {confidence[0]:.3f} | **Fake:** {confidence[1]:.3f}")

        # Explainability Section
        with st.spinner("🧩 Generating explanation..."):
            heatmap, preds = explain_prediction(model, features)
            pred_class = np.argmax(preds)
            reason_text, keyword_info = generate_text_reason(pred_class, heatmap, features)

            # Show heatmap overlay
            st.subheader("🌈 Model Focus (Grad-CAM Heatmap)")
            fig, ax = plt.subplots(figsize=(8, 3))
            librosa.display.specshow(features.T, sr=sr, x_axis='time', y_axis='mel', cmap='magma', ax=ax)
            ax.imshow(heatmap[np.newaxis, :], aspect='auto', cmap='jet', alpha=0.6,
                    extent=[0, features.shape[0], 0, features.shape[1]])
            ax.set_title("Regions influencing prediction")
            st.pyplot(fig)

            # Main explanation text
            st.subheader("🧾 Explanation")
            st.markdown(reason_text)

            # Keyword definitions in table
            if keyword_info:
                st.markdown("### 📘 Keyword Definitions & Real-vs-Fake Behavior")

                # Create Markdown table
                table_md = "| Keyword | Definition | Real vs Fake Behavior |\n"
                table_md += "|:--|:--|:--|\n"

                for kw, details in keyword_info.items():
                    table_md += f"| **{kw.capitalize()}** | {details['definition']} | {details['real_vs_fake']} |\n"

                st.markdown(table_md)

        # Clean temp file
        os.remove(tmp_path)
    else:
        st.info("👆 Upload an audio file to start analysis.")
