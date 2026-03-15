# compare_page.py
import streamlit as st
import tempfile
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

from utils.preprocessing import preprocess_audio
from utils.plotting import plot_waveform, plot_energy, plot_pitch, plot_band_energy

def compare_page():
    st.title("⚖️ Compare Real vs Fake Audio")
    st.markdown("Upload one **real** and one **fake** audio file to visually compare their patterns.")

    col1, col2 = st.columns(2)
    with col1:
        real_file = st.file_uploader("🎙️ Upload Real Audio", type=["wav", "mp3"], key="real")
    with col2:
        fake_file = st.file_uploader("🤖 Upload Fake Audio", type=["wav", "mp3"], key="fake")

    if real_file and fake_file:
        # Save both temporarily
        with tempfile.NamedTemporaryFile(delete=False) as real_tmp:
            real_tmp.write(real_file.read())
            real_path = real_tmp.name
        with tempfile.NamedTemporaryFile(delete=False) as fake_tmp:
            fake_tmp.write(fake_file.read())
            fake_path = fake_tmp.name

        # Preprocess both
        real_features, real_sr = preprocess_audio(real_path)
        fake_features, fake_sr = preprocess_audio(fake_path)
        real_y, _ = librosa.load(real_path, sr=real_sr)
        fake_y, _ = librosa.load(fake_path, sr=fake_sr)

        st.success("✅ Both audios processed successfully!")

        # Layout — side by side comparison
        st.markdown("## 📊 Visual Comparison")

        # 1️⃣ Waveform
        st.markdown("### 🔊 Waveform Comparison")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Real Audio**")
            st.pyplot(plot_waveform(real_y, real_sr))
        with c2:
            st.markdown("**Fake Audio**")
            st.pyplot(plot_waveform(fake_y, fake_sr))

        # 2️⃣ Energy
        st.markdown("### ⚡ Energy Over Time")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Real Audio**")
            st.pyplot(plot_energy(real_y, real_sr))
        with c2:
            st.markdown("**Fake Audio**")
            st.pyplot(plot_energy(fake_y, fake_sr))

        # 3️⃣ Pitch Contour
        st.markdown("### 🎵 Pitch Contour")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Real Audio**")
            st.pyplot(plot_pitch(real_y, real_sr))
        with c2:
            st.markdown("**Fake Audio**")
            st.pyplot(plot_pitch(fake_y, fake_sr))

        # 4️⃣ Band Energy
        st.markdown("### 🎚️ Frequency Band Energy Distribution")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Real Audio**")
            st.pyplot(plot_band_energy(real_features, real_sr))
        with c2:
            st.markdown("**Fake Audio**")
            st.pyplot(plot_band_energy(fake_features, fake_sr))

        # 5️⃣ Mel Spectrogram
        st.markdown("### 🎼 Mel Spectrogram Comparison")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Real Audio**")
            fig, ax = plt.subplots(figsize=(8, 3))
            librosa.display.specshow(real_features.T, sr=real_sr, x_axis='time', y_axis='mel', cmap='magma', ax=ax)
            ax.set_title("Real Audio")
            st.pyplot(fig)
        with c2:
            st.markdown("**Fake Audio**")
            fig, ax = plt.subplots(figsize=(8, 3))
            librosa.display.specshow(fake_features.T, sr=fake_sr, x_axis='time', y_axis='mel', cmap='magma', ax=ax)
            ax.set_title("Fake Audio")
            st.pyplot(fig)

        # Clean up temp files
        os.remove(real_path)
        os.remove(fake_path)

    else:
        st.info("👆 Please upload both a Real and a Fake audio file to begin comparison.")
