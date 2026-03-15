import streamlit as st
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

def info_page():
    st.title("🎧 Deepfake Audio Detection – Understanding the Basics")
    st.markdown("---")

    # Section 1: What is Deepfake
    st.header("🧠 What is a Deepfake?")
    st.write("""
    Deepfakes are synthetic media — videos, images, or audios — generated using Artificial Intelligence 
    that mimic real human behavior. These models learn from real samples and then recreate speech or visuals 
    that appear genuine but are artificially produced.
    """)

    # Section 2: What is Audio Deepfake
    st.header("🎙️ What is an Audio Deepfake?")
    st.write("""
    Audio deepfakes are **AI-generated voices** that sound like real people. 
    Modern deep learning models (like voice cloning and speech synthesis systems) 
    can imitate tone, pitch, speaking style, and emotion.
    """)

    st.info("""
    ⚠️ These can be used maliciously — for impersonation, misinformation, or fraud.
    Hence, understanding their **acoustic differences** from real speech is crucial.
    """)

    # Section 3: Features Considered
    st.header("🔍 Key Audio Features We Analyze")
    st.write("""
    To distinguish real and fake audios, we extract and visualize several acoustic features:
    """)
    st.markdown("""
    - **Waveform** → shows raw amplitude variations  
    - **Spectrogram** → displays frequency energy over time  
    - **MFCCs** → capture speech patterns and vocal timbre  
    - **Pitch (F₀)** → represents the perceived tone of voice  
    - **Energy** → measures loudness variation  
    - **Formants (F₁–F₃)** → vocal tract resonances that define vowel quality  
    - **Spectral Centroid** → indicates where the sound’s "brightness" lies  
    - **Zero-Crossing Rate (ZCR)** → measures how frequently the signal changes sign
    """)
    
    st.header("🔍 Key Audio Features Used for Deepfake Detection")

    st.write("""
    To differentiate real and fake audios, we analyze several important **acoustic features**:
    """)

    # Create a small sample audio visualization (demo only)
    sr = 22050
    t = np.linspace(0, 1, sr)
    y = 0.5 * np.sin(2 * np.pi * 220 * t)  # simple sine wave

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Waveform")
        fig, ax = plt.subplots(figsize=(3, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Waveform Example")
        st.pyplot(fig)
        st.caption("Shows amplitude variation over time.")

    with col2:
        st.subheader("Spectrogram")
        fig, ax = plt.subplots(figsize=(3, 2))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        ax.set_title("Spectrogram Example")
        st.pyplot(fig)
        st.caption("Displays frequency energy across time.")

    with col3:
        st.subheader("MFCCs")
        fig, ax = plt.subplots(figsize=(3, 2))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfccs, x_axis='time', ax=ax)
        ax.set_title("MFCC Example")
        fig.colorbar(img, ax=ax)
        st.pyplot(fig)
        st.caption("Encodes vocal timbre and tone characteristics.")
    

    # Section 4: Real vs Fake Comparison
    st.header("⚖️ How Real and Fake Audio Differ")
    st.markdown("""
    | Feature | Real Audio | Deepfake Audio |
    |:--|:--|:--|
    | **Waveform** | Natural amplitude variation | Over-smoothed or clipped |
    | **Spectrogram** | Rich harmonics, smooth transitions | Missing harmonics or abrupt jumps |
    | **MFCC** | Varies naturally across frames | Too stable or repetitive |
    | **Pitch (F₀)** | Small fluctuations (vibrato, emotion) | Monotone or unnaturally smooth |
    | **Energy** | Rises and falls with speech | Often flat |
    | **Formants** | Naturally moving resonance peaks | Static or inconsistent |
    | **Spectral Centroid** | Varies with articulation | May stay constant |
    | **ZCR** | Phoneme-dependent variation | Often steady or noisy |
    """, unsafe_allow_html=True)

    # Closing note
    st.markdown("---")
    st.success("""
    ✅ This project uses these features with a CNN–LSTM model and additional forensic analysis
    (pitch, formants, and variability metrics) to detect deepfake audio with high accuracy.
    """)

    st.caption("Developed as part of the Capstone Deepfake Audio Detection Project 🧩")

