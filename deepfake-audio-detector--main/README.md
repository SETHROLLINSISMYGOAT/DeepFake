# 🎙️ Deepfake Audio Detection for KYC Authentication

Detect and explain **AI-generated (deepfake) voices** using a **CNN–LSTM model**, audio forensics, and explainability visualization — all in an interactive **Streamlit web app**.

🔗 **Live Demo:** [https://ruhkjhtofjqcjwprzkabmc.streamlit.app/](https://ruhkjhtofjqcjwprzkabmc.streamlit.app/)

---

## 🚀 Features

| Category | Description |
|-----------|--------------|
| 🎧 **Single Audio Detection** | Upload an audio clip and get a real/fake prediction with model confidence |
| ⚖️ **Audio Comparison Mode** | Compare real and fake audios side by side with synchronized visualizations |
| 🔥 **Explainability (Grad-CAM)** | View heatmaps showing which spectrogram regions influenced model decisions |
| 🔍 **Forensic Analysis** | Examine handcrafted forensic metrics — spectral bursts, pitch jitter, harmonicity, etc. |
| 🧠 **Advanced Feature Extraction** | Visualize MFCCs, spectral roll-off, pitch contour, and other advanced audio features |

---

## 💼 Project Structure

```bash
project/
│
├── app.py                       🎯  Main Streamlit entry file (handles routing + sidebar)
│
├── single_audio_page.py          🎧  Detect and explain deepfake for a single uploaded audio
├── compare_page.py               ⚖️  Compare real vs fake audios side by side
├── advanced_features_page.py     🧠  Perform forensic and advanced acoustic analyses
│
├── cnn_lstm_deepfake_model.h5    🧩  Trained CNN-LSTM model (real vs fake classifier)
│
├── utils/                        ⚙️  Core utility modules
│   ├── preprocessing.py          🔊  Audio loading, trimming, feature extraction (MFCCs, etc.)
│   ├── plotting.py               📊  Visualization helpers (waveform, spectrogram, MFCC plots)
│   ├── model_utils.py            🧠  Model loading, inference, and caching utilities
│   ├── explainability.py         🔥  Grad-CAM heatmaps and explainability visualizations
│   ├── advanced_features.py      🎵  Extracts advanced spectral and prosodic features
│   └── forensics.py              🔍  Forensic metrics (pitch jitter, harmonicity, fade mismatch)
│
├── requirements.txt              📦  Dependency list for Streamlit or local environment
├── README.md                     📘  Project documentation (overview, setup, usage)
└── screenshots/ (optional)       🖼️  Demo images for README or Streamlit Cloud preview
```
---

🧩 Model Overview
Architecture: CNN + LSTM hybrid
Input: Mel-spectrogram features (250 × 64)
Output: Binary classification → Real / Fake
Framework: TensorFlow / Keras
Trained On: Real vs synthetic speech samples from KYC-style datasets

---

⚙️ Installation & Setup

1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/deepfake-audio-detector.git
cd deepfake-audio-detector
```
2️⃣ Create a Virtual Environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```
3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
4️⃣ Run the Streamlit App
```bash
streamlit run app.py
```
---

🔍 Forensic Feature Set
| Feature                     | Description                                                           |
| --------------------------- | --------------------------------------------------------------------- |
| **Spectral Burst Fraction** | Detects sudden spectral spikes — may indicate synthesis artifacts     |
| **Fade Mismatch Score**     | Detects unnatural fade-in/out — signs of splicing or generation       |
| **Pitch Jitter Score**      | Measures unnatural pitch steadiness, common in TTS voices             |
| **Harmonicity Ratio**       | Checks periodic consistency — low values often mean vocoder artifacts |
| **Formant Variability**     | Low variability suggests vocoder smoothing or synthetic resonance     |

---

📊 Output Examples
🎵 Waveform & Spectrogram
🔥 Grad-CAM Heatmap (Model Attention)
📈 Confidence Bar (Real vs Fake)
🔍 Forensic Scores Table
⚖️ Side-by-Side Real vs Fake Comparison

---

☁️ Deployment
You can deploy this project seamlessly on:
🌐 Streamlit Cloud
🤗 Hugging Face Spaces

App entry command:
```bash
streamlit run app.py
```
---

🧩 Requirements
```bash
streamlit
librosa
numpy
matplotlib
tensorflow
scipy
praat-parselmouth
soundfile
scikit-learn
```
---

🧑‍💻 Author

Capstone Project (Aug 2025 – Oct 2025)
Developed by [Capstone team VIT-AP]
Focus: Deepfake Audio Detection for KYC Authentication Systems

---

🌟 Future Enhancements

🎤 Integrate speaker verification embeddings

🌍 Expand to multilingual datasets

🧠 Add attention-based Grad-CAM++ visualization

🪶 Optimize lightweight model for mobile/on-edge deployment

---
💡 Emoji Legend
| Emoji | Meaning                  |
| :---: | :----------------------- |
|   🎯  | Main App Entry           |
|   🎧  | Single Audio Page        |
|   ⚖️  | Comparison Page          |
|   🧠  | Advanced / Forensic Page |
|   🔊  | Audio Preprocessing      |
|   📊  | Plotting / Visualization |
|   🔥  | Explainability           |
|   🎵  | Advanced Features        |
|   🔍  | Forensic Analysis        |
|   📦  | Dependencies             |
|   📘  | Documentation            |
|  🖼️  | Screenshots / Demo Media |
