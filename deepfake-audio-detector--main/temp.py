import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import io
import tempfile
import os
from tensorflow.keras import Model  # <-- ✅ You missed this import

st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎧", layout="wide")

# Sidebar Navigation
page = st.sidebar.radio(
    "📂 Choose a Page:",
    ["🔍 Single Audio Analysis", "⚖️ Real vs Fake Comparison"]
)




# ===============================
# 1️⃣ Load Model
# ===============================
@st.cache_resource
def load_trained_model():
    model_path = "cnn_lstm_deepfake_model.h5"
    model = tf.keras.models.load_model(model_path, compile=False)

    # ⚡ Force-build the model by passing a dummy input through it
    input_shape = (250, 64)
    dummy_input = np.zeros((1, *input_shape), dtype=np.float32)
    _ = model.predict(dummy_input, verbose=0)

    # ✅ Rewrap the model as a Functional model so .inputs/.outputs are defined
    inputs = tf.keras.Input(shape=input_shape)
    outputs = model(inputs)
    functional_model = tf.keras.Model(inputs, outputs)

    return functional_model


model = load_trained_model()

# Force build (important for Grad-CAM)
try:
    input_shape = model.input_shape[1:]
    dummy = np.zeros((1, *input_shape), dtype=np.float32)
    _ = model.predict(dummy)
except Exception as e:
    st.warning(f"⚠️ Could not initialize model automatically: {e}")


# ===============================
# 2️⃣ Preprocess Audio Function
# ===============================
def preprocess_audio(file, n_mels=64, max_len=250):
    y, sr = librosa.load(file, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec).T

    # Pad or truncate
    if len(mel_db) < max_len:
        mel_db = np.pad(mel_db, ((0, max_len - len(mel_db)), (0, 0)), mode='constant')
    else:
        mel_db = mel_db[:max_len]

    return mel_db.astype(np.float32), sr

# ===============================
# 3️⃣ Prediction Function
# ===============================
def predict_audio(model, features):
    sample = features[np.newaxis, :, :]
    preds = model.predict(sample, verbose=0)
    confidence = preds[0]
    label = np.argmax(confidence)
    label_name = "Real" if label == 0 else "Fake"
    return label_name, confidence

# ===============================
# 4️⃣ Grad-CAM Explanation Utils
# ===============================
def explain_prediction(model, features, class_index=None, layer_name=None):
    sample = features[np.newaxis, :, :]

    # ✅ Try to detect Conv1D layer
    conv_layers = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv1D)]
    if not conv_layers:
        st.warning("❌ No Conv1D layers found. Using a random activation map instead (for visualization only).")
        fake_heatmap = np.random.rand(features.shape[0])
        preds = model.predict(sample, verbose=0)
        return fake_heatmap, preds

    # use last Conv1D layer
    layer_name = conv_layers[-1]

    # Create Grad model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(sample)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        class_output = preds[:, class_index]

    grads = tape.gradient(class_output, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_out), axis=-1)
    heatmap = np.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy(), preds.numpy()



def generate_text_reason(pred_class, heatmap, features):
    """Generate textual reason + keyword definitions based on Grad-CAM analysis."""

    # Compute stats from Grad-CAM
    high_attention_ratio = np.mean(heatmap > 0.6)
    low_attention_ratio = np.mean(heatmap < 0.2)
    avg_activation = np.mean(heatmap)
    std_activation = np.std(heatmap)

    temporal_focus = np.argmax(np.mean(heatmap, axis=-1)) if heatmap.ndim > 1 else np.argmax(heatmap)
    freq_focus = np.argmax(np.mean(heatmap, axis=0)) if heatmap.ndim > 1 else None

    # Common keyword dictionary (definition + how it differs)
    keyword_defs = {
        "spectral bursts": {
            "definition": "Short, intense increases in energy across frequencies, often appearing as bright vertical bands in a spectrogram.",
            "real_vs_fake": "In real audio, bursts are smooth and correspond to plosive sounds (like 'p', 't'); in fake audio, they often look abrupt or unnaturally sharp due to synthesis artifacts."
        },
        "temporal irregularities": {
            "definition": "Inconsistent timing or rhythm in how energy changes over time.",
            "real_vs_fake": "Real speech has smooth timing transitions; fake audio may have jitter or uneven segment durations due to poor time alignment in generation."
        },
        "frequency inconsistencies": {
            "definition": "Unnatural emphasis or suppression in certain frequency bands.",
            "real_vs_fake": "Real speech maintains a balanced harmonic spread; fake clips often exaggerate highs/lows or lose midrange clarity due to vocoder limitations."
        },
        "harmonic structure": {
            "definition": "The pattern of overtones or multiples of a base frequency that give a voice its natural timbre.",
            "real_vs_fake": "Real voices have stable harmonic spacing; fake ones show jitter or missing harmonics because synthesis models struggle with fine spectral detail."
        },
        "fade-in/fade-out mismatches": {
            "definition": "Sudden starts or endings in an audio clip without natural energy transitions.",
            "real_vs_fake": "Human recordings naturally ramp up/down; generated clips often start or stop abruptly."
        }
    }

    # --- CASE 1: FAKE AUDIO ---
    if pred_class == 1:
        reasons = []
        used_keywords = set()

        if high_attention_ratio > 0.35 and std_activation > 0.25:
            reasons.append(
                "High-energy **spectral bursts** and irregular **temporal patches** detected — "
                "often created when neural vocoders generate transitions between phonemes."
            )
            used_keywords.update(["spectral bursts", "temporal irregularities"])

        if freq_focus is not None and (freq_focus < features.shape[1] * 0.2 or freq_focus > features.shape[1] * 0.8):
            reasons.append(
                "Unnatural **frequency inconsistencies** observed — "
                "strong emphasis in very low or high frequency bands."
            )
            used_keywords.add("frequency inconsistencies")

        if temporal_focus < features.shape[0] * 0.2 or temporal_focus > features.shape[0] * 0.8:
            reasons.append(
                "Attention concentrated at clip boundaries, suggesting **fade-in/fade-out mismatches**."
            )
            used_keywords.add("fade-in/fade-out mismatches")

        if avg_activation < 0.3 and high_attention_ratio < 0.25:
            reasons.append(
                "Mild **harmonic structure** irregularities detected — "
                "consistent with partially synthesized voice characteristics."
            )
            used_keywords.add("harmonic structure")

        if not reasons:
            reasons.append("Detected subtle inconsistencies resembling synthetic speech generation.")

        reason_text = "🧠 **Reasons the model classified as FAKE:**\n- " + "\n- ".join(reasons)
        defs = {k: keyword_defs[k] for k in used_keywords if k in keyword_defs}
        return reason_text, defs

    # --- CASE 2: REAL AUDIO ---
    else:
        reasons = []
        used_keywords = set(["harmonic structure"])  # Always relevant

        if low_attention_ratio > 0.5:
            reasons.append(
                "Smooth **harmonic structure** with consistent spectral flow — "
                "energy changes are gradual and balanced."
            )
        else:
            reasons.append(
                "Stable **harmonic structure** and natural **frequency consistency** observed."
            )
            used_keywords.add("frequency inconsistencies")

        reason_text = "✅ **Reasons the model classified as REAL:**\n- " + "\n- ".join(reasons)
        defs = {k: keyword_defs[k] for k in used_keywords if k in keyword_defs}
        return reason_text, defs

# ===============================
# 📊 Visualization Functions
# ===============================
def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(8, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='cyan')
    ax.set_title("Waveform (Amplitude over Time)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return fig


def plot_energy(y, sr, frame_length=2048, hop_length=512):
    energy = np.array([
        sum(abs(y[i:i+frame_length]**2))
        for i in range(0, len(y), hop_length)
    ])
    frames = range(len(energy))
    t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(t, energy, color='orange')
    ax.set_title("Energy Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy")
    return fig


def plot_pitch(y, sr):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = pitches[magnitudes > np.median(magnitudes)]
    if pitch_mean.size == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Pitch not detected", ha='center')
        return fig

    fig, ax = plt.subplots(figsize=(8, 2))
    times = librosa.times_like(pitches)
    pitch_values = np.mean(pitches, axis=0)
    ax.plot(times, pitch_values, color='green')
    ax.set_title("Pitch Contour (Fundamental Frequency)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    return fig


def plot_band_energy(mel_db, sr, n_mels=64):
    """Show average energy in Low, Mid, and High frequency bands."""
    low_band = np.mean(mel_db[:, :int(n_mels*0.33)])
    mid_band = np.mean(mel_db[:, int(n_mels*0.33):int(n_mels*0.66)])
    high_band = np.mean(mel_db[:, int(n_mels*0.66):])

    fig, ax = plt.subplots(figsize=(5, 3))
    bands = ['Low', 'Mid', 'High']
    energies = [low_band, mid_band, high_band]
    ax.bar(bands, energies, color=['#66c2a5', '#fc8d62', '#8da0cb'])
    ax.set_title("Frequency Band Energy Distribution")
    ax.set_ylabel("Average dB")
    return fig


# ===============================
# 5️⃣ Streamlit UI
# ===============================
if page == "🔍 Single Audio Analysis":
    st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎧", layout="wide")

    st.title("🎵 Deepfake Audio Detection App")
    st.markdown("Upload an audio file (.wav, .mp3) and let the model predict whether it's **Real** or **Fake**.")

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
        
        # ===============================
        # 🎧 Visualization Section (Vertical Layout)
        # ===============================
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

        # 🔍 Explainability Section
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

            # 🧾 Main explanation text
            st.subheader("🧾 Explanation")
            st.markdown(reason_text)

            # 📚 Keyword definitions in table
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

elif page == "⚖️ Real vs Fake Comparison":
    st.title("⚖️ Compare Real vs Fake Audio")
    st.markdown("Upload one **real** and one **fake** audio file to visually compare their patterns.")

    # Upload section
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
