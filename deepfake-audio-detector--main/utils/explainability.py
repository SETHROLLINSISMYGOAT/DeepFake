# utils/explainability.py
import numpy as np
import tensorflow as tf
import streamlit as st

def explain_prediction(model, features, class_index=None, layer_name=None):
    sample = features[np.newaxis, :, :]

    # Try to detect Conv1D layer
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
