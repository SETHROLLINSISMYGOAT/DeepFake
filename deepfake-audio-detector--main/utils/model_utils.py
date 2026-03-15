# utils/model_utils.py
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras import Model  # kept because original code referenced it

@st.cache_resource
def load_trained_model(model_path="cnn_lstm_deepfake_model.h5"):
    model = tf.keras.models.load_model(model_path, compile=False)

    # Force-build the model by passing a dummy input through it
    input_shape = (250, 64)
    dummy_input = np.zeros((1, *input_shape), dtype=np.float32)
    _ = model.predict(dummy_input, verbose=0)

    # Rewrap as functional so .inputs/.outputs defined
    inputs = tf.keras.Input(shape=input_shape)
    outputs = model(inputs)
    functional_model = tf.keras.Model(inputs, outputs)

    return functional_model


def predict_audio(model, features):
    sample = features[np.newaxis, :, :]
    preds = model.predict(sample, verbose=0)
    confidence = preds[0]
    label = np.argmax(confidence)
    label_name = "Real" if label == 0 else "Fake"
    return label_name, confidence
