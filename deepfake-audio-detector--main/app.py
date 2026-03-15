# app.py
import streamlit as st
from single_audio_page import single_audio_page
from compare_page import compare_page
from advanced_features_page import advanced_features_page
from info_page import info_page

st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎧", layout="wide")

menu = st.sidebar.radio("Navigation", ["Home", "Single Audio Analysis", "Compare Audios", "Forensic Analysis", "About Deepfake Audio"])

if menu == "Home" or menu == "About Deepfake Audio":
    info_page()
elif menu == "Single Audio Analysis":
    single_audio_page()
elif menu == "Compare Audios":
    compare_page()
elif menu == "Forensic Analysis":
    advanced_features_page()