# utils/preprocessing.py
import librosa
import numpy as np

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
