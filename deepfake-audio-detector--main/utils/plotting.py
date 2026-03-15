# utils/plotting.py
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

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
    ax.plot(t, energy)
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
    ax.plot(times, pitch_values)
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
    ax.bar(bands, energies)
    ax.set_title("Frequency Band Energy Distribution")
    ax.set_ylabel("Average dB")
    return fig
