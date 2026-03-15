# utils/forensics.py
import numpy as np
import librosa
import scipy.signal

def spectral_burst_score(y, sr, frame_length=2048, hop_length=256, burst_threshold_db=20):
    S = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    frame_energy = np.mean(S_db, axis=0)
    med = scipy.signal.medfilt(frame_energy, kernel_size=7)
    spikes = frame_energy - med
    burst_frames = np.sum(spikes > burst_threshold_db)
    return float(burst_frames / len(frame_energy))

def fade_mismatch_score(y, sr, window_ms=50):
    window = int(sr * window_ms / 1000)
    if len(y) < 2 * window:
        return 0.0
    start_rms = np.sqrt(np.mean(y[:window]**2))
    end_rms = np.sqrt(np.mean(y[-window:]**2))
    mid_rms = np.sqrt(np.mean(y[window:-window]**2)) + 1e-10
    start_ratio = start_rms / mid_rms
    end_ratio = end_rms / mid_rms
    return float(max(0, 1 - start_ratio) + max(0, 1 - end_ratio)) / 2.0

def pitch_jitter_score(y, sr, hop_length=256):
    pitches, mags = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)
    pitch_vals = []
    for i in range(pitches.shape[1]):
        col = pitches[:, i]
        mag_col = mags[:, i]
        if mag_col.max() > np.median(mags):
            p = col[mag_col.argmax()]
            if p > 0:
                pitch_vals.append(p)
    if len(pitch_vals) < 3:
        return 0.0
    pitch_vals = np.array(pitch_vals)
    return float(np.std(pitch_vals) / (np.mean(pitch_vals) + 1e-10))

def formant_variability_score(formant_tracks):
    scores = []
    for k in ["F1", "F2", "F3"]:
        arr = np.array(formant_tracks.get(k, []))
        if arr.size == 0:
            continue
        scores.append(float(np.nanstd(arr) / (np.nanmean(arr) + 1e-10)))
    return float(np.mean(scores)) if scores else 0.0

def harmonicity_score(y, sr, frame_length=2048):
    y = librosa.util.normalize(y)
    hop = frame_length // 2
    ac_vals = []
    for i in range(0, len(y) - frame_length, hop):
        frame = y[i:i + frame_length]
        ac = np.correlate(frame, frame, mode="full")
        ac = ac[ac.size // 2:]
        if ac[0] == 0:
            continue
        peak = np.max(ac[1:]) if ac[1:].size else 0.0
        ac_vals.append(float(peak / (ac[0] + 1e-12)))
    return float(np.mean(ac_vals)) if ac_vals else 0.0

def run_forensic_checks(y, sr, formant_tracks=None):
    results = {}
    results["spectral_burst_frac"] = (
        spectral_burst_score(y, sr),
        "Fraction of frames with sudden spectral spikes (higher may indicate artifacts)."
    )
    results["fade_mismatch"] = (
        fade_mismatch_score(y, sr),
        "Abrupt start/end mismatch (may indicate splice or synthesis)."
    )
    results["pitch_jitter"] = (
        pitch_jitter_score(y, sr),
        "Normalized pitch variability; extreme steadiness can suggest TTS."
    )
    results["harmonicity"] = (
        harmonicity_score(y, sr),
        "Mean autocorrelation ratio; low or odd values may suggest synthetic harmonics."
    )
    if formant_tracks is not None:
        results["formant_variability"] = (
            formant_variability_score(formant_tracks),
            "Variation in formant movement; low variability can indicate vocoder smoothing."
        )
    return results
