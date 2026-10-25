"""Fluency and prosody measurements computed directly from the waveform.

Everything here is deterministic signal processing - no model and no LLM is
involved, so the numbers are reproducible and explainable.
"""
from __future__ import annotations

import numpy as np
import torch
import torchaudio.functional as AF

from src.config import SR

FRAME = 0.025
HOP = 0.010


def _frames(x: np.ndarray, sr: int):
    n, h = int(FRAME * sr), int(HOP * sr)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    idx = np.arange(0, len(x) - n + 1, h)
    return np.stack([x[i:i + n] for i in idx]) if len(idx) else np.zeros((1, n), np.float32)


def energy_vad(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """Adaptive-threshold voice activity, returned per 10 ms frame."""
    fr = _frames(x, sr)
    rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    # threshold relative to the utterance's own loud/quiet spread
    hi, lo = np.percentile(db, 95), np.percentile(db, 10)
    thr = max(lo + 0.35 * (hi - lo), hi - 32.0)
    v = db > thr
    # close 1-frame gaps so single glottal dips don't split a word
    for i in range(1, len(v) - 1):
        if not v[i] and v[i - 1] and v[i + 1]:
            v[i] = True
    return v


def pauses(voiced: np.ndarray, min_pause: float = 0.25) -> list[tuple[float, float]]:
    out, start = [], None
    for i, v in enumerate(voiced):
        if not v and start is None:
            start = i
        elif v and start is not None:
            if (i - start) * HOP >= min_pause:
                out.append((start * HOP, i * HOP))
            start = None
    if start is not None and (len(voiced) - start) * HOP >= min_pause:
        out.append((start * HOP, len(voiced) * HOP))
    return out


def pitch_stats(x: np.ndarray, sr: int = SR) -> dict:
    try:
        f0 = AF.detect_pitch_frequency(torch.from_numpy(x).unsqueeze(0), sr,
                                       freq_low=60, freq_high=400).squeeze(0).numpy()
        f0 = f0[np.isfinite(f0) & (f0 > 60) & (f0 < 400)]
    except Exception:
        f0 = np.array([])
    if f0.size < 5:
        return {"f0_mean": 0.0, "f0_std": 0.0, "f0_range_semitones": 0.0}
    st = 12 * np.log2(np.maximum(f0, 1e-6) / np.median(f0))
    return {
        "f0_mean": float(np.median(f0)),
        "f0_std": float(np.std(f0)),
        # semitone range is speaker-normalised, so it compares across voices
        "f0_range_semitones": float(np.percentile(st, 90) - np.percentile(st, 10)),
    }


def analyse(x: np.ndarray, n_words: int, n_syllables: int, sr: int = SR) -> dict:
    dur = len(x) / sr
    voiced = energy_vad(x, sr)
    ps = pauses(voiced)
    pause_time = sum(b - a for a, b in ps)
    speech_time = max(dur - pause_time, 1e-6)
    out = {
        "duration_s": round(dur, 2),
        "speech_time_s": round(speech_time, 2),
        "pause_count": len(ps),
        "pause_time_s": round(pause_time, 2),
        "pause_ratio": round(pause_time / max(dur, 1e-6), 3),
        "longest_pause_s": round(max([b - a for a, b in ps], default=0.0), 2),
        "words_per_min": round(60 * n_words / max(dur, 1e-6), 1),
        # articulation rate excludes silence -> measures speaking speed itself
        "articulation_syl_per_s": round(n_syllables / speech_time, 2),
        "speech_rate_syl_per_s": round(n_syllables / max(dur, 1e-6), 2),
    }
    out.update({k: round(v, 2) for k, v in pitch_stats(x, sr).items()})
    return out
