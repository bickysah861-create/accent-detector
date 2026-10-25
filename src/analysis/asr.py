"""Speech recognition via faster-whisper (CTranslate2, int8 on CPU).

Used only to find out *what* was said, with word timings. Every score in the
system is computed from the audio itself; the transcript is an input to that,
not a judgement of it.
"""
from __future__ import annotations

import re
import numpy as np

from src.config import ASR_MODEL

_VOWEL_RUN = re.compile(r"[aeiouy]+", re.I)


def count_syllables(word: str) -> int:
    """Cheap English syllable estimate; good enough for speaking-rate metrics."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    n = len(_VOWEL_RUN.findall(w))
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee", "ye")):
        n -= 1
    return max(n, 1)


class Transcriber:
    def __init__(self, threads: int = 4):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8",
                                  cpu_threads=threads)

    def __call__(self, wav: np.ndarray) -> dict:
        segs, info = self.model.transcribe(
            wav, language="en", beam_size=5, word_timestamps=True,
            vad_filter=True, condition_on_previous_text=False,
        )
        words, texts, probs = [], [], []
        for s in segs:
            texts.append(s.text.strip())
            for w in (s.words or []):
                tok = w.word.strip()
                if not tok:
                    continue
                words.append({"word": tok, "start": round(w.start, 3),
                              "end": round(w.end, 3), "prob": round(w.probability, 3)})
                probs.append(w.probability)
        text = " ".join(texts).strip()
        clean = re.sub(r"[^A-Za-z' ]", " ", text)
        toks = [t for t in clean.split() if t]
        return {
            "text": text,
            "words": words,
            "n_words": len(toks),
            "n_syllables": sum(count_syllables(t) for t in toks),
            # mean word posterior: low values flag slurred or unclear delivery
            "asr_confidence": round(float(np.mean(probs)), 3) if probs else 0.0,
            "language_prob": round(float(getattr(info, "language_probability", 0.0)), 3),
        }
