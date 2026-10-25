"""End to end analysis of one spoken utterance.

Order matters: recognition first (what was said), then everything that needs a
reference transcript. Models are loaded once and reused across requests.
"""
from __future__ import annotations

import threading
import numpy as np

from src.analysis.accent import AccentClassifier
from src.analysis.asr import Transcriber
from src.analysis.phonemes import PronunciationScorer, weakest_groups
from src.analysis.prosody import analyse as prosody_analyse
from src.analysis.scoring import fluency_score, grade, overall
from src.analysis.words import coachable_words


class Analyzer:
    """Lazily loaded, thread safe holder for the three models."""

    def __init__(self):
        self._lock = threading.Lock()
        self._asr = None
        self._pron = None
        self._acc = None

    @property
    def asr(self):
        with self._lock:
            if self._asr is None:
                self._asr = Transcriber(threads=4)
        return self._asr

    @property
    def pron(self):
        with self._lock:
            if self._pron is None:
                self._pron = PronunciationScorer(threads=4)
        return self._pron

    @property
    def accent(self):
        with self._lock:
            if self._acc is None:
                self._acc = AccentClassifier(threads=6)
        return self._acc

    def run(self, wav: np.ndarray) -> dict:
        if wav.size < 1600:
            return {"ok": False, "reason": "recording too short"}

        asr = self.asr(wav)
        if not asr["text"].strip():
            return {"ok": False, "reason": "no speech detected",
                    "prosody": prosody_analyse(wav, 0, 0)}

        pros = prosody_analyse(wav, asr["n_words"], asr["n_syllables"])
        gop = self.pron.score(wav, asr["text"])
        acc = self.accent.predict(wav)

        pron_score = gop["overall"] if gop.get("ok") else None
        weak = weakest_groups(gop["phones"]) if gop.get("ok") else []
        flu = fluency_score(pros)
        clarity = round(100 * asr["asr_confidence"], 1)
        total = overall(pron_score, flu["fluency"], clarity)

        worst_words = coachable_words(gop["words"]) if gop.get("ok") else []

        return {
            "ok": True,
            "transcript": asr["text"],
            "asr": {"n_words": asr["n_words"], "n_syllables": asr["n_syllables"],
                    "confidence": asr["asr_confidence"]},
            "accent": acc,
            "pronunciation": {"score": pron_score,
                              "weak_groups": weak,
                              "weak_words": worst_words,
                              "detail_available": bool(gop.get("ok")),
                              "reason": gop.get("reason")},
            "prosody": pros,
            "fluency": flu,
            "clarity": clarity,
            "overall": total,
            "grade": grade(total),
        }
