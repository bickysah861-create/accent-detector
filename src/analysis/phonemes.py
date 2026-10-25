"""Phoneme-level pronunciation scoring via Goodness of Pronunciation (GOP).

Method (the standard GOP formulation, Witt & Young 1997):
  1. a CTC acoustic model gives frame-level posteriors over 39 ARPAbet phones;
  2. the expected phone sequence comes from the recognised words via CMUdict g2p;
  3. CTC forced alignment assigns each expected phone a span of frames;
  4. GOP(p) = mean_t [ log P(p | x_t) - max_q log P(q | x_t) ] over that span.

A value of 0 means the intended phone was also the most likely one; large
negative values mean the acoustics point somewhere else, i.e. a mispronunciation.
No language model or LLM is involved - the score is read off the acoustics.
"""
from __future__ import annotations

import os, re
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

import nltk
nltk.data.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".venv", "nltk_data"))

from src.config import PHONEME_MODEL, SR

_STRESS = re.compile(r"\d")
_WORD = re.compile(r"[A-Za-z][A-Za-z']*")

# Articulatory groupings used to turn raw GOP into actionable advice.
PHONE_GROUPS = {
    "th": "dental fricatives (th)", "dh": "dental fricatives (th)",
    "r": "rhotic /r/", "er": "rhotic /r/",
    "l": "lateral /l/",
    "v": "labiodental /v/-/w/", "w": "labiodental /v/-/w/",
    "z": "voiced sibilants", "zh": "voiced sibilants", "jh": "voiced sibilants",
    "s": "voiceless sibilants", "sh": "voiceless sibilants", "ch": "voiceless sibilants",
    "iy": "tense/lax vowel pairs", "ih": "tense/lax vowel pairs",
    "uw": "tense/lax vowel pairs", "uh": "tense/lax vowel pairs",
    "ae": "open front vowels", "eh": "open front vowels", "ah": "open front vowels",
    "aa": "open back vowels", "ao": "open back vowels",
    "ng": "velar nasal (ng)", "n": "nasals", "m": "nasals",
    "p": "stops", "b": "stops", "t": "stops", "d": "stops", "k": "stops", "g": "stops",
}

VOWELS = {"aa","ae","ah","aw","ay","eh","er","ey","ih","iy","ow","oy","uh","uw"}


def _load_gop_k(default: float = 3.2) -> float:
    try:
        import json
        from src.config import MODELS
        return float(json.load(open(MODELS / "gop_calibration.json"))["k"])
    except Exception:
        return default


_GOP_K = _load_gop_k()


class PronunciationScorer:
    def __init__(self, threads: int = 4):
        from transformers import AutoModelForCTC, AutoProcessor
        torch.set_num_threads(threads)
        self.proc = AutoProcessor.from_pretrained(PHONEME_MODEL)
        self.model = AutoModelForCTC.from_pretrained(PHONEME_MODEL).eval()
        self.vocab = self.proc.tokenizer.get_vocab()
        self.blank = self.vocab.get("<pad>", 0)
        self.id2ph = {v: k for k, v in self.vocab.items()}
        from g2p_en import G2p
        self.g2p = G2p()

    # ---------------------------------------------------------------- g2p
    def expected(self, text: str) -> list[tuple[str, list[str]]]:
        """-> [(word, [ARPAbet phones present in the model vocab]), ...].

        The word string is carried alongside its phones rather than recovered by
        position later. Words with no CMUdict entry yield no phones and are
        dropped, so any positional pairing against a separate word list silently
        shifts and attributes a low score to the wrong word.
        """
        out = []
        for raw in _WORD.findall(text):
            phones = [
                p for p in (_STRESS.sub("", t).lower() for t in self.g2p(raw))
                if p in self.vocab
            ]
            if phones:
                out.append((raw, phones))
        return out

    @torch.inference_mode()
    def score(self, wav: np.ndarray, text: str) -> dict:
        words = self.expected(text)
        flat = [p for _, ph in words for p in ph]
        if not flat:
            return {"ok": False, "reason": "no pronounceable words recognised"}

        iv = self.proc(wav, sampling_rate=SR, return_tensors="pt")
        logits = self.model(iv.input_values).logits[0]
        logp = F.log_softmax(logits, dim=-1)           # (T, V)
        T = logp.shape[0]
        if T < len(flat):                               # audio too short to carry them
            return {"ok": False, "reason": "utterance too short for its transcript"}

        tgt = torch.tensor([[self.vocab[p] for p in flat]])
        aligned, _ = AF.forced_align(logp.unsqueeze(0), tgt, blank=self.blank)
        spans = AF.merge_tokens(aligned[0], torch.zeros(T), blank=self.blank)

        best = logp.max(dim=-1).values                  # (T,) best log-posterior per frame
        gops, i = [], 0
        for sp in spans:
            if i >= len(flat):
                break
            s, e = sp.start, max(sp.end, sp.start + 1)
            pid = self.vocab[flat[i]]
            g = float((logp[s:e, pid] - best[s:e]).mean())
            gops.append({"phone": flat[i], "gop": g,
                         "start": round(s * 0.02, 3), "end": round(e * 0.02, 3)})
            i += 1
        while i < len(flat):                            # unaligned tail = not realised
            gops.append({"phone": flat[i], "gop": -8.0, "start": None, "end": None})
            i += 1

        # regroup flat phones back into their words
        wout, k = [], 0
        for word, ph_expected in words:
            ph = gops[k:k + len(ph_expected)]; k += len(ph_expected)
            if not ph: continue
            wout.append({"word": word, "phones": ph,
                         "score": self._pct(np.mean([p["gop"] for p in ph]))})
        return {"ok": True, "phones": gops, "words": wout,
                "overall": self._pct(np.mean([g["gop"] for g in gops]))}

    @staticmethod
    def _pct(gop: float) -> float:
        """Map GOP, which is at most 0 with 0 being perfect, onto 0 to 100.

        The constant is not guessed. `calibrate_gop.py` measures mean GOP across
        native English speakers in this corpus and solves for the value that puts
        the median native speaker at the target score, writing it to
        models/gop_calibration.json. Without that file a conservative default is
        used, but the calibrated constant is what makes a score comparable to
        real speech rather than to an arbitrary curve.
        """
        return round(float(100.0 * np.exp(gop / _GOP_K)), 1)


def weakest_groups(phones: list[dict], k: int = 3) -> list[dict]:
    """Aggregate phone-level GOP into the articulatory groups worth practising."""
    agg: dict[str, list[float]] = {}
    for p in phones:
        g = PHONE_GROUPS.get(p["phone"])
        if g:
            agg.setdefault(g, []).append(p["gop"])
    rows = [{"group": g, "score": PronunciationScorer._pct(float(np.mean(v))), "n": len(v)}
            for g, v in agg.items() if len(v) >= 2]
    rows.sort(key=lambda r: r["score"])
    return rows[:k]
