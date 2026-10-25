"""Accent inference using the trained ensemble head on frozen WavLM features."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.config import ENCODE_MAX_DUR, HEAD_DIR, NATIVE_VARIETIES, SR
from src.features.encoder import WavLMFeatures
from src.train.model import AccentHead, LinearProbe


class AccentClassifier:
    available = False

    def __init__(self, threads: int = 6):
        self.feats = WavLMFeatures(threads=threads)
        path = HEAD_DIR / "head.pt"
        if not path.exists():
            return
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.accents = ck["accents"]
        self.mu, self.sd = ck["mu"], ck["sd"]
        self.metrics = ck.get("metrics", {})
        cfg = ck["cfg"]
        self.models = []
        for sdict in ck["state_dicts"]:
            m = (AccentHead(len(self.accents), cfg["hidden"], cfg["dropout"])
                 if cfg["arch"] == "mlp"
                 else LinearProbe(len(self.accents), self.mu.shape[1], cfg["dropout"]))
            m.load_state_dict(sdict); m.eval()
            self.models.append(m)
        self.available = True

    def _windows(self, wav: np.ndarray) -> list[np.ndarray]:
        """Split long audio into encoder sized windows and average their votes.

        A single centre crop throws away most of a long utterance; averaging
        posteriors over consecutive windows uses all of it and is markedly
        steadier on spontaneous speech.
        """
        lim = int(ENCODE_MAX_DUR * SR)
        if len(wav) <= lim:
            return [wav]
        hop = int(lim * 0.75)
        out = [wav[s:s + lim] for s in range(0, max(1, len(wav) - lim + 1), hop)]
        return out[:8] or [wav[:lim]]

    def predict(self, wav: np.ndarray) -> dict:
        if not self.available:
            return {"available": False,
                    "reason": "accent model not trained yet"}
        emb = self.feats(self._windows(wav))
        x = (emb - self.mu) / self.sd
        xt = torch.from_numpy(x.astype(np.float32))
        with torch.inference_mode():
            probs = np.mean([F.softmax(m(xt), -1).numpy() for m in self.models], axis=0)
        p = probs.mean(0)                       # average across windows
        order = np.argsort(-p)
        top = [{"accent": self.accents[i], "prob": round(float(p[i]), 4)} for i in order[:5]]
        native = float(sum(p[i] for i, a in enumerate(self.accents) if a in NATIVE_VARIETIES))
        # margin between first and second choice signals how decisive the call is
        margin = float(p[order[0]] - p[order[1]])
        return {
            "available": True,
            "top": top,
            "predicted": self.accents[order[0]],
            "confidence": round(float(p[order[0]]), 4),
            "margin": round(margin, 4),
            "native_variety_prob": round(native, 4),
            "windows": int(emb.shape[0]),
            "certainty": ("high" if margin > 0.25 else "moderate" if margin > 0.10 else "low"),
        }
