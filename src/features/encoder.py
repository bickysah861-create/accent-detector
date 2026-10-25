"""Frozen WavLM feature extractor with multi-layer statistics pooling.

Design rationale
----------------
Fine-tuning a 94M-param SSL model end-to-end is out of reach on a CPU-only box
inside a 2-3 h budget. Instead we freeze the encoder, extract embeddings once,
and spend the remaining budget training a small head on the cached features.
That makes many epochs, hyper-parameter search and seed-ensembling essentially
free, which buys back far more accuracy than one partial fine-tuning pass.

Middle transformer layers carry the most phonetic/accent information; the top
layers drift toward the masked-prediction pretraining objective. We therefore
read layers 4/6/8/10 and drop 11-12 entirely, which also cuts ~17% of compute.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel

from src.config import SSL_ENCODER

LAYERS = [4, 6, 8, 10]      # 0 = conv feature projection output
MAX_LAYER = max(LAYERS)
DIM_PER_LAYER = 768 * 2     # mean + std
EMB_DIM = len(LAYERS) * DIM_PER_LAYER


class WavLMFeatures:
    def __init__(self, threads: int = 10):
        torch.set_num_threads(threads)
        m = AutoModel.from_pretrained(SSL_ENCODER)
        # Layers above MAX_LAYER are never read -> delete them.
        m.encoder.layers = m.encoder.layers[:MAX_LAYER]
        m.eval()
        self.model = m

    @torch.inference_mode()
    def __call__(self, batch: list[np.ndarray]) -> np.ndarray:
        """batch: list of float32 mono 16k waveforms -> (B, EMB_DIM) float32."""
        lens = [len(x) for x in batch]
        T = max(lens)
        x = torch.zeros(len(batch), T, dtype=torch.float32)
        mask = torch.zeros(len(batch), T, dtype=torch.long)
        for i, w in enumerate(batch):
            # per-utterance normalisation, matching the model's preprocessing
            w = (w - w.mean()) / (w.std() + 1e-7)
            x[i, : len(w)] = torch.from_numpy(w)
            mask[i, : len(w)] = 1

        out = self.model(x, attention_mask=mask, output_hidden_states=True)
        hs = out.hidden_states
        # frame-level validity mask for correct masked statistics pooling
        fl = self.model._get_feat_extract_output_lengths(mask.sum(-1)).clamp(min=1)
        Tf = hs[0].shape[1]
        fmask = (torch.arange(Tf)[None, :] < fl[:, None]).float().unsqueeze(-1)
        denom = fmask.sum(1).clamp(min=1.0)

        feats = []
        for li in LAYERS:
            h = hs[li] * fmask
            mean = h.sum(1) / denom
            var = (h.pow(2).sum(1) / denom - mean.pow(2)).clamp(min=1e-8)
            feats.append(torch.cat([mean, var.sqrt()], dim=-1))
        return torch.cat(feats, dim=-1).numpy().astype(np.float32)
