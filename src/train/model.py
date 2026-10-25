"""Classifier head trained on frozen WavLM statistics-pooled embeddings."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.encoder import DIM_PER_LAYER, LAYERS


class AccentHead(nn.Module):
    """Learned layer-weighting + MLP.

    The four cached layers are not equally informative and the best mix is not
    known a priori, so a softmax-weighted sum over layers is learned jointly
    with the classifier instead of being fixed by hand.
    """

    def __init__(self, n_classes: int, hidden: int = 512, dropout: float = 0.35,
                 n_layers: int = len(LAYERS), dim: int = DIM_PER_LAYER):
        super().__init__()
        self.n_layers, self.dim = n_layers, dim
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden // 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = x.view(b, self.n_layers, self.dim)
        w = F.softmax(self.layer_logits, dim=0).view(1, -1, 1)
        x = (x * w).sum(1)
        return self.out(self.net(self.norm(x)))

    def layer_weights(self) -> list[float]:
        return F.softmax(self.layer_logits.detach(), dim=0).tolist()


class LinearProbe(nn.Module):
    """Plain linear probe on the concatenated layers.

    A strong, very cheap baseline: linear probes on SSL features are often
    within a point or two of far larger heads, so it is worth measuring rather
    than assuming the MLP wins.
    """

    def __init__(self, n_classes: int, in_dim: int, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(self.drop(self.norm(x)))
