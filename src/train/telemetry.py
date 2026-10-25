"""Live telemetry for the training dashboard.

Training writes a small JSON snapshot after every epoch (and periodically
inside an epoch). The dashboard polls that file, so the visualiser stays
completely decoupled from the training loop: if the dashboard is closed, or
opened halfway through, nothing about training changes.

Writes are atomic (temp file plus rename) so the dashboard can never read a
half written snapshot.
"""
from __future__ import annotations

import json, os, time
from pathlib import Path

import numpy as np
import torch

LIVE = Path(__file__).resolve().parents[2] / "logs" / "live"
LIVE.mkdir(parents=True, exist_ok=True)
STATE = LIVE / "train_state.json"


class Telemetry:
    def __init__(self, accents: list[str], ssl_layers: list[int], total_epochs: int,
                 run_label: str = ""):
        self.t0 = time.time()
        self.state: dict = {
            "status": "starting",
            "run_label": run_label,
            "accents": accents,
            "ssl_layers": [f"L{i}" for i in ssl_layers],
            "total_epochs": total_epochs,
            "epoch": 0,
            "phase": "",
            "history": [],          # per epoch scalars
            "step_loss": [],        # fine grained loss trace
            "layer_weights": [],    # learned softmax weights over SSL layers
            "grad_norms": {},       # per module gradient magnitude
            "act_stats": {},        # per module activation mean/std
            "per_class_f1": [],
            "confusion": [],
            "best": {},
            "trials": [],           # hyper parameter search results
            "seeds": [],            # ensemble member scores
            "elapsed_s": 0.0,
            "message": "",
        }
        self.flush()

    # ------------------------------------------------------------------
    def set(self, **kw):
        self.state.update(kw)

    def new_run(self, label: str, total_epochs: int) -> None:
        """Begin a fresh curve.

        The search trains dozens of models one after another. Appending them all
        to one history draws a sawtooth that says nothing; each run gets its own
        curve, and the trials and seeds tables carry the cross run comparison.
        """
        self.state["history"] = []
        self.state["step_loss"] = []
        self.state["phase"] = label
        self.state["total_epochs"] = total_epochs
        self.state["epoch"] = 0
        self.state["run_label"] = label
        self.flush()

    def epoch_end(self, epoch: int, train_loss: float, lr: float, metrics: dict,
                  layer_weights: list[float] | None, grad_norms: dict,
                  act_stats: dict, per_class_f1: list[float]):
        self.state["epoch"] = epoch
        self.state["history"].append({
            "epoch": epoch, "loss": round(float(train_loss), 5), "lr": float(lr),
            "val_acc": round(float(metrics["acc"]), 5),
            "val_bal_acc": round(float(metrics["bal_acc"]), 5),
            "val_macro_f1": round(float(metrics["macro_f1"]), 5),
        })
        # An empty list is meaningful: the active model has no layer mixing, so
        # the panel must clear rather than keep showing the previous model's mix.
        if layer_weights is not None:
            self.state["layer_weights"] = [round(float(w), 5) for w in layer_weights]
        self.state["grad_norms"] = {k: round(float(v), 6) for k, v in grad_norms.items()}
        self.state["act_stats"] = act_stats
        self.state["per_class_f1"] = [round(float(x), 4) for x in per_class_f1]
        self.state["elapsed_s"] = round(time.time() - self.t0, 1)
        self.flush()

    def step(self, loss: float, keep: int = 600):
        self.state["step_loss"].append(round(float(loss), 5))
        if len(self.state["step_loss"]) > keep:      # keep the trace bounded
            self.state["step_loss"] = self.state["step_loss"][-keep:]

    def flush(self):
        self.state["elapsed_s"] = round(time.time() - self.t0, 1)
        tmp = STATE.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(self.state, fh)
        os.replace(tmp, STATE)       # atomic: readers never see a partial file


# ---------------------------------------------------------------- probes
class ActivationProbe:
    """Forward hooks that record per module activation statistics.

    This is what lets the dashboard show each layer actually *doing* something:
    a layer whose activations collapse toward zero, or whose gradients vanish,
    is visible immediately rather than only via a flat loss curve.
    """

    def __init__(self, model: torch.nn.Module, names: dict[str, torch.nn.Module]):
        self.stats: dict[str, dict] = {}
        self.handles = []
        for name, mod in names.items():
            self.handles.append(mod.register_forward_hook(self._make(name)))

    def _make(self, name):
        def hook(_m, _i, out):
            if not isinstance(out, torch.Tensor):
                return
            with torch.no_grad():
                o = out.detach()
                self.stats[name] = {
                    "mean": round(float(o.mean()), 5),
                    "std": round(float(o.std()), 5),
                    # fraction of units that are effectively off this batch
                    "dead_frac": round(float((o.abs() < 1e-3).float().mean()), 4),
                    "absmax": round(float(o.abs().max()), 4),
                }
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def grad_norms(model: torch.nn.Module) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            out[name] = float(p.grad.detach().norm())
    return out
