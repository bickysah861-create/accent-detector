"""Train the accent classifier head on cached WavLM embeddings.

Because the encoder is frozen and its features are cached, one training run
costs seconds rather than hours. The time budget is therefore spent on a small
architecture search plus a seed ensemble instead of a single long pass.

Every model selection decision uses the speaker disjoint validation split. The
test split is evaluated once, at the very end.

Training publishes live telemetry (loss, learned SSL layer weights, per module
activation and gradient statistics, per class F1, confusion matrix) for the
dashboard at web/train.html.
"""
from __future__ import annotations

import argparse, itertools, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)

from src.config import ACCENTS, CACHE, HEAD_DIR
from src.features.encoder import LAYERS
from src.train.model import AccentHead, LinearProbe
from src.train.telemetry import ActivationProbe, Telemetry, grad_norms

LAYER_LR_MULT = 40.0


def load(split: str):
    X = np.load(CACHE / f"emb_{split}.npy")
    meta = [json.loads(l) for l in open(CACHE / f"meta_{split}.jsonl")]
    y = np.array([m["accent"] for m in meta], np.int64)
    keep = np.abs(X).sum(1) > 0          # drop clips that failed to encode
    return X[keep], y[keep]


def standardise(Xtr, *others):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    return [(Xtr - mu) / sd] + [(o - mu) / sd for o in others], (mu, sd)


def evaluate(model, X, y, bs=1024):
    model.eval()
    ps = []
    with torch.inference_mode():
        for i in range(0, len(X), bs):
            ps.append(F.softmax(model(torch.from_numpy(X[i:i + bs])), -1).numpy())
    p = np.concatenate(ps)
    pred = p.argmax(1)
    return {"acc": accuracy_score(y, pred),
            "bal_acc": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0)}, p


def probe_targets(model):
    if isinstance(model, AccentHead):
        return {"layer_mix": model.norm, "mlp_block1": model.net[0],
                "mlp_block2": model.net[3], "classifier": model.out}
    return {"norm": model.norm, "classifier": model.fc}


def train_one(Xtr, ytr, Xva, yva, *, arch, hidden, dropout, lr, wd, epochs,
              mixup, smooth, seed, n_classes, tel: Telemetry | None = None,
              phase: str = ""):
    torch.manual_seed(seed); np.random.seed(seed)
    model = (AccentHead(n_classes, hidden, dropout) if arch == "mlp"
             else LinearProbe(n_classes, Xtr.shape[1], dropout))
    # The layer mixing logits receive gradients roughly two orders of magnitude
    # smaller than the MLP weights, because the classifier can compensate for a
    # poor mix internally. At a shared learning rate they never leave their
    # uniform initialisation, so the "learned" layer weighting would be a
    # decoration rather than a result. They get their own, much larger step size
    # and no weight decay (decay would actively pull the mix back to uniform).
    mix = [p_ for n_, p_ in model.named_parameters() if n_ == "layer_logits"]
    rest = [p_ for n_, p_ in model.named_parameters() if n_ != "layer_logits"]
    groups = [{"params": rest, "lr": lr, "weight_decay": wd}]
    if mix:
        groups.append({"params": mix, "lr": lr * LAYER_LR_MULT, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
    # inverse frequency weights so rare accents are not simply ignored
    cnt = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    w = torch.from_numpy((cnt.sum() / np.maximum(cnt, 1)) ** 0.5)
    w = w / w.mean()

    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    n, bs = len(Xt), 128
    steps = max(1, n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups],
        total_steps=epochs * steps, pct_start=0.25)
    probe = ActivationProbe(model, probe_targets(model)) if tel else None
    if tel:
        tel.new_run(phase or "training", epochs)
    best, best_state = -1.0, None

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        run_loss, gn = 0.0, {}
        for s in range(steps):
            idx = perm[s * bs:(s + 1) * bs]
            xb, yb = Xt[idx], yt[idx]
            if mixup > 0:
                lam = float(np.random.beta(mixup, mixup))
                pi = torch.randperm(len(xb))
                xb = lam * xb + (1 - lam) * xb[pi]
                lg = model(xb)
                loss = (lam * F.cross_entropy(lg, yb, weight=w, label_smoothing=smooth)
                        + (1 - lam) * F.cross_entropy(lg, yb[pi], weight=w, label_smoothing=smooth))
            else:
                loss = F.cross_entropy(model(xb), yb, weight=w, label_smoothing=smooth)
            opt.zero_grad(); loss.backward()
            if tel and s == steps - 1:
                gn = grad_norms(model)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            run_loss += float(loss)
            if tel:
                tel.step(float(loss))

        m, p = evaluate(model, Xva, yva)
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if tel:
            pcf1 = f1_score(yva, p.argmax(1), average=None,
                            labels=range(n_classes), zero_division=0)
            tel.set(phase=phase, status="training",
                    best={"val_macro_f1": round(best, 5)})
            lw = model.layer_weights() if isinstance(model, AccentHead) else []
            tel.set(arch=arch)
            tel.epoch_end(ep, run_loss / steps, sched.get_last_lr()[0], m,
                          lw, gn, probe.stats if probe else {}, list(pcf1))
            tel.state["confusion"] = confusion_matrix(
                yva, p.argmax(1), labels=range(n_classes)).tolist()
            tel.flush()

    if probe:
        probe.close()
    model.load_state_dict(best_state)
    return model, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--ensemble", type=int, default=7)
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--search-epochs", type=int, default=30,
                    help="shorter runs for screening; the winner is retrained in full")
    a = ap.parse_args()

    Xtr, ytr = load("train"); Xva, yva = load("val"); Xte, yte = load("test")
    (Xtr, Xva, Xte), (mu, sd) = standardise(Xtr, Xva, Xte)
    n_classes = len(ACCENTS)
    print(f"train {Xtr.shape} val {Xva.shape} test {Xte.shape} classes {n_classes}", flush=True)

    tel = Telemetry(ACCENTS, LAYERS, a.epochs, run_label="accent head")
    tel.set(status="running", message=f"train {Xtr.shape[0]} / val {Xva.shape[0]} / test {Xte.shape[0]}")
    tel.flush()

    grid = [dict(arch="mlp", hidden=512, dropout=0.35, lr=3e-4, wd=1e-2, mixup=0.4, smooth=0.1)]
    if a.search:
        # `hidden` is meaningless for the linear probe, so enumerating it there
        # would just retrain identical models. Build the two arms separately.
        grid = [dict(arch="mlp", hidden=H, dropout=D, lr=L, wd=1e-2, mixup=M, smooth=0.1)
                for H, D, L, M in itertools.product(
                    [512, 1024], [0.3, 0.45], [3e-4, 1e-3], [0.0, 0.4])]
        grid += [dict(arch="linear", hidden=512, dropout=D, lr=L, wd=1e-2, mixup=M, smooth=0.1)
                 for D, L, M in itertools.product([0.2, 0.4], [1e-3, 3e-3], [0.0, 0.4])]

    results = []
    for gi, cfg in enumerate(grid, 1):
        t0 = time.time()
        # Screening runs are short. Ranking is stable well before convergence,
        # and the winning configuration is retrained for the full schedule below.
        ep = a.search_epochs if len(grid) > 1 else a.epochs
        _, f1 = train_one(Xtr, ytr, Xva, yva, epochs=ep, seed=0, n_classes=n_classes,
                          tel=tel, phase=f"search {gi}/{len(grid)}", **cfg)
        results.append((f1, cfg))
        tel.state["trials"].append({"cfg": cfg, "val_macro_f1": round(f1, 5),
                                    "secs": round(time.time() - t0, 1)})
        tel.flush()
        print(f"  [{gi}/{len(grid)}] val macroF1 {f1:.4f} ({time.time()-t0:.0f}s) {cfg}", flush=True)

    results.sort(key=lambda r: -r[0])
    best_cfg = results[0][1]
    print(f"\nBEST val macroF1 {results[0][0]:.4f}: {best_cfg}", flush=True)
    tel.set(message=f"best config macro F1 {results[0][0]:.4f}"); tel.flush()

    models, pv_all, pt_all = [], [], []
    for s in range(a.ensemble):
        m, _ = train_one(Xtr, ytr, Xva, yva, epochs=a.epochs, seed=s, n_classes=n_classes,
                         tel=tel, phase=f"ensemble seed {s+1}/{a.ensemble}", **best_cfg)
        mv, pv = evaluate(m, Xva, yva); mt, pt = evaluate(m, Xte, yte)
        models.append(m); pv_all.append(pv); pt_all.append(pt)
        tel.state["seeds"].append({"seed": s, "val_macro_f1": round(mv["macro_f1"], 5),
                                   "test_macro_f1": round(mt["macro_f1"], 5)})
        tel.flush()
        print(f"  seed {s}: val F1 {mv['macro_f1']:.4f} | test F1 {mt['macro_f1']:.4f}", flush=True)

    def agg(ps, y):
        pred = np.mean(ps, 0).argmax(1)
        return dict(acc=accuracy_score(y, pred), bal_acc=balanced_accuracy_score(y, pred),
                    macro_f1=f1_score(y, pred, average="macro", zero_division=0))

    ev, et = agg(pv_all, yva), agg(pt_all, yte)
    pred_te = np.mean(pt_all, 0).argmax(1)
    print(f"\nENSEMBLE val : acc {ev['acc']:.4f} bal {ev['bal_acc']:.4f} macroF1 {ev['macro_f1']:.4f}")
    print(f"ENSEMBLE test: acc {et['acc']:.4f} bal {et['bal_acc']:.4f} macroF1 {et['macro_f1']:.4f}\n")
    print(classification_report(yte, pred_te, labels=range(n_classes),
                                target_names=ACCENTS, zero_division=0, digits=3))
    if best_cfg["arch"] == "mlp":
        print("learned SSL layer weights:",
              dict(zip([f"L{i}" for i in LAYERS],
                       [round(x, 3) for x in models[0].layer_weights()])))

    HEAD_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dicts": [m.state_dict() for m in models], "cfg": best_cfg,
                "mu": mu, "sd": sd, "accents": ACCENTS,
                "metrics": {"val": ev, "test": et}}, HEAD_DIR / "head.pt")
    json.dump({"val": ev, "test": et, "cfg": best_cfg,
               "confusion_test": confusion_matrix(yte, pred_te).tolist(),
               "per_class_f1_test": f1_score(yte, pred_te, average=None,
                                             labels=range(n_classes), zero_division=0).tolist()},
              open(HEAD_DIR / "metrics.json", "w"), indent=2)

    tel.set(status="done", phase="finished",
            message=(f"test macro F1 {et['macro_f1']:.4f} | acc {et['acc']:.4f} | "
                     f"balanced acc {et['bal_acc']:.4f}"),
            final={"val": ev, "test": et})
    tel.state["confusion"] = confusion_matrix(yte, pred_te, labels=range(n_classes)).tolist()
    tel.state["per_class_f1"] = f1_score(yte, pred_te, average=None,
                                         labels=range(n_classes), zero_division=0).tolist()
    tel.flush()
    print(f"\nsaved {HEAD_DIR/'head.pt'}")


if __name__ == "__main__":
    main()
