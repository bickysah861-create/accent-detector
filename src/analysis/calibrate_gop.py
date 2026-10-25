"""Calibrate the GOP to score mapping against real native speech.

The raw GOP is an unbounded log ratio, so turning it into a 0 to 100 score
needs a scale. Picking that constant by eye is guesswork: the acoustic model is
wav2vec2 trained on TIMIT read speech, while this corpus is spontaneous
conversation, so raw GOP values sit lower than intuition suggests and a guessed
constant makes every speaker look bad.

This script measures the distribution of mean GOP over native English speakers
in the corpus and solves for the constant that places the median native speaker
at a chosen target score. Scores then mean something: "around the median native
speaker" rather than "an arbitrary function of a log ratio".

Run after the shards are present:
    python -m src.analysis.calibrate_gop --n 120 --target 86
"""
from __future__ import annotations

import argparse, glob, io, json, math, os, random
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from src.config import ACCENTS, DATA, MODELS, NATIVE_VARIETIES, SR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="clips to measure")
    ap.add_argument("--target", type=float, default=86.0,
                    help="score the median native speaker should receive")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()

    have = {os.path.basename(p) for p in glob.glob(str(DATA / "shards" / "*.parquet"))}
    rows = []
    for split in ("train", "validation", "test"):
        f = DATA / f"index_{split}.jsonl"
        if not f.exists():
            continue
        for line in open(f):
            r = json.loads(line)
            if r["shard"] in have and ACCENTS[r["accent"]] in NATIVE_VARIETIES \
               and 3.0 <= r["dur"] <= 10.0:
                rows.append(r)
    random.Random(0).shuffle(rows)
    rows = rows[: a.n]
    print(f"measuring {len(rows)} native English clips")

    from src.analysis.phonemes import PronunciationScorer
    ps = PronunciationScorer(threads=a.threads)

    # Group by shard/row-group so each row group is read once.
    by = {}
    for r in rows:
        by.setdefault((r["shard"], r["rg"]), []).append(r)

    gops = []
    for (shard, rg), items in by.items():
        tbl = pq.ParquetFile(str(DATA / "shards" / shard)).read_row_group(
            rg, columns=["audio", "raw_text"])
        for r in items:
            b = tbl.column("audio")[r["i"]].as_py()["bytes"]
            txt = tbl.column("raw_text")[r["i"]].as_py()
            x, sr = sf.read(io.BytesIO(b), dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            # Use the corpus reference transcript, not ASR output, so the
            # measurement reflects pronunciation alone and not recognition error.
            out = ps.score(x, txt)
            if out.get("ok"):
                gops.append(float(np.mean([p["gop"] for p in out["phones"]])))
        print(f"  {shard} rg{rg}: {len(gops)} measured", flush=True)

    g = np.array(gops)
    med = float(np.median(g))
    # 100 * exp(med / k) = target  ->  k = med / ln(target/100)
    k = med / math.log(a.target / 100.0)
    print(f"\nmean GOP over native speech: median {med:.3f} "
          f"p10 {np.percentile(g,10):.3f} p90 {np.percentile(g,90):.3f} (n={len(g)})")
    print(f"calibration constant k = {k:.3f} "
          f"(was 3.2) so median native scores {a.target}")
    for q in (10, 25, 50, 75, 90):
        v = float(np.percentile(g, q))
        print(f"  p{q:<3d} gop {v:7.3f} -> score {100*math.exp(v/k):5.1f}")

    MODELS.mkdir(parents=True, exist_ok=True)
    out = MODELS / "gop_calibration.json"
    json.dump({"k": k, "median_native_gop": med, "target": a.target,
               "n": len(g), "percentiles": {str(q): float(np.percentile(g, q))
                                            for q in (5, 10, 25, 50, 75, 90, 95)}},
              open(out, "w"), indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
