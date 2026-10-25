"""End to end check of the trained system on held out audio.

Picks clips from the speaker disjoint TEST manifest, so every clip comes from a
speaker the model never heard, and runs the complete pipeline that the app runs.
"""
from __future__ import annotations

import io, json, random, sys, time
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from src.config import ACCENTS, DATA
from src.analysis.pipeline import Analyzer
from src.llm import offline_reply


def main(n: int = 5) -> None:
    rows = [json.loads(l) for l in open(DATA / "manifest_test.jsonl")]
    rows = [r for r in rows if 4.0 <= r["dur"] <= 10.0]
    random.Random(7).shuffle(rows)

    an = Analyzer()
    print("loading models", flush=True)
    t0 = time.time(); an.asr; an.pron; an.accent
    print(f"models ready in {time.time()-t0:.0f}s | accent model: {an.accent.available}\n")

    hits = 0; done = 0
    for r in rows:
        if done >= n:
            break
        try:
            tbl = pq.ParquetFile(str(DATA / "shards" / r["shard"])).read_row_group(
                r["rg"], columns=["audio"])
        except Exception:
            continue
        x, sr = sf.read(io.BytesIO(tbl.column("audio")[r["i"]].as_py()["bytes"]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        truth = ACCENTS[r["accent"]]
        t0 = time.time(); a = an.run(x); el = time.time() - t0
        done += 1
        if not a.get("ok"):
            print(f"[{done}] truth {truth:14s} -> not analysed: {a.get('reason')}")
            continue
        pred = a["accent"].get("predicted", "n/a")
        ok = pred == truth
        hits += ok
        print(f"[{done}] truth {truth:14s} pred {pred:14s} {'HIT ' if ok else 'miss'} "
              f"conf {a['accent'].get('confidence',0):.2f} | pron {a['pronunciation']['score']} "
              f"| fluency {a['fluency']['fluency']} | overall {a['overall']} | {el:.1f}s")
        print(f"     heard: {a['transcript'][:90]}")
    print(f"\ntop 1 accent on {done} held out clips: {hits}/{done}")
    if done:
        print("\n--- sample coaching reply ---")
        print(offline_reply(a, 0)[:900])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
