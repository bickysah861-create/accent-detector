"""Extract and cache WavLM embeddings for every clip in the manifests.

Download (network-bound, ~80 min) and encoding (CPU-bound, ~77 min) are
complementary, so a prefetch thread pulls shard N+1 while the main thread
encodes shard N. Wall time is therefore ~max(download, encode) rather than
their sum.
"""
from __future__ import annotations

import argparse, io, json, os, queue, threading, time, collections
import numpy as np
import pyarrow.parquet as pq
import requests
import soundfile as sf

from src.config import (CACHE, DATA, ENCODE_MAX_DUR, HF_REPO_URL, SHARD_DIR, SR)
from src.features.encoder import EMB_DIM, WavLMFeatures

SPLITS = ("train", "val", "test")


def load_manifests() -> dict[str, list[dict]]:
    out = {}
    for s in SPLITS:
        out[s] = [json.loads(l) for l in open(DATA / f"manifest_{s}.jsonl")]
    return out


def download(name: str, keep: bool) -> str:
    """Fetch a shard to local disk with resume support."""
    dest = SHARD_DIR / name
    url = f"{HF_REPO_URL}/data/{name}"
    head = requests.head(url, allow_redirects=True, timeout=60)
    total = int(head.headers["Content-Length"])
    if dest.exists() and dest.stat().st_size == total:
        return str(dest)
    for attempt in range(8):
        have = dest.stat().st_size if dest.exists() else 0
        if have == total:
            return str(dest)
        try:
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(head.url, headers=headers, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "ab" if have else "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
        except Exception as e:
            print(f"  [dl] {name} attempt {attempt} failed: {type(e).__name__}; resuming", flush=True)
            time.sleep(3)
    if not dest.exists() or dest.stat().st_size != total:
        raise RuntimeError(f"could not download {name}")
    return str(dest)


def crop(x: np.ndarray) -> np.ndarray:
    lim = int(ENCODE_MAX_DUR * SR)
    if len(x) > lim:                      # centre crop keeps the steadiest speech
        s = (len(x) - lim) // 2
        x = x[s:s + lim]
    return x


def save_cache(arrs: dict, done: dict, meta: dict) -> None:
    """Write embeddings and metadata atomically so a reader never sees a partial file."""
    CACHE.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        # numpy appends .npy when the name lacks it, so the temp name must
        # already end in .npy or the rename target will not exist.
        tmp = CACHE / f"emb_{s}.tmp.npy"
        np.save(tmp, arrs[s])
        os.replace(tmp, CACHE / f"emb_{s}.npy")
        tmpm = CACHE / f"meta_{s}.jsonl.tmp"
        with open(tmpm, "w") as fh:
            for r in meta[s]:
                fh.write(json.dumps(r) + "\n")
        os.replace(tmpm, CACHE / f"meta_{s}.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--keep-shards", action="store_true")
    a = ap.parse_args()

    mans = load_manifests()
    # index every wanted row by shard -> row-group -> row offset
    want: dict[str, dict[int, list[tuple]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    meta: dict[str, list] = {s: [] for s in SPLITS}
    slot: dict[tuple, tuple] = {}
    for s in SPLITS:
        for j, r in enumerate(mans[s]):
            want[r["shard"]][r["rg"]].append((r["i"], s, j))
            meta[s].append(r)
    arrs = {s: np.zeros((len(mans[s]), EMB_DIM), np.float32) for s in SPLITS}
    done = {s: np.zeros(len(mans[s]), bool) for s in SPLITS}

    shard_names = sorted(want)
    print(f"shards to process: {len(shard_names)}, clips: {sum(len(v) for v in mans.values())}")

    # ---- prefetch thread: keeps at most 2 shards queued ahead of the encoder
    q: queue.Queue = queue.Queue(maxsize=2)

    def producer():
        for n in shard_names:
            try:
                q.put((n, download(n, a.keep_shards)))
            except Exception as e:
                q.put((n, None)); print(f"  [dl] FAILED {n}: {e}", flush=True)
        q.put((None, None))

    threading.Thread(target=producer, daemon=True).start()

    enc = WavLMFeatures(threads=a.threads)
    t_start = time.time()
    n_done = 0
    total_clips = sum(len(v) for v in mans.values())
    audio_s = 0.0
    BUF = 384
    buf_w: list = []
    buf_t: list = []

    def flush(bw, bt) -> int:
        if not bw:
            return 0
        order = sorted(range(len(bw)), key=lambda k: len(bw[k]))
        n = 0
        for b0 in range(0, len(order), a.batch):
            idx = order[b0:b0 + a.batch]
            emb = enc([bw[k] for k in idx])
            for row, k in zip(emb, idx):
                s_, j_ = bt[k]
                arrs[s_][j_] = row; done[s_][j_] = True
            n += len(idx)
        bw.clear(); bt.clear()
        return n

    while True:
        name, path = q.get()
        if name is None:
            break
        if path is None:
            continue
        pf = pq.ParquetFile(path)
        for rg, items in sorted(want[name].items()):
            tbl = pf.read_row_group(rg, columns=["audio"])
            col = tbl.column("audio")
            for i, s, j in items:
                b = col[i].as_py()["bytes"]
                x, sr = sf.read(io.BytesIO(b), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(1)
                x = crop(x)
                buf_w.append(x); buf_t.append((s, j)); audio_s += len(x) / SR
            # A row group holds only ~25 wanted clips, so batching within one
            # spends most of each batch on padding. Buffer across row groups and
            # sort globally by length: near-uniform batches, far less waste.
            if len(buf_w) >= BUF:
                n_done += flush(buf_w, buf_t)
            el = time.time() - t_start
            if rg % 6 == 0:
                rate = audio_s / el if el else 0
                eta = (total_clips - n_done) / max(n_done / el, 1e-9) / 60
                print(f"  {name} rg{rg:02d} | {n_done}/{total_clips} clips "
                      f"| {rate:.1f}x realtime | ETA {eta:.0f} min", flush=True)
        # Checkpoint after every shard. Download bandwidth is the bottleneck, so
        # this makes the cache usable at any moment: rows that are not encoded
        # yet stay all zero and the trainer drops them automatically.
        n_done += flush(buf_w, buf_t)
        save_cache(arrs, done, meta)
        print(f"  [checkpoint] {name} done | "
              + " ".join(f"{s_}:{int(done[s_].sum())}/{len(done[s_])}" for s_ in SPLITS),
              flush=True)
        if not a.keep_shards:
            os.remove(path)

    n_done += flush(buf_w, buf_t)
    save_cache(arrs, done, meta)
    for s in SPLITS:
        print(f"{s}: {int(done[s].sum())}/{len(done[s])} encoded -> emb_{s}.npy {arrs[s].shape}")
    print(f"TOTAL {time.time()-t_start:.0f}s for {audio_s/3600:.2f} h audio")


if __name__ == "__main__":
    main()
