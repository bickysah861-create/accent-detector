"""Standalone resumable shard downloader.

The extractor's prefetch queue is shallow, so while it encodes shards already on
disk it is not downloading anything. Running the large downloads separately lets
network and CPU work at the same time instead of taking turns.
"""
from __future__ import annotations

import argparse, os, sys, time
import requests

from src.config import HF_REPO_URL, SHARD_DIR


def fetch(name: str, retries: int = 200) -> None:
    dest = SHARD_DIR / name
    url = f"{HF_REPO_URL}/data/{name}"
    head = requests.head(url, allow_redirects=True, timeout=60)
    total = int(head.headers["Content-Length"])
    t0, start_have = time.time(), (dest.stat().st_size if dest.exists() else 0)
    for attempt in range(retries):
        have = dest.stat().st_size if dest.exists() else 0
        if have >= total:
            print(f"{name}: complete ({total/1e6:.0f} MB)", flush=True)
            return
        try:
            h = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(head.url, headers=h, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "ab" if have else "wb") as fh:
                    last = time.time()
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                        if time.time() - last > 30:
                            got = fh.tell()
                            rate = (got - start_have) / max(time.time() - t0, 1)
                            eta = (total - got) / max(rate, 1) / 60
                            print(f"{name}: {got/1e6:.0f}/{total/1e6:.0f} MB "
                                  f"{rate/1024:.0f} KB/s ETA {eta:.0f} min", flush=True)
                            last = time.time()
        except Exception as e:
            print(f"{name}: {type(e).__name__}, resuming", flush=True)
            time.sleep(4)
    print(f"{name}: gave up", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    a = ap.parse_args()
    for s in a.shards:
        fetch(s)
    print("ALL SHARDS DONE", flush=True)
