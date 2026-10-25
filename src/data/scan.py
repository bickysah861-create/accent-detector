"""Pass 1: scan remote parquet metadata to build a lightweight sample index.

Reads only the `accent`, `speaker_id` and `duration` columns (a few MB across
all 28 shards) so we can choose a balanced, speaker-disjoint subset *before*
downloading any audio.
"""
from __future__ import annotations

import argparse
import json
import sys

import pyarrow.parquet as pq

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from src.config import ACCENTS, DATA, HF_REPO_URL, N_TRAIN_SHARDS
from src.data.http_file import HttpRangeFile

META_COLS = ["accent", "speaker_id", "duration"]


def shard_urls(split: str) -> list[tuple[str, str]]:
    if split == "train":
        return [
            (f"train-{i:05d}-of-{N_TRAIN_SHARDS:05d}.parquet",
             f"{HF_REPO_URL}/data/train-{i:05d}-of-{N_TRAIN_SHARDS:05d}.parquet")
            for i in range(N_TRAIN_SHARDS)
        ]
    name = f"{split}-00000-of-00001.parquet"
    return [(name, f"{HF_REPO_URL}/data/{name}")]


def scan_split(split: str) -> list[dict]:
    rows: list[dict] = []
    for name, url in shard_urls(split):
        f = HttpRangeFile(url)
        pf = pq.ParquetFile(f)
        n_rg = pf.metadata.num_row_groups
        offset = 0
        for rg in range(n_rg):
            tbl = pf.read_row_group(rg, columns=META_COLS)
            acc = tbl.column("accent").to_pylist()
            spk = tbl.column("speaker_id").to_pylist()
            dur = tbl.column("duration").to_pylist()
            for i, (a, s, d) in enumerate(zip(acc, spk, dur)):
                rows.append({
                    "shard": name, "rg": rg, "i": i, "gidx": offset + i,
                    "accent": int(a), "speaker": s, "dur": float(d),
                })
            offset += tbl.num_rows
        print(f"  {name}: {n_rg} row-groups, {offset} rows "
              f"(running total {len(rows)})", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["validation", "test", "train"])
    args = ap.parse_args()

    for split in args.splits:
        print(f"=== scanning {split}", flush=True)
        rows = scan_split(split)
        out = DATA / f"index_{split}.jsonl"
        with out.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        # quick distribution report
        from collections import Counter
        c = Counter(ACCENTS[r["accent"]] for r in rows)
        spk = len({r["speaker"] for r in rows})
        hrs = sum(r["dur"] for r in rows) / 3600
        print(f"--> {out.name}: {len(rows)} rows, {spk} speakers, {hrs:.1f} h")
        for k, v in c.most_common():
            print(f"     {k:15s} {v}")


if __name__ == "__main__":
    main()
