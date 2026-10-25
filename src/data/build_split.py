"""Pass 2: build a speaker-disjoint, accent-stratified subset.

Why not use the dataset's own splits: every one of the 250 official test
speakers (and 230/232 validation speakers) also appears in train. Because each
speaker carries exactly one accent label, a model can score well there by
recognising voices rather than accents. We therefore re-split by speaker and
keep the official split only as a secondary, optimistic reference number.

Per-speaker caps matter as much as per-class caps: e.g. one Lithuanian speaker
owns 140 of that class's 178 clips, so without a cap the class collapses into
a single-voice prototype.
"""
from __future__ import annotations

import argparse, json, random, collections
from src.config import ACCENTS, DATA, MIN_DUR, MAX_DUR

TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}


def load_pool(allow: set[str] | None = None) -> list[dict]:
    """Pool every indexed clip, optionally restricted to a set of shards.

    The shard filter exists because download bandwidth, not compute, turned out
    to be the binding constraint: it lets training use the shards actually on
    disk instead of waiting for all 8.85 GB.
    """
    rows = []
    for s in ("train", "validation", "test"):
        for line in open(DATA / f"index_{s}.jsonl"):
            r = json.loads(line)
            if allow is not None and r["shard"] not in allow:
                continue
            r["orig_split"] = s
            rows.append(r)
    return [r for r in rows if MIN_DUR <= r["dur"] <= 30.0]


def assign_speakers(pool: list[dict], seed: int) -> dict[str, str]:
    """Greedily assign whole speakers to splits, per accent, by clip quota."""
    rng = random.Random(seed)
    by_acc: dict[int, dict[str, int]] = collections.defaultdict(collections.Counter)
    for r in pool:
        by_acc[r["accent"]][r["speaker"]] += 1

    spk_split: dict[str, str] = {}
    for acc, spks in by_acc.items():
        names = list(spks)
        rng.shuffle(names)
        # Largest speakers first makes the greedy quota fill far more even.
        names.sort(key=lambda s: -spks[s])
        total = sum(spks.values())
        quota = {k: v * total for k, v in TARGET.items()}
        got = {k: 0 for k in TARGET}
        # Guarantee val/test each receive at least one speaker when possible.
        order = ["train", "val", "test"]
        for i, s in enumerate(names):
            if len(names) >= 3 and i < 3:
                split = order[i] if i else "train"
                if i == 1: split = "val"
                if i == 2: split = "test"
            else:
                split = max(TARGET, key=lambda k: quota[k] - got[k])
            spk_split[s] = split
            got[split] += spks[s]
    return spk_split


def sample(rows: list[dict], per_class: int, per_speaker: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_acc = collections.defaultdict(list)
    for r in rows:
        by_acc[r["accent"]].append(r)
    out = []
    for acc, rs in by_acc.items():
        by_spk = collections.defaultdict(list)
        for r in rs:
            by_spk[r["speaker"]].append(r)
        for v in by_spk.values():
            rng.shuffle(v)
            # prefer clips in the model's comfortable duration band
            v.sort(key=lambda r: 0 if MIN_DUR <= r["dur"] <= MAX_DUR else 1)
        picked, si = [], 0
        spks = list(by_spk)
        rng.shuffle(spks)
        # round-robin across speakers -> even voice coverage inside each class
        while len(picked) < per_class:
            progressed = False
            for s in spks:
                bucket = by_spk[s]
                used = sum(1 for p in picked if p["speaker"] == s)
                if used < per_speaker and used < len(bucket):
                    picked.append(bucket[used]); progressed = True
                    if len(picked) >= per_class: break
            if not progressed: break
        out.extend(picked)
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--train-per-class", type=int, default=700)
    ap.add_argument("--train-per-speaker", type=int, default=110)
    ap.add_argument("--eval-per-class", type=int, default=160)
    ap.add_argument("--eval-per-speaker", type=int, default=45)
    ap.add_argument("--shards", nargs="*", default=None,
                    help="restrict to these shard filenames")
    a = ap.parse_args()

    pool = load_pool(set(a.shards) if a.shards else None)
    spk_split = assign_speakers(pool, a.seed)
    buckets = collections.defaultdict(list)
    for r in pool:
        buckets[spk_split[r["speaker"]]].append(r)

    manifests = {}
    for split in ("train", "val", "test"):
        pc, ps = ((a.train_per_class, a.train_per_speaker) if split == "train"
                  else (a.eval_per_class, a.eval_per_speaker))
        manifests[split] = sample(buckets[split], pc, ps, a.seed)

    for split, rows in manifests.items():
        out = DATA / f"manifest_{split}.jsonl"
        with out.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        hrs = sum(min(r["dur"], MAX_DUR) for r in rows) / 3600
        c = collections.Counter(ACCENTS[r["accent"]] for r in rows)
        print(f"\n== {split}: {len(rows)} clips, {len({r['speaker'] for r in rows})} speakers, "
              f"{hrs:.2f} h (capped at {MAX_DUR}s)")
        print("   " + ", ".join(f"{k}:{v}" for k, v in c.most_common()))

    # sanity: splits must share no speaker
    ss = {k: {r["speaker"] for r in v} for k, v in manifests.items()}
    print("\nspeaker overlap train/val:", len(ss["train"] & ss["val"]),
          "train/test:", len(ss["train"] & ss["test"]),
          "val/test:", len(ss["val"] & ss["test"]))
    tot = sum(sum(min(r["dur"], MAX_DUR) for r in v) for v in manifests.values()) / 3600
    print(f"TOTAL audio to encode: {tot:.2f} h  -> ~{tot/18.4*60:.0f} min at 18.4x realtime")


if __name__ == "__main__":
    main()
