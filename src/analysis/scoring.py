"""Turns raw measurements into the rubric the learner actually sees.

Each sub score is a bounded, monotonic function of one measured quantity, so a
number can always be traced back to the evidence that produced it.
"""
from __future__ import annotations


def _band(x: float, lo: float, hi: float) -> float:
    """1.0 inside [lo, hi], decaying smoothly outside it."""
    if lo <= x <= hi:
        return 1.0
    d = (lo - x) / max(lo, 1e-6) if x < lo else (x - hi) / max(hi, 1e-6)
    return max(0.0, 1.0 - min(d, 1.0))


def fluency_score(pros: dict) -> dict:
    # Comfortable conversational English sits near 3.5 to 5.5 syllables/second
    # of articulation, with pauses under about a quarter of total time.
    rate = _band(pros.get("articulation_syl_per_s", 0.0), 3.2, 5.6)
    pause = _band(pros.get("pause_ratio", 1.0), 0.0, 0.28)
    longest = _band(pros.get("longest_pause_s", 0.0), 0.0, 1.1)
    expr = _band(pros.get("f0_range_semitones", 0.0), 4.0, 16.0)
    total = 100 * (0.34 * rate + 0.28 * pause + 0.16 * longest + 0.22 * expr)
    return {
        "fluency": round(total, 1),
        "components": {
            "pace": round(100 * rate, 1),
            "pausing": round(100 * pause, 1),
            "continuity": round(100 * longest, 1),
            "expressiveness": round(100 * expr, 1),
        },
    }


def overall(pron: float | None, fluency: float, clarity: float) -> float:
    if pron is None:
        return round(0.55 * fluency + 0.45 * clarity, 1)
    return round(0.45 * pron + 0.3 * fluency + 0.25 * clarity, 1)


def grade(score: float) -> str:
    for cut, name in ((88, "excellent"), (76, "strong"), (62, "solid"),
                      (48, "developing")):
        if score >= cut:
            return name
    return "early"
