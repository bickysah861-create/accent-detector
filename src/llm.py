"""Conversational reply layer.

This module is the ONLY place a language model may be used, and it is used for
wording alone. Every number it mentions is handed to it already computed by the
analysis pipeline; it never estimates, scores or judges pronunciation itself.

With no API key present the offline writer runs, which composes the reply from
the same metrics. The product therefore works fully offline, and enabling an
API only changes phrasing, never scores.
"""
from __future__ import annotations

import os
import random

PROMPTS = [
    "What did you do this weekend?",
    "Describe the room you are sitting in right now.",
    "What is something you would like to learn this year?",
    "Tell me about a meal you really enjoyed recently.",
    "If you could travel anywhere tomorrow, where would you go?",
    "What is a film or book you would recommend, and why?",
    "Describe your morning routine.",
    "What is the most useful thing you own?",
    "Talk about a person who has influenced you.",
    "What would you change about your city?",
]

GROUP_TIPS = {
    "dental fricatives (th)":
        "Let the tongue tip touch the back of the upper teeth and keep the air flowing, "
        "as in think and this. Many languages replace these with t, d, s or z.",
    "rhotic /r/":
        "Curl the tongue back slightly without letting it touch the roof of the mouth, and "
        "keep the lips a little rounded.",
    "lateral /l/":
        "Press the tongue tip firmly to the ridge behind the upper teeth and let the sound "
        "escape around the sides.",
    "labiodental /v/-/w/":
        "For v the upper teeth rest on the lower lip; for w the lips round with no teeth "
        "contact at all.",
    "tense/lax vowel pairs":
        "Contrast the long tense vowel in sheep with the short lax vowel in ship. The "
        "difference is tongue height and length, not just duration.",
    "open front vowels":
        "Drop the jaw further for the vowel in cat, and keep it higher and more relaxed for bet.",
    "open back vowels":
        "Keep the tongue low and back, with the lips relaxed rather than rounded.",
    "velar nasal (ng)":
        "Let the back of the tongue meet the soft palate and release the sound through the "
        "nose without adding a hard g at the end.",
    "voiced sibilants":
        "Keep the vocal folds vibrating right through the sound so z does not harden into s.",
    "voiceless sibilants":
        "Aim a narrow, steady stream of air along the centre of the tongue.",
    "nasals": "Let the sound resonate through the nose while the lips or tongue hold the closure.",
    "stops": "Make a complete closure, then release it cleanly with a small burst of air.",
}


def _pace_note(p: dict) -> str | None:
    a = p.get("articulation_syl_per_s", 0)
    if a and a < 3.2:
        return (f"You are speaking at about {a} syllables per second, which is on the "
                "deliberate side. A little more momentum will sound more natural.")
    if a > 5.6:
        return (f"You are moving quickly at about {a} syllables per second. Easing off "
                "slightly will give your consonants room to land.")
    return None


def _pause_note(p: dict) -> str | None:
    if p.get("pause_ratio", 0) > 0.32:
        return (f"Silence took up {int(p['pause_ratio']*100)} percent of the recording, "
                f"with the longest gap at {p['longest_pause_s']} seconds. Try planning a "
                "whole phrase before you begin it.")
    return None


def _pitch_note(p: dict) -> str | None:
    r = p.get("f0_range_semitones", 0)
    if r and r < 4:
        return (f"Your pitch moved across roughly {r} semitones, which reads as quite flat. "
                "Lifting key words will make your meaning easier to follow.")
    return None


def offline_reply(a: dict, turn: int) -> str:
    """Compose the coaching reply from computed metrics alone."""
    if not a.get("ok"):
        return ("I could not hear speech in that recording. Check the microphone and try "
                "again, speaking for at least three or four seconds.")

    out: list[str] = []
    t = a["transcript"].strip()
    out.append(f'I heard: "{t}"')

    acc = a.get("accent", {})
    if acc.get("available"):
        top = acc["top"][0]
        pct = int(top["prob"] * 100)
        line = (f"Your accent profile points to **{top['accent']}** at {pct} percent "
                f"confidence ({acc['certainty']} certainty)")
        if len(acc["top"]) > 1 and acc["certainty"] != "high":
            second = acc["top"][1]
            line += f", with {second['accent']} the next closest at {int(second['prob']*100)} percent"
        out.append(line + ".")

    p = a["pronunciation"]
    if p.get("score") is not None:
        out.append(f"Pronunciation scored **{p['score']} out of 100**, measured phoneme by "
                   f"phoneme against what the words should sound like.")
        if p["weak_groups"]:
            g = p["weak_groups"][0]
            tip = GROUP_TIPS.get(g["group"], "")
            out.append(f"The weakest area was {g['group']} at {g['score']} out of 100. {tip}")
        if p["weak_words"]:
            ws = ", ".join(f"{w['word']} ({w['score']})" for w in p["weak_words"][:3])
            out.append(f"Words worth repeating: {ws}.")
    elif p.get("reason"):
        out.append(f"Pronunciation detail was unavailable because the {p['reason']}.")

    f = a["fluency"]
    out.append(f"Fluency came out at **{f['fluency']} out of 100** "
               f"(pace {f['components']['pace']}, pausing {f['components']['pausing']}, "
               f"expressiveness {f['components']['expressiveness']}).")
    for note in (_pace_note(a["prosody"]), _pause_note(a["prosody"]), _pitch_note(a["prosody"])):
        if note:
            out.append(note)

    out.append(f"Overall this is **{a['overall']} out of 100**, which I would call {a['grade']}.")
    out.append("")
    out.append(random.Random(turn).choice(PROMPTS))
    return "\n\n".join(out)


def reply(analysis: dict, turn: int = 0) -> dict:
    """Return the coaching reply plus the backend that wrote it."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        try:
            return {"text": _api_reply(analysis, key), "backend": "anthropic"}
        except Exception:
            pass          # any API problem falls back to the offline writer
    return {"text": offline_reply(analysis, turn), "backend": "offline"}


def _api_reply(a: dict, key: str) -> str:
    """Phrase the already computed metrics. The model is told not to invent any."""
    import json, urllib.request

    sys_prompt = (
        "You are a warm, concise pronunciation coach. You are given measurements that were "
        "already computed by a speech model. Use ONLY those numbers. Never invent a score, "
        "never re-judge pronunciation yourself, and never contradict the measurements. "
        "Write four short paragraphs: what you heard, the accent reading, the single most "
        "useful pronunciation fix with concrete articulation advice, and fluency. "
        "Finish with one short question that keeps the conversation going. "
        "Do not use dash punctuation."
    )
    body = json.dumps({
        "model": "claude-sonnet-5",
        "max_tokens": 700,
        "system": sys_prompt,
        "messages": [{"role": "user", "content": json.dumps(a)[:6000]}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"content-type": "application/json", "x-api-key": key,
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read())["content"][0]["text"]
