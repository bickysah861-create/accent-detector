"""Pick the words actually worth practising out of the per word GOP scores.

Unstressed function words (a, to, of, and) reduce heavily in natural speech:
the vowel becomes a schwa and consonants elide. CMUdict supplies the citation
form, so these words score badly for every speaker including natives. Surfacing
them as pronunciation advice is noise, and asking a learner to practise "a"
teaches nothing. Only content words long enough to carry a real articulation
target are offered.
"""
from __future__ import annotations

# Closed class words whose reduced forms are correct, not errors.
FUNCTION_WORDS = {
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "am", "do", "does", "did", "so",
    "as", "if", "it", "its", "he", "she", "they", "them", "we", "us", "you",
    "your", "i", "me", "my", "his", "her", "their", "our", "that", "this",
    "then", "than", "there", "here", "with", "from", "by", "up", "out", "not",
    "no", "yes", "can", "will", "would", "could", "should", "have", "has",
    "had", "just", "like", "um", "uh", "oh", "okay", "ok", "well", "yeah",
}

MIN_PHONES = 3          # below this a GOP average is dominated by one frame span
SCORE_CUTOFF = 72.0
MAX_WORDS = 4


def coachable_words(words: list[dict]) -> list[dict]:
    """Lowest scoring content words, one entry per distinct word.

    A word repeated in the utterance (or duplicated by the recogniser) would
    otherwise fill the whole list with the same practice target.
    """
    out, seen = [], set()
    for g in sorted(words, key=lambda g: g["score"]):
        w = g.get("word", "")
        if w.lower() in FUNCTION_WORDS or w.lower() in seen:
            continue
        seen.add(w.lower())
        if len(g.get("phones", [])) < MIN_PHONES:
            continue
        if g["score"] >= SCORE_CUTOFF:
            continue
        out.append({"word": w, "score": g["score"],
                    "phones": [p["phone"] for p in g["phones"]]})
        if len(out) >= MAX_WORDS:
            break
    return out
