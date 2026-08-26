"""normalization + fuzzy scoring, used by both channels so they're judged the
same way. scores still never get compared across channels though, see fuse.py.
"""
from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz


def normalize(text: str) -> str:
    """nfkc, casefold, strip punctuation, collapse whitespace. so a stray
    comma or OCR misread doesn't fail an otherwise correct match.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


MIN_MATCH_LENGTH = 5  # chars, after normalizing
MIN_COVERAGE_FRACTION = 0.6


def score(candidate: str, target: str) -> float:
    """fuzzy score, 0-100. partial_ratio for when the candidate has extra
    text around the match, token_set_ratio for word-order stuff, we just
    take whichever's higher.

    the two length checks below exist because partial_ratio has a real
    problem: it scores the best local window with zero penalty for how much
    of the target that window covers, so a tiny fragment can score 100 by
    pure accident. MIN_MATCH_LENGTH catches the absolute-tiny case,
    MIN_COVERAGE_FRACTION catches a short-but-real word that's still only a
    small chunk of a longer target.
    """
    c = normalize(candidate)
    t = normalize(target)
    if not c or not t:
        return 0.0
    if len(c) < min(MIN_MATCH_LENGTH, len(t)):
        return 0.0
    if len(c) < len(t) and len(c) < MIN_COVERAGE_FRACTION * len(t):
        return 0.0
    return max(fuzz.partial_ratio(c, t), fuzz.token_set_ratio(c, t))


def alignment_ratio(candidate: str, target: str) -> float:
    """whole-string similarity, 0-100. unlike score() this only peaks when
    candidate and target are actually close in length/content, not just when
    one contains the other. used as a tie-breaker in locate.py, not a gate.
    """
    c = normalize(candidate)
    t = normalize(target)
    if not c or not t:
        return 0.0
    return fuzz.ratio(c, t)
