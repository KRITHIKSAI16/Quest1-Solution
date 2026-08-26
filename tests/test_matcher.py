import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.matcher import normalize, score


def test_normalize_casefold_and_punctuation():
    assert normalize("My Mind Rebels at Stagnation!") == "my mind rebels at stagnation"


def test_normalize_collapses_whitespace_and_newlines():
    assert normalize("My mind\nrebels   at stagnation") == "my mind rebels at stagnation"


def test_normalize_nfkc():
    # fullwidth chars should normalise to ascii-equivalent
    assert normalize("ＭＹ ＭＩＮＤ") == "my mind"


def test_score_exact_match_is_high():
    assert score("My mind rebels at stagnation", "My mind rebels at stagnation") >= 99


def test_score_tolerates_punctuation_difference():
    assert score("My mind rebels at stagnation!", "My mind rebels at stagnation") >= 95


def test_score_tolerates_surrounding_text():
    # candidate has extra context around the target line (e.g. a subtitle card
    # or transcript segment with more than just the target)
    candidate = "Watson, my mind rebels at stagnation, give me problems"
    assert score(candidate, "my mind rebels at stagnation") >= 85


def test_score_low_for_unrelated_text():
    # rapidfuzz's token_set_ratio is generous with short common words, so this
    # sits below the pipeline's 70-point uncertain threshold rather than near 0
    assert score("The weather today is quite pleasant", "My mind rebels at stagnation") < 60


def test_score_empty_candidate_is_zero():
    assert score("", "My mind rebels at stagnation") == 0.0


def test_score_tolerates_single_char_ocr_noise():
    # 'rebeis' instead of 'rebels' — one character swapped
    assert score("My mind rebeis at stagnation", "My mind rebels at stagnation") >= 85


def test_score_rejects_tiny_degenerate_candidate_against_long_target():
    """Regression test for a real bug found via manual testing: a single
    stray OCR-misread character ('c') scored 100.0 against a long target
    purely because rapidfuzz's partial_ratio treats any short candidate as a
    "perfect match" whenever it appears as a substring anywhere in the
    target — here, because "much" contains a "c". See
    internal/INTERVIEW_NOTES.md §B11."""
    target = "I think your love would be too much"
    assert score("c", target) < 50
    assert score("x", target) < 50
    assert score("mu", target) < 50  # still short enough to be noise, not a partial read


def test_score_still_matches_when_target_itself_is_short():
    """The degenerate-candidate guard must not break legitimate short
    targets — a short candidate matching a short target is not noise."""
    assert score("OK", "OK") >= 99
    assert score("ok", "OK") >= 99
    assert score("Stop!", "Stop") >= 90


def test_score_genuine_partial_read_still_matches():
    """A real truncated OCR/ASR read (much longer than noise, just missing
    the start of the line) must still score well — the length guard targets
    tiny noise fragments, not legitimate partial visibility."""
    target = "I think your love would be too much"
    candidate = "love would be too much"  # first half of the subtitle cut off
    assert score(candidate, target) >= 85


def test_score_rejects_short_word_that_is_literal_substring_of_target():
    """Regression test for a real bug found via manual testing on a real
    video: candidate "SUBTITLES" (9 chars, well above MIN_MATCH_LENGTH)
    scored 100.0 against target "Generating Subtitles..." (20 chars) purely
    because "subtitles" is a perfect, literal substring of the target —
    despite covering under half of it. The B11 absolute-length floor alone
    doesn't catch this: the candidate is a real, meaningful-length word,
    just not most of the target. See internal/INTERVIEW_NOTES.md §B13."""
    target = "Generating Subtitles..."
    assert score("SUBTITLES", target) < 50
    assert score("subtitles", target) < 50


def test_score_coverage_guard_does_not_penalise_longer_candidate():
    """The relative-coverage guard only applies when the candidate is
    SHORTER than the target — a candidate that's longer (e.g. a subtitle
    card with extra surrounding text) is the case partial_ratio legitimately
    exists to handle, and must be unaffected."""
    candidate = "Watson, my mind rebels at stagnation, give me problems"
    target = "my mind rebels at stagnation"
    assert score(candidate, target) >= 85


def test_score_coverage_guard_allows_majority_coverage():
    """A candidate covering most (not all) of a longer target should still
    match — the guard targets small fragments, not any imperfect read."""
    target = "Generating Subtitles for this video now"
    candidate = "Generating Subtitles for this video"  # missing trailing word
    assert score(candidate, target) >= 85
