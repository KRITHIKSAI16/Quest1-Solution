"""Unit tests for core/locate.py's _word_level_onset — the audio channel's
speech-onset finder within a matched transcript segment.

No coverage existed for this function before internal/INTERVIEW_NOTES.md
§B18: a real user run found it returning the segment's START time almost
regardless of where the target word actually falls, because partial_ratio
saturates at 100.0 for any span merely containing the target, and the old
strict `>` scan (growing spans left-to-right from start=0) let the first
such span win before the scan ever reached the real target word. These
tests construct TranscriptSegment/Word directly (plain dataclasses, see
core/asr.py) — no ASR model, no audio, no video, fully deterministic.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.asr import TranscriptSegment, Word
from core.locate import _word_level_onset


def _make_segment(text: str, start_s: float = 0.0, word_duration_s: float = 0.35) -> TranscriptSegment:
    """Builds a TranscriptSegment with evenly-spaced word timings, so each
    word's start_s is independently predictable for assertions."""
    words = []
    t = start_s
    for w in text.split():
        words.append(Word(text=w, start_s=round(t, 3), end_s=round(t + word_duration_s, 3)))
        t += word_duration_s
    return TranscriptSegment(text=text, start_s=start_s, end_s=t, words=words)


def test_single_word_mid_segment_returns_that_words_own_onset_not_segment_start():
    """The exact regression from §B18: target is one word buried well after
    the segment's first word — must return that word's own start time, not
    the segment's start (the bug returned the segment start here)."""
    seg = _make_segment(
        "Overwhelmed means you have more work stress or emotions than you can handle",
        start_s=7.74,
    )
    onset = _word_level_onset(seg, "emotions")
    target_word = next(w for w in seg.words if w.text == "emotions")
    assert onset == target_word.start_s
    assert onset != seg.start_s


def test_multi_word_target_returns_its_own_first_word_onset():
    seg = _make_segment(
        "Overwhelmed means you have more work stress or emotions than you can handle",
        start_s=7.74,
    )
    onset = _word_level_onset(seg, "stress or emotions")
    expected = next(w for w in seg.words if w.text == "stress")
    assert onset == expected.start_s


def test_target_at_segment_start_still_works():
    seg = _make_segment("My mind rebels at stagnation", start_s=5.0)
    onset = _word_level_onset(seg, "my mind")
    assert onset == seg.words[0].start_s


def test_repeated_target_word_returns_earliest_occurrence():
    """When the target word appears more than once in a segment, the
    earliest-start tie-break must win — 'first appears' semantics, not an
    arbitrary later occurrence."""
    seg = _make_segment("emotions run high in this scene with more emotions everywhere", start_s=0.0)
    onset = _word_level_onset(seg, "emotions")
    assert onset == seg.words[0].start_s


def test_onset_of_exactly_zero_is_not_discarded():
    """Regression guard for the separate falsy-zero bug at the call site in
    run_audio_channel (`_word_level_onset(...) or seg.start_s`): this
    function itself must return a real 0.0, distinct from None."""
    seg = _make_segment("start of the clip", start_s=0.0)
    onset = _word_level_onset(seg, "start")
    assert onset == 0.0
    assert onset is not None


def test_target_longer_than_old_twelve_word_span_cap_is_still_matched():
    """Before the fix, no span could ever exceed 12 words, so a target with
    more words than that could never be matched at all."""
    long_target = "more work stress or emotions than you can handle right now today"  # 13 words
    seg = _make_segment(
        "Overwhelmed means you have " + long_target,
        start_s=0.0,
    )
    onset = _word_level_onset(seg, long_target)
    expected = next(w for w in seg.words if w.text == "more")
    assert onset == expected.start_s


def test_empty_words_returns_none():
    seg = TranscriptSegment(text="", start_s=1.0, end_s=1.0, words=[])
    assert _word_level_onset(seg, "anything") is None
