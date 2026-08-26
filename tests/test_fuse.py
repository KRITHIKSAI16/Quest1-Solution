import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.fuse import Candidate, NearMiss, Outcome, fuse, resolve_strict

CONF = 85.0
TOL = 2.0


def _cand(channel, ts, conf=90.0, text="my mind rebels at stagnation"):
    return Candidate(channel=channel, timestamp_s=ts, frame_index=int(ts * 25), confidence=conf, matched_text=text, matching_method="test")


def test_confirmed_by_audio_when_only_audio_confident():
    result = fuse([], [_cand("audio", 10.0)], [], [], CONF, TOL)
    assert result.outcome == Outcome.CONFIRMED_BY_AUDIO
    assert result.primary.channel == "audio"


def test_confirmed_by_visual_when_only_visual_confident():
    result = fuse([_cand("visual", 10.0)], [], [], [], CONF, TOL)
    assert result.outcome == Outcome.CONFIRMED_BY_VISUAL
    assert result.primary.channel == "visual"


def test_corroborated_when_timestamps_agree():
    result = fuse([_cand("visual", 10.0)], [_cand("audio", 10.5)], [], [], CONF, TOL)
    assert result.outcome == Outcome.CORROBORATED
    assert result.primary is not None
    assert len(result.all_candidates) == 2


def test_ambiguous_when_timestamps_disagree_and_never_silently_resolved():
    """This is the test that proves the architecture works: two channels
    landing far apart must NOT collapse to a single silent answer."""
    result = fuse([_cand("visual", 10.0, conf=88.0)], [_cand("audio", 45.0, conf=90.0)], [], [], CONF, TOL)
    assert result.outcome == Outcome.AMBIGUOUS
    assert result.primary is None  # never silently picked
    assert len(result.all_candidates) == 2
    timestamps = {c.timestamp_s for c in result.all_candidates}
    assert timestamps == {10.0, 45.0}


def test_ambiguous_is_not_resolved_by_score_even_when_scores_differ():
    """Regression guard for the rejected 'highest score wins' design: even
    though audio's score is higher, disagreement must still be AMBIGUOUS."""
    result = fuse([_cand("visual", 10.0, conf=99.0)], [_cand("audio", 45.0, conf=99.9)], [], [], CONF, TOL)
    assert result.outcome == Outcome.AMBIGUOUS
    assert result.primary is None


def test_not_found_when_neither_confident():
    near_v = [NearMiss(channel="visual", timestamp_s=5.0, confidence=40.0, text="foo")]
    near_a = [NearMiss(channel="audio", timestamp_s=8.0, confidence=50.0, text="bar")]
    result = fuse([], [], near_v, near_a, CONF, TOL)
    assert result.outcome == Outcome.NOT_FOUND
    assert result.primary is None
    assert len(result.near_misses) == 2


def test_resolve_strict_marks_uncertain_on_ambiguous():
    result = fuse([_cand("visual", 10.0, conf=88.0)], [_cand("audio", 45.0, conf=90.0)], [], [], CONF, TOL)
    resolved = resolve_strict(result)
    assert resolved is not None
    assert result.uncertain is True


def test_resolve_strict_does_not_have_fixed_visual_preference():
    """Regression guard for the rejected 'visual always wins ties' design:
    when audio has higher confidence, strict mode must not force visual."""
    result = fuse([_cand("visual", 10.0, conf=86.0)], [_cand("audio", 45.0, conf=99.0)], [], [], CONF, TOL)
    resolved = resolve_strict(result)
    assert resolved.channel == "audio"  # strongest independent validation, not a fixed preference
