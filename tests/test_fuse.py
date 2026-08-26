import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.fuse import Candidate, NearMiss, Outcome, fuse, resolve_strict

CONF = 85.0
TOL = 2.0


def _cand(channel, ts, conf=90.0, text="my mind rebels at stagnation", face=None, motion=None):
    return Candidate(channel=channel, timestamp_s=ts, frame_index=int(ts * 25), confidence=conf,
                      matched_text=text, matching_method="test", face_detected=face, mouth_motion=motion)


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


def test_audio_only_with_confirmed_face_is_plain_confirmed_by_audio():
    result = fuse([], [_cand("audio", 10.0, face=True)], [], [], CONF, TOL)
    assert result.outcome == Outcome.CONFIRMED_BY_AUDIO


def test_audio_only_with_no_face_becomes_off_screen_outcome():
    result = fuse([], [_cand("audio", 10.0, face=False)], [], [], CONF, TOL)
    assert result.outcome == Outcome.CONFIRMED_BY_AUDIO_OFF_SCREEN
    assert result.primary.timestamp_s == 10.0  # still returns the real match, just relabeled
    assert result.ambiguity_status


def test_audio_only_with_face_check_unavailable_behaves_like_before():
    """face_detected=None means the check never ran at all — must not be
    treated the same as a confirmed absence."""
    result = fuse([], [_cand("audio", 10.0, face=None)], [], [], CONF, TOL)
    assert result.outcome == Outcome.CONFIRMED_BY_AUDIO


def test_face_confirmed_candidate_preferred_over_higher_confidence_faceless_one():
    """the 'keep looking' behavior: the target is said twice, once with a
    face, once without and higher-scoring — the on-screen one should win."""
    faceless = _cand("audio", 5.0, conf=99.0, face=False)
    on_screen = _cand("audio", 40.0, conf=90.0, face=True)
    result = fuse([], [faceless, on_screen], [], [], CONF, TOL)
    assert result.outcome == Outcome.CONFIRMED_BY_AUDIO
    assert result.primary.timestamp_s == 40.0


def test_off_screen_only_fires_when_audio_is_the_only_evidence():
    """visual also confident at the same time -> CORROBORATED still wins,
    the off-screen outcome shouldn't shadow a genuine agreement."""
    result = fuse([_cand("visual", 10.0)], [_cand("audio", 10.5, face=False)], [], [], CONF, TOL)
    assert result.outcome == Outcome.CORROBORATED


def test_mouth_motion_is_purely_informational_and_never_affects_the_outcome():
    """report-only by design: a face confirmed but no speech-like motion
    still counts as CONFIRMED_BY_AUDIO, not something weaker — this field
    doesn't participate in fuse()'s decision at all."""
    result = fuse([], [_cand("audio", 10.0, face=True, motion=False)], [], [], CONF, TOL)
    assert result.outcome == Outcome.CONFIRMED_BY_AUDIO
    assert result.primary.mouth_motion is False  # carried through untouched, not overridden
