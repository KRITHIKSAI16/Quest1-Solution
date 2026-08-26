"""Face-presence check for the audio channel's on-screen verification.

Runs the real bundled YuNet model on blank frames (fast, deterministic, no
mocking needed to prove "no face" on genuinely faceless input) and mocks
the detector's own .detect() call where it's testing this module's logic
(the window search, the fail-safe bypass) rather than detection accuracy.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.face import FacePresenceDetector, _dist
from core.video import VideoSource


@pytest.fixture
def blank_video_path(tmp_path):
    path = str(tmp_path / "blank.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for _ in range(20):
        writer.write(np.zeros((64, 64, 3), dtype="uint8"))
    writer.release()
    return path


@pytest.fixture
def longer_blank_video_path(tmp_path):
    # enough runway for a speech window + a baseline window offset before it
    path = str(tmp_path / "blank_long.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for _ in range(50):
        writer.write(np.zeros((64, 64, 3), dtype="uint8"))
    writer.release()
    return path


def test_model_file_is_present_and_loadable():
    # the whole feature depends on this file actually being committed to
    # the repo, not on a network fetch at runtime — confirm it loads clean
    assert FacePresenceDetector().available is True


def test_has_face_near_returns_none_when_detector_unavailable(blank_video_path):
    detector = FacePresenceDetector()
    detector._detector = None  # simulate a model that failed to load
    with VideoSource(blank_video_path) as video:
        assert detector.has_face_near(video, 5, 0.5) is None


def test_has_face_near_returns_false_on_blank_frames(blank_video_path):
    # real model, real inference — a blank frame genuinely has no face
    detector = FacePresenceDetector()
    with VideoSource(blank_video_path) as video:
        assert detector.has_face_near(video, 5, 0.5) is False


def test_has_face_near_finds_a_face_off_the_exact_onset_frame(blank_video_path):
    """the whole point of checking a window, not one frame: a face present
    a couple frames away from the literal onset index still counts.
    """
    detector = FacePresenceDetector()
    call_count = {"n": 0}

    def _fake_detect(image):
        call_count["n"] += 1
        # first call (the exact onset frame) finds nothing, a later one
        # (an offset frame) does — proves the window search doesn't stop
        # at frame 0 and doesn't just get lucky on the first call either
        if call_count["n"] == 3:
            return 1, np.zeros((1, 15), dtype="float32")
        return 0, None

    with VideoSource(blank_video_path) as video:
        with patch.object(cv2.FaceDetectorYN, "detect", side_effect=_fake_detect):
            assert detector.has_face_near(video, 5, 0.5) is True
    assert call_count["n"] >= 3


def test_has_face_near_handles_out_of_range_offsets(blank_video_path):
    """onset near the very start of the video: some window offsets go
    negative, must be skipped rather than raising.
    """
    detector = FacePresenceDetector()
    with VideoSource(blank_video_path) as video:
        assert detector.has_face_near(video, 0, 0.5) is False


def test_constructor_never_raises_when_model_load_fails():
    with patch("cv2.FaceDetectorYN_create", side_effect=RuntimeError("boom")):
        detector = FacePresenceDetector()
    assert detector.available is False


# --- mouth-motion signal (report-only) --------------------------------------

def _fake_row(right_eye, left_eye, right_mouth, left_mouth, score=0.9):
    return np.array([0, 0, 200, 200, *right_eye, *left_eye, 0, 0, *right_mouth, *left_mouth, score], dtype="float32")


def test_mouth_metric_none_without_a_face():
    detector = FacePresenceDetector()
    with patch.object(cv2.FaceDetectorYN, "detect", return_value=(0, None)):
        assert detector._mouth_metric(np.zeros((64, 64, 3), dtype="uint8")) is None


def test_mouth_metric_normalizes_by_inter_ocular_distance():
    detector = FacePresenceDetector()
    # eyes 100px apart, mouth corners 40px apart -> ratio 0.4, independent of face size
    row = _fake_row((0, 0), (100, 0), (30, 80), (70, 80))
    with patch.object(cv2.FaceDetectorYN, "detect", return_value=(1, np.array([row]))):
        ratio = detector._mouth_metric(np.zeros((200, 200, 3), dtype="uint8"))
    assert ratio == pytest.approx(0.4)
    assert type(ratio) is float  # not numpy.float32/float64 — see the _dist regression test below


def test_dist_returns_native_python_float_not_numpy_scalar():
    """regression test for a real bug: _dist() on numpy inputs (which is
    what it always gets — row slices straight from a real detect() call)
    returned numpy.float32, not a Python float. That survived all the way
    through _mouth_metric -> _window_motion_energy -> the boolean
    comparison in mouth_motion_elevated() as a numpy.bool_, which crashed
    write_result_json() with "TypeError: Object of type bool is not JSON
    serializable" on a real run. None of the other tests caught it because
    they all mocked at a level that bypassed this exact arithmetic.
    """
    a = np.array([0.0, 0.0], dtype="float32")
    b = np.array([3.0, 4.0], dtype="float32")
    d = _dist(a, b)
    assert d == 5.0
    assert type(d) is float
    json.dumps({"d": d})  # the actual failure mode: must not raise


def test_mouth_motion_elevated_is_json_serializable_through_the_real_numpy_path(longer_blank_video_path):
    """same regression, exercised end-to-end through mouth_motion_elevated()
    with a real numpy detect() row (not mocking _window_motion_energy or
    _mouth_metric away — those mocks are exactly what let the original bug
    ship without a test catching it) — the actual failure mode a real run hit.
    """
    detector = FacePresenceDetector()
    call_count = {"n": 0}

    def _fake_detect(image):
        call_count["n"] += 1
        width = 30.0 if call_count["n"] % 2 == 0 else 60.0  # oscillating mouth width = real motion
        row = _fake_row((0, 0), (100, 0), (50 - width / 2, 80), (50 + width / 2, 80))
        return 1, np.array([row])

    with patch.object(cv2.FaceDetectorYN, "detect", side_effect=_fake_detect):
        with VideoSource(longer_blank_video_path) as video:
            result = detector.mouth_motion_elevated(video, 2.5, 0.5, 1.5, 10.0, 1.6)

    assert result in (True, False)
    assert type(result) is bool  # not numpy.bool_
    json.dumps({"mouth_motion": result})  # the actual failure mode: must not raise


def test_mouth_metric_picks_the_largest_face_when_more_than_one():
    detector = FacePresenceDetector()
    small = np.array([0, 0, 50, 50, 0, 0, 10, 0, 0, 0, 2, 5, 8, 5, 0.7], dtype="float32")  # tiny, background extra
    large = _fake_row((0, 0), (100, 0), (30, 80), (70, 80))  # ratio 0.4, the main subject
    with patch.object(cv2.FaceDetectorYN, "detect", return_value=(2, np.array([small, large]))):
        ratio = detector._mouth_metric(np.zeros((200, 200, 3), dtype="uint8"))
    assert ratio == pytest.approx(0.4)


def test_window_motion_energy_none_when_face_never_detected(blank_video_path):
    detector = FacePresenceDetector()
    with patch.object(FacePresenceDetector, "_mouth_metric", return_value=None):
        with VideoSource(blank_video_path) as video:
            assert detector._window_motion_energy(video, 0.0, 1.0, 10.0) is None


def test_window_motion_energy_computes_mean_frame_to_frame_delta(blank_video_path):
    from itertools import cycle
    detector = FacePresenceDetector()
    values = cycle([0.2, 0.4])  # alternating, so every step's delta is exactly 0.2
    with patch.object(FacePresenceDetector, "_mouth_metric", side_effect=lambda img: next(values)):
        with VideoSource(blank_video_path) as video:
            energy = detector._window_motion_energy(video, 0.0, 0.3, 10.0)
    assert energy == pytest.approx(0.2, abs=1e-6)


def test_mouth_motion_elevated_none_when_detector_unavailable(longer_blank_video_path):
    detector = FacePresenceDetector()
    detector._detector = None
    with VideoSource(longer_blank_video_path) as video:
        assert detector.mouth_motion_elevated(video, 2.5, 0.5, 1.5, 10.0, 1.6) is None


def test_mouth_motion_elevated_true_when_speech_window_has_more_motion(longer_blank_video_path):
    detector = FacePresenceDetector()
    with patch.object(FacePresenceDetector, "_window_motion_energy", side_effect=[0.20, 0.05]):
        with VideoSource(longer_blank_video_path) as video:
            assert detector.mouth_motion_elevated(video, 2.5, 0.5, 1.5, 10.0, 1.6) is True


def test_mouth_motion_elevated_false_when_not_meaningfully_higher(longer_blank_video_path):
    detector = FacePresenceDetector()
    with patch.object(FacePresenceDetector, "_window_motion_energy", side_effect=[0.06, 0.05]):
        with VideoSource(longer_blank_video_path) as video:
            assert detector.mouth_motion_elevated(video, 2.5, 0.5, 1.5, 10.0, 1.6) is False


def test_mouth_motion_elevated_none_when_either_window_unmeasurable(longer_blank_video_path):
    detector = FacePresenceDetector()
    with patch.object(FacePresenceDetector, "_window_motion_energy", side_effect=[0.2, None]):
        with VideoSource(longer_blank_video_path) as video:
            assert detector.mouth_motion_elevated(video, 2.5, 0.5, 1.5, 10.0, 1.6) is None


def test_mouth_motion_elevated_handles_near_zero_baseline(longer_blank_video_path):
    """a baseline with essentially no motion at all shouldn't divide weirdly
    — any real motion during speech should still count as elevated."""
    detector = FacePresenceDetector()
    with patch.object(FacePresenceDetector, "_window_motion_energy", side_effect=[0.001, 0.0]):
        with VideoSource(longer_blank_video_path) as video:
            assert detector.mouth_motion_elevated(video, 2.5, 0.5, 1.5, 10.0, 1.6) is True


def test_mouth_metric_real_model_on_reference_frame():
    """sanity check against the real reference frame from the actual
    reference-video run, not just fabricated landmark math.
    """
    frame_path = Path(__file__).parent.parent / "output_face_check" / "frame_7771.png"
    if not frame_path.is_file():
        pytest.skip("reference frame only exists after a real end-to-end run")
    img = cv2.imread(str(frame_path))
    ratio = FacePresenceDetector()._mouth_metric(img)
    assert ratio is not None
    assert 0.0 < ratio < 2.0
