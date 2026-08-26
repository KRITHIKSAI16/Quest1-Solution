"""on-screen presence check for the audio channel. a line only counts as
on-screen dialogue if the speaker is actually visible in the shot, so ASR
matching the transcript isn't enough on its own, that just proves the line
was said somewhere in the audio.

Uses OpenCV's YuNet face detector (cv2.FaceDetectorYN), backed by a small
bundled ONNX model (core/models/face_detection_yunet_2023mar.onnx, ~227KB,
from the OpenCV Zoo, MIT licensed). Wasn't the first choice: mediapipe has no
wheel for Python 3.13 yet on any platform, and the classic Haar-cascade
detector this project depended on for years is gone as of OpenCV 5.0 (which
this project already runs) — 5.0 dropped both cv2.CascadeClassifier and the
bundled cascade XML files from the Python wheel entirely. YuNet is actually
more accurate than Haar cascade anyway, so this isn't a downgrade.

Also does a lightweight mouth-motion check (report-only, doesn't gate
anything) — see mouth_motion_elevated() below. Face presence alone can't
tell a silently-listening on-screen character apart from one actually
delivering the line; this adds a temporal signal on top, using landmarks
YuNet already computes on every call, at no extra model cost.
"""
from __future__ import annotations

from pathlib import Path

import cv2

from core.video import VideoSource

_MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"

# detect()'s row layout, verified directly against a real detection: 15
# columns, bbox(4) + 5 landmarks as (x, y) pairs (right eye, left eye, nose
# tip, right mouth corner, left mouth corner) + score. only the eye and
# mouth corner columns are used here.
_RIGHT_EYE, _LEFT_EYE = slice(4, 6), slice(6, 8)
_RIGHT_MOUTH, _LEFT_MOUTH = slice(10, 12), slice(12, 14)


def _dist(a, b) -> float:
    # a/b are slices of a numpy row straight from detect() — the arithmetic
    # below produces numpy.float32/float64, not a real Python float, even
    # though the type hint says so. json.dump() can't serialize that (it
    # shows up as "TypeError: Object of type bool is not JSON serializable"
    # once it propagates through a comparison in mouth_motion_elevated() —
    # a real bug that reached a real run before this cast was here). float()
    # forces it to an actual Python float at the one place it's computed, so
    # nothing downstream needs to know or care.
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


class FacePresenceDetector:
    """fails open, not closed: if the model can't load for any reason,
    has_face_near() always returns None (bypass) instead of raising. a face
    check that's unavailable should never take down the whole run.
    """

    def __init__(self, score_threshold: float = 0.6):
        self._detector = None
        try:
            if _MODEL_PATH.is_file():
                self._detector = cv2.FaceDetectorYN_create(
                    str(_MODEL_PATH), "", (0, 0),
                    score_threshold=score_threshold, nms_threshold=0.3,
                )
        except Exception:
            self._detector = None

    @property
    def available(self) -> bool:
        return self._detector is not None

    def _has_face(self, image_bgr) -> bool:
        h, w = image_bgr.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(image_bgr)
        return faces is not None and len(faces) > 0

    def has_face_near(self, video: VideoSource, frame_index: int, window_s: float) -> bool | None:
        """checks a small window of frames around frame_index rather than
        trusting one exact frame, same reason the visual channel's onset
        walk doesn't trust a single OCR read: a face can be turned away or
        motion-blurred for one specific frame while still genuinely being on
        screen a few frames earlier or later in the same line.

        True if any frame in the window shows a face, False if none do,
        None if the detector isn't available at all (bypass, not a "no face"
        verdict, those mean different things downstream).
        """
        if self._detector is None:
            return None

        step = max(1, round(video.meta.fps * window_s / 3))
        offsets = sorted({0, -step, step, -2 * step, 2 * step})
        checked_any = False
        for offset in offsets:
            idx = frame_index + offset
            if idx < 0 or idx >= video.meta.frame_count:
                continue
            try:
                frame = video.frame_at_index(idx)
            except IndexError:
                continue
            checked_any = True
            if self._has_face(frame.image):
                return True

        return False if checked_any else None

    def _mouth_metric(self, image_bgr) -> float | None:
        """mouth-corner distance normalized by inter-ocular distance — a
        standard scale-invariance trick in landmark analysis, so the metric
        means the same thing regardless of how close the face is to camera.
        picks the largest face in frame if there's more than one (the main
        subject, not a background extra) since there's no tracker here to
        keep identity consistent frame to frame otherwise.
        """
        h, w = image_bgr.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(image_bgr)
        if faces is None or len(faces) == 0:
            return None
        row = max(faces, key=lambda f: f[2] * f[3])
        inter_ocular = _dist(row[_RIGHT_EYE], row[_LEFT_EYE])
        if inter_ocular < 1e-3:
            return None
        return _dist(row[_RIGHT_MOUTH], row[_LEFT_MOUTH]) / inter_ocular

    def _window_motion_energy(self, video: VideoSource, t0_s: float, t1_s: float, sample_fps: float) -> float | None:
        metrics = [m for m in (
            self._mouth_metric(frame.image) for frame in video.sample_range(t0_s, t1_s, sample_fps)
        ) if m is not None]
        if len(metrics) < 3:
            # too few valid reads to say anything — a face that was mostly
            # occluded or off-angle through this window, not a real signal
            return None
        return sum(abs(metrics[i] - metrics[i - 1]) for i in range(1, len(metrics))) / (len(metrics) - 1)

    def mouth_motion_elevated(self, video: VideoSource, onset_s: float, window_s: float,
                               baseline_offset_s: float, sample_fps: float, ratio: float) -> bool | None:
        """compares mouth-motion energy during the speech window against a
        same-length silent baseline window shortly before it, rather than
        against one fixed global threshold — self-calibrates to this video's
        own idle-motion level (compression noise, camera shake, blink rate)
        instead of guessing a universal cutoff that would drift across
        different footage.

        this is a motion signal, not real audio-visual sync — it can't tell
        "speaking" apart from "nodding, reacting, or otherwise moving during
        this window" the way a learned synchrony model could. report-only:
        never feeds into CONFIRMED_BY_AUDIO vs CONFIRMED_BY_AUDIO_OFF_SCREEN,
        purely an extra signal alongside the actual presence check.

        None if the detector's unavailable, or either window came up too
        thin to compare (occlusion, video edge) — not the same as "False".
        """
        if self._detector is None:
            return None

        speech = self._window_motion_energy(video, onset_s - window_s, onset_s + window_s, sample_fps)
        baseline_center = onset_s - baseline_offset_s
        baseline = self._window_motion_energy(video, baseline_center - window_s, baseline_center + window_s, sample_fps)
        if speech is None or baseline is None:
            return None
        if baseline < 1e-6:
            # essentially no idle motion in the baseline at all — any real
            # motion during speech counts as elevated relative to that
            return bool(speech > 1e-4)
        return bool(speech >= baseline * ratio)
