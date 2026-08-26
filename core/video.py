"""video metadata + frame access, wraps cv2.VideoCapture. frame numbers come
from the decoder's own position counter, not t*fps, so this still works on
containers with a weird timebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoMeta:
    fps: float
    frame_count: int
    duration_s: float
    width: int
    height: int
    is_vfr: bool  # best-effort: True if fps*duration disagrees with frame_count by >2%


@dataclass
class Frame:
    index: int
    timestamp_s: float
    image: np.ndarray  # BGR, HxWx3


class VideoSource:
    """wraps cv2.VideoCapture with the access patterns we actually need:
    metadata, uniform sampling, and a bounded native-fps window for refine.
    """

    def __init__(self, path: str):
        self.path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"Could not open video: {path}")
        self.meta = self._read_meta()

    def _read_meta(self) -> VideoMeta:
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = frame_count / fps if fps > 0 else 0.0

        # rough VFR check: seek to the last frame, see if its timestamp lines
        # up with what fps*count would predict. way off = variable frame rate.
        is_vfr = False
        if frame_count > 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            ok = self._cap.grab()
            if ok:
                actual_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                expected_ms = (frame_count - 1) / fps * 1000 if fps > 0 else 0
                if expected_ms > 0 and abs(actual_ms - expected_ms) / expected_ms > 0.02:
                    is_vfr = True
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return VideoMeta(fps, frame_count, duration_s, width, height, is_vfr)

    def sample_at(self, sample_fps: float) -> Iterator[Frame]:
        """yields frames at roughly sample_fps, using the decoder's own frame
        index so seeks stay accurate.
        """
        if sample_fps <= 0:
            raise ValueError("sample_fps must be > 0")
        step = max(1, round(self.meta.fps / sample_fps))
        idx = 0
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while idx < self.meta.frame_count:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, img = self._cap.read()
            if not ok:
                break
            actual_idx = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            ts = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield Frame(index=actual_idx, timestamp_s=ts, image=img)
            idx += step

    def iter_range(self, t0_s: float, t1_s: float) -> Iterator[Frame]:
        """every frame in [t0_s, t1_s], native frame rate."""
        t0_s = max(0.0, t0_s)
        t1_s = min(self.meta.duration_s, t1_s)
        start_idx = max(0, int(t0_s * self.meta.fps))
        end_idx = min(self.meta.frame_count - 1, int(t1_s * self.meta.fps) + 1)

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        idx = start_idx
        while idx <= end_idx:
            ok, img = self._cap.read()
            if not ok:
                break
            actual_idx = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            ts = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield Frame(index=actual_idx, timestamp_s=ts, image=img)
            idx += 1

    def sample_range(self, t0_s: float, t1_s: float, sample_fps: float) -> Iterator[Frame]:
        """same as iter_range but at sample_fps instead of native rate. used
        by refine, costs some onset precision (worst case 1/sample_fps) but
        it's a lot cheaper.
        """
        if sample_fps <= 0 or sample_fps >= self.meta.fps:
            yield from self.iter_range(t0_s, t1_s)
            return

        t0_s = max(0.0, t0_s)
        t1_s = min(self.meta.duration_s, t1_s)
        start_idx = max(0, int(t0_s * self.meta.fps))
        end_idx = min(self.meta.frame_count - 1, int(t1_s * self.meta.fps) + 1)
        step = max(1, round(self.meta.fps / sample_fps))

        idx = start_idx
        while idx <= end_idx:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, img = self._cap.read()
            if not ok:
                break
            actual_idx = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            ts = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield Frame(index=actual_idx, timestamp_s=ts, image=img)
            idx += step

    def frame_at_index(self, index: int) -> Frame:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, img = self._cap.read()
        if not ok:
            raise IndexError(f"Could not read frame {index}")
        ts = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        return Frame(index=index, timestamp_s=ts, image=img)

    def close(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
