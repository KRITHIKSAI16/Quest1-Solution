"""cheap filters to skip OCR when we don't need it. dedup does most of the
work here (~30% rejection on real footage). the text-presence gate barely
adds anything on its own, a generic edge-density check can't really tell
"legible text" from "busy scene" apart, but it's cheap so it stays.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GateStats:
    total: int = 0
    rejected_dedup: int = 0
    rejected_no_text: int = 0

    @property
    def passed(self) -> int:
        return self.total - self.rejected_dedup - self.rejected_no_text


class FrameGate:
    """keeps the last ROI around so it has something to diff the next one against."""

    def __init__(self, mad_threshold: float, min_edge_density: float):
        self.mad_threshold = mad_threshold
        self.min_edge_density = min_edge_density
        self._last_roi_gray: np.ndarray | None = None
        self.stats = GateStats()

    def should_ocr(self, roi_bgr: np.ndarray) -> bool:
        self.stats.total += 1
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        if self._is_duplicate(gray):
            self.stats.rejected_dedup += 1
            return False

        if not self._has_text_like_content(gray):
            self.stats.rejected_no_text += 1
            return False

        self._last_roi_gray = gray
        return True

    def _is_duplicate(self, gray: np.ndarray) -> bool:
        if self._last_roi_gray is None or self._last_roi_gray.shape != gray.shape:
            return False
        mad = float(np.mean(np.abs(gray.astype(np.int16) - self._last_roi_gray.astype(np.int16))))
        return mad < self.mad_threshold

    def _has_text_like_content(self, gray: np.ndarray) -> bool:
        """rendered text has a lot of edges, plain backgrounds don't. crude but fast."""
        edges = cv2.Canny(gray, 80, 160)
        density = float(np.count_nonzero(edges)) / edges.size
        return density >= self.min_edge_density
