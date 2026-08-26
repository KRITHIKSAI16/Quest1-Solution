"""OCR engine interface + the EasyOCR implementation. behind a Protocol so
swapping engines later doesn't touch locate.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class OcrBox:
    text: str
    x: float  # top-left x of the bounding box, in ROI-local pixels
    y: float  # top-left y


class OcrEngine(Protocol):
    def read(self, image: np.ndarray) -> list[OcrBox]: ...


class EasyOcrEngine:
    def __init__(self, languages: list[str] | None = None, gpu: bool = False):
        import easyocr  # deferred: heavy import, only pay for it if this engine is used

        self._reader = easyocr.Reader(languages or ["en"], gpu=gpu, verbose=False)

    def read(self, image: np.ndarray) -> list[OcrBox]:
        results = self._reader.readtext(image)
        boxes = []
        for bbox, text, _conf in results:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            boxes.append(OcrBox(text=text, x=min(xs), y=min(ys)))
        return boxes


def assemble_line(boxes: list[OcrBox]) -> str:
    """joins detected boxes into one string, top-to-bottom then left-to-right,
    so a two-line subtitle matches as one thing instead of two fragments.
    """
    ordered = sorted(boxes, key=lambda b: (round(b.y / 10), b.x))
    return " ".join(b.text for b in ordered)
