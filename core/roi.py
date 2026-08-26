"""ROI cropping for the visual channel. this is just a fast path, locate.py
falls back to a full-frame sweep if the ROI pass comes up empty.
"""
from __future__ import annotations

import cv2
import numpy as np


def downscale_for_ocr(image: np.ndarray, max_dimension: int) -> np.ndarray:
    """caps the longest edge before it hits OCR, keeps aspect ratio, never
    upscales. on 4K footage this was most of the OCR cost and full res
    wasn't buying us anything, captions are big enough to read shrunk down.
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dimension:
        return image
    scale = max_dimension / longest
    new_size = (round(w * scale), round(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def crop_roi(image: np.ndarray, bottom_fraction: float) -> np.ndarray:
    """Return the bottom `bottom_fraction` of the frame, where subtitles
    conventionally render.
    """
    h = image.shape[0]
    y0 = int(h * (1.0 - bottom_fraction))
    return image[y0:h, :]


def full_frame(image: np.ndarray) -> np.ndarray:
    return image
