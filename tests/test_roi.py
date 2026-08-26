import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.roi import crop_roi, downscale_for_ocr, full_frame


def test_crop_roi_returns_bottom_fraction():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    cropped = crop_roi(img, 0.4)
    assert cropped.shape == (40, 200, 3)


def test_crop_roi_full_height_at_one():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    cropped = crop_roi(img, 1.0)
    assert cropped.shape == (100, 200, 3)


def test_full_frame_is_identity():
    img = np.random.randint(0, 255, (50, 60, 3), dtype=np.uint8)
    assert np.array_equal(full_frame(img), img)


def test_crop_roi_takes_bottom_not_top():
    img = np.zeros((100, 10, 3), dtype=np.uint8)
    img[80:, :, 0] = 255  # mark the bottom 20 rows
    cropped = crop_roi(img, 0.2)
    assert cropped.shape[0] == 20
    assert np.all(cropped[:, :, 0] == 255)


def test_downscale_for_ocr_caps_longest_edge():
    """Regression test for B14 — real 4K (3840x2160) footage measured
    ~2.5-3.5s/OCR call vs. ~1.1s on lower-resolution footage, and this was
    the dominant cost in both the coarse sweep and the refine window."""
    img = np.zeros((2160, 3840, 3), dtype=np.uint8)
    out = downscale_for_ocr(img, max_dimension=1280)
    assert max(out.shape[:2]) == 1280


def test_downscale_for_ocr_preserves_aspect_ratio():
    img = np.zeros((2160, 3840, 3), dtype=np.uint8)  # 16:9
    out = downscale_for_ocr(img, max_dimension=1280)
    h, w = out.shape[:2]
    assert abs((w / h) - (3840 / 2160)) < 0.01


def test_downscale_for_ocr_never_upscales():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    out = downscale_for_ocr(img, max_dimension=1280)
    assert out.shape == img.shape
