"""Deterministic end-to-end test of the visual channel against a synthetic
clip with known ground truth (see tests/generate_synthetic.py). This is the
only precise check of the coarse sweep + refine + backward-onset-walk chain
— everything else in this suite either mocks pieces of it or relies on real
footage where we don't control the ground truth exactly.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import Config
from core.locate import run_visual_channel
from core.video import VideoSource
from tests.generate_synthetic import FPS, TARGET_TEXT, TEXT_END_FRAME, TEXT_START_FRAME

FIXTURE = str(Path(__file__).parent / "fixtures" / "synthetic_clip.mp4")


@pytest.fixture(scope="module", autouse=True)
def ensure_fixture():
    if not os.path.exists(FIXTURE):
        from tests.generate_synthetic import generate
        generate(FIXTURE)


def test_visual_channel_finds_exact_onset_frame():
    cfg = Config(sample_fps=5.0, roi_bottom_fraction=1.0)  # full-height ROI: synthetic text isn't in a real lower-third
    with VideoSource(FIXTURE) as video:
        candidates, near_misses, _diag = run_visual_channel(video, TARGET_TEXT, cfg)

    assert len(candidates) == 1, f"expected exactly one match cluster, got {candidates}"
    c = candidates[0]
    assert c.channel == "visual"
    # Onset must land exactly on TEXT_START_FRAME, or very close — OCR
    # detection quality near the fade boundary means "very close" is a
    # legitimate outcome, not just an exact hit.
    assert abs(c.frame_index - TEXT_START_FRAME) <= 2, (
        f"expected onset near frame {TEXT_START_FRAME}, got {c.frame_index}"
    )
    expected_ts = TEXT_START_FRAME / FPS
    assert abs(c.timestamp_s - expected_ts) <= 0.15


def test_visual_channel_no_match_for_unrelated_text():
    cfg = Config(sample_fps=5.0, roi_bottom_fraction=1.0)
    with VideoSource(FIXTURE) as video:
        candidates, near_misses, _diag = run_visual_channel(video, "COMPLETELY DIFFERENT PHRASE", cfg)
    assert candidates == []


def test_force_full_frame_skips_roi_pass_entirely():
    """Regression test: --full-frame previously only disabled the
    fallback-after-miss while still running the ROI pass first, contradicting
    its own documented behaviour ("skip the ROI fast path"). This asserts the
    diagnostics shape directly: forcing full-frame must produce a
    'full_frame_forced' key and no 'roi_sweep' key at all.
    """
    cfg = Config(sample_fps=5.0, roi_bottom_fraction=1.0, force_full_frame=True)
    with VideoSource(FIXTURE) as video:
        candidates, near_misses, diag = run_visual_channel(video, TARGET_TEXT, cfg)

    assert "full_frame_forced" in diag
    assert "roi_sweep" not in diag
    assert len(candidates) == 1
