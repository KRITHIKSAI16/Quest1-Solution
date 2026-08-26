"""Regression tests for locate()'s mode-branching logic — which channel(s)
run, what diagnostics they produce, and that a real failure in either
channel isn't silently swallowed.

Channel-level concurrency (running visual + audio simultaneously via a
thread pool) was tried and reverted: measured slower end-to-end on real
footage (2m5s vs. 1m40s sequential, same clip, same result) because
EasyOCR/PyTorch and faster-whisper/CTranslate2 both want significant CPU
threads for their own internal work, and running them at once caused
contention that cost more than the concurrency saved. See
internal/INTERVIEW_NOTES.md T9 for the full measurement trail. locate() is
sequential again; these tests cover what's actually still true — the mode
logic itself — using mocked channel functions so they run fast and portably
without depending on the real-footage fixtures used for manual verification.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import Config
from core.locate import locate


def _fake_visual_channel(video, target, cfg):
    return [], [], {"roi_sweep": {"total": 1, "rejected_dedup": 0, "rejected_no_text": 0, "ocr_calls": 1}}


def _fake_audio_channel(video_path, video, target, cfg):
    return [], [], {"segments_transcribed": 1}


@pytest.fixture
def tiny_video_path(tmp_path):
    """A real, minimal video file — locate() opens it via VideoSource before
    either channel runs, so it needs to exist and be readable, even though
    both channels are mocked and never touch its actual content."""
    import cv2
    import numpy as np
    path = str(tmp_path / "tiny.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 32))
    for _ in range(5):
        writer.write(np.zeros((32, 32, 3), dtype="uint8"))
    writer.release()
    return path


def test_auto_mode_diagnostics_contains_both_channels(tiny_video_path):
    cfg = Config()
    with patch("core.locate.run_visual_channel", side_effect=_fake_visual_channel), \
         patch("core.locate.run_audio_channel", side_effect=_fake_audio_channel):
        _fusion, _primary, diagnostics = locate(tiny_video_path, "target", cfg, mode="auto")

    assert "visual" in diagnostics
    assert "audio" in diagnostics
    assert diagnostics["visual"]["roi_sweep"]["ocr_calls"] == 1
    assert diagnostics["audio"]["segments_transcribed"] == 1


def test_visual_only_mode_never_calls_audio_channel(tiny_video_path):
    cfg = Config()
    with patch("core.locate.run_visual_channel", side_effect=_fake_visual_channel) as visual_mock, \
         patch("core.locate.run_audio_channel", side_effect=_fake_audio_channel) as audio_mock:
        _fusion, _primary, diagnostics = locate(tiny_video_path, "target", cfg, mode="visual")

    assert visual_mock.called
    assert not audio_mock.called
    assert "visual" in diagnostics
    assert "audio" not in diagnostics


def test_audio_only_mode_never_calls_visual_channel(tiny_video_path):
    cfg = Config()
    with patch("core.locate.run_visual_channel", side_effect=_fake_visual_channel) as visual_mock, \
         patch("core.locate.run_audio_channel", side_effect=_fake_audio_channel) as audio_mock:
        _fusion, _primary, diagnostics = locate(tiny_video_path, "target", cfg, mode="audio")

    assert audio_mock.called
    assert not visual_mock.called
    assert "audio" in diagnostics
    assert "visual" not in diagnostics


def test_auto_mode_propagates_exception_from_either_channel(tiny_video_path):
    def _raise(*a, **kw):
        raise RuntimeError("simulated channel failure")

    cfg = Config()
    with patch("core.locate.run_visual_channel", side_effect=_raise), \
         patch("core.locate.run_audio_channel", side_effect=_fake_audio_channel):
        with pytest.raises(RuntimeError, match="simulated channel failure"):
            locate(tiny_video_path, "target", cfg, mode="auto")
