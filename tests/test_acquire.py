"""Unit tests for core/acquire.py's robustness logic.

Deliberately does NOT hit real network/yt-dlp — that would be slow, flaky,
and isn't what's being verified here. Instead:
- Classification and URL/path detection are pure functions, tested directly.
- The retry wrapper's *policy* (retry transient, don't retry permanent,
  clean up a bad cache entry) is tested by monkeypatching acquire.py's
  internal _download_once/_verify_complete rather than by faking a whole
  yt-dlp session.
- The completeness check's math is tested against the real synthetic clip
  fixture (tests/fixtures/synthetic_clip.mp4, 10.0s) with a fabricated
  info_duration, exercising the actual VideoSource read path.
"""
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.acquire import (
    AcquisitionError,
    InvalidSourceError,
    MissingDependencyError,
    NetworkUnreachableError,
    IncompleteDownloadError,
    MAX_ATTEMPTS,
    _classify,
    _looks_like_url,
    _verify_complete,
    acquire,
)
from tests.generate_synthetic import DURATION_S as SYNTHETIC_DURATION_S

FIXTURE = str(Path(__file__).parent / "fixtures" / "synthetic_clip.mp4")


@pytest.fixture(scope="module", autouse=True)
def ensure_fixture():
    if not os.path.exists(FIXTURE):
        from tests.generate_synthetic import generate
        generate(FIXTURE)


# --- _looks_like_url ---------------------------------------------------------

def test_looks_like_url_accepts_http_https():
    assert _looks_like_url("https://example.com/video")
    assert _looks_like_url("http://example.com/video")


def test_looks_like_url_rejects_local_paths():
    assert not _looks_like_url(r"C:\videos\file.mp4")
    assert not _looks_like_url("relative/path.mp4")
    assert not _looks_like_url("not a url at all")


def test_looks_like_url_rejects_empty_string():
    assert not _looks_like_url("")


# --- _classify ----------------------------------------------------------------

def test_classify_unsupported_url_is_permanent():
    assert _classify(Exception("Unsupported URL: https://example.com")) is InvalidSourceError


def test_classify_no_video_formats_is_permanent():
    assert _classify(Exception("No video formats found!")) is InvalidSourceError


def test_classify_video_unavailable_is_permanent():
    assert _classify(Exception("Video unavailable")) is InvalidSourceError


def test_classify_connection_reset_is_transient():
    assert _classify(Exception("Connection reset by peer")) is NetworkUnreachableError


def test_classify_unrecognised_message_defaults_transient_not_swallowed():
    """An unrecognised failure must default to retryable, not silently
    treated as permanent — a wrong 'permanent' classification would give up
    on a source that might work on retry."""
    assert _classify(Exception("some completely novel yt-dlp error")) is NetworkUnreachableError


# --- acquire(): local file / invalid source paths (no network involved) -----

def test_acquire_returns_local_file_path_unchanged():
    assert acquire(FIXTURE) == FIXTURE


def test_acquire_raises_invalid_source_for_nonexistent_non_url():
    with pytest.raises(InvalidSourceError):
        acquire("this is neither a file nor a url")


def test_acquire_raises_invalid_source_for_empty_string():
    with pytest.raises(InvalidSourceError):
        acquire("")


# --- acquire(): retry policy (network layer mocked) --------------------------

def test_acquire_does_not_retry_invalid_source_from_downloader(tmp_path):
    """If the downloader itself raises something classified as permanent,
    acquire() must fail on the first attempt, not burn the retry budget."""
    call_count = {"n": 0}

    def fake_download(source, cached):
        call_count["n"] += 1
        raise Exception("Unsupported URL: nope")

    with patch("core.acquire._download_once", side_effect=fake_download):
        with pytest.raises(InvalidSourceError):
            acquire("https://example.com/not-a-video", cache_dir=str(tmp_path))

    assert call_count["n"] == 1, "permanent failures must not be retried"


def test_acquire_does_not_retry_missing_dependency_and_reports_it_correctly(tmp_path):
    """A missing module (e.g. imageio_ffmpeg not installed in the active
    interpreter) is not a network failure and must not be misreported as
    one, nor retried — see internal/INTERVIEW_NOTES.md §B16."""
    call_count = {"n": 0}

    def fake_download(source, cached):
        call_count["n"] += 1
        raise ModuleNotFoundError("No module named 'imageio_ffmpeg'")

    with patch("core.acquire._download_once", side_effect=fake_download):
        with pytest.raises(MissingDependencyError):
            acquire("https://example.com/video", cache_dir=str(tmp_path))

    assert call_count["n"] == 1, "missing-dependency failures must not be retried"


def test_acquire_retries_transient_failure_then_succeeds(tmp_path):
    call_count = {"n": 0}
    cache_dir = tmp_path

    def fake_download(source, cached):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise Exception("Connection reset by peer")
        cached.write_bytes(b"fake video bytes")
        return {"duration": None}  # skip completeness check for this test

    with patch("core.acquire._download_once", side_effect=fake_download), \
         patch("core.acquire.time.sleep"):  # don't actually wait during tests
        result = acquire("https://example.com/video", cache_dir=str(cache_dir))

    assert call_count["n"] == 2
    assert os.path.exists(result)


def test_acquire_gives_up_after_max_attempts_with_persistent_failure(tmp_path):
    call_count = {"n": 0}

    def fake_download(source, cached):
        call_count["n"] += 1
        raise Exception("Connection reset by peer")

    with patch("core.acquire._download_once", side_effect=fake_download), \
         patch("core.acquire.time.sleep"):
        with pytest.raises(NetworkUnreachableError):
            acquire("https://example.com/video", cache_dir=str(tmp_path))

    assert call_count["n"] == MAX_ATTEMPTS


def test_acquire_cleans_up_incomplete_download_before_retry(tmp_path):
    """Regression guard for the gap found while designing this: a bad
    cached file must not survive to false-positive as a cache hit on the
    next attempt."""
    cache_dir = tmp_path
    call_count = {"n": 0}

    def fake_download(source, cached):
        call_count["n"] += 1
        cached.write_bytes(b"truncated garbage")
        return {"duration": 999999}  # forces the completeness check to fail every time

    with patch("core.acquire._download_once", side_effect=fake_download), \
         patch("core.acquire._verify_complete", side_effect=IncompleteDownloadError("too short")), \
         patch("core.acquire.time.sleep"):
        with pytest.raises(AcquisitionError):
            acquire("https://example.com/video", cache_dir=str(cache_dir))

    assert call_count["n"] == MAX_ATTEMPTS, "an incomplete download must be retried, not accepted"


# --- _verify_complete: real duration math against the synthetic fixture -----

def test_verify_complete_passes_when_duration_matches():
    info = {"duration": SYNTHETIC_DURATION_S}
    _verify_complete(Path(FIXTURE), info)  # must not raise


def test_verify_complete_raises_on_large_duration_mismatch():
    info = {"duration": SYNTHETIC_DURATION_S * 3}  # source claims 3x longer than actual
    with pytest.raises(IncompleteDownloadError):
        _verify_complete(Path(FIXTURE), info)


def test_verify_complete_skips_honestly_when_source_reports_no_duration():
    info = {"duration": None}
    _verify_complete(Path(FIXTURE), info)  # must not raise — nothing to compare against
