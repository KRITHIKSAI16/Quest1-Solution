"""Input acquisition. Figures out if `source` is a yt-dlp URL, a direct media
URL, or a local path, and gets us a local file either way.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp


# acquire() always raises one of these, never a raw yt-dlp exception, so
# callers can branch on type instead of parsing error text.

class AcquisitionError(Exception):
    """base for everything below."""


class InvalidSourceError(AcquisitionError):
    """not a local file, not a URL, or a URL with nothing playable at it. don't retry these."""


class NetworkUnreachableError(AcquisitionError):
    """connection kept failing across the whole retry budget."""


class MissingDependencyError(AcquisitionError):
    """a package we need isn't importable. retrying won't help, it'll fail the same way every time."""


class IncompleteDownloadError(AcquisitionError):
    """yt-dlp said it succeeded but the file's actual duration doesn't match what the source reported."""


MAX_ATTEMPTS = 3
BACKOFF_SCHEDULE_S = (2, 6, 18)  # wait before attempt 2, 3, 4
DURATION_TOLERANCE = 0.05  # 5% off from reported duration = call it incomplete


def _cache_path(cache_dir: str, url: str) -> Path:
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"{key}.mp4"


def _looks_like_url(s: str) -> bool:
    try:
        parsed = urlparse(s)
        return parsed.scheme in ("http", "https")
    except ValueError:
        return False


def _classify(exc: Exception) -> type[AcquisitionError]:
    """yt-dlp doesn't give us typed error causes, so this is just sniffing the
    message text. anything we don't recognize gets treated as transient (worth
    retrying) rather than assumed permanent.
    """
    msg = str(exc).lower()
    permanent_markers = (
        "unsupported url",
        "is not a valid url",
        "no video formats found",
        "unable to extract",
        "video unavailable",
        "this video is not available",
        "requested format is not available",
    )
    if any(m in msg for m in permanent_markers):
        return InvalidSourceError
    return NetworkUnreachableError


class _SilentLogger:
    """yt-dlp prints its own ERROR line even with quiet/no_warnings set. we
    already raise a proper exception for this, so shut it up.
    """
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


def _download_once(source: str, cached: Path) -> dict:
    """one attempt. classification/retry is handled by the caller, this just
    lets the yt-dlp exception through as-is.
    """
    import imageio_ffmpeg

    ydl_opts = {
        "outtmpl": str(cached.with_suffix("")) + ".%(ext)s",
        # most sites don't serve one combined stream anymore, so grab the best
        # video+audio and let ffmpeg merge them. capped at 1080p so a casual
        # test run doesn't pull down a 4K file by accident.
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "merge_output_format": "mp4",
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _SilentLogger(),
        # the merge step's own ffmpeg subprocess prints straight to the
        # terminal, ignores quiet/no_warnings above. needed on both inputs,
        # not just the output, or the noise still gets through.
        "postprocessor_args": {
            "merger+ffmpeg_i1": ["-loglevel", "fatal"],
            "merger+ffmpeg_i2": ["-loglevel", "fatal"],
            "default": ["-loglevel", "fatal"],
        },
        "retries": 5,
        "fragment_retries": 10,
        "continuedl": True,
        # fixes an MPEG-TS/AAC timestamp issue on HLS sources
        "hls_use_mpegts": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source, download=True)
        downloaded = ydl.prepare_filename(info)

    downloaded_path = Path(downloaded)
    if downloaded_path != cached:
        downloaded_path.replace(cached)
    return info


def _verify_complete(cached: Path, info: dict) -> None:
    """checks the downloaded file's real duration against what the source said
    it should be. skips the check if the source didn't report a duration,
    that seemed better than pretending it passed.
    """
    expected = info.get("duration")
    if not expected:
        return

    from core.video import VideoSource 

    with VideoSource(str(cached)) as v:
        actual = v.meta.duration_s

    if abs(actual - expected) / expected > DURATION_TOLERANCE:
        raise IncompleteDownloadError(
            f"Downloaded file duration ({actual:.1f}s) disagrees with the "
            f"source's reported duration ({expected:.1f}s) by more than "
            f"{DURATION_TOLERANCE:.0%} — the download is likely incomplete "
            f"(a partial fragment failure that yt-dlp did not treat as fatal)."
        )


def acquire(source: str, cache_dir: str = ".cache") -> str:
    """local path -> returned as-is. URL -> fetched via yt-dlp, cached by hash.
    retries transient failures, gives up immediately on permanent ones.
    """
    if os.path.isfile(source):
        return source

    if not _looks_like_url(source):
        raise InvalidSourceError(
            f"'{source}' is not an existing local file and not a URL."
        )

    os.makedirs(cache_dir, exist_ok=True)
    cached = _cache_path(cache_dir, source)
    if cached.exists() and cached.stat().st_size > 0:
        return str(cached)

    last_error: AcquisitionError | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            info = _download_once(source, cached)
            _verify_complete(cached, info)
            return str(cached)
        except InvalidSourceError:
            raise  # never retried
        except IncompleteDownloadError as e:
            # don't leave a bad file in the cache, it'd look like a hit next attempt
            cached.unlink(missing_ok=True)
            last_error = e
        except ImportError as e:
            # not a yt-dlp thing, classify() would just misfile it as a network error
            raise MissingDependencyError(
                f"{e}. If you're running this from an activated virtual "
                f"environment, its packages may not be installed "
                f"(`pip install -r requirements.txt`); if not, make sure "
                f"you're running the project's venv interpreter rather than "
                f"a system-wide `python`."
            ) from e
        except Exception as e:
            error_cls = _classify(e)
            if error_cls is InvalidSourceError:
                raise InvalidSourceError(str(e)) from e
            cached.unlink(missing_ok=True)
            last_error = NetworkUnreachableError(str(e))

        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(BACKOFF_SCHEDULE_S[attempt])

    assert last_error is not None
    raise last_error
