"""audio channel, faster-whisper wrapper with word-level timestamps. the
first matched word's start time is what this channel reports as the onset.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Word:
    text: str
    start_s: float
    end_s: float


@dataclass
class TranscriptSegment:
    text: str
    start_s: float
    end_s: float
    words: list[Word]


class WhisperTranscriber:
    def __init__(self, model_size: str = "small.en", device: str = "cpu", compute_type: str = "int8"):
        from faster_whisper import WhisperModel  # deferred: heavy import

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str) -> list[TranscriptSegment]:
        segments, _info = self._model.transcribe(audio_path, word_timestamps=True)
        out = []
        for seg in segments:
            words = [
                Word(text=w.word, start_s=w.start, end_s=w.end)
                for w in (seg.words or [])
            ]
            out.append(TranscriptSegment(text=seg.text, start_s=seg.start, end_s=seg.end, words=words))
        return out


def extract_audio(video_path: str, out_wav_path: str) -> str:
    """mono 16kHz wav for whisper. uses imageio-ffmpeg's bundled binary so we
    don't need ffmpeg installed on the system.
    """
    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    import subprocess

    subprocess.run(
        [
            ffmpeg_exe,
            "-loglevel", "fatal",  # suppress decoder noise; success is checked via check=True
            "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000",
            out_wav_path,
        ],
        check=True,
    )
    return out_wav_path
