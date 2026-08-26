"""runs both channels, fuses the results, produces a Result."""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass

from core.asr import WhisperTranscriber, extract_audio
from core.config import Config
from core.fuse import Candidate, FusionResult, NearMiss, Outcome, fuse, resolve_strict
from core.gate import FrameGate
from core.matcher import alignment_ratio, score as fuzzy_score
from core.ocr import EasyOcrEngine, assemble_line
from core.roi import crop_roi, downscale_for_ocr, full_frame
from core.video import Frame, VideoSource

# tried parallelizing OCR calls (cfg.ocr_workers) at one point, it was
# actually slower. EasyOCR already uses multiple threads internally per
# call. channel-level concurrency below is fine though, different libraries,
# no contention.


@dataclass
class VisualHit:
    frame: Frame
    text: str
    score: float


def _visual_sweep(video: VideoSource, target: str, cfg: Config, use_roi: bool, ocr: EasyOcrEngine) -> tuple[list[VisualHit], FrameGate]:
    """coarse sweep, scores whatever frames make it past the gates."""
    gate = FrameGate(cfg.dedup_mad_threshold, cfg.text_presence_min_edge_density)
    hits: list[VisualHit] = []

    for frame in video.sample_at(cfg.sample_fps):
        region = crop_roi(frame.image, cfg.roi_bottom_fraction) if use_roi else full_frame(frame.image)
        region = downscale_for_ocr(region, cfg.max_ocr_dimension)
        if not gate.should_ocr(region):
            continue
        boxes = ocr.read(region)
        text = assemble_line(boxes)
        if not text:
            continue
        s = fuzzy_score(text, target)
        if s >= cfg.uncertain_threshold:
            hits.append(VisualHit(frame=frame, text=text, score=s))

    return hits, gate


def _refine_visual_onset(video: VideoSource, target: str, approx_time_s: float, cfg: Config, ocr: EasyOcrEngine) -> VisualHit | None:
    """pin down where the text actually starts. re-decodes a small window
    around the coarse hit (at refine_sample_fps, not native, way cheaper),
    then walks backward from the anchor while the score stays confident.
    """
    t0 = approx_time_s - cfg.refine_window_s
    t1 = approx_time_s + cfg.refine_window_s
    scored: list[VisualHit] = []
    for frame in video.sample_range(t0, t1, cfg.refine_sample_fps):
        region = crop_roi(frame.image, cfg.roi_bottom_fraction)
        region = downscale_for_ocr(region, cfg.max_ocr_dimension)
        boxes = ocr.read(region)
        text = assemble_line(boxes)
        s = fuzzy_score(text, target) if text else 0.0
        scored.append(VisualHit(frame=frame, text=text, score=s))

    scored.sort(key=lambda h: h.frame.index)
    # find the frame closest to approx_time_s that's confident, then walk backward
    confident = [h for h in scored if h.score >= cfg.confident_threshold]
    if not confident:
        return None
    anchor = min(confident, key=lambda h: abs(h.frame.timestamp_s - approx_time_s))
    anchor_pos = scored.index(anchor)

    onset = anchor
    walked_s = 0.0
    i = anchor_pos - 1
    while i >= 0 and walked_s < cfg.onset_walk_cap_s:
        h = scored[i]
        if h.score < cfg.confident_threshold:
            break
        onset = h
        walked_s = anchor.frame.timestamp_s - h.frame.timestamp_s
        i -= 1

    return onset


def run_visual_channel(video: VideoSource, target: str, cfg: Config) -> tuple[list[Candidate], list[NearMiss], dict]:
    """coarse sweep -> refine -> onset walk. also returns the gate stats so we
    can actually see the search-space reduction instead of just claiming it.
    """
    t0 = time.time()
    ocr = EasyOcrEngine()  # reuse across sweep + refine, don't reload the model every call
    t_engine_done = time.time()

    if cfg.force_full_frame:
        hits, gate = _visual_sweep(video, target, cfg, use_roi=False, ocr=ocr)
        diag = {"full_frame_forced": {"total": gate.stats.total, "rejected_dedup": gate.stats.rejected_dedup,
                                       "rejected_no_text": gate.stats.rejected_no_text, "ocr_calls": gate.stats.passed}}
    else:
        hits, gate = _visual_sweep(video, target, cfg, use_roi=True, ocr=ocr)
        diag = {"roi_sweep": {"total": gate.stats.total, "rejected_dedup": gate.stats.rejected_dedup,
                               "rejected_no_text": gate.stats.rejected_no_text, "ocr_calls": gate.stats.passed}}

        if not hits and cfg.full_frame_on_miss:
            hits, gate2 = _visual_sweep(video, target, cfg, use_roi=False, ocr=ocr)
            diag["full_frame_fallback"] = {"total": gate2.stats.total, "rejected_dedup": gate2.stats.rejected_dedup,
                                            "rejected_no_text": gate2.stats.rejected_no_text, "ocr_calls": gate2.stats.passed}

    t_sweep_done = time.time()
    diag["timing_s"] = {"engine_load": round(t_engine_done - t0, 2),
                         "coarse_sweep": round(t_sweep_done - t_engine_done, 2)}

    if not hits:
        return [], [], diag

    # group nearby hits together in case the target shows up more than once
    hits.sort(key=lambda h: h.frame.timestamp_s)
    clusters: list[list[VisualHit]] = [[hits[0]]]
    for h in hits[1:]:
        if h.frame.timestamp_s - clusters[-1][-1].frame.timestamp_s <= cfg.refine_window_s * 2:
            clusters[-1].append(h)
        else:
            clusters.append([h])

    candidates: list[Candidate] = []
    near_misses: list[NearMiss] = []
    t_refine_total = 0.0
    refine_calls = 0
    for cluster in clusters:
        best_in_cluster = max(cluster, key=lambda h: h.score)
        if best_in_cluster.score < cfg.confident_threshold:
            near_misses.append(NearMiss(
                channel="visual", timestamp_s=best_in_cluster.frame.timestamp_s,
                confidence=best_in_cluster.score, text=best_in_cluster.text,
            ))
            continue
        t_refine_start = time.time()
        onset = _refine_visual_onset(video, target, best_in_cluster.frame.timestamp_s, cfg, ocr=ocr)
        t_refine_total += time.time() - t_refine_start
        refine_calls += 1
        if onset is None:
            continue
        candidates.append(Candidate(
            channel="visual", timestamp_s=onset.frame.timestamp_s,
            frame_index=onset.frame.index, confidence=onset.score,
            matched_text=onset.text, matching_method="partial_ratio/token_set_ratio",
        ))

    diag["timing_s"]["refine_total"] = round(t_refine_total, 2)
    diag["timing_s"]["refine_calls"] = refine_calls

    return candidates, near_misses, diag


def run_audio_channel(video_path: str, video: VideoSource, target: str, cfg: Config) -> tuple[list[Candidate], list[NearMiss], dict]:
    """extract audio, transcribe with word timestamps, fuzzy-match against
    segments, then pin the onset down to the actual word.
    """
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "audio.wav")
        extract_audio(video_path, wav_path)
        transcriber = WhisperTranscriber(cfg.whisper_model, cfg.whisper_device, cfg.whisper_compute_type)
        segments = transcriber.transcribe(wav_path)

    diag = {"segments_transcribed": len(segments)}
    candidates: list[Candidate] = []
    near_misses: list[NearMiss] = []

    for seg in segments:
        s = fuzzy_score(seg.text, target)
        if s < cfg.uncertain_threshold:
            continue
        if s < cfg.confident_threshold:
            near_misses.append(NearMiss(channel="audio", timestamp_s=seg.start_s, confidence=s, text=seg.text))
            continue

        onset_word_s = _word_level_onset(seg, target)
        # careful with `or` here, a real onset of 0.0 would get treated as falsy
        onset_s = onset_word_s if onset_word_s is not None else seg.start_s
        # this is derived from timestamp*fps rather than a real decoder frame
        # index like the visual channel gets, so it's approximate on VFR video.
        # round() not int(), truncating always picks a frame too early.
        frame_idx = round(onset_s * video.meta.fps)
        candidates.append(Candidate(
            channel="audio", timestamp_s=onset_s, frame_index=frame_idx,
            confidence=s, matched_text=seg.text, matching_method="partial_ratio/token_set_ratio",
        ))

    near_misses.sort(key=lambda n: -n.confidence)
    return candidates, near_misses[:cfg.near_miss_top_n], diag


def _word_level_onset(segment, target: str) -> float | None:
    """finds the tightest run of words in the segment that matches the target,
    returns when that run starts. scored by (fuzzy_score, alignment_ratio) so
    a wide span that just happens to contain the target doesn't win over the
    actual tight match sitting further into the segment.
    """
    words = segment.words
    if not words:
        return None
    n = len(words)
    max_span = max(12, len(target.split()) + 4)
    best_start_idx = 0
    best_key = (-1.0, -1.0)
    for start in range(n):
        joined = ""
        for end in range(start, min(start + max_span, n)):
            joined = (joined + " " + words[end].text).strip()
            key = (fuzzy_score(joined, target), alignment_ratio(joined, target))
            if key > best_key:
                best_key = key
                best_start_idx = start
    return words[best_start_idx].start_s


def locate(video_path: str, target: str, cfg: Config, mode: str = "auto") -> tuple[FusionResult, Candidate | None, dict]:
    """entrypoint. mode is "auto" / "visual" / "audio".

    tried running both channels concurrently in a thread pool at one point,
    ended up slower, not faster. both libraries are already pulling on a lot
    of CPU threads internally and running them at the same time just causes
    contention. sequential it is.
    """
    with VideoSource(video_path) as video:
        visual_candidates: list[Candidate] = []
        visual_near: list[NearMiss] = []
        audio_candidates: list[Candidate] = []
        audio_near: list[NearMiss] = []
        diagnostics: dict = {}

        if mode in ("auto", "visual"):
            visual_candidates, visual_near, diagnostics["visual"] = run_visual_channel(video, target, cfg)
        if mode in ("auto", "audio"):
            audio_candidates, audio_near, diagnostics["audio"] = run_audio_channel(video_path, video, target, cfg)

        result = fuse(
            visual_candidates, audio_candidates, visual_near, audio_near,
            cfg.confident_threshold, cfg.corroboration_tolerance_s,
        )
        primary = result.primary
        return result, primary, diagnostics
