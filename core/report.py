"""builds the Result, writes result.json, saves the frame PNG. everything's
forced UTF-8 since extracted text can be any unicode and Windows' default
console codepage chokes on it otherwise.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from core.fuse import Candidate, FusionResult, NearMiss, Outcome


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


@dataclass
class Result:
    outcome: str
    timestamp: str | None
    frame_number: int | None
    text: str | None
    channel: str | None
    confidence: float | None
    ambiguity_status: str
    uncertain: bool
    all_candidates: list[dict] = field(default_factory=list)
    near_misses: list[dict] = field(default_factory=list)
    image_path: str | None = None
    diagnostics: dict = field(default_factory=dict)
    target_text: str | None = None  # what the user searched for, separate from `text` (what we actually found)

    def to_dict(self) -> dict:
        return asdict(self)


def build_result(fusion: FusionResult, primary: Candidate | None, target: str, diagnostics: dict | None = None) -> Result:
    return Result(
        outcome=fusion.outcome.value,
        timestamp=format_timestamp(primary.timestamp_s) if primary else None,
        frame_number=primary.frame_index if primary else None,
        text=primary.matched_text if primary else None,
        channel=primary.channel if primary else None,
        confidence=primary.confidence if primary else None,
        ambiguity_status=fusion.ambiguity_status,
        uncertain=fusion.uncertain,
        all_candidates=[asdict(c) for c in fusion.all_candidates],
        near_misses=[asdict(n) for n in fusion.near_misses],
        diagnostics=diagnostics or {},
        target_text=target,
    )


def save_frame(image: np.ndarray, out_dir: str, frame_index: int | None, timestamp_s: float) -> str:
    os.makedirs(out_dir, exist_ok=True)
    tag = frame_index if frame_index is not None else f"t{timestamp_s:.3f}"
    path = os.path.join(out_dir, f"frame_{tag}.png")
    cv2.imwrite(path, image)
    return path


def write_result_json(result: Result, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "result.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    return path


def print_report(result: Result):
    """prints the console report, forced UTF-8 for the cp1252 console case."""
    import sys
    import io

    out = sys.stdout
    wrapper = None
    if getattr(out, "encoding", "").lower() != "utf-8" and hasattr(out, "buffer"):
        # detach() before this goes out of scope, otherwise its __del__
        # closes the real stdout buffer and some unrelated atexit hook
        # (colorama, in this case) blows up trying to write to it
        wrapper = io.TextIOWrapper(out.buffer, encoding="utf-8", errors="replace")
        out = wrapper

    print(f"Outcome    : {result.outcome}", file=out)
    if result.target_text is not None:
        print(f"Target     : \"{result.target_text}\"", file=out)
    if result.timestamp:
        print(f"Timestamp  : {result.timestamp}", file=out)
        print(f"Frame      : {result.frame_number}", file=out)
        print(f"Text       : \"{result.text}\"", file=out)
        print(f"Channel    : {result.channel}", file=out)
        print(f"Confidence : {result.confidence:.1f}", file=out)
    if len(result.all_candidates) > 1:
        # both channels found something (CORROBORATED or AMBIGUOUS), show both
        print("Per-channel evidence:", file=out)
        for c in result.all_candidates:
            print(f"  [{c['channel']}] {format_timestamp(c['timestamp_s'])}  "
                  f"{c['confidence']:.1f}  \"{c['matched_text']}\"", file=out)
    if result.ambiguity_status:
        print(f"Note       : {result.ambiguity_status}", file=out)
    if result.uncertain:
        print("Uncertain  : true", file=out)
    if result.image_path:
        print(f"Image      : {result.image_path}", file=out)
    if not result.timestamp and result.near_misses:
        print("No confident match. Nearest candidates:", file=out)
        for nm in result.near_misses:
            print(f"  [{nm['channel']}] {format_timestamp(nm['timestamp_s'])}  "
                  f"score={nm['confidence']:.1f}  \"{nm['text']}\"", file=out)

    if result.diagnostics:
        print(file=out)
        print("Search diagnostics:", file=out)
        vd = result.diagnostics.get("visual", {})
        if "roi_sweep" in vd:
            s = vd["roi_sweep"]
            print(f"  visual (ROI sweep)   : {s['total']} frames sampled -> "
                  f"{s['rejected_dedup']} deduped, {s['rejected_no_text']} no-text, "
                  f"{s['ocr_calls']} OCR calls", file=out)
        if "full_frame_fallback" in vd:
            s = vd["full_frame_fallback"]
            print(f"  visual (full-frame)  : {s['total']} frames sampled -> "
                  f"{s['rejected_dedup']} deduped, {s['rejected_no_text']} no-text, "
                  f"{s['ocr_calls']} OCR calls  [ROI sweep found nothing, fell back]", file=out)
        if "full_frame_forced" in vd:
            s = vd["full_frame_forced"]
            print(f"  visual (full-frame)  : {s['total']} frames sampled -> "
                  f"{s['rejected_dedup']} deduped, {s['rejected_no_text']} no-text, "
                  f"{s['ocr_calls']} OCR calls  [--full-frame: ROI pass skipped]", file=out)
        if "timing_s" in vd:
            t = vd["timing_s"]
            line = f"  visual timing        : engine_load={t.get('engine_load')}s, coarse_sweep={t.get('coarse_sweep')}s"
            if "refine_total" in t:
                line += f", refine={t['refine_total']}s over {t.get('refine_calls', 0)} cluster(s)"
            print(line, file=out)
        ad = result.diagnostics.get("audio", {})
        if "segments_transcribed" in ad:
            print(f"  audio                : {ad['segments_transcribed']} transcript segments", file=out)

    out.flush()
    if wrapper is not None:
        wrapper.detach()
