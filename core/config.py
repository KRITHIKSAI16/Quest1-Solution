"""all the tunables in one place instead of scattered magic numbers."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- visual channel ---
    sample_fps: float = 1.0
    roi_bottom_fraction: float = 0.40  # fraction of frame height, from the bottom
    force_full_frame: bool = False  # --full-frame: skip the ROI pass entirely
    full_frame_on_miss: bool = True  # automatic fallback when the ROI pass finds nothing
    dedup_mad_threshold: float = 4.0  # mean-abs-diff below this = "same frame as last"
    text_presence_min_edge_density: float = 0.006
    ocr_workers: int = 4
    max_ocr_dimension: int = 1280  # cap the longest edge fed to OCR
    refine_window_s: float = 2.0
    refine_sample_fps: float = 8.0  # sample the refine window, not every native frame
    onset_walk_cap_s: float = 10.0

    # --- audio channel ---
    whisper_model: str = "small.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    face_check_enabled: bool = True
    face_check_window_s: float = 0.5  # how far around the onset frame to look for a face
    mouth_motion_check_enabled: bool = True  # report-only signal, never gates the outcome
    mouth_motion_baseline_offset_s: float = 1.5  # how far before the onset to sample the "quiet" comparison window
    mouth_motion_sample_fps: float = 12.0  # dense-sampling rate within each window, for frame-to-frame motion
    mouth_motion_ratio: float = 1.6  # speech-window motion must exceed baseline by this factor to count as "elevated"

    # --- matching (shared) ---
    confident_threshold: float = 85.0
    uncertain_threshold: float = 70.0
    near_miss_top_n: int = 5

    # --- fusion ---
    corroboration_tolerance_s: float = 2.0

    # --- io ---
    cache_dir: str = ".cache"
    out_dir: str = "output"


DEFAULT = Config()
