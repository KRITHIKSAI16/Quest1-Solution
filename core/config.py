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
