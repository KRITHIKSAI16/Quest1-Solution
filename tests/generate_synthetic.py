"""Generates a short synthetic video with burned-in text at a known frame and
known timestamp — exact ground truth, fully offline and deterministic. This
is the backbone of the visual-channel integration test: it validates the
coarse sweep, ROI/gate filters, and the backward onset walk against a result
we know for certain, rather than trusting the pipeline's own output on real
footage (see APPROACH.md "Verification").
"""
from __future__ import annotations

import cv2
import numpy as np

FPS = 25
WIDTH, HEIGHT = 320, 240
DURATION_S = 10
TARGET_TEXT = "HELLO WORLD TEST"
# Text is on screen from frame TEXT_START to TEXT_END inclusive.
TEXT_START_FRAME = 100  # 4.0s
TEXT_END_FRAME = 150  # 6.0s


def generate(path: str):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, FPS, (WIDTH, HEIGHT))
    total_frames = FPS * DURATION_S

    for i in range(total_frames):
        frame = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)  # dark gray background
        # some visual noise so early frames aren't literally identical (more
        # representative of real footage, and exercises the dedup gate)
        noise = np.random.randint(0, 10, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)

        if TEXT_START_FRAME <= i <= TEXT_END_FRAME:
            cv2.putText(
                frame, TARGET_TEXT, (20, HEIGHT - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
            )
        writer.write(frame)

    writer.release()


if __name__ == "__main__":
    generate("tests/fixtures/synthetic_clip.mp4")
    print(f"Generated tests/fixtures/synthetic_clip.mp4: text on screen frames "
          f"{TEXT_START_FRAME}-{TEXT_END_FRAME} ({TEXT_START_FRAME/FPS:.2f}s-{TEXT_END_FRAME/FPS:.2f}s)")
