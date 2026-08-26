"""merges evidence from both channels. compares timestamps first, scores
second. an OCR score and a Whisper score aren't measuring the same kind of
error, so they're not really comparable numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Outcome(Enum):
    CONFIRMED_BY_AUDIO = "CONFIRMED_BY_AUDIO"
    CONFIRMED_BY_VISUAL = "CONFIRMED_BY_VISUAL"
    CORROBORATED = "CORROBORATED"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_FOUND = "NOT_FOUND"


@dataclass
class Candidate:
    channel: str  # "visual" | "audio"
    timestamp_s: float
    frame_index: int | None
    confidence: float
    matched_text: str
    matching_method: str  # e.g. "partial_ratio", "token_set_ratio"


@dataclass
class NearMiss:
    channel: str
    timestamp_s: float
    confidence: float
    text: str


@dataclass
class FusionResult:
    outcome: Outcome
    primary: Candidate | None
    all_candidates: list[Candidate] = field(default_factory=list)
    near_misses: list[NearMiss] = field(default_factory=list)
    ambiguity_status: str = ""
    uncertain: bool = False


def fuse(
    visual_candidates: list[Candidate],
    audio_candidates: list[Candidate],
    visual_near_misses: list[NearMiss],
    audio_near_misses: list[NearMiss],
    confident_threshold: float,
    corroboration_tolerance_s: float,
) -> FusionResult:
    v_confident = [c for c in visual_candidates if c.confidence >= confident_threshold]
    a_confident = [c for c in audio_candidates if c.confidence >= confident_threshold]

    v_best = max(v_confident, key=lambda c: c.confidence, default=None)
    a_best = max(a_confident, key=lambda c: c.confidence, default=None)

    if v_best is None and a_best is None:
        near = sorted(visual_near_misses + audio_near_misses, key=lambda n: -n.confidence)
        return FusionResult(
            outcome=Outcome.NOT_FOUND,
            primary=None,
            near_misses=near[:5],
            ambiguity_status="no confident match in either channel",
        )

    if v_best is None:
        return FusionResult(
            outcome=Outcome.CONFIRMED_BY_AUDIO,
            primary=a_best,
            all_candidates=[a_best],
        )

    if a_best is None:
        return FusionResult(
            outcome=Outcome.CONFIRMED_BY_VISUAL,
            primary=v_best,
            all_candidates=[v_best],
        )

    # both confident, so compare where they landed
    delta = abs(v_best.timestamp_s - a_best.timestamp_s)
    if delta <= corroboration_tolerance_s:
        # they agree, use visual as primary since it's frame-accurate. not
        # picking a "preferred" channel, just the more precise value here
        return FusionResult(
            outcome=Outcome.CORROBORATED,
            primary=v_best,
            all_candidates=[v_best, a_best],
            ambiguity_status=f"channels agree within {delta:.2f}s",
        )

    # they don't agree. not picking one, just reporting both.
    return FusionResult(
        outcome=Outcome.AMBIGUOUS,
        primary=None,
        all_candidates=[v_best, a_best],
        ambiguity_status=f"channels disagree: visual={v_best.timestamp_s:.2f}s "
                          f"audio={a_best.timestamp_s:.2f}s (delta={delta:.2f}s)",
    )


def resolve_strict(result: FusionResult) -> Candidate | None:
    """only gets called under --strict, when a single answer is required.
    no fixed channel preference here, on purpose.
    """
    if result.primary is not None:
        return result.primary

    if result.outcome == Outcome.AMBIGUOUS and result.all_candidates:
        # pick whichever channel was more confident, but flag it as
        # uncertain, forcing an answer shouldn't hide that it was a guess
        best = max(result.all_candidates, key=lambda c: c.confidence)
        best_marked = Candidate(**{**best.__dict__})
        result.uncertain = True
        return best_marked

    return None
