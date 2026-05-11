"""Trajectory and rollout-summary probes."""

from __future__ import annotations

from typing import Any

from .common import string_list
from .contract import ProbeResult


def object_region_motion_probe(
    *,
    object_name: str,
    region_name: str,
    summary: dict[str, Any],
    failure_modes: list[str] | tuple[str, ...] | None = None,
) -> ProbeResult:
    """Convert object-to-region rollout diagnostics into probe evidence."""

    distance_reduced = _float_or_none(summary.get("distance_reduced"))
    final_distance = _float_or_none(summary.get("final_distance"))
    modes = string_list(failure_modes)
    passed = bool(distance_reduced is not None and distance_reduced > 0.0)
    solved_like = final_distance is not None and final_distance <= float(summary.get("threshold") or 0.0)
    if solved_like:
        passed = True
    return ProbeResult(
        name="object_moves_toward_region",
        passed=passed,
        tier_evidence=5 if solved_like else (4 if passed else 2),
        objects=(object_name, region_name),
        metrics={**summary, "failure_modes": modes},
        diagnosis="object_moved_toward_region" if passed else "object_failed_to_approach_region",
        repair=(
            "Object moved toward the target region."
            if passed
            else "Stage the agent behind the object, prevent sideways drift, and reduce mass/friction if forward velocity is weak."
        ),
        severity="info" if passed else "error",
    )


def agent_target_motion_probe(
    *,
    target_name: str,
    summary: dict[str, Any],
    failure_modes: list[str] | tuple[str, ...] | None = None,
) -> ProbeResult:
    """Convert agent-to-target rollout diagnostics into probe evidence."""

    distance_reduced = _float_or_none(summary.get("distance_reduced"))
    final_distance = _float_or_none(summary.get("final_distance"))
    threshold = _float_or_none(summary.get("threshold"))
    modes = string_list(failure_modes)
    solved_like = (
        final_distance is not None
        and threshold is not None
        and final_distance <= threshold
    )
    progress = bool(distance_reduced is not None and distance_reduced > 5.0)
    return ProbeResult(
        name="agent_moves_toward_target",
        passed=solved_like or progress,
        tier_evidence=5 if solved_like else (4 if progress else 2),
        objects=(target_name,),
        metrics={**summary, "failure_modes": modes},
        diagnosis="agent_reached_target" if solved_like else ("agent_progressed_toward_target" if progress else "agent_failed_to_progress"),
        repair=(
            "Agent reached/progressed toward the target."
            if solved_like or progress
            else "Move the target closer, clear the path, or increase agent_strength."
        ),
        severity="info" if solved_like or progress else "error",
    )


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
