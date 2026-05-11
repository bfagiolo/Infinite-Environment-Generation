"""Trajectory-change probes for bounce, deflection, and aiming objectives."""

from __future__ import annotations

from typing import Any

from .contract import ProbeResult


def trajectory_change_probe(
    *,
    object_name: str,
    summary: dict[str, Any],
    min_heading_change_degrees: float = 15.0,
    min_speed_change: float = 10.0,
) -> ProbeResult:
    """Convert rollout trajectory telemetry into reusable probe evidence."""

    heading_change = _float_or_none(summary.get("heading_change_degrees"))
    speed_change = _float_or_none(summary.get("speed_change"))
    impact_count = _float_or_none(summary.get("impact_count"))
    heading_passed = heading_change is not None and abs(heading_change) >= min_heading_change_degrees
    speed_passed = speed_change is not None and abs(speed_change) >= min_speed_change
    contact_passed = impact_count is not None and impact_count > 0
    passed = heading_passed or speed_passed or contact_passed
    return ProbeResult(
        name="trajectory_change",
        passed=passed,
        tier_evidence=4 if passed else 2,
        objects=(object_name,),
        metrics={
            "heading_change_degrees": heading_change,
            "speed_change": speed_change,
            "impact_count": impact_count,
            "recommended_min_heading_change_degrees": float(min_heading_change_degrees),
            "recommended_min_speed_change": float(min_speed_change),
        },
        diagnosis="trajectory_changed" if passed else "trajectory_did_not_change",
        repair=(
            "Trajectory changed during rollout."
            if passed
            else "Place the bumper/field directly in the moving object's path and reduce damping/friction that prevents meaningful deflection."
        ),
        severity="info" if passed else "warning",
    )


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
