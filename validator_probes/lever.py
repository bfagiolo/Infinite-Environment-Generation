"""Lever, seesaw, and launch probes for jointed mechanism validation."""

from __future__ import annotations

from typing import Any

from .contract import ProbeResult


def pivot_mechanism_probe(
    *,
    plank_name: str,
    weight_name: str | None,
    has_pivot_constraint: bool,
    plank_is_dynamic: bool,
    weight_is_dynamic: bool | None,
) -> ProbeResult:
    """Check whether a declared lever has the required physical structure."""

    passed = bool(has_pivot_constraint and plank_is_dynamic and (weight_is_dynamic is not False))
    failures: list[str] = []
    if not plank_is_dynamic:
        failures.append("plank_not_dynamic")
    if not has_pivot_constraint:
        failures.append("missing_pivot_constraint")
    if weight_is_dynamic is False:
        failures.append("weight_not_dynamic")
    return ProbeResult(
        name="pivot_mechanism",
        passed=passed,
        tier_evidence=3 if passed else 1,
        objects=tuple(item for item in (plank_name, weight_name) if item),
        metrics={
            "has_pivot_constraint": bool(has_pivot_constraint),
            "plank_is_dynamic": bool(plank_is_dynamic),
            "weight_is_dynamic": weight_is_dynamic,
            "failures": failures,
        },
        diagnosis="pivot_mechanism_valid" if passed else "pivot_mechanism_invalid",
        repair=(
            "Pivot mechanism has a dynamic plank and registered pivot constraint."
            if passed
            else "Use create_dynamic_box for the plank and register_constraint(type='PivotJoint', body_a=plank_name, body_b=None, anchor_a=(pivot_x, pivot_y))."
        ),
        severity="info" if passed else "error",
    )


def plank_rotation_probe(
    *,
    plank_name: str,
    summary: dict[str, Any],
    min_angle_delta: float = 0.08,
) -> ProbeResult:
    """Check whether the lever plank rotated meaningfully during rollout."""

    angle_delta = _float_or_none(summary.get("plank_angle_delta_abs"))
    max_angular_velocity = _float_or_none(summary.get("max_plank_angular_velocity_abs"))
    passed = (
        angle_delta is not None
        and angle_delta >= min_angle_delta
    ) or (
        max_angular_velocity is not None
        and max_angular_velocity >= 0.3
    )
    return ProbeResult(
        name="plank_rotates",
        passed=passed,
        tier_evidence=4 if passed else 2,
        objects=(plank_name,),
        metrics={
            "plank_angle_delta_abs": angle_delta,
            "max_plank_angular_velocity_abs": max_angular_velocity,
            "recommended_min_angle_delta": float(min_angle_delta),
        },
        diagnosis="plank_rotation_observed" if passed else "plank_rotation_insufficient",
        repair=(
            "Plank rotation was observed."
            if passed
            else "Increase mechanical advantage: move the weight nearer the load side, reduce plank mass/friction, lengthen the lever arm, or ensure the weight actually contacts the plank."
        ),
        severity="info" if passed else "warning",
    )


def launch_progress_probe(
    *,
    agent_name: str,
    target_name: str | None,
    summary: dict[str, Any],
    min_agent_lift: float = 45.0,
    min_target_progress: float = 20.0,
) -> ProbeResult:
    """Check whether the agent gained useful launch/target progress."""

    agent_lift = _float_or_none(summary.get("agent_lift_toward_target"))
    target_progress = _float_or_none(summary.get("agent_target_distance_reduced"))
    max_velocity = _float_or_none(summary.get("max_agent_velocity_toward_target"))
    passed = (
        agent_lift is not None
        and agent_lift >= min_agent_lift
    ) or (
        target_progress is not None
        and target_progress >= min_target_progress
    ) or (
        max_velocity is not None
        and max_velocity >= 80.0
    )
    objects = (agent_name,) if target_name is None else (agent_name, target_name)
    return ProbeResult(
        name="launch_progress",
        passed=passed,
        tier_evidence=4 if passed else 2,
        objects=objects,
        metrics={
            "agent_lift_toward_target": agent_lift,
            "agent_target_distance_reduced": target_progress,
            "max_agent_velocity_toward_target": max_velocity,
            "recommended_min_agent_lift": float(min_agent_lift),
            "recommended_min_target_progress": float(min_target_progress),
        },
        diagnosis="launch_progress_observed" if passed else "launch_progress_insufficient",
        repair=(
            "Agent launch/target progress was observed."
            if passed
            else "Move the agent onto the launch side, enlarge/align the high goal sensor, or increase lever energy through better weight placement and lever arm length."
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
