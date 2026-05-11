"""Push-interaction probes for contact dynamics and force transfer."""

from __future__ import annotations

from typing import Any

from .contract import ProbeResult


def collision_filter_probe(
    *,
    agent_name: str,
    object_name: str,
    filter_summary: dict[str, Any],
) -> ProbeResult:
    """Check whether shape filters and sensor flags permit physical contact."""

    blocks = _string_list(filter_summary.get("blocking_reasons"))
    passed = not blocks
    return ProbeResult(
        name="agent_object_collision_allowed",
        passed=passed,
        tier_evidence=3 if passed else 1,
        objects=(agent_name, object_name),
        metrics=filter_summary,
        diagnosis="collision_allowed" if passed else "collision_filter_blocks_contact",
        repair=(
            "Agent and object are allowed to collide."
            if passed
            else "Make agent/object shapes non-sensor and ensure ShapeFilter groups/categories/masks allow agent-box collision."
        ),
        severity="info" if passed else "error",
    )


def push_contact_probe(
    *,
    agent_name: str,
    object_name: str,
    summary: dict[str, Any],
) -> ProbeResult:
    """Check whether agent/object contact or near-contact actually occurred."""

    min_gap = _float_or_none(summary.get("min_surface_gap"))
    contact_count = int(summary.get("contact_sample_count") or 0)
    passed = contact_count > 0 or (min_gap is not None and min_gap <= 2.0)
    return ProbeResult(
        name="agent_object_contact",
        passed=passed,
        tier_evidence=3 if passed else 2,
        objects=(agent_name, object_name),
        metrics={
            "min_surface_gap": min_gap,
            "contact_sample_count": contact_count,
            "recommended_max_surface_gap": 2.0,
        },
        diagnosis="agent_contacted_object" if passed else "agent_never_contacted_object",
        repair=(
            "Agent reached contact with the push object."
            if passed
            else "Place the agent closer behind the object or make the push controller press through the contact point instead of stopping at staging distance."
        ),
        severity="info" if passed else "error",
    )


def push_force_probe(
    *,
    agent_name: str,
    object_name: str,
    summary: dict[str, Any],
) -> ProbeResult:
    """Check whether the controller commanded useful force along the push axis."""

    max_force = _float_or_none(summary.get("max_applied_force"))
    max_forward_force = _float_or_none(summary.get("max_applied_force_toward_region"))
    passed = max_forward_force is not None and max_forward_force > 1.0
    return ProbeResult(
        name="agent_force_applied",
        passed=passed,
        tier_evidence=3 if passed else 1,
        objects=(agent_name, object_name),
        metrics={
            "max_applied_force": max_force,
            "max_applied_force_toward_region": max_forward_force,
            "recommended_min_forward_force": 1.0,
        },
        diagnosis="forward_force_commanded" if passed else "no_forward_force_commanded",
        repair=(
            "Controller commanded forward force."
            if passed
            else "Increase agent_strength or adjust the push policy so it applies force through the object toward the target region."
        ),
        severity="info" if passed else "error",
    )


def agent_motion_under_force_probe(
    *,
    agent_name: str,
    summary: dict[str, Any],
) -> ProbeResult:
    """Check whether applied force actually moved the agent body."""

    displacement = _float_or_none(summary.get("agent_displacement"))
    max_velocity = _float_or_none(summary.get("max_agent_velocity_toward_region"))
    max_force = _float_or_none(summary.get("max_applied_force"))
    force_was_applied = max_force is not None and max_force > 1.0
    passed = (
        not force_was_applied
        or (displacement is not None and displacement > 1.0)
        or (max_velocity is not None and abs(max_velocity) > 1.0)
    )
    return ProbeResult(
        name="agent_moves_under_force",
        passed=passed,
        tier_evidence=3 if passed else 2,
        objects=(agent_name,),
        metrics={
            "agent_displacement": displacement,
            "max_agent_velocity_toward_region": max_velocity,
            "max_applied_force": max_force,
            "recommended_min_agent_displacement": 1.0,
        },
        diagnosis="agent_moved_under_force" if passed else "agent_force_did_not_move_agent",
        repair=(
            "Agent body moved under commanded force."
            if passed
            else "Reduce agent/floor friction, remove pinning geometry, or increase agent mobility so applied force moves the agent into contact."
        ),
        severity="info" if passed else "error",
    )


def push_impulse_probe(
    *,
    object_name: str,
    summary: dict[str, Any],
) -> ProbeResult:
    """Estimate whether contact/force transferred useful motion to the object."""

    max_velocity = _float_or_none(summary.get("max_object_velocity_toward_region"))
    max_speed_delta = _float_or_none(summary.get("max_object_velocity_delta"))
    distance_reduced = _float_or_none(summary.get("distance_reduced"))
    contact_count = int(summary.get("contact_sample_count") or 0)
    passed = (
        contact_count > 0
        and (
            (max_velocity is not None and max_velocity > 5.0)
            or (max_speed_delta is not None and max_speed_delta > 2.0)
            or (distance_reduced is not None and distance_reduced > 5.0)
        )
    )
    return ProbeResult(
        name="contact_impulse_observed",
        passed=passed,
        tier_evidence=4 if passed else 2,
        objects=(object_name,),
        metrics={
            "max_object_velocity_toward_region": max_velocity,
            "max_object_velocity_delta": max_speed_delta,
            "distance_reduced": distance_reduced,
            "contact_sample_count": contact_count,
            "recommended_min_velocity_toward_region": 5.0,
            "recommended_min_distance_reduced": 5.0,
        },
        diagnosis="impulse_transferred_to_object" if passed else "no_useful_impulse_transfer",
        repair=(
            "Contact transferred useful motion to the object."
            if passed
            else "If contact occurred, reduce object friction/mass, remove pinning geometry, or increase sustained agent force so object velocity changes."
        ),
        severity="info" if passed else "error",
    )


def box_motion_probe(
    *,
    object_name: str,
    region_name: str,
    summary: dict[str, Any],
) -> ProbeResult:
    """Check whether the pushed object moved at all, independent of target progress."""

    displacement = _float_or_none(summary.get("object_displacement"))
    distance_reduced = _float_or_none(summary.get("distance_reduced"))
    passed = (
        (distance_reduced is not None and distance_reduced > 1.0)
        or (displacement is not None and displacement > 1.0)
    )
    return ProbeResult(
        name="box_moved_at_all",
        passed=passed,
        tier_evidence=3 if passed else 2,
        objects=(object_name, region_name),
        metrics={
            "object_displacement": displacement,
            "distance_reduced": distance_reduced,
            "recommended_min_distance_reduced": 1.0,
            "recommended_min_displacement": 1.0,
        },
        diagnosis="object_moved" if passed else "object_stationary",
        repair=(
            "Object moved during rollout."
            if passed
            else "Check for missing contact, collision filtering, pinning geometry, excessive friction, or insufficient agent force."
        ),
        severity="info" if passed else "error",
    )


def box_blockage_probe(
    *,
    object_name: str,
    region_name: str,
    summary: dict[str, Any],
) -> ProbeResult:
    """Check whether the object appears blocked or pinned."""

    blockers = _string_list(summary.get("blockers"))
    overlap_count = int(summary.get("static_overlap_count") or 0)
    passed = not blockers and overlap_count == 0
    return ProbeResult(
        name="box_free_to_move",
        passed=passed,
        tier_evidence=3 if passed else 2,
        objects=(object_name, region_name),
        metrics={
            "blockers": blockers,
            "static_overlap_count": overlap_count,
        },
        diagnosis="object_not_blocked" if passed else "object_blocked_or_pinned",
        repair=(
            "No obvious blocker/pinning geometry was detected."
            if passed
            else "Remove, sensorize, or relocate named blockers/overlapping static geometry so the object can translate toward the region."
        ),
        severity="info" if passed else "error",
    )


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]
