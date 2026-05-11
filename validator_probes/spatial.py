"""Spatial affordance and target-resolution probes."""

from __future__ import annotations

from typing import Any
import math

import pymunk

from .common import body_type_name, record_by_name, record_is_sensor, record_radius
from .contract import ProbeResult


def concrete_target_probe(
    env: Any,
    *,
    subgoal_kind: str,
    target_name: str | None,
) -> ProbeResult:
    """Ensure a subgoal references a concrete registered object/region."""

    if not target_name:
        return ProbeResult(
            name="concrete_target",
            passed=False,
            tier_evidence=0,
            diagnosis="missing_target_name",
            repair=f"{subgoal_kind} must name a concrete registered target/object.",
            severity="error",
        )
    record = record_by_name(env, str(target_name))
    passed = record is not None
    return ProbeResult(
        name="concrete_target",
        passed=passed,
        tier_evidence=1 if passed else 0,
        objects=(str(target_name),),
        metrics={"target_name": str(target_name), "registered": passed},
        diagnosis="target_registered" if passed else "target_not_registered",
        repair=(
            "Target is registered."
            if passed
            else f"Register an object/region named {target_name!r}; avoid aliases like any_bumper."
        ),
        severity="info" if passed else "error",
    )


def object_region_affordance_probe(
    env: Any,
    *,
    agent_name: str,
    object_name: str,
    region_name: str,
    agent_radius: float = 12.0,
    max_object_to_region: float = 140.0,
    max_agent_to_object: float = 220.0,
    max_alignment_degrees: float = 35.0,
) -> ProbeResult:
    """Measure generic geometry for moving an object into a region."""

    agent = record_by_name(env, agent_name)
    object_record = record_by_name(env, object_name)
    region_record = record_by_name(env, region_name)
    missing = [
        name
        for name, record in (
            (agent_name, agent),
            (object_name, object_record),
            (region_name, region_record),
        )
        if record is None
    ]
    if missing:
        return ProbeResult(
            name="object_region_affordance",
            passed=False,
            tier_evidence=0,
            objects=tuple(name for name in (agent_name, object_name, region_name) if name),
            metrics={"missing": missing},
            diagnosis="missing_affordance_objects",
            repair="Create all named objects/regions before declaring this subgoal.",
            severity="error",
        )

    agent_to_object = object_record.body.position - agent.body.position
    object_to_region = region_record.body.position - object_record.body.position
    alignment = _angle_between_degrees(agent_to_object, object_to_region)
    object_distance = float(object_to_region.length)
    agent_distance = float(agent_to_object.length)
    object_mass = float(object_record.body.mass)
    agent_mass = max(float(agent.body.mass), 0.001)
    metrics = {
        "agent_position": [
            round(float(agent.body.position.x), 3),
            round(float(agent.body.position.y), 3),
        ],
        "object_position": [
            round(float(object_record.body.position.x), 3),
            round(float(object_record.body.position.y), 3),
        ],
        "region_position": [
            round(float(region_record.body.position.x), 3),
            round(float(region_record.body.position.y), 3),
        ],
        "agent_to_object_distance": round(agent_distance, 3),
        "object_to_region_distance": round(object_distance, 3),
        "alignment_angle_degrees": round(alignment, 3),
        "agent_mass": round(agent_mass, 3),
        "object_mass": round(object_mass, 3),
        "object_radius": round(record_radius(object_record, agent_radius), 3),
        "region_is_sensor": record_is_sensor(region_record),
        "object_body_type": body_type_name(object_record.body),
        "recommended_object_to_region_max": float(max_object_to_region),
        "recommended_agent_to_object_max": float(max_agent_to_object),
        "recommended_alignment_angle_max": float(max_alignment_degrees),
    }
    failures: list[str] = []
    if object_record.body.body_type != pymunk.Body.DYNAMIC:
        failures.append("object_not_dynamic")
    if not record_is_sensor(region_record):
        failures.append("region_not_sensor")
    if object_distance > max_object_to_region:
        failures.append("object_too_far_from_region")
    if agent_distance > max_agent_to_object:
        failures.append("agent_too_far_from_object")
    if alignment > max_alignment_degrees:
        failures.append("poor_push_alignment")
    if object_mass / agent_mass > 5.0:
        failures.append("object_too_heavy")
    passed = not failures
    return ProbeResult(
        name="object_region_affordance",
        passed=passed,
        tier_evidence=3 if passed else 2,
        objects=(agent_name, object_name, region_name),
        metrics={**metrics, "failures": failures},
        diagnosis="object_region_affordance_valid" if passed else ",".join(failures),
        repair=(
            "Object-region geometry is validator-friendly."
            if passed
            else "Move/stage the agent, object, and region into a short aligned lane, keep the region sensor=True, and use reasonable mass/friction."
        ),
        severity="info" if passed else "error",
    )


def _angle_between_degrees(first: pymunk.Vec2d, second: pymunk.Vec2d) -> float:
    if first.length <= 1e-6 or second.length <= 1e-6:
        return 0.0
    cosine = max(-1.0, min(1.0, float(first.normalized().dot(second.normalized()))))
    return math.degrees(math.acos(cosine))
