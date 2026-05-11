"""Deterministic patch layer for small measurable physics/layout failures.

The auto-repair pass is intentionally conservative. It only changes local,
numeric affordance knobs when validator telemetry gives a concrete physical
reason. Broad semantic or code-contract errors remain LLM repair territory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from pathlib import Path
from typing import Any

from validator import ValidationResult


@dataclass(frozen=True)
class AutoRepairResult:
    """Outcome of a deterministic auto-repair attempt."""

    applied: bool
    source_path: str
    output_path: str | None = None
    reason: str = ""
    actions: tuple[str, ...] = ()
    failure_category: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "applied": self.applied,
            "source_path": self.source_path,
            "output_path": self.output_path,
            "reason": self.reason,
            "actions": list(self.actions),
            "failure_category": self.failure_category,
        }


def auto_repair_generated_env(
    source_path: str | Path,
    validation: ValidationResult,
    output_path: str | Path,
) -> AutoRepairResult:
    """Patch a generated env when the failure is local and numeric."""

    source = Path(source_path)
    output = Path(output_path)
    category = str(validation.details.get("failure_category") or "")
    objective_type = str(validation.details.get("objective_type") or "")
    if not _is_repairable(validation):
        return AutoRepairResult(
            applied=False,
            source_path=str(source),
            reason="Failure is not a safe deterministic auto-repair candidate.",
            failure_category=category,
        )

    try:
        code = source.read_text(encoding="utf-8")
    except OSError as exc:
        return AutoRepairResult(
            applied=False,
            source_path=str(source),
            reason=f"Could not read source: {exc}",
            failure_category=category,
        )

    actions: list[str] = []
    if _is_push_object_region_failure(validation):
        code, push_actions = _repair_push_object_region(code, validation)
        actions.extend(push_actions)
    if _is_agent_reach_region_failure(validation):
        code, reach_actions = _repair_agent_reach_region(code, validation)
        actions.extend(reach_actions)
    if _is_hazard_navigation_route_failure(validation):
        code, hazard_route_actions = _repair_hazard_navigation_route(code, validation)
        actions.extend(hazard_route_actions)
    if _is_field_force_failure(validation):
        code, field_actions = _repair_field_force_interaction(code, validation)
        actions.extend(field_actions)
    if _is_semantic_falling_hazard_failure(validation):
        code, semantic_actions = _repair_semantic_falling_hazards(code, validation)
        actions.extend(semantic_actions)
    if objective_type == "survival" or _has_subgoal(validation, "survive_duration"):
        code, survival_actions = _repair_survival_duration(code, validation)
        actions.extend(survival_actions)

    if not actions or code == source.read_text(encoding="utf-8"):
        return AutoRepairResult(
            applied=False,
            source_path=str(source),
            reason="No supported numeric/layout knob was found to patch.",
            failure_category=category,
        )

    header = (
        "# AUTO-REPAIRED by Harness Alpha deterministic layout repair.\n"
        "# This patch only adjusts measured numeric physics affordances.\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(header + code, encoding="utf-8")
    return AutoRepairResult(
        applied=True,
        source_path=str(source),
        output_path=str(output),
        reason="Applied deterministic physics/layout auto-repair.",
        actions=tuple(actions),
        failure_category=category,
    )


def _is_repairable(validation: ValidationResult) -> bool:
    category = str(validation.details.get("failure_category") or "")
    objective_type = str(validation.details.get("objective_type") or "")
    if objective_type == "survival" or _has_subgoal(validation, "survive_duration"):
        return True
    if category in {
        "subgoal_affordance_failure",
        "contact_control_failure",
        "subgoal_execution_failure",
        "semantic_dynamics_failure",
    }:
        return True
    reason = validation.reason.lower()
    return any(
        marker in reason
        for marker in (
            "object_not_passively_stable",
            "object_outside_region",
            "survive_duration",
            "survival",
            "duration",
            "generic_push_controller_failed",
            "field_force_interaction",
            "force_zone",
            "field_effect",
            "field force",
            "pressure_plate",
            "poor_push_alignment",
            "alignment",
            "too far from",
            "starts",
            "did not complete",
            "drifts",
            "passively stable",
            "passive stability",
            "semantic dynamics validation failed",
            "did not fall downward enough",
        )
    )


def _is_semantic_falling_hazard_failure(validation: ValidationResult) -> bool:
    if "did not fall downward enough" in validation.reason.lower():
        return True
    if str(validation.details.get("failure_category") or "") != "semantic_dynamics_failure":
        return False
    for probe in validation.details.get("semantic_failures") or []:
        if not isinstance(probe, dict):
            continue
        metrics = probe.get("metrics")
        if not isinstance(metrics, dict):
            continue
        requirement = metrics.get("requirement")
        if not isinstance(requirement, dict):
            continue
        motion = str(requirement.get("motion") or "").lower()
        direction = str(requirement.get("direction") or "").lower()
        if motion in {"falling", "dropping", "raining"} or direction == "down":
            return True
    return False


def _is_push_object_region_failure(validation: ValidationResult) -> bool:
    failed = validation.details.get("failed_subgoal")
    if isinstance(failed, dict):
        return str(failed.get("kind") or "") in {
            "move_object_to_region",
            "strike_object_to_region",
            "ballistic_object_to_region",
        }
    return (
        _has_subgoal(validation, "move_object_to_region")
        or _has_subgoal(validation, "strike_object_to_region")
        or _has_subgoal(validation, "ballistic_object_to_region")
        or any(
            marker in validation.reason.lower()
            for marker in (
                "object_not_passively_stable",
                "object_outside_region",
                "generic_push_controller_failed",
                "pressure_plate",
                "ballistic",
                "clearance",
            )
        )
    )


def _is_agent_reach_region_failure(validation: ValidationResult) -> bool:
    failed = validation.details.get("failed_subgoal")
    if isinstance(failed, dict) and str(failed.get("kind") or "") == "agent_reach_region":
        return True
    if isinstance(failed, dict):
        return False
    if _has_subgoal(validation, "agent_reach_region") and str(
        validation.details.get("failure_category") or ""
    ) in {"post_subgoal_navigation_failure", "post_mechanism_navigation_failure"}:
        return True
    reason = validation.reason.lower()
    return "agent_reach_region" in reason or "agent reach/touch subgoal" in reason


def _is_hazard_navigation_route_failure(validation: ValidationResult) -> bool:
    if str(validation.details.get("objective_type") or "") != "navigation_goal":
        return False
    category = str(validation.details.get("failure_category") or "")
    if category not in {
        "physics_instability",
        "post_subgoal_navigation_failure",
        "post_mechanism_navigation_failure",
        "subgoal_execution_failure",
    }:
        return False
    profile = validation.details.get("objective_profile")
    if not isinstance(profile, dict):
        profile = {}
    fields = []
    for key in ("failure_modes", "progress_metrics", "required_capabilities"):
        value = profile.get(key)
        if isinstance(value, list):
            fields.extend(str(item) for item in value)
    fields.extend([str(validation.reason or ""), str(profile.get("objective_description") or "")])
    text = " ".join(fields).lower()
    has_hazard_objective = any(token in text for token in ("hazard", "fire", "lava", "avoid", "hit_by"))
    has_falling_requirement = any(
        isinstance(item, dict)
        and str(item.get("motion") or "").lower() in {"falling", "dropping", "raining"}
        for item in validation.details.get("semantic_requirements") or []
    )
    return has_hazard_objective or has_falling_requirement


def _is_field_force_failure(validation: ValidationResult) -> bool:
    failed = validation.details.get("failed_subgoal")
    if isinstance(failed, dict) and str(failed.get("kind") or "") == "field_force_interaction":
        return True
    observed_text = validation.reason.lower()
    if "field_force_interaction" in observed_text or "field_effect" in observed_text:
        return True
    for event in validation.details.get("progress_events") or []:
        if not isinstance(event, dict):
            continue
        subgoal = event.get("subgoal")
        if isinstance(subgoal, dict) and str(subgoal.get("kind") or "") == "field_force_interaction":
            return True
    return _has_subgoal(validation, "field_force_interaction") and any(
        marker in observed_text
        for marker in (
            "force zone",
            "field",
            "barely moved",
            "did not improve target progress",
        )
    )


def _has_subgoal(validation: ValidationResult, kind: str) -> bool:
    for subgoal in _iter_subgoals(validation.details):
        if str(subgoal.get("kind") or "") == kind:
            return True
    return False


def _iter_subgoals(details: dict[str, Any]):
    for key in ("failed_subgoal", "subgoal"):
        value = details.get(key)
        if isinstance(value, dict):
            yield value
    objective_profile = details.get("objective_profile")
    if isinstance(objective_profile, dict):
        subgoals = objective_profile.get("subgoals")
        if isinstance(subgoals, list):
            for subgoal in subgoals:
                if isinstance(subgoal, dict):
                    yield subgoal
    route = details.get("validator_route")
    if isinstance(route, dict):
        subgoals = route.get("subgoals")
        if isinstance(subgoals, list):
            for subgoal in subgoals:
                if isinstance(subgoal, dict):
                    yield subgoal
    failures = details.get("affordance_failures")
    if isinstance(failures, list):
        for failure in failures:
            if isinstance(failure, dict) and isinstance(failure.get("subgoal"), dict):
                yield failure["subgoal"]


def _first_subgoal(details: dict[str, Any], kind: str) -> dict[str, Any] | None:
    for subgoal in _iter_subgoals(details):
        if str(subgoal.get("kind") or "") == kind:
            return subgoal
    return None


def _repair_push_object_region(
    code: str,
    validation: ValidationResult,
) -> tuple[str, list[str]]:
    actions: list[str] = []
    metrics = _push_metrics(validation.details)
    stability = _passive_stability_metrics(validation.details)
    failed = validation.details.get("failed_subgoal")
    if not isinstance(failed, dict):
        failed = (
            _first_subgoal(validation.details, "move_object_to_region")
            or _first_subgoal(validation.details, "strike_object_to_region")
            or _first_subgoal(validation.details, "ballistic_object_to_region")
            or {}
        )
    kind = str(failed.get("kind") or "")
    is_ballistic = kind == "ballistic_object_to_region"
    object_name = str(failed.get("object") or "")
    final_distance = _float_or_none(metrics.get("final_distance")) or _float_or_none(
        metrics.get("distance")
    )
    object_to_region = _float_or_none(metrics.get("object_to_region_distance"))
    target_distance = final_distance or object_to_region

    if _needs_vertical_stability_repair(stability):
        end_position = stability.get("end_position")
        if isinstance(end_position, list | tuple) and len(end_position) >= 2:
            settled_y = _float_or_none(end_position[1])
            if settled_y is not None:
                code, changed = _replace_layout_tuple_y(
                    code,
                    "blue_box_center",
                    settled_y,
                )
                if changed:
                    actions.append(f"lowered blue_box_center y to settled support height {settled_y:.1f}px")
                code, changed = _replace_named_tuple_y(
                    code,
                    "box_start",
                    settled_y,
                )
                if changed:
                    actions.append(f"lowered box_start y to settled support height {settled_y:.1f}px")

    if not is_ballistic:
        for blocker_name in _solid_blockers_between(validation.details):
            code, changed = _sensorize_named_static_helper(code, blocker_name)
            if changed:
                actions.append(f"sensorized blocker {blocker_name} between object and target")

    if stability:
        end_position = _point_or_none(stability.get("end_position"))
        displacement = _float_or_none(stability.get("object_displacement"))
        if end_position is not None and displacement is not None and displacement > 24.0:
            for key in _object_start_key_candidates(object_name):
                code, changed = _replace_layout_tuple_value(code, key, end_position)
                if changed:
                    actions.append(
                        f"restaged {key} at measured passive-stable settled position"
                    )
                    break

    code, geometry_actions = _repair_push_lane_geometry(code, validation, metrics)
    actions.extend(geometry_actions)
    if is_ballistic:
        code, ballistic_actions = _repair_ballistic_relation_geometry(code, validation, metrics)
        actions.extend(ballistic_actions)

    code, changed = _replace_numeric_assignment(
        code,
        "self.agent_strength",
        lambda value: min(max(value * 1.35, value + 800.0), 9000.0),
    )
    if changed:
        actions.append("increased self.agent_strength for measured push failure")

    code, changed = _replace_numeric_assignment(
        code,
        "self.box_mass",
        lambda value: max(0.8, min(value, 1.2)),
    )
    if changed:
        actions.append("reduced self.box_mass toward validator-friendly manipulation range")

    if target_distance is not None:
        desired_plate_width = min(240.0, max(120.0, (target_distance + 24.0) * 2.0))
    else:
        desired_plate_width = 160.0
    code, changed = _replace_tuple_assignment_min_width(
        code,
        "self.plate_size",
        desired_plate_width,
    )
    if changed:
        actions.append(f"expanded pressure plate sensor width to at least {desired_plate_width:.1f}px")

    desired_offset = 55.0
    if object_to_region is not None:
        desired_offset = min(70.0, max(40.0, object_to_region * 0.55))
    code, changed = _replace_named_offset(
        code,
        "box_start",
        "plate_center[0]",
        desired_offset,
    )
    if changed:
        actions.append(f"shortened box-to-plate start offset to {desired_offset:.1f}px")

    code, changed = _replace_keyword_numeric(
        code,
        keyword="friction",
        old_min=0.7,
        new_value=0.35,
        nearby="self.blue_box_name",
    )
    if changed:
        actions.append("reduced blue_box dynamic friction for controllable pushing")

    return code, actions


def _repair_push_lane_geometry(
    code: str,
    validation: ValidationResult,
    metrics: dict[str, Any],
) -> tuple[str, list[str]]:
    actions: list[str] = []
    failed = validation.details.get("failed_subgoal")
    if not isinstance(failed, dict):
        failed = (
            _first_subgoal(validation.details, "move_object_to_region")
            or _first_subgoal(validation.details, "strike_object_to_region")
            or _first_subgoal(validation.details, "ballistic_object_to_region")
        )
    if not isinstance(failed, dict):
        return code, actions

    kind = str(failed.get("kind") or "")
    is_strike = kind == "strike_object_to_region"
    is_ballistic = kind == "ballistic_object_to_region"
    object_name = str(failed.get("object") or "")
    region_name = str(failed.get("region") or "")
    subgoal_metrics = _metrics_for_subgoal(validation.details, failed)
    if subgoal_metrics:
        metrics = {**metrics, **subgoal_metrics}
    object_position = _point_or_none(metrics.get("object_position"))
    region_position = _point_or_none(metrics.get("region_position"))
    agent_position = _point_or_none(metrics.get("agent_position"))
    if object_position is None or region_position is None:
        return code, actions

    axis = _unit_vector(object_position, region_position)
    if axis is None:
        return code, actions

    object_to_region = _float_or_none(metrics.get("object_to_region_distance"))
    agent_to_object = _float_or_none(metrics.get("agent_to_object_distance"))
    alignment = _float_or_none(metrics.get("alignment_angle_degrees"))
    surface_gap = _float_or_none(metrics.get("min_surface_gap"))
    object_radius = _float_or_none(metrics.get("object_radius")) or 20.0
    desired_object_gap = 230.0 if is_ballistic else (150.0 if is_strike else 78.0)

    max_object_gap = 340.0 if is_ballistic else (260.0 if is_strike else 140.0)
    object_moved = False
    if object_to_region is not None and object_to_region > max_object_gap:
        desired_object = (
            region_position[0] - axis[0] * desired_object_gap,
            region_position[1] - axis[1] * desired_object_gap,
        )
        for key in _object_start_key_candidates(object_name):
            code, changed = _replace_layout_point_value(code, key, desired_object)
            if changed:
                actions.append(
                    f"moved {key} to {desired_object_gap:.0f}px before {region_name}"
                )
                object_position = desired_object
                object_to_region = desired_object_gap
                object_moved = True
                break
            code, changed = _replace_named_tuple_value(code, key, desired_object)
            if changed:
                actions.append(
                    f"moved local {key} to {desired_object_gap:.0f}px before {region_name}"
                )
                object_position = desired_object
                object_to_region = desired_object_gap
                object_moved = True
                break
        if not object_moved and abs(axis[1]) < 0.25:
            for key in _object_x_key_candidates(object_name):
                code, changed = _replace_layout_numeric_key_exact(code, key, desired_object[0])
                if changed:
                    actions.append(
                        f"moved {key} to place {object_name} {desired_object_gap:.0f}px before {region_name}"
                    )
                    object_position = desired_object
                    object_to_region = desired_object_gap
                    object_moved = True
                    break

    should_restage_agent = (
        object_moved
        or
        alignment is not None
        and alignment > (75.0 if is_ballistic else (25.0 if is_strike else 35.0))
        or agent_to_object is not None
        and agent_to_object > (190.0 if is_ballistic else (170.0 if is_strike else 220.0))
        or surface_gap is not None
        and surface_gap > 12.0
        or "agent_never_contacted_object" in str(metrics.get("failure_modes") or "")
        or "agent never physically contacted" in validation.reason.lower()
    )
    if should_restage_agent:
        desired_agent_gap = object_radius + 15.0 + (3.0 if is_ballistic else (2.0 if is_strike else 4.0))
        desired_agent = (
            object_position[0] - axis[0] * desired_agent_gap,
            object_position[1] - axis[1] * desired_agent_gap,
        )
        code, changed = _replace_layout_point_value(code, "agent_start", desired_agent)
        if changed:
            actions.append("restaged agent directly behind movable object on push axis")
        else:
            code, changed = _replace_named_tuple_value(code, "agent_start", desired_agent)
            if changed:
                actions.append("restaged local agent_start behind movable object on push axis")
        if not changed:
            code, changed = _replace_layout_tuple_y(code, "agent_start", desired_agent[1])
            if changed:
                actions.append("restaged agent y-position behind movable object on push axis")
        if not changed and abs(axis[1]) < 0.25:
            for key in ("agent_x", "agent_start_x"):
                code, changed = _replace_layout_numeric_key_exact(code, key, desired_agent[0])
                if changed:
                    actions.append("restaged agent x-position behind shot/push object")
                    break

    static_overlaps = metrics.get("static_overlaps")
    if isinstance(static_overlaps, list) and any("gate" in str(item).lower() for item in static_overlaps):
        desired_gate = (
            region_position[0] + axis[0] * 115.0,
            region_position[1] + axis[1] * 115.0,
        )
        code, changed = _replace_layout_point_value(code, "gate_center", desired_gate)
        if changed:
            actions.append("moved overlapping gate beyond the push target so it no longer pins the object")
        else:
            code, changed = _replace_layout_tuple_y(code, "gate_center", desired_gate[1])
            if changed:
                actions.append("moved overlapping gate y-position beyond the push target")

    return code, actions


def _repair_ballistic_relation_geometry(
    code: str,
    validation: ValidationResult,
    metrics: dict[str, Any],
) -> tuple[str, list[str]]:
    """Tune reusable over-barrier projectile knobs without deleting the barrier."""

    actions: list[str] = []
    failure_modes = " ".join(str(item) for item in metrics.get("failure_modes") or [])
    reason = validation.reason.lower()
    needs_arc_help = any(
        marker in f"{failure_modes} {reason}"
        for marker in (
            "ballistic",
            "apex",
            "clearance",
            "drifted_sideways",
            "blocked_or_pinned",
            "did not complete",
            "objective not satisfied",
        )
    )
    if not needs_arc_help:
        return code, actions

    code, changed = _replace_layout_tuple_max_dimensions(code, "barrier_size", 34.0, 96.0)
    if changed:
        actions.append("lowered/capped ballistic barrier_size so a visible arc is validator-solvable")
    code, changed = _replace_named_tuple_max_dimensions(code, "barrier_size", 34.0, 96.0)
    if changed:
        actions.append("lowered/capped local ballistic barrier_size so a visible arc is validator-solvable")
    code, changed = _replace_numeric_assignment(
        code,
        "barrier_height",
        lambda value: min(value, 96.0),
    )
    if changed:
        actions.append("lowered local barrier_height for validator-solvable over-wall arc")

    code, changed = _replace_layout_tuple_min_dimensions(code, "target_size", 190.0, 170.0)
    if changed:
        actions.append("expanded ballistic target sensor to catch valid over-wall arcs")
    code, changed = _replace_named_tuple_min_dimensions(code, "target_size", 190.0, 170.0)
    if changed:
        actions.append("expanded local ballistic target_size to catch valid over-wall arcs")

    code, changed = _replace_layout_numeric_key_exact(code, "object_friction", 0.035)
    if changed:
        actions.append("reduced ballistic object_friction for crisp impulse transfer")
    code, changed = _replace_layout_numeric_key_exact(code, "object_elasticity", 0.24)
    if changed:
        actions.append("increased ballistic object_elasticity for a readable lob")
    code, changed = _replace_numeric_assignment(
        code,
        "self.agent_strength",
        lambda value: min(max(value * 1.4, 6200.0), 9500.0),
    )
    if changed:
        actions.append("increased agent_strength for ballistic kick impulse")
    return code, actions


def _needs_vertical_stability_repair(metrics: dict[str, Any]) -> bool:
    diagnosis = str(metrics.get("diagnosis") or "").lower()
    if diagnosis and diagnosis != "object_not_passively_stable":
        return False
    vertical = abs(_float_or_none(metrics.get("vertical_displacement")) or 0.0)
    horizontal = abs(_float_or_none(metrics.get("horizontal_displacement")) or 0.0)
    settled = bool(metrics.get("settled_on_support"))
    return not settled and vertical > 24.0 and horizontal < max(8.0, vertical * 0.5)


def _repair_agent_reach_region(
    code: str,
    validation: ValidationResult,
) -> tuple[str, list[str]]:
    actions: list[str] = []
    metrics = _agent_reach_metrics(validation.details)
    for key in ("initial_blocking_object", "final_blocking_object"):
        blocker = metrics.get(key)
        if isinstance(blocker, str) and blocker:
            code, changed = _sensorize_named_static_helper(code, blocker)
            if changed:
                actions.append(f"sensorized post-subgoal path blocker {blocker}")
    final_distance = _float_or_none(metrics.get("final_distance"))
    threshold = _float_or_none(metrics.get("threshold"))
    if final_distance is None or threshold is None:
        return code, actions
    if final_distance > 220.0:
        return code, actions

    desired_goal_width = min(260.0, max(140.0, (final_distance + 16.0) * 2.0))
    code, changed = _replace_tuple_assignment_min_width(
        code,
        "self.goal_size",
        desired_goal_width,
    )
    if changed:
        actions.append(f"expanded final goal sensor width to at least {desired_goal_width:.1f}px")
    code, changed = _replace_layout_tuple_min_dimensions(
        code,
        "goal_size",
        desired_goal_width,
        desired_goal_width,
    )
    if changed:
        actions.append(f"expanded layout goal_size to at least {desired_goal_width:.1f}px")
    return code, actions


def _repair_hazard_navigation_route(
    code: str,
    validation: ValidationResult,
) -> tuple[str, list[str]]:
    """Repair coupled 'avoid hazards, then reach exit' layout failures."""

    actions: list[str] = []
    metrics = _agent_reach_metrics(validation.details)
    for key in ("initial_blocking_object", "final_blocking_object"):
        blocker = metrics.get(key)
        if isinstance(blocker, str) and blocker:
            code, changed = _sensorize_named_static_helper(code, blocker)
            if changed:
                actions.append(f"sensorized hazard-navigation route blocker {blocker}")
    for route_blocker in ("inner_left", "inner_right"):
        code, changed = _sensorize_named_static_helper(code, route_blocker)
        if changed:
            actions.append(f"sensorized decorative route blocker {route_blocker}")
    for route_blocker in (
        "ledge_mid",
        "platform_mid",
        "middle_ledge",
        "main_ramp",
        "ramp_0",
        "ramp_1",
        "ramp_2",
        "column_0",
        "column_1",
        "column_2",
    ):
        code, changed = _sensorize_named_static_helper(code, route_blocker)
        if changed:
            actions.append(f"sensorized measured/decorative hazard route blocker {route_blocker}")

    for key in (
        "column_centers_sizes",
        "columns",
        "route_columns",
        "decorative_columns",
        "obstacle_centers_sizes",
        "route_obstacles",
        "floating_obstacles",
        "solid_obstacles",
    ):
        code, changed = _replace_layout_sequence_empty(code, key)
        if changed:
            actions.append(f"removed layout {key} from hazard-navigation route")

    final_distance = _float_or_none(metrics.get("final_distance"))
    if final_distance is not None and final_distance > 300.0:
        # Low-gravity hazard navigation often fails because the LLM starts the
        # agent inside a decorative side wall or leaves the exit embedded in a
        # rim. Restage to a visible open route while preserving the top-right
        # exit intent.
        code, changed = _replace_layout_tuple_value(code, "agent_start", (480.0, 90.0))
        if changed:
            actions.append("moved agent_start into the open hazard-navigation lane")
        code, changed = _replace_layout_tuple_value(code, "goal_center", (820.0, 560.0))
        if changed:
            actions.append("moved goal_center onto a reachable upper exit lane")
        code, changed = _replace_layout_tuple_min_dimensions(code, "goal_size", 150.0, 150.0)
        if changed:
            actions.append("expanded goal_size for final reach tolerance")
        code, changed = _replace_numeric_assignment(
            code,
            "self.agent_strength",
            lambda value: min(max(value * 1.45, 4800.0), 9000.0),
        )
        if changed:
            actions.append("increased agent thrust for low-gravity hazard navigation")

    # Keep the fireballs visually meaningful but out of the most direct
    # center-to-exit route, so the validator can prove reachability without
    # deleting the requested falling-hazard behavior.
    side_lane_xs = (160.0, 240.0, 320.0, 400.0)
    for key in ("hazard_spawn_xs", "fireball_spawn_xs", "falling_spawn_xs"):
        code, changed = _replace_layout_numeric_list_exact(code, key, side_lane_xs)
        if changed:
            actions.append(f"moved {key} to avoidable side fall lanes")
    code, changed = _replace_layout_point_list_visible_fall_lanes(
        code,
        "fireball_starts",
        side_lane_xs,
    )
    if changed:
        actions.append("moved fireball_starts to avoidable visible side fall lanes")

    code, changed = _replace_layout_numeric_keys(
        code,
        ("hit_slack", "hit_extra_margin"),
        lambda value: max(value, 8.0),
    )
    if changed:
        actions.append("made hazard hit threshold less brittle while preserving avoidance")
    return code, actions


def _repair_survival_duration(
    code: str,
    validation: ValidationResult,
) -> tuple[str, list[str]]:
    actions: list[str] = []

    # Preserve the semantic duration requested by the user. Older harness builds
    # shortened survival objectives to fit a fixed 600-step rollout; the
    # validator now expands its horizon for duration tasks, so deterministic
    # repair should adjust fairness/readability knobs instead of changing the
    # win condition.
    code, changed = _replace_assignment_leading_number(
        code,
        "self.agent_strength",
        lambda value: min(max(value * 1.25, 3600.0), 9000.0),
    )
    if changed:
        actions.append("increased agent thrust for survival dodge control")

    code, changed = _replace_assignment_leading_number(
        code,
        "self.agent_max_speed",
        lambda value: min(max(value * 1.12, 440.0), 720.0),
    )
    if changed:
        actions.append("raised agent max speed for survival evasion")

    code, changed = _replace_assignment_leading_number(
        code,
        "self.projectile_speed",
        lambda value: max(value * 0.65, 0.35),
    )
    if changed:
        actions.append("reduced projectile speed for dodgeable survival")

    code, changed = _replace_assignment_leading_number(
        code,
        "self.enemy_max_speed",
        lambda value: max(value * 0.75, 0.35),
    )
    if changed:
        actions.append("reduced enemy pursuit speed for survival fairness")

    code, changed = _replace_assignment_leading_number(
        code,
        "self.max_simultaneous_projectiles",
        lambda value: max(4.0, min(value, 8.0)),
    )
    if changed:
        actions.append("capped simultaneous projectile density")

    code, changed = _replace_int_round_seconds_assignment(
        code,
        "self.warmup_steps",
        lambda value: max(value, 3.5),
    )
    if changed:
        actions.append("increased initial survival warmup")

    code, changed = _replace_int_round_seconds_assignment(
        code,
        "self.telegraph_steps",
        lambda value: max(value, 0.45),
    )
    if changed:
        actions.append("increased projectile telegraph window")

    code, changed = _replace_int_round_seconds_assignment(
        code,
        "self.projectile_lifetime_steps",
        lambda value: min(value, 2.4),
    )
    if changed:
        actions.append("shortened projectile lifetime to reduce arena saturation")

    code, changed = _replace_layout_numeric_keys(
        code,
        ("projectile_radius", "shot_radius", "laser_radius", "bolt_radius"),
        lambda value: max(3.0, min(value * 0.72, 6.0)),
    )
    if changed:
        actions.append("reduced projectile hit radius while preserving visible hazards")

    code, changed = _replace_layout_numeric_keys(
        code,
        ("max_simultaneous_projectiles", "active_projectiles", "projectile_cap"),
        lambda value: max(4.0, min(value, 8.0)),
    )
    if changed:
        actions.append("reduced declared projectile density cap")

    code, changed = _replace_layout_numeric_keys(
        code,
        ("warmup_seconds", "warmup_s"),
        lambda value: max(value, 3.5),
    )
    if changed:
        actions.append("raised declared projectile warmup seconds")

    code, changed = _replace_layout_numeric_keys(
        code,
        ("telegraph_seconds", "pre_fire_cue_seconds"),
        lambda value: max(value, 0.45),
    )
    if changed:
        actions.append("raised declared projectile telegraph seconds")

    return code, actions


def _repair_field_force_interaction(
    code: str,
    validation: ValidationResult,
) -> tuple[str, list[str]]:
    """Patch local field-force affordance knobs from measured telemetry."""

    actions: list[str] = []
    metrics = _field_metrics(validation.details)
    failed = validation.details.get("failed_subgoal")
    object_name = str(failed.get("object") or "") if isinstance(failed, dict) else ""
    field_name = str(failed.get("field") or "") if isinstance(failed, dict) else ""
    region_name = str(failed.get("region") or failed.get("target") or "") if isinstance(failed, dict) else ""

    # If the force effect was present but insufficient, prefer making the field
    # bigger/stronger before asking the LLM to redesign the task.
    code, changed = _replace_layout_numeric_keys(
        code,
        ("force_strength", "field_strength", "mag_force_strength", "magnetic_strength"),
        lambda value: min(max(value * 1.75, value + 1200.0), 12000.0),
    )
    if changed:
        actions.append("increased registered force-zone strength for measured field under-effect")

    code, changed = _replace_layout_numeric_keys(
        code,
        ("force_falloff", "field_falloff", "mag_force_falloff", "magnetic_falloff"),
        lambda value: max(0.0, min(value, 0.03)),
    )
    if changed:
        actions.append("reduced force-zone falloff so the effect persists across the zone")

    field_distance = _float_or_none(metrics.get("object_to_field_distance")) or _float_or_none(
        metrics.get("start_distance")
    )
    min_field_width = 220.0
    min_field_height = 160.0
    if field_distance is not None:
        min_field_width = min(360.0, max(min_field_width, field_distance * 2.2))
        min_field_height = min(280.0, max(min_field_height, field_distance * 1.4))
    for key in _field_size_key_candidates(field_name, region_name):
        code, changed = _replace_layout_tuple_min_dimensions(
            code,
            key,
            min_field_width,
            min_field_height,
        )
        if changed:
            actions.append(
                f"expanded {key} to keep the affected object inside the force zone longer"
            )
        code, changed = _replace_named_tuple_min_dimensions(
            code,
            f"self.{key}",
            min_field_width,
            min_field_height,
        )
        if changed:
            actions.append(
                f"expanded self.{key} to keep the affected object inside the force zone longer"
            )

    # Many generated field worlds use a sensor region plus a force-zone with a
    # different name. Keep their sizes aligned so the visible affordance matches
    # the actual physics affordance.
    for key in _region_size_key_candidates(region_name):
        code, changed = _replace_layout_tuple_min_dimensions(
            code,
            key,
            min_field_width,
            min_field_height,
        )
        if changed:
            actions.append(f"expanded {key} to match the repaired force-zone footprint")
        code, changed = _replace_named_tuple_min_dimensions(
            code,
            f"self.{key}",
            min_field_width,
            min_field_height,
        )
        if changed:
            actions.append(f"expanded self.{key} to match the repaired force-zone footprint")

    code, changed = _replace_layout_numeric_keys(
        code,
        _object_mass_key_candidates(object_name),
        lambda value: max(0.75, min(value, 1.1)),
    )
    if changed:
        actions.append("reduced affected object mass for stronger deterministic field response")

    code, changed = _replace_layout_numeric_keys(
        code,
        _object_friction_key_candidates(object_name),
        lambda value: max(0.05, min(value, 0.25)),
    )
    if changed:
        actions.append("reduced affected object friction so field motion is not damped out")

    # If the object starts too far from the field and both positions are literal
    # layout tuples, move the object just before the zone along the measured axis.
    field_position = _point_or_none(metrics.get("field_position"))
    object_position = _point_or_none(metrics.get("object_position"))
    if field_distance is not None and field_distance > 220.0 and field_position and object_position:
        desired = _point_toward(object_position, field_position, max_distance=120.0)
        for key in _object_start_key_candidates(object_name):
            code, changed = _replace_layout_tuple_value(code, key, desired)
            if changed:
                actions.append(
                    f"moved {key} closer to force zone from {field_distance:.1f}px separation"
                )
                break

    # When a target is barely missed, the subgoal threshold can stay semantic
    # while the sensor footprint gets a little more forgiving.
    final_distance = _float_or_none(metrics.get("final_distance"))
    if final_distance is not None and final_distance < 110.0:
        for key in _region_size_key_candidates(region_name):
            code, changed = _replace_layout_tuple_min_dimensions(
                code,
                key,
                final_distance * 2.4,
                final_distance * 1.8,
            )
            if changed:
                actions.append(f"expanded {key} for near-miss field completion")
            code, changed = _replace_named_tuple_min_dimensions(
                code,
                f"self.{key}",
                final_distance * 2.4,
                final_distance * 1.8,
            )
            if changed:
                actions.append(f"expanded self.{key} for near-miss field completion")

    return code, actions


def _repair_semantic_falling_hazards(
    code: str,
    validation: ValidationResult,
) -> tuple[str, list[str]]:
    """Patch local numeric knobs when falling hazards are blocked or too slow."""

    actions: list[str] = []
    metrics = _first_semantic_failure_metrics(validation)
    requirement = metrics.get("requirement") if isinstance(metrics, dict) else {}
    if not isinstance(requirement, dict):
        requirement = {}
    per_object = metrics.get("per_object") if isinstance(metrics, dict) else []
    if not isinstance(per_object, list):
        per_object = []

    max_start_y = max(
        (
            _float_or_none((item.get("start") or [None, None])[1])
            for item in per_object
            if isinstance(item, dict) and isinstance(item.get("start"), list | tuple)
        ),
        default=None,
    )
    max_end_y = max(
        (
            _float_or_none((item.get("end") or [None, None])[1])
            for item in per_object
            if isinstance(item, dict) and isinstance(item.get("end"), list | tuple)
        ),
        default=None,
    )
    effective_requirement = metrics.get("effective_requirement") if isinstance(metrics, dict) else None
    if not isinstance(effective_requirement, dict):
        effective_requirement = requirement
    required_drop = (
        _float_or_none(effective_requirement.get("min_displacement_y"))
        or _float_or_none(requirement.get("min_displacement_y"))
        or 160.0
    )

    desired_spawn_y = None
    if max_start_y is not None:
        desired_spawn_y = min(max_start_y, 590.0)
        if max_end_y is not None and max_start_y - max_end_y < required_drop:
            desired_spawn_y = min(
                desired_spawn_y,
                max(180.0, max_end_y + required_drop + 70.0),
            )

    if desired_spawn_y is not None:
        for key in (
            "fireball_spawn_y",
            "hazard_spawn_y",
            "falling_spawn_y",
            "meteor_spawn_y",
            "rock_spawn_y",
        ):
            code, changed = _replace_layout_numeric_key_exact(code, key, desired_spawn_y)
            if changed:
                actions.append(f"moved {key} inside playable area to y={desired_spawn_y:.1f}px")

    code, changed = _replace_falling_hazard_pos_y_expressions(code)
    if changed:
        actions.append("moved inline falling hazard pos y expressions from off-screen to visible sky lane")

    code, changed = _replace_spawn_y_plus_offsets(code)
    if changed:
        actions.append("removed offscreen spawn_y offsets so pooled hazards begin in visible drop lanes")

    code, changed = _replace_layout_numeric_key_exact(code, "start_safe_x_max", 0.0)
    if changed:
        actions.append("relaxed start_safe_x_max so semantic hazard probe can trigger staggered drops")

    desired_xs = _desired_falling_hazard_xs(code, per_object)
    if desired_xs:
        for key in (
            "fireball_spawn_xs",
            "hazard_spawn_xs",
            "falling_spawn_xs",
            "meteor_spawn_xs",
            "rock_spawn_xs",
        ):
            code, changed = _replace_layout_numeric_list_exact(code, key, desired_xs)
            if changed:
                actions.append(
                    f"moved {key} into open fall lanes: {tuple(round(x, 1) for x in desired_xs)}"
                )

        code, changed = _replace_layout_point_list_visible_fall_lanes(
            code,
            "fireball_starts",
            desired_xs,
        )
        if changed:
            actions.append("moved fireball_starts into visible open fall lanes")

    code, changed = _replace_layout_numeric_keys(
        code,
        ("fall_strength", "hazard_fall_strength", "downward_force", "fall_force"),
        lambda old: max(old * 2.0, 9000.0),
    )
    if changed:
        actions.append("increased downward hazard fall force for semantic falling motion")

    code, changed = _replace_semantic_numeric_value(
        code,
        "min_displacement_y",
        max(required_drop, 160.0),
    )
    if changed:
        actions.append("raised semantic min_displacement_y so falling hazards must visibly descend")

    return code, actions


def _desired_falling_hazard_xs(code: str, per_object: list[Any]) -> tuple[float, ...]:
    count = max(1, len([item for item in per_object if isinstance(item, dict)]))
    corridor_left = _layout_number(code, "corridor_x_left")
    corridor_right = _layout_number(code, "corridor_x_right")
    wall_thickness = _layout_number(code, "corridor_wall_thickness") or 0.0
    if corridor_left is not None and corridor_right is not None and corridor_right - corridor_left > 120:
        left = corridor_left + wall_thickness * 0.5 + 50.0
        right = corridor_right - wall_thickness * 0.5 - 50.0
    else:
        left, right = 180.0, 780.0
    if count == 1:
        return ((left + right) * 0.5,)

    # Default to side-lane skyfall positions instead of the exact center.
    # Generated worlds often place a large central obstacle/platform; center
    # hazards can satisfy a tiny motion check while visually landing almost
    # immediately. Side lanes make "falling from the sky" visible and robust.
    if corridor_left is None or corridor_right is None:
        # Prefer laterally open lanes first. Many generated "lava/falling"
        # worlds place a decorative top/platform block across the upper-left or
        # center; right-side lanes are less likely to be immediately supported.
        candidates = [650.0, 730.0, 220.0, 340.0, 560.0, 460.0, 160.0]
        while len(candidates) < count:
            candidates.append(480.0 + 70.0 * (len(candidates) - count / 2.0))
        return tuple(candidates[:count])

    step = (right - left) / max(1, count - 1)
    return tuple(left + step * index for index in range(count))


def _replace_falling_hazard_pos_y_expressions(code: str) -> tuple[str, bool]:
    """Patch common LLM pattern: {"pos": (x, h + N)} for falling hazards.

    Off-screen spawns can move in physics yet remain invisible to the user.
    Keeping them just below the top boundary makes skyfall visible while still
    preserving the generated x-layout and hazard count.
    """

    pattern = re.compile(
        r"([\"']pos[\"']\s*:\s*\(\s*[^,\n]+,\s*)"
        r"h\s*\+\s*[-+]?[0-9]+(?:\.[0-9]+)?"
        r"(\s*\))"
    )
    patched, count = pattern.subn(lambda match: f"{match.group(1)}h - 70.0{match.group(2)}", code)
    return patched, count > 0


def _replace_spawn_y_plus_offsets(code: str) -> tuple[str, bool]:
    """Patch pooled hazards parked at spawn_y + huge offsets above ceilings."""

    pattern = re.compile(r"spawn_y\s*\+\s*[0-9]+(?:\.[0-9]+)?")
    patched, count = pattern.subn("spawn_y", code)
    return patched, count > 0


def _replace_layout_point_list_visible_fall_lanes(
    code: str,
    key: str,
    xs: tuple[float, ...],
) -> tuple[str, bool]:
    pattern = re.compile(
        rf"([\"']{re.escape(key)}[\"']\s*:\s*\[\s*)"
        r"(?P<body>.*?)"
        r"(\s*\]\s*,)",
        re.DOTALL,
    )
    match = pattern.search(code)
    if not match:
        return code, False
    count = max(1, _tuple_count(match.group("body")) or len(xs))
    chosen_xs = list(xs)
    while len(chosen_xs) < count:
        chosen_xs.append(chosen_xs[-1] + 55.0 if chosen_xs else 650.0)
    entries = ",\n".join(
        f"                ({_format_number(x)}, h - 70.0)"
        for x in chosen_xs[:count]
    )
    replacement = f"{match.group(1)}\n{entries},\n            {match.group(3)}"
    patched = code[: match.start()] + replacement + code[match.end() :]
    return patched, patched != code


def _tuple_count(text: str) -> int:
    return len(re.findall(r"\([^\)]*,[^\)]*\)", text))


def _layout_number(code: str, key: str) -> float | None:
    pattern = re.compile(
        rf"[\"']{re.escape(key)}[\"']\s*:\s*(?P<value>[-+]?[0-9]+(?:\.[0-9]+)?)"
    )
    match = pattern.search(code)
    if not match:
        return None
    return _float_or_none(match.group("value"))


def _first_semantic_failure_metrics(validation: ValidationResult) -> dict[str, Any]:
    failures = validation.details.get("semantic_failures")
    if not isinstance(failures, list):
        return {}
    for failure in failures:
        if isinstance(failure, dict) and isinstance(failure.get("metrics"), dict):
            return failure["metrics"]
    return {}


def _push_metrics(details: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    diagnostics = details.get("subgoal_diagnostics")
    if isinstance(diagnostics, dict):
        failure_modes = diagnostics.get("failure_modes")
        if isinstance(failure_modes, list):
            merged["failure_modes"] = list(failure_modes)
        for key in ("summary", "initial", "final"):
            value = diagnostics.get(key)
            if isinstance(value, dict):
                merged.update(value)
    for event in details.get("progress_events") or []:
        if not isinstance(event, dict):
            continue
        diagnostics = event.get("diagnostics")
        if isinstance(diagnostics, dict):
            failure_modes = diagnostics.get("failure_modes")
            if isinstance(failure_modes, list):
                merged["failure_modes"] = list(failure_modes)
            summary = diagnostics.get("summary")
            if isinstance(summary, dict):
                merged.update(summary)
            initial = diagnostics.get("initial")
            if isinstance(initial, dict):
                for key in ("agent_position", "object_position", "region_position"):
                    merged.setdefault(key, initial.get(key))
            final = diagnostics.get("final")
            if isinstance(final, dict):
                for key, value in final.items():
                    if key in {"agent_position", "object_position", "region_position"}:
                        continue
                    merged[key] = value
    for failure in details.get("affordance_failures") or []:
        if not isinstance(failure, dict):
            continue
        if isinstance(failure.get("subgoal"), dict) and str(failure["subgoal"].get("kind") or "") in {
            "move_object_to_region",
            "strike_object_to_region",
            "ballistic_object_to_region",
        }:
            metrics = failure.get("metrics")
            if isinstance(metrics, dict):
                merged.update(metrics)
                probe_result = metrics.get("probe_result")
                if isinstance(probe_result, dict):
                    probe_metrics = probe_result.get("metrics")
                    if isinstance(probe_metrics, dict):
                        merged.update(probe_metrics)
    for probe in _collect_probe_dicts(details):
        if str(probe.get("name")) == "object_inside_region":
            metrics = probe.get("metrics")
            if isinstance(metrics, dict):
                merged.update(metrics)
        if str(probe.get("name")) == "object_region_affordance":
            metrics = probe.get("metrics")
            if isinstance(metrics, dict):
                merged.update(metrics)
    return merged


def _metrics_for_subgoal(
    details: dict[str, Any],
    subgoal: dict[str, Any],
) -> dict[str, Any]:
    wanted_kind = str(subgoal.get("kind") or "")
    wanted_object = str(subgoal.get("object") or "")
    wanted_region = str(subgoal.get("region") or subgoal.get("target") or "")
    diagnostics = details.get("subgoal_diagnostics")
    if isinstance(diagnostics, dict) and str(diagnostics.get("kind") or "") == wanted_kind:
        result: dict[str, Any] = {}
        for key in ("summary", "initial", "final"):
            value = diagnostics.get(key)
            if isinstance(value, dict):
                result.update(value)
        if result:
            return result
    for failure in details.get("affordance_failures") or []:
        if not isinstance(failure, dict):
            continue
        failure_subgoal = failure.get("subgoal")
        if not isinstance(failure_subgoal, dict):
            continue
        if str(failure_subgoal.get("kind") or "") != wanted_kind:
            continue
        if wanted_object and str(failure_subgoal.get("object") or "") != wanted_object:
            continue
        failure_region = str(failure_subgoal.get("region") or failure_subgoal.get("target") or "")
        if wanted_region and failure_region != wanted_region:
            continue
        metrics = failure.get("metrics")
        if not isinstance(metrics, dict):
            continue
        result = dict(metrics)
        probe_result = metrics.get("probe_result")
        if isinstance(probe_result, dict):
            probe_metrics = probe_result.get("metrics")
            if isinstance(probe_metrics, dict):
                result.update(probe_metrics)
        return result
    return {}


def _field_metrics(details: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    diagnostics = details.get("subgoal_diagnostics")
    if isinstance(diagnostics, dict) and diagnostics.get("kind") == "field_force_interaction":
        for key in ("summary", "initial", "final"):
            value = diagnostics.get(key)
            if isinstance(value, dict):
                merged.update(value)
    for event in details.get("progress_events") or []:
        if not isinstance(event, dict):
            continue
        subgoal = event.get("subgoal")
        diagnostics = event.get("diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        if isinstance(subgoal, dict) and str(subgoal.get("kind") or "") != "field_force_interaction":
            continue
        for key in ("summary", "initial", "final"):
            value = diagnostics.get(key)
            if isinstance(value, dict):
                merged.update(value)
    for failure in details.get("affordance_failures") or []:
        if not isinstance(failure, dict):
            continue
        if str(failure.get("code") or "").startswith(("object_too_far_from_force", "force_zone")):
            metrics = failure.get("metrics")
            if isinstance(metrics, dict):
                merged.update(metrics)
    for probe in _collect_probe_dicts(details):
        if str(probe.get("name")) == "field_effect":
            metrics = probe.get("metrics")
            if isinstance(metrics, dict):
                merged.update(metrics)
    return merged


def _passive_stability_metrics(details: dict[str, Any]) -> dict[str, Any]:
    for failure in details.get("affordance_failures") or []:
        if not isinstance(failure, dict):
            continue
        if failure.get("code") == "object_not_passively_stable":
            metrics = failure.get("metrics")
            if isinstance(metrics, dict):
                return dict(metrics)
    for probe in _collect_probe_dicts(details):
        if str(probe.get("name")) == "passive_stability" and not bool(probe.get("passed")):
            metrics = probe.get("metrics")
            if isinstance(metrics, dict):
                result = dict(metrics)
                result["diagnosis"] = probe.get("diagnosis")
                return result
    return {}


def _agent_reach_metrics(details: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    diagnostics = details.get("subgoal_diagnostics")
    if isinstance(diagnostics, dict):
        summary = diagnostics.get("summary")
        if isinstance(summary, dict):
            merged.update(summary)
        final = diagnostics.get("final")
        if isinstance(final, dict):
            merged.update(final)
    for event in details.get("progress_events") or []:
        if not isinstance(event, dict):
            continue
        diagnostics = event.get("diagnostics")
        if isinstance(diagnostics, dict):
            summary = diagnostics.get("summary")
            if isinstance(summary, dict):
                merged.update(summary)
    return merged


def _solid_blockers_between(details: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for failure in details.get("affordance_failures") or []:
        if not isinstance(failure, dict):
            continue
        if failure.get("code") != "solid_blocker_between_object_and_region":
            continue
        message = str(failure.get("message") or "")
        match = re.search(r"solid blocker\s+([A-Za-z0-9_]+)\s+lies between", message)
        if match:
            blockers.append(match.group(1))
    return list(dict.fromkeys(blockers))


def _collect_probe_dicts(value: Any, *, _depth: int = 0) -> list[dict[str, Any]]:
    if _depth > 8:
        return []
    probes: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "name" in value and "passed" in value:
            probes.append(value)
        for child in value.values():
            probes.extend(_collect_probe_dicts(child, _depth=_depth + 1))
    elif isinstance(value, list):
        for child in value:
            probes.extend(_collect_probe_dicts(child, _depth=_depth + 1))
    return probes


def _survival_step_budget(validation: ValidationResult) -> float:
    total_steps = _float_or_none(validation.details.get("total_steps"))
    if total_steps and total_steps > 30:
        return min(420.0, max(180.0, total_steps))
    return 240.0


def _replace_assignment_leading_number(
    code: str,
    name: str,
    transform,
) -> tuple[str, bool]:
    """Patch the first numeric factor in a simple assignment.

    This handles both `self.x = 18` and ratio assignments such as
    `self.projectile_speed = 0.7 * self.agent_max_speed`.
    """

    escaped = re.escape(name)
    pattern = re.compile(
        rf"(^\s*{escaped}\s*=\s*)([0-9]+(?:\.[0-9]+)?)(?P<tail>[^\n]*$)",
        re.MULTILINE,
    )

    def repl(match: re.Match[str]) -> str:
        old = float(match.group(2))
        new = float(transform(old))
        if abs(new - old) < 1e-6:
            return match.group(0)
        return f"{match.group(1)}{_format_number(new)}{match.group('tail')}"

    patched, count = pattern.subn(repl, code, count=1)
    return patched, patched != code and count > 0


def _replace_int_round_seconds_assignment(
    code: str,
    name: str,
    transform,
) -> tuple[str, bool]:
    """Patch assignments like `self.warmup_steps = int(round(2.0 * ...))`."""

    escaped = re.escape(name)
    pattern = re.compile(
        rf"(^\s*{escaped}\s*=\s*int\(round\()\s*"
        r"([0-9]+(?:\.[0-9]+)?)"
        r"(?P<tail>\s*\*[^\n]*$)",
        re.MULTILINE,
    )

    def repl(match: re.Match[str]) -> str:
        old = float(match.group(2))
        new = float(transform(old))
        if abs(new - old) < 1e-6:
            return match.group(0)
        return f"{match.group(1)}{_format_number(new)}{match.group('tail')}"

    patched, count = pattern.subn(repl, code, count=1)
    return patched, patched != code and count > 0


def _replace_numeric_assignment(
    code: str,
    name: str,
    transform,
) -> tuple[str, bool]:
    escaped = re.escape(name)
    pattern = re.compile(rf"(^\s*{escaped}\s*=\s*)([0-9]+(?:\.[0-9]+)?)(\s*(?:#.*)?$)", re.MULTILINE)

    def repl(match: re.Match[str]) -> str:
        old = float(match.group(2))
        new = float(transform(old))
        if abs(new - old) < 1e-6:
            return match.group(0)
        return f"{match.group(1)}{_format_number(new)}{match.group(3)}"

    patched, count = pattern.subn(repl, code, count=1)
    return patched, patched != code and count > 0


def _replace_tuple_assignment_min_width(
    code: str,
    name: str,
    min_width: float,
) -> tuple[str, bool]:
    escaped = re.escape(name)
    pattern = re.compile(
        rf"(^\s*{escaped}\s*=\s*\()\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)(\s*\)\s*(?:#.*)?$)",
        re.MULTILINE,
    )

    def repl(match: re.Match[str]) -> str:
        width = float(match.group(2))
        height = float(match.group(3))
        if width >= min_width:
            return match.group(0)
        return f"{match.group(1)}{_format_number(min_width)}, {_format_number(height)}{match.group(4)}"

    patched, count = pattern.subn(repl, code, count=1)
    return patched, patched != code and count > 0


def _replace_named_tuple_min_dimensions(
    code: str,
    name: str,
    min_width: float,
    min_height: float,
) -> tuple[str, bool]:
    escaped = re.escape(name)
    pattern = re.compile(
        rf"(^\s*{escaped}\s*=\s*\()\s*"
        r"(?P<x>[0-9]+(?:\.[0-9]+)?)\s*,\s*"
        r"(?P<y>[0-9]+(?:\.[0-9]+)?)(\s*\)\s*(?:#.*)?$)",
        re.MULTILINE,
    )

    def repl(match: re.Match[str]) -> str:
        width = float(match.group("x"))
        height = float(match.group("y"))
        new_width = max(width, float(min_width))
        new_height = max(height, float(min_height))
        if abs(new_width - width) < 1e-6 and abs(new_height - height) < 1e-6:
            return match.group(0)
        return (
            f"{match.group(1)}{_format_number(new_width)}, "
            f"{_format_number(new_height)}{match.group(4)}"
        )

    patched, count = pattern.subn(repl, code, count=1)
    return patched, patched != code and count > 0


def _replace_named_tuple_max_dimensions(
    code: str,
    name: str,
    max_width: float,
    max_height: float,
) -> tuple[str, bool]:
    escaped = re.escape(name)
    pattern = re.compile(
        rf"(^\s*{escaped}\s*=\s*\()\s*"
        r"(?P<x>[0-9]+(?:\.[0-9]+)?)\s*,\s*"
        r"(?P<y>[0-9]+(?:\.[0-9]+)?)(\s*\)\s*(?:#.*)?$)",
        re.MULTILINE,
    )

    def repl(match: re.Match[str]) -> str:
        width = float(match.group("x"))
        height = float(match.group("y"))
        new_width = min(width, float(max_width))
        new_height = min(height, float(max_height))
        if abs(new_width - width) < 1e-6 and abs(new_height - height) < 1e-6:
            return match.group(0)
        return (
            f"{match.group(1)}{_format_number(new_width)}, "
            f"{_format_number(new_height)}{match.group(4)}"
        )

    patched, count = pattern.subn(repl, code, count=1)
    return patched, patched != code and count > 0


def _replace_named_tuple_value(
    code: str,
    name: str,
    value: tuple[float, float],
) -> tuple[str, bool]:
    escaped = re.escape(name)
    pattern = re.compile(
        rf"(^\s*{escaped}\s*=\s*\(\s*)"
        r"(?P<x>[^,\n]+)\s*,\s*"
        r"(?P<y>[^)\n]+)(\s*\)\s*(?:#.*)?$)",
        re.MULTILINE,
    )

    def repl(match: re.Match[str]) -> str:
        old_x = _float_or_none(str(match.group("x")).strip())
        new_x = _format_number(value[0]) if old_x is not None else str(match.group("x")).strip()
        old_y = _float_or_none(str(match.group("y")).strip())
        new_y = _format_number(value[1]) if old_y is not None else str(match.group("y")).strip()
        return (
            f"{match.group(1)}{new_x}, "
            f"{new_y}{match.group(4)}"
        )

    patched, count = pattern.subn(repl, code, count=1)
    return patched, patched != code and count > 0


def _replace_named_offset(
    code: str,
    variable_name: str,
    anchor_expr: str,
    desired_offset: float,
) -> tuple[str, bool]:
    pattern = re.compile(
        rf"(^\s*{re.escape(variable_name)}\s*=\s*\(\s*{re.escape(anchor_expr)}\s*-\s*)([0-9]+(?:\.[0-9]+)?)(\s*,)",
        re.MULTILINE,
    )

    def repl(match: re.Match[str]) -> str:
        old = float(match.group(2))
        if old <= desired_offset:
            return match.group(0)
        return f"{match.group(1)}{_format_number(desired_offset)}{match.group(3)}"

    patched, count = pattern.subn(repl, code, count=1)
    return patched, patched != code and count > 0


def _replace_keyword_numeric(
    code: str,
    *,
    keyword: str,
    old_min: float,
    new_value: float,
    nearby: str,
) -> tuple[str, bool]:
    index = code.find(nearby)
    if index < 0:
        return code, False
    window_end = min(len(code), index + 700)
    before = code[:index]
    window = code[index:window_end]
    after = code[window_end:]
    pattern = re.compile(rf"({re.escape(keyword)}\s*=\s*)([0-9]+(?:\.[0-9]+)?)")

    def repl(match: re.Match[str]) -> str:
        old = float(match.group(2))
        if old < old_min:
            return match.group(0)
        return f"{match.group(1)}{_format_number(new_value)}"

    patched_window, count = pattern.subn(repl, window, count=1)
    patched = before + patched_window + after
    return patched, patched != code and count > 0


def _replace_dict_numeric_value(
    code: str,
    key: str,
    max_value: float,
) -> tuple[str, bool]:
    pattern = re.compile(
        rf"([\"']{re.escape(key)}[\"']\s*:\s*)([0-9]+(?:\.[0-9]+)?)"
    )

    def repl(match: re.Match[str]) -> str:
        old = float(match.group(2))
        if old <= max_value:
            return match.group(0)
        return f"{match.group(1)}{_format_number(max_value)}"

    patched, count = pattern.subn(repl, code)
    return patched, patched != code and count > 0


def _sensorize_named_static_helper(code: str, object_name: str) -> tuple[str, bool]:
    name_literal_1 = f'"{object_name}"'
    name_literal_2 = f"'{object_name}'"
    name_index = code.find(name_literal_1)
    if name_index < 0:
        name_index = code.find(name_literal_2)
    if name_index < 0:
        return code, False
    call_start_candidates = [
        code.rfind("self.create_static_box(", 0, name_index),
        code.rfind("self.create_static_segment(", 0, name_index),
    ]
    call_start = max(call_start_candidates)
    if call_start < 0:
        return code, False
    call_end = _matching_call_end(code, code.find("(", call_start))
    if call_end is None:
        return code, False
    call = code[call_start : call_end + 1]
    if "sensor=True" in call:
        return code, False
    if re.search(r"\bsensor\s*=", call):
        patched_call = re.sub(r"\bsensor\s*=\s*False", "sensor=True", call, count=1)
    else:
        insert_at = call.rfind(")")
        indent_match = re.search(r"\n(\s*)\)$", call)
        indent = indent_match.group(1) if indent_match else " " * 12
        prefix = call[:insert_at].rstrip()
        suffix = call[insert_at:]
        separator = "" if prefix.endswith(",") else ","
        patched_call = f"{prefix}{separator}\n{indent}sensor=True,{suffix}"
    if patched_call == call:
        return code, False
    return code[:call_start] + patched_call + code[call_end + 1 :], True


def _matching_call_end(code: str, open_paren_index: int) -> int | None:
    if open_paren_index < 0:
        return None
    depth = 0
    quote: str | None = None
    escape = False
    for index in range(open_paren_index, len(code)):
        char = code[index]
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _replace_layout_numeric_keys(
    code: str,
    key_fragments: tuple[str, ...],
    transform,
) -> tuple[str, bool]:
    """Patch numeric self.layout dict entries whose keys match any fragment."""

    fragments = tuple(fragment.lower() for fragment in key_fragments if fragment)
    if not fragments:
        return code, False
    pattern = re.compile(
        r"([\"'](?P<key>[^\"']+)[\"']\s*:\s*)(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<suffix>\s*,?)"
    )

    def repl(match: re.Match[str]) -> str:
        key = match.group("key").lower()
        if not any(fragment in key for fragment in fragments):
            return match.group(0)
        old = float(match.group("value"))
        new = float(transform(old))
        if abs(new - old) < 1e-6:
            return match.group(0)
        return f"{match.group(1)}{_format_number(new)}{match.group('suffix')}"

    patched, count = pattern.subn(repl, code)
    return patched, patched != code and count > 0


def _replace_layout_numeric_key_exact(
    code: str,
    key: str,
    value: float,
) -> tuple[str, bool]:
    pattern = re.compile(
        rf"([\"']{re.escape(key)}[\"']\s*:\s*)(?P<value>[-+]?[0-9]+(?:\.[0-9]+)?)(?P<suffix>\s*,?)"
    )

    def repl(match: re.Match[str]) -> str:
        old = float(match.group("value"))
        if abs(old - value) < 1e-6:
            return match.group(0)
        return f"{match.group(1)}{_format_number(value)}{match.group('suffix')}"

    patched, count = pattern.subn(repl, code, count=1)
    return patched, patched != code and count > 0


def _replace_semantic_numeric_value(
    code: str,
    key: str,
    value: float,
) -> tuple[str, bool]:
    pattern = re.compile(
        rf"([\"']{re.escape(key)}[\"']\s*:\s*)(?P<value>[-+]?[0-9]+(?:\.[0-9]+)?)(?P<suffix>\s*,?)"
    )

    def repl(match: re.Match[str]) -> str:
        old = float(match.group("value"))
        if old <= value:
            return match.group(0)
        return f"{match.group(1)}{_format_number(value)}{match.group('suffix')}"

    patched, count = pattern.subn(repl, code)
    return patched, patched != code and count > 0


def _replace_layout_numeric_list_exact(
    code: str,
    key: str,
    values: tuple[float, ...],
) -> tuple[str, bool]:
    pattern = re.compile(
        rf"([\"']{re.escape(key)}[\"']\s*:\s*\[)(?P<values>[^\]]+)(\]\s*,?)",
        re.DOTALL,
    )
    replacement = ", ".join(_format_number(value) for value in values)

    def repl(match: re.Match[str]) -> str:
        old_values = tuple(
            _float_or_none(item.strip())
            for item in re.split(r",", match.group("values"))
            if item.strip()
        )
        old_values_clean = tuple(value for value in old_values if value is not None)
        if len(old_values_clean) == len(values) and all(
            abs(old - new) < 1e-6 for old, new in zip(old_values_clean, values)
        ):
            return match.group(0)
        return f"{match.group(1)}{replacement}{match.group(3)}"

    patched, count = pattern.subn(repl, code, count=1)
    return patched, patched != code and count > 0


def _replace_layout_sequence_empty(code: str, key: str) -> tuple[str, bool]:
    pattern = re.compile(
        rf"([\"']{re.escape(key)}[\"']\s*:\s*)\[[\s\S]*?\](\s*,)",
        re.MULTILINE,
    )
    patched, count = pattern.subn(r"\1[]\2", code, count=1)
    return patched, patched != code and count > 0


def _replace_layout_tuple_min_dimensions(
    code: str,
    key: str,
    min_width: float,
    min_height: float,
) -> tuple[str, bool]:
    pattern = re.compile(
        rf"([\"']{re.escape(key)}[\"']\s*:\s*\(\s*)"
        r"(?P<x>[0-9]+(?:\.[0-9]+)?)\s*,\s*"
        r"(?P<y>[0-9]+(?:\.[0-9]+)?)(\s*\))"
    )

    def repl(match: re.Match[str]) -> str:
        width = float(match.group("x"))
        height = float(match.group("y"))
        new_width = max(width, float(min_width))
        new_height = max(height, float(min_height))
        if abs(new_width - width) < 1e-6 and abs(new_height - height) < 1e-6:
            return match.group(0)
        return (
            f"{match.group(1)}{_format_number(new_width)}, "
            f"{_format_number(new_height)}{match.group(4)}"
        )

    patched, count = pattern.subn(repl, code, count=1)
    return patched, patched != code and count > 0


def _replace_layout_tuple_max_dimensions(
    code: str,
    key: str,
    max_width: float,
    max_height: float,
) -> tuple[str, bool]:
    pattern = re.compile(
        rf"([\"']{re.escape(key)}[\"']\s*:\s*\(\s*)"
        r"(?P<x>[0-9]+(?:\.[0-9]+)?)\s*,\s*"
        r"(?P<y>[0-9]+(?:\.[0-9]+)?)(\s*\))"
    )

    def repl(match: re.Match[str]) -> str:
        width = float(match.group("x"))
        height = float(match.group("y"))
        new_width = min(width, float(max_width))
        new_height = min(height, float(max_height))
        if abs(new_width - width) < 1e-6 and abs(new_height - height) < 1e-6:
            return match.group(0)
        return (
            f"{match.group(1)}{_format_number(new_width)}, "
            f"{_format_number(new_height)}{match.group(4)}"
        )

    patched, count = pattern.subn(repl, code, count=1)
    return patched, patched != code and count > 0


def _replace_layout_tuple_value(
    code: str,
    key: str,
    value: tuple[float, float],
) -> tuple[str, bool]:
    pattern = re.compile(
        rf"([\"']{re.escape(key)}[\"']\s*:\s*\(\s*)"
        r"(?P<x>[^,\n]+)\s*,\s*"
        r"(?P<y>[-+]?[0-9]+(?:\.[0-9]+)?)(\s*\))"
    )
    if not pattern.search(code):
        return code, False

    def repl(match: re.Match[str]) -> str:
        old_x = _float_or_none(str(match.group("x")).strip())
        new_x = _format_number(value[0]) if old_x is not None else str(match.group("x")).strip()
        return (
            f"{match.group(1)}{new_x}, "
            f"{_format_number(value[1])}{match.group(4)}"
        )

    patched, count = pattern.subn(repl, code, count=1)
    return patched, patched != code and count > 0


def _replace_layout_point_value(
    code: str,
    key: str,
    value: tuple[float, float],
) -> tuple[str, bool]:
    """Replace a repairable layout point written as either tuple or list syntax."""

    patched, changed = _replace_layout_tuple_value(code, key, value)
    if changed:
        return patched, True
    return _replace_layout_list_value(code, key, value)


def _replace_layout_list_value(
    code: str,
    key: str,
    value: tuple[float, float],
) -> tuple[str, bool]:
    pattern = re.compile(
        rf"([\"']{re.escape(key)}[\"']\s*:\s*\[\s*)"
        r"(?P<x>[-+]?[0-9]+(?:\.[0-9]+)?)\s*,\s*"
        r"(?P<y>[-+]?[0-9]+(?:\.[0-9]+)?)(\s*\])"
    )

    def repl(match: re.Match[str]) -> str:
        old_x = _float_or_none(match.group("x"))
        old_y = _float_or_none(match.group("y"))
        if old_x is not None and old_y is not None:
            if abs(old_x - value[0]) < 1.0 and abs(old_y - value[1]) < 1.0:
                return match.group(0)
        return f"{match.group(1)}{_format_number(value[0])}, {_format_number(value[1])}{match.group(4)}"

    patched, count = pattern.subn(repl, code, count=1)
    return patched, patched != code and count > 0


def _replace_layout_tuple_y(
    code: str,
    key: str,
    new_y: float,
) -> tuple[str, bool]:
    pattern = re.compile(
        rf"([\"']{re.escape(key)}[\"']\s*:\s*\(\s*([^,\n]+)\s*,\s*)([^,\)\n]+)(\s*\))"
    )

    def repl(match: re.Match[str]) -> str:
        old = _float_or_none(match.group(3))
        if old is not None and abs(old - new_y) < 1.0:
            return match.group(0)
        return f"{match.group(1)}{_format_number(new_y)}{match.group(4)}"

    patched, count = pattern.subn(repl, code, count=1)
    return patched, patched != code and count > 0


def _replace_named_tuple_y(
    code: str,
    variable_name: str,
    new_y: float,
) -> tuple[str, bool]:
    pattern = re.compile(
        rf"(^\s*{re.escape(variable_name)}\s*=\s*\(\s*([^,\n]+)\s*,\s*)([^,\)\n]+)(\s*\).*$)",
        re.MULTILINE,
    )

    def repl(match: re.Match[str]) -> str:
        old = _float_or_none(match.group(3))
        if old is not None and abs(old - new_y) < 1.0:
            return match.group(0)
        return f"{match.group(1)}{_format_number(new_y)}{match.group(4)}"

    patched, count = pattern.subn(repl, code, count=1)
    return patched, patched != code and count > 0


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _point_or_none(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list | tuple) or len(value) < 2:
        return None
    x = _float_or_none(value[0])
    y = _float_or_none(value[1])
    if x is None or y is None:
        return None
    return (x, y)


def _point_toward(
    start: tuple[float, float],
    target: tuple[float, float],
    *,
    max_distance: float,
) -> tuple[float, float]:
    dx = start[0] - target[0]
    dy = start[1] - target[1]
    distance = (dx * dx + dy * dy) ** 0.5
    if distance <= max_distance or distance <= 1e-6:
        return start
    scale = max_distance / distance
    return (target[0] + dx * scale, target[1] + dy * scale)


def _unit_vector(
    start: tuple[float, float],
    target: tuple[float, float],
) -> tuple[float, float] | None:
    dx = target[0] - start[0]
    dy = target[1] - start[1]
    distance = (dx * dx + dy * dy) ** 0.5
    if distance <= 1e-6:
        return None
    return (dx / distance, dy / distance)


def _name_tokens(name: str) -> tuple[str, ...]:
    tokens = [token for token in re.split(r"[^A-Za-z0-9]+", name.lower()) if token]
    # Drop generic suffixes so "charged_ball" can match "ball_start" and
    # "magnetic_zone_force" can match "magnetic_size".
    generic = {"zone", "force", "field", "region", "object", "dynamic"}
    useful = [token for token in tokens if token not in generic]
    return tuple(dict.fromkeys([*tokens, *useful]))


def _object_start_key_candidates(object_name: str) -> tuple[str, ...]:
    tokens = _name_tokens(object_name)
    candidates = [f"{object_name}_start"] if object_name else []
    for token in tokens:
        candidates.extend([f"{token}_start", f"{token}_center", f"{token}_pos"])
    candidates.extend(["object_start", "ball_start", "box_start"])
    return tuple(dict.fromkeys(candidates))


def _object_x_key_candidates(object_name: str) -> tuple[str, ...]:
    tokens = _name_tokens(object_name)
    candidates = [f"{object_name}_x"] if object_name else []
    for token in tokens:
        candidates.extend([f"{token}_x", f"{token}_start_x", f"{token}_center_x"])
    candidates.extend(["object_x", "ball_x", "box_x"])
    return tuple(dict.fromkeys(candidates))


def _object_mass_key_candidates(object_name: str) -> tuple[str, ...]:
    tokens = _name_tokens(object_name)
    candidates = [f"{object_name}_mass"] if object_name else []
    candidates.extend(f"{token}_mass" for token in tokens)
    candidates.extend(["object_mass", "ball_mass", "box_mass"])
    return tuple(dict.fromkeys(candidates))


def _object_friction_key_candidates(object_name: str) -> tuple[str, ...]:
    tokens = _name_tokens(object_name)
    candidates = [f"{object_name}_friction"] if object_name else []
    candidates.extend(f"{token}_friction" for token in tokens)
    candidates.extend(["object_friction", "ball_friction", "box_friction"])
    return tuple(dict.fromkeys(candidates))


def _field_size_key_candidates(field_name: str, region_name: str) -> tuple[str, ...]:
    candidates: list[str] = []
    for name in (field_name, region_name):
        if name:
            candidates.append(f"{name}_size")
        for token in _name_tokens(name):
            candidates.extend([f"{token}_size", f"{token}_zone_size", f"{token}_field_size"])
    candidates.extend(["field_size", "force_zone_size", "magnetic_size", "wind_size"])
    return tuple(dict.fromkeys(candidates))


def _region_size_key_candidates(region_name: str) -> tuple[str, ...]:
    candidates = [f"{region_name}_size"] if region_name else []
    for token in _name_tokens(region_name):
        candidates.extend([f"{token}_size", f"{token}_zone_size"])
    candidates.extend(["region_size", "goal_size", "magnetic_size"])
    return tuple(dict.fromkeys(candidates))


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))}.0"
    return f"{value:.3f}".rstrip("0").rstrip(".")
