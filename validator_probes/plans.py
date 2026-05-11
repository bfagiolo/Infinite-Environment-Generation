"""Subgoal-to-probe planning for reusable physics validation."""

from __future__ import annotations

from .contract import ProbePlan


PROBES_BY_SUBGOAL = {
    "agent_reach_region": (
        "concrete_target",
        "path_reachability",
        "agent_moves_toward_target",
        "target_containment",
    ),
    "agent_touch_object": (
        "concrete_target",
        "agent_moves_toward_target",
        "contact_or_proximity",
    ),
    "move_object_to_region": (
        "passive_stability",
        "physical_parameter_bounds",
        "object_region_affordance",
        "path_reachability",
        "agent_object_collision_allowed",
        "agent_force_applied",
        "agent_moves_under_force",
        "agent_object_contact",
        "contact_impulse_observed",
        "box_moved_at_all",
        "box_free_to_move",
        "object_moves_toward_region",
        "object_inside_region",
    ),
    "low_friction_slide_to_region": (
        "passive_stability",
        "physical_parameter_bounds",
        "object_region_affordance",
        "agent_object_collision_allowed",
        "agent_force_applied",
        "agent_moves_under_force",
        "agent_object_contact",
        "contact_impulse_observed",
        "box_moved_at_all",
        "box_free_to_move",
        "object_moves_toward_region",
        "object_inside_region",
    ),
    "strike_object_to_region": (
        "passive_stability",
        "physical_parameter_bounds",
        "object_region_affordance",
        "agent_object_collision_allowed",
        "agent_force_applied",
        "agent_moves_under_force",
        "agent_object_contact",
        "contact_impulse_observed",
        "momentum_transfer",
        "box_moved_at_all",
        "box_free_to_move",
        "object_moves_toward_region",
        "object_inside_region",
    ),
    "ballistic_object_to_region": (
        "passive_stability",
        "physical_parameter_bounds",
        "object_region_affordance",
        "agent_object_collision_allowed",
        "agent_force_applied",
        "agent_moves_under_force",
        "agent_object_contact",
        "contact_impulse_observed",
        "momentum_transfer",
        "barrier_clearance",
        "apex_height",
        "trajectory_change",
        "object_moves_toward_region",
        "object_inside_region",
    ),
    "support_exit_freefall": (
        "passive_stability",
        "physical_parameter_bounds",
        "object_region_affordance",
        "agent_object_collision_allowed",
        "agent_force_applied",
        "agent_moves_under_force",
        "agent_object_contact",
        "contact_impulse_observed",
        "box_free_to_move",
        "object_moves_toward_region",
        "trajectory_change",
    ),
    "classify_objects_to_regions": (
        "concrete_target",
        "class_membership",
        "object_moves_toward_region",
        "object_inside_region",
    ),
    "bounce_to_target": (
        "concrete_target",
        "contact_or_proximity",
        "trajectory_change",
        "target_contact",
    ),
    "field_force_interaction": (
        "passive_or_field_motion",
        "object_displacement",
        "progress_metric_change",
    ),
    "lever_launch": (
        "pivot_mechanism",
        "passive_stability",
        "plank_rotates",
        "launch_progress",
    ),
    "activate_mechanism": (
        "concrete_target",
        "activation_state",
        "post_activation_reachability",
    ),
    "survive_duration": (
        "stability",
        "duration_condition",
    ),
    "maintain_balance": (
        "angle_or_height_metric",
        "stability",
    ),
}


def probe_plan_for_subgoal(subgoal: dict[str, object]) -> ProbePlan:
    kind = str(subgoal.get("kind") or "")
    return ProbePlan(kind, tuple(PROBES_BY_SUBGOAL.get(kind, ("custom_progress_metric",))))
