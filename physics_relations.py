"""Composable physics-relation graph for prompt-to-world generation.

This layer is intentionally not a task-template registry. It translates prompt
meaning into small physical relations that can be composed into many tasks and
then projected into validator-readable subgoals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any


@dataclass(frozen=True)
class PhysicsRelation:
    """A small, probeable physical relation between objects/regions."""

    type: str
    actors: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    parameter_constraints: dict[str, Any] = field(default_factory=dict)
    probes: list[str] = field(default_factory=list)
    repair_knobs: list[str] = field(default_factory=list)
    repair: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "type": self.type,
            "actors": list(self.actors),
            "objects": list(self.objects),
            "regions": list(self.regions),
            "fields": list(self.fields),
            "parameters": dict(self.parameters),
            "parameter_constraints": dict(self.parameter_constraints),
            "probes": list(self.probes),
            "repair_knobs": list(self.repair_knobs),
            "repair": self.repair,
        }
        return {key: value for key, value in payload.items() if value not in ([], {}, "")}


def infer_physics_relation_graph(
    prompt: str,
    *,
    simulation_brief: dict[str, Any] | None = None,
    gameplay_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Infer a composable relation graph from prompt and run-level briefs."""

    text = " ".join(
        [
            prompt,
            _json_text(simulation_brief),
            _json_text(gameplay_profile),
        ]
    ).lower()
    objective = _dict_field(simulation_brief, "objective")
    objective_type = str(objective.get("type") or "").lower()
    route_text = " ".join(
        [
            prompt,
            objective_type,
            str(objective.get("description") or objective.get("success") or ""),
        ]
    ).lower()
    world_context = _dict_field(simulation_brief, "world_context")
    relations: list[PhysicsRelation] = []

    if _is_agent_projectile_impact_prompt(route_text):
        relations.extend(_agent_projectile_impact_relations(route_text))
    elif _mentions_whole_phrase(route_text, "pressure plate", "plate", "button", "gate", "door", "switch"):
        relations.extend(_pressure_plate_gate_relations(route_text))
    elif _is_boundary_exit_prompt(route_text, objective_type):
        relations.extend(_boundary_exit_relations(route_text))
    elif _is_ballistic_prompt(route_text, objective_type):
        relations.extend(_ballistic_relations(route_text))
    elif _mentions(route_text, "magnetic", "wind", "current", "conveyor", "force field", "gravity well"):
        relations.extend(_field_force_relations(route_text))
    elif _mentions(route_text, "seesaw", "see-saw", "lever", "catapult", "balance beam", "plank"):
        relations.extend(_lever_relations(route_text))
    elif _is_navigation_avoidance_prompt(route_text, objective_type):
        relations.extend(_navigation_avoidance_relations(route_text))
    elif _is_hazard_or_survival_prompt(route_text, objective_type):
        relations.extend(_hazard_escape_relations(route_text))
    elif _mentions(route_text, "touch", "collect", "crystal", "asteroid", "rock", "gem"):
        relations.extend(_touch_collect_relations(route_text, world_context))
    elif _mentions(route_text, "push", "shove", "move", "slide"):
        relations.extend(_generic_push_relations(route_text))
    else:
        relations.append(
            PhysicsRelation(
                "agent_reaches_region",
                actors=["agent"],
                regions=["goal"],
                probes=["path_reachability", "agent_moves_toward_target", "check_objective"],
                repair="Place the target on a reachable route and keep the region non-blocking.",
            )
        )

    relation_dicts = _dedupe_relations([relation.to_dict() for relation in relations])
    subgoals = _project_relations_to_subgoals(relation_dicts)
    physical_parameters = _physical_parameter_profile(relation_dicts, text, world_context)
    repair_knobs = _repair_knob_manifest(relation_dicts, physical_parameters)
    return {
        "version": "physics_relation_graph.v2",
        "source": "deterministic_relation_inference",
        "intent_summary": _short_text(
            str((simulation_brief or {}).get("intent_summary") or prompt),
            280,
        ),
        "world_context": world_context,
        "physical_parameters": physical_parameters,
        "relations": relation_dicts,
        "suggested_subgoals": subgoals,
        "repair_knobs": repair_knobs,
        "validator_strategy": {
            "primary": "validate physics_relations, then objective_profile.subgoals",
            "repair_unit": "failed relation, not failed task template",
            "parameter_policy": "validate physical properties such as mass, friction, restitution, gravity, slope, clearance, and impulse before broad regeneration",
            "template_policy": "helpers are optional constructors; relations are the conceptual source of truth",
        },
    }


def format_physics_relation_guidance(graph: dict[str, Any] | None) -> str:
    """Render graph guidance for the Architect prompt."""

    if not graph:
        return "- No physics relation graph was provided."
    return (
        "- Treat this as the mechanic source of truth. It is a relation graph, not a template.\n"
        "- Define `self.physics_relations = {...}` in the mandatory registry and export it in ground truth.\n"
        "- Project the relations into `objective_profile['subgoals']` using the suggested_subgoals when they fit.\n"
        "- Treat `physical_parameters` as the physical-property contract: mass, friction, restitution, gravity, slope, clearance, damping, and impulse assumptions must match the world.\n"
        "- Store tunable parameters in `self.layout` so deterministic repair can adjust the listed `repair_knobs`.\n"
        "- If a helper conflicts with the relations, obey the relation graph and use lower-level BaseEnv primitives.\n"
        "- Repair should preserve passed relations and only edit the failed relation.\n\n"
        f"{json.dumps(graph, indent=2, sort_keys=True)}"
    )


def _pressure_plate_gate_relations(text: str) -> list[PhysicsRelation]:
    moving = _object_name(text, default="blue_box", choices=("box", "crate", "rock", "ball"))
    trigger = "pressure_plate" if "plate" in text else "trigger"
    return [
        PhysicsRelation(
            "contact_push",
            actors=["agent"],
            objects=[moving],
            parameter_constraints={
                "object_mass": {"min": 0.6, "max": 2.0, "why": "movable by agent without becoming weightless"},
                "object_friction": {"min": 0.05, "max": 0.45, "why": "allows sustained push progress"},
                "support_friction": {"min": 0.45, "max": 0.95, "why": "keeps agent grounded while pushing"},
            },
            probes=["agent_object_contact", "contact_impulse_observed", "object_moves_toward_region"],
            repair_knobs=["agent_start", "object_start", "trigger_center", "object_mass", "object_friction", "agent_strength"],
            repair="Align agent -> object -> trigger and keep the object supported, close, and movable.",
        ),
        PhysicsRelation(
            "object_enters_region",
            objects=[moving],
            regions=[trigger],
            probes=["object_region_affordance", "object_inside_region"],
            repair="Make the trigger a non-blocking sensor and place it on the object's push lane.",
        ),
        PhysicsRelation(
            "mechanism_activation",
            objects=[moving],
            regions=[trigger, "goal"],
            probes=["activation_state", "path_reachability"],
            repair="Keep the gate after the trigger and make the opened route physically passable.",
        ),
    ]


def _agent_projectile_impact_relations(text: str) -> list[PhysicsRelation]:
    projectile = _object_name(
        text,
        default="agent_bullet",
        choices=("bullet", "projectile", "missile", "laser", "blaster"),
    )
    if projectile in {"bullet", "projectile", "missile", "laser", "blaster"}:
        projectile = f"agent_{projectile}"
    target = _object_name(
        text,
        default="target_stack",
        choices=("pile", "stack", "tower", "squares", "blocks", "target"),
    )
    if target in {"pile", "stack", "tower", "squares", "blocks", "target"}:
        target = "target_stack"
    return [
        PhysicsRelation(
            "agent_fires_projectile",
            actors=["agent"],
            objects=[projectile],
            parameters={
                "projectile_is_agent_tool": True,
                "not_incoming_hazard": True,
                "continuous_survival_not_required": True,
            },
            parameter_constraints={
                "projectile_speed": {"min": 240, "why": "shot must visibly travel across open space"},
                "projectile_mass": {"min": 0.1, "max": 1.5, "why": "enough impulse to move target without destabilizing the world"},
            },
            probes=["projectile_spawned_near_agent", "projectile_travels_forward"],
            repair_knobs=["projectile_start", "projectile_speed", "projectile_mass", "agent_start"],
            repair="Create an agent-fired dynamic projectile with a clear travel lane toward the target structure.",
        ),
        PhysicsRelation(
            "projectile_impact_transfer",
            actors=["agent"],
            objects=[projectile, target],
            regions=[target],
            parameter_constraints={
                "object_to_region_distance": {"max": 360, "why": "shot should reach target in the validator window"},
                "target_mass": {"min": 0.2, "max": 2.5, "why": "stack pieces should be toppleable by projectile impulse"},
                "target_friction": {"min": 0.0, "max": 0.55, "why": "impact should create visible displacement/rotation"},
            },
            probes=["projectile_target_contact", "contact_impulse_observed", "target_displacement_or_rotation"],
            repair_knobs=["projectile_start", "target_center", "projectile_speed", "projectile_mass", "target_mass", "target_friction"],
            repair="Align projectile lane with the toppleable target; tune projectile speed/mass and target friction so impact visibly moves it.",
        ),
    ]


def _ballistic_relations(text: str) -> list[PhysicsRelation]:
    obj = _object_name(text, default="ball", choices=("basketball", "soccer_ball", "ball", "puck"))
    target = "hoop" if "hoop" in text or "basket" in text else "goal_line"
    requires_arc = target == "hoop" or _mentions(text, "throw", "lob", "launch", "arc", "over", "wall", "barrier", "defender")
    relation_type = "ballistic_arc_to_region" if requires_arc else "impulse_transfer_to_region"
    return [
        PhysicsRelation(
            "agent_approaches_object",
            actors=["agent"],
            objects=[obj],
            probes=["concrete_target", "agent_moves_toward_target", "contact_or_proximity"],
            repair="Spawn the object reachable by the agent without hidden blockers.",
        ),
        PhysicsRelation(
            "impulse_transfer",
            actors=["agent"],
            objects=[obj],
            parameter_constraints={
                "object_mass": {"min": 0.25, "max": 0.85, "why": "light enough for visible impulse transfer"},
                "object_friction": {"min": 0.0, "max": 0.12, "why": "prevents the ball/puck from acting like a crate"},
                "object_restitution": {"min": 0.08, "max": 0.35, "why": "keeps contact lively but controllable"},
            },
            probes=["agent_object_contact", "contact_impulse_observed", "trajectory_change"],
            repair_knobs=["agent_start", "object_start", "object_mass", "object_friction", "object_elasticity", "agent_strength"],
            repair="Keep the object light, dynamic, separate from the agent, and close enough for contact impulse.",
        ),
        PhysicsRelation(
            relation_type,
            objects=[obj],
            regions=[target],
            fields=["gravity"],
            parameters={
                "requires_arc": requires_arc,
                "target_should_be_sensor": True,
                "barrier_clearance_required": _mentions(text, "over", "wall", "barrier", "defender"),
            },
            parameter_constraints={
                "object_to_region_distance": {"max": 340, "why": "generic ballistic validator can prove the arc"},
                "target_sensor": {"required": True, "why": "code-level objective should be non-blocking"},
                "clearance_margin": {"min": 45, "why": "projectile should clear the barrier visibly"},
                "gravity": {"allowed": ["normal", "low_g", "custom"], "why": "arc needs gravity or explicit field"},
            },
            probes=["trajectory_change", "barrier_clearance", "apex_height", "object_region_affordance", "object_inside_region"],
            repair_knobs=["object_start", "target_center", "target_size", "barrier_height", "object_mass", "object_friction", "object_elasticity", "agent_strength"],
            repair="Use a non-blocking target sensor, clear arc space, and enough impulse/low friction for the object to reach the region.",
        ),
    ]


def _boundary_exit_relations(text: str) -> list[PhysicsRelation]:
    obj = _object_name(text, default="rock", choices=("rock", "boulder", "ball", "crate"))
    boundary = "cliff_edge_boundary" if _mentions(text, "cliff", "mountain", "edge") else "exit_boundary"
    drop = "open_air_drop_zone"
    return [
        PhysicsRelation(
            "contact_push",
            actors=["agent"],
            objects=[obj],
            probes=["agent_object_contact", "contact_impulse_observed", "object_moves_toward_region"],
            repair="Place the agent behind the object relative to the exit boundary and tune mass/friction so contact starts motion.",
        ),
        PhysicsRelation(
            "support_boundary_exit",
            objects=[obj],
            regions=[boundary],
            parameter_constraints={
                "object_to_boundary_distance": {"max": 220, "why": "agent can push object across support edge"},
                "support_friction": {"min": 0.4, "max": 0.95, "why": "stable support before edge exit"},
                "boundary_sensor": {"required": True, "why": "edge crossing must be code-visible"},
            },
            probes=["object_crosses_boundary", "support_exit"],
            repair_knobs=["object_start", "edge_x", "boundary_center", "support_width", "object_friction", "agent_strength"],
            repair="The boundary must align with the true end of support; support terrain should not be treated as a blocker before the edge.",
        ),
        PhysicsRelation(
            "freefall_after_support_exit",
            objects=[obj],
            regions=[drop],
            fields=["gravity"],
            parameter_constraints={
                "gravity": {"allowed": ["normal", "high_g", "custom"], "why": "freefall must produce downward velocity"},
                "drop_clearance": {"min": 90, "why": "object needs visible unsupported fall distance"},
            },
            probes=["downward_velocity", "unsupported_motion", "object_inside_region"],
            repair_knobs=["drop_height", "drop_zone_center", "gravity", "object_mass", "object_friction"],
            repair="Keep open air below the edge clear and let gravity produce visible downward displacement after crossing.",
        ),
    ]


def _field_force_relations(text: str) -> list[PhysicsRelation]:
    obj = _object_name(text, default="charged_ball", choices=("ball", "crate", "rock", "puck"))
    return [
        PhysicsRelation(
            "contact_push",
            actors=["agent"],
            objects=[obj],
            regions=["force_zone"],
            probes=["object_region_affordance", "agent_object_contact"],
            repair="Stage agent, object, and field entrance on one reachable lane.",
        ),
        PhysicsRelation(
            "field_force_transfer",
            objects=[obj],
            fields=["force_zone"],
            regions=["target_zone"],
            parameter_constraints={
                "field_strength": {"min": 800, "why": "field must measurably alter object trajectory"},
                "affected_object_mass": {"max": 2.5, "why": "field should visibly influence the target object"},
            },
            probes=["field_effect", "trajectory_change", "object_moves_toward_region"],
            repair_knobs=["field_center", "field_size", "field_strength", "object_mass", "target_center"],
            repair="Register a BaseEnv force zone and place the object close enough to enter it.",
        ),
    ]


def _lever_relations(text: str) -> list[PhysicsRelation]:
    return [
        PhysicsRelation(
            "pivot_constraint",
            objects=["seesaw_plank"],
            parameter_constraints={
                "joint_clearance_px": {"min": 2, "why": "prevents initial overlap jitter"},
                "angular_damping": {"min": 0.3, "max": 0.8, "why": "stable readable lever motion"},
            },
            probes=["pivot_mechanism", "plank_rotates"],
            repair_knobs=["pivot_anchor", "plank_mass", "plank_length", "angular_damping"],
            repair="Use a dynamic plank with a registered PivotJoint and separated connected shapes.",
        ),
        PhysicsRelation(
            "load_side_impulse_or_weight",
            objects=["heavy_ball", "seesaw_plank"],
            probes=["contact_impulse_observed", "plank_rotates"],
            repair="Stage the weight where it can rotate the plank without being pinned or unsupported.",
        ),
        PhysicsRelation(
            "lever_launch",
            actors=["agent"],
            objects=["seesaw_plank", "heavy_ball"],
            regions=["goal"],
            probes=["launch_progress", "agent_moves_toward_target"],
            repair="Align the launch side with a generous high goal sensor and avoid fragile push-to-impact unless required.",
        ),
    ]


def _hazard_escape_relations(text: str) -> list[PhysicsRelation]:
    relations: list[PhysicsRelation] = []
    if _mentions(text, "exit", "escape", "reach", "navigate", "goal"):
        relations.append(
            PhysicsRelation(
                "agent_reaches_region",
                actors=["agent"],
                regions=["exit"],
                probes=["path_reachability", "agent_moves_toward_target", "check_objective"],
                repair="Provide a clear reachable route and place the exit sensor on the route.",
            )
        )
    relations.extend(
        [
        PhysicsRelation(
            "hazard_motion",
            objects=["hazard"],
            fields=["gravity"],
            parameters={"recurring": True, "staggered": True},
            parameter_constraints={
                "hazard_speed_y": {"min_abs": 160, "max_abs": 520, "why": "visible but readable hazard motion"},
                "phase_gap_steps": {"min": 18, "why": "hazards should not all fall simultaneously"},
                "vertical_clearance": {"min": 160, "why": "falling hazards must visibly travel"},
            },
            probes=["semantic_dynamics", "trajectory_change"],
            repair_knobs=["hazard_spawn_positions", "hazard_speed_y", "phase_gap_steps", "hazard_sensor", "clearance_gap"],
            repair="Create dynamic hazards with recurring/staggered motion and enough clearance for visible travel.",
        ),
        PhysicsRelation(
            "avoid_hazard",
            actors=["agent"],
            objects=["hazard"],
            parameters={"seconds": _duration_seconds(text) or 8},
            parameter_constraints={
                "safe_lane_count": {"min": 1, "why": "survival must be possible for a simple controller"},
                "hazard_density": {"max": 6, "why": "avoid unavoidable dense crossfire"},
            },
            probes=["survival_duration", "min_distance_to_hazard"],
            repair_knobs=["duration_seconds", "hazard_count", "hazard_speed", "spawn_spacing", "safe_lane_center"],
            repair="Leave fair dodge lanes and avoid unavoidable starting contact.",
        ),
        ]
    )
    return relations


def _navigation_avoidance_relations(text: str) -> list[PhysicsRelation]:
    target = "exit" if _mentions(text, "exit", "escape") else "goal"
    avoid_role = "chaser" if _mentions(text, "chase", "chaser", "angry", "enemy", "enemies") else "hazard"
    return [
        PhysicsRelation(
            "agent_reaches_region",
            actors=["agent"],
            regions=[target],
            probes=["path_reachability", "agent_moves_toward_target", "check_objective"],
            repair="Provide a clear reachable route and place the exit/goal sensor on the route.",
        ),
        PhysicsRelation(
            "avoid_contact_until_goal",
            actors=["agent"],
            objects=[avoid_role],
            parameters={"terminal_condition": "failure_on_contact", "paired_with_goal": target},
            parameter_constraints={
                "enemy_spawn_distance_from_start": {"min": 240, "why": "avoid immediate unfair contact"},
                "enemy_spawn_distance_from_goal": {"min": 220, "why": "avoid turning the goal sensor into a guarded blocker"},
                "enemy_max_speed_relative_to_agent": {"max": 0.7, "why": "simple route-following oracle can still prove reachability"},
                "critical_path_occupancy": {"allowed": "brief_crossing_only", "why": "avoidance is pressure, not a permanent path blocker"},
            },
            probes=["spawn_separation", "collision_clearance", "route_reachability"],
            repair_knobs=["enemy_spawns", "start_clearance", "corridor_width", "goal_center", "agent_strength"],
            repair=(
                "Treat avoidance as a safety constraint paired with the reach-goal objective. "
                "Do not convert this into a standalone survival timer unless the prompt gives a duration."
            ),
        ),
    ]


def _touch_collect_relations(text: str, world_context: dict[str, Any]) -> list[PhysicsRelation]:
    base = _collectible_base_name(text)
    count = _requested_target_count(text)
    target_names = [f"{base}_{index}" for index in range(1, count + 1)] if count > 1 else [base]
    return [
        PhysicsRelation(
            "agent_reaches_or_intercepts_object",
            actors=["agent"],
            objects=[target],
            fields=[str(world_context.get("gravity") or world_context.get("gravity_model") or "")],
            parameter_constraints={
                "target_sensor": {
                    "required": False,
                    "why": "touchable collectibles may be dynamic bodies or non-blocking sensors, but must be concrete registered names",
                },
                "object_to_region_distance": {
                    "max": 420,
                    "why": "multi-target collection should stage each target within reachable/interceptable range",
                },
            },
            probes=["concrete_target", "agent_moves_toward_target", "contact_or_proximity"],
            repair_knobs=[f"{target}_start" for target in target_names] + ["agent_start", "agent_strength"],
            repair="Place finite concrete targets in reachable/interceptable positions; do not use placeholder aggregate names like 'targets'.",
        )
        for target in target_names
    ]


def _collectible_base_name(text: str) -> str:
    if _mentions(text, "crystal", "crystals"):
        return "crystal"
    if _mentions(text, "asteroid", "asteroids", "rock", "rocks"):
        return "rock"
    if _mentions(text, "gem", "gems"):
        return "gem"
    if _mentions(text, "artifact", "artifacts"):
        return "artifact"
    if _mentions(text, "coin", "coins"):
        return "coin"
    return "target"


def _requested_target_count(text: str) -> int:
    word_counts = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
    }
    for word, count in word_counts.items():
        if re.search(rf"\b{word}\b", text):
            return count
    match = re.search(r"\b([2-6])\b", text)
    if match:
        return int(match.group(1))
    return 3 if _mentions(text, "multiple", "collect", "all") else 1


def _generic_push_relations(text: str) -> list[PhysicsRelation]:
    obj = _object_name(text, default="push_object", choices=("box", "crate", "rock", "ball", "block"))
    return [
        PhysicsRelation(
            "contact_push",
            actors=["agent"],
            objects=[obj],
            regions=["target_region"],
            parameter_constraints={
                "object_mass": {"min": 0.6, "max": 2.0, "why": "movable without trivializing contact"},
                "object_friction": {"min": 0.05, "max": 0.45, "why": "push should be visibly possible"},
                "alignment_angle_degrees": {"max": 35, "why": "push controller needs clean contact geometry"},
            },
            probes=["agent_object_contact", "contact_impulse_observed", "object_moves_toward_region"],
            repair_knobs=["agent_start", "object_start", "target_region_center", "object_mass", "object_friction", "agent_strength"],
            repair="Align agent behind object, support the object, and tune mass/friction for visible displacement.",
        ),
        PhysicsRelation(
            "object_enters_region",
            objects=[obj],
            regions=["target_region"],
            probes=["object_region_affordance", "object_inside_region"],
            repair="Use a non-blocking target sensor and keep it close enough or mechanically guided.",
        ),
    ]


def _project_relations_to_subgoals(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    subgoals: list[dict[str, Any]] = []
    support_boundary_by_object: dict[str, str] = {}
    for relation in relations:
        relation_type = str(relation.get("type") or "")
        objects = [str(item) for item in relation.get("objects") or []]
        regions = [str(item) for item in relation.get("regions") or []]
        if relation_type == "support_boundary_exit" and objects and regions:
            support_boundary_by_object[objects[0]] = regions[0]

    for relation in relations:
        relation_type = str(relation.get("type") or "")
        objects = [str(item) for item in relation.get("objects") or []]
        regions = [str(item) for item in relation.get("regions") or []]
        obj = objects[0] if objects else None
        region = regions[0] if regions else None
        if relation_type in {"agent_approaches_object", "agent_reaches_or_intercepts_object"} and obj:
            subgoals.append({"kind": "agent_touch_object", "target": obj})
        elif relation_type == "agent_reaches_region" and region:
            subgoals.append({"kind": "agent_reach_region", "target": region})
        elif relation_type == "contact_push" and obj and region:
            subgoals.append({"kind": "move_object_to_region", "object": obj, "region": region, "interaction": "push_contact"})
        elif relation_type == "object_enters_region" and obj and region:
            subgoals.append({"kind": "move_object_to_region", "object": obj, "region": region, "interaction": "push_contact"})
        elif relation_type in {"ballistic_arc_to_region", "impulse_transfer_to_region"} and obj and region:
            subgoals.append({"kind": "ballistic_object_to_region", "object": obj, "region": region, "interaction": "impulse_transfer"})
        elif relation_type == "projectile_impact_transfer" and obj and region:
            subgoals.append(
                {
                    "kind": "ballistic_object_to_region",
                    "object": obj,
                    "region": region,
                    "interaction": "agent_fired_projectile_impact",
                    "impact_required": True,
                }
            )
        elif relation_type == "support_boundary_exit" and obj and region:
            continue
        elif relation_type == "freefall_after_support_exit" and obj:
            boundary = support_boundary_by_object.get(obj, "exit_boundary")
            subgoals.append(
                {
                    "kind": "support_exit_freefall",
                    "object": obj,
                    "boundary": boundary,
                    "region": region or "open_air_drop_zone",
                    "min_downward_velocity": 40,
                    "min_fall_distance": 90,
                }
            )
        elif relation_type == "field_force_transfer" and obj:
            field = (relation.get("fields") or ["force_zone"])[0]
            subgoals.append({"kind": "field_force_interaction", "object": obj, "field": field, "target": region or "target_zone"})
        elif relation_type == "lever_launch":
            subgoals.append({"kind": "lever_launch", "plank": "seesaw_plank", "weight": "heavy_ball", "agent": "agent", "target": region or "goal"})
        elif relation_type == "hazard_motion":
            continue
        elif relation_type == "avoid_contact_until_goal":
            continue
        elif relation_type == "avoid_hazard":
            params = relation.get("parameters") if isinstance(relation.get("parameters"), dict) else {}
            subgoals.append({"kind": "survive_duration", "seconds": int(params.get("seconds") or 8), "avoid_role": "hazard"})
        elif relation_type == "mechanism_activation":
            trigger = regions[0] if regions else "pressure_plate"
            subgoals.append({"kind": "activate_mechanism", "trigger": trigger, "effect": "gate_open", "mechanism": "gate_mechanism"})
    return _dedupe_subgoals(subgoals)


def _physical_parameter_profile(
    relations: list[dict[str, Any]],
    text: str,
    world_context: dict[str, Any],
) -> dict[str, Any]:
    """Summarize the physical-property assumptions needed by the graph."""

    gravity = str(world_context.get("gravity") or world_context.get("gravity_model") or "").lower()
    if not gravity:
        gravity = "zero_g" if _mentions(text, "zero gravity", "zero-g", "space", "asteroid") else "normal"
    profile: dict[str, Any] = {
        "gravity": {
            "model": gravity,
            "explicit": _mentions(text, "zero gravity", "zero-g", "no gravity", "normal gravity", "earth gravity", "low gravity", "high gravity"),
            "repair_knob": "EnvConfig.gravity",
        },
        "agent": {
            "strength_policy": "scale with max scene mass and gravity",
            "friction_policy": "high enough for contact tasks; unrestricted thrust only in zero_g",
            "repair_knobs": ["agent_strength", "agent_radius", "agent_start"],
        },
        "objects": {},
        "surfaces": {},
        "fields": {},
        "clearance": {},
    }
    for relation in relations:
        relation_type = str(relation.get("type") or "")
        objects = [str(item) for item in relation.get("objects") or []]
        constraints = relation.get("parameter_constraints") if isinstance(relation.get("parameter_constraints"), dict) else {}
        for object_name in objects:
            object_profile = profile["objects"].setdefault(object_name, {"relations": [], "constraints": {}, "repair_knobs": []})
            object_profile["relations"].append(relation_type)
            object_profile["constraints"].update(
                {
                    key: value
                    for key, value in constraints.items()
                    if key.startswith("object_") or key.startswith("affected_object")
                }
            )
            object_profile["repair_knobs"] = sorted(
                set(object_profile["repair_knobs"]) | {knob for knob in relation.get("repair_knobs") or [] if "object" in str(knob) or "mass" in str(knob) or "friction" in str(knob)}
            )
        if relation_type in {"contact_push", "support_boundary_exit", "freefall_after_support_exit"}:
            profile["surfaces"].setdefault(
                relation_type,
                {
                    "constraints": {
                        key: value
                        for key, value in constraints.items()
                        if "support" in key or "friction" in key or "clearance" in key
                    },
                    "repair_knobs": [knob for knob in relation.get("repair_knobs") or [] if any(token in str(knob) for token in ("support", "edge", "drop", "boundary"))],
                },
            )
        if relation_type in {"ballistic_arc_to_region", "impulse_transfer_to_region"}:
            profile["clearance"].setdefault(
                "ballistic_path",
                {
                    "constraints": {
                        key: value
                        for key, value in constraints.items()
                        if key in {"clearance_margin", "object_to_region_distance", "target_sensor", "gravity"}
                    },
                    "repair_knobs": [knob for knob in relation.get("repair_knobs") or [] if any(token in str(knob) for token in ("barrier", "target", "object", "elasticity", "strength"))],
                },
            )
        if relation_type == "field_force_transfer":
            profile["fields"].setdefault(
                "force_zone",
                {
                    "constraints": constraints,
                    "repair_knobs": relation.get("repair_knobs") or [],
                },
            )
    return profile


def _repair_knob_manifest(
    relations: list[dict[str, Any]],
    physical_parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return deduplicated repair knobs with the relation they tune."""

    manifest: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for relation in relations:
        relation_type = str(relation.get("type") or "unknown_relation")
        for knob in relation.get("repair_knobs") or []:
            knob_name = str(knob)
            key = (relation_type, knob_name)
            if key in seen:
                continue
            seen.add(key)
            manifest.append(
                {
                    "knob": knob_name,
                    "relation": relation_type,
                    "scope": _knob_scope(knob_name),
                    "policy": "safe numeric/layout tuning only; preserve prompt semantics",
                }
            )
    if not manifest:
        manifest.append(
            {
                "knob": "layout",
                "relation": "custom_physics",
                "scope": "layout",
                "policy": "add explicit JSON-like layout values for repairable physical parameters",
            }
        )
    return manifest


def _knob_scope(knob: str) -> str:
    text = knob.lower()
    if any(token in text for token in ("mass", "friction", "elasticity", "damping", "strength", "gravity")):
        return "physical_parameter"
    if any(token in text for token in ("start", "center", "x", "y", "height", "size", "width", "edge", "barrier")):
        return "geometry"
    if any(token in text for token in ("speed", "phase", "duration", "count")):
        return "temporal_dynamics"
    return "layout"


def _is_ballistic_prompt(text: str, objective_type: str) -> bool:
    return objective_type in {"strike_object_to_region", "ballistic_object_to_region"} or (
        _mentions(text, "throw", "toss", "hurl", "lob", "basketball", "hoop")
        or (_mentions(text, "kick", "score", "shoot", "strike", "slam") and not _mentions(text, "spaceship", "laser", "bullet", "missile", "turret", "enemy shot"))
    )


def _is_agent_projectile_impact_prompt(text: str) -> bool:
    agent_fires_at_object = bool(
        re.search(
            r"\b(agent|player|person|character|robot)\b.{0,35}\b(shoots|shoot|fires|fire|firing)\b.{0,45}\b(bullet|projectile|missile|laser|blaster)\b",
            text,
        )
    ) and _mentions(
        text,
        "knock",
        "knocks",
        "knock over",
        "topple",
        "break",
        "hit",
        "hits",
        "pile",
        "stack",
        "tower",
        "target",
        "squares",
        "blocks",
    )
    incoming_or_enemy_fire = _mentions(
        text,
        "enemy",
        "enemies",
        "turret",
        "turrets",
        "incoming",
        "avoid",
        "avoiding",
        "dodge",
        "dodging",
        "survive",
        "survival",
        "at the agent",
        "toward the agent",
    )
    return agent_fires_at_object and not incoming_or_enemy_fire


def _is_boundary_exit_prompt(text: str, objective_type: str) -> bool:
    return objective_type in {"manipulation_to_boundary_exit", "boundary_exit"} or (
        _mentions(text, "off the", "off of", "cliff", "mountain", "edge", "ledge")
        and _mentions(text, "push", "shove", "roll", "slide")
    )


def _is_hazard_or_survival_prompt(text: str, objective_type: str) -> bool:
    if objective_type in {"survival", "hazard_escape", "avoidance"}:
        return True
    agent_fires_at_object = bool(
        re.search(
            r"\b(agent|player|person|character|robot)\b.{0,35}\b(shoots|shoot|fires|fire|firing)\b.{0,45}\b(bullet|projectile|missile|laser|blaster)\b",
            text,
        )
    ) and _mentions(
        text,
        "knock",
        "knocks",
        "knock over",
        "topple",
        "break",
        "hit",
        "hits",
        "pile",
        "stack",
        "tower",
        "target",
    )
    incoming_or_enemy_fire = _mentions(
        text,
        "enemy",
        "enemies",
        "turret",
        "turrets",
        "incoming",
        "avoid",
        "avoiding",
        "dodge",
        "dodging",
        "survive",
        "survival",
        "at the agent",
        "toward the agent",
    )
    if agent_fires_at_object and not incoming_or_enemy_fire:
        return False
    if _is_navigation_avoidance_prompt(text, objective_type):
        return False
    return _mentions(
        text,
        "falling",
        "raining",
        "dropping",
        "meteors",
        "fireballs",
        "rocks rain",
        "survive",
        "enemy shot",
        "enemy shots",
        "projectile",
        "laser",
        "missile",
        "bullet",
    ) or bool(re.search(r"\bfor\s+\d+\s*(seconds|second|secs|sec|s)\b", text))


def _is_navigation_avoidance_prompt(text: str, objective_type: str) -> bool:
    has_goal = objective_type in {"reach_exit_avoid_contact", "navigation_goal"} or _mentions(
        text,
        "exit",
        "escape",
        "reach goal",
        "get to",
        "navigate to",
    )
    has_contact_avoidance = _mentions(
        text,
        "without getting touched",
        "without being touched",
        "avoid getting touched",
        "avoid being touched",
        "avoid contact",
        "not get caught",
        "without getting caught",
    ) or (_mentions(text, "avoid") and _mentions(text, "enemy", "enemies", "angry", "chaser", "chasers"))
    has_explicit_survival_duration = bool(
        re.search(r"\b(for|survive)\s+\d+\s*(seconds|second|secs|sec|s)\b", text)
    )
    return has_goal and has_contact_avoidance and not has_explicit_survival_duration


def _duration_seconds(text: str) -> int | None:
    match = re.search(r"\bfor\s+(\d+)\s*(seconds|second|secs|sec|s)\b", text)
    if not match:
        return None
    try:
        return max(1, min(60, int(match.group(1))))
    except ValueError:
        return None


def _mentions(text: str, *phrases: str) -> bool:
    lowered = text.lower()
    for phrase in phrases:
        escaped = re.escape(phrase.lower())
        suffix = r"(?:s|es|ed|ing)?" if re.fullmatch(r"[a-z]+", phrase.lower()) else ""
        if re.search(rf"(?<![a-z0-9]){escaped}{suffix}(?![a-z0-9])", lowered):
            return True
    return False


def _mentions_whole_phrase(text: str, *phrases: str) -> bool:
    """Match mechanics keywords without substring false positives like gate/navigate."""

    for phrase in phrases:
        if " " in phrase:
            if phrase in text:
                return True
        elif re.search(rf"\b{re.escape(phrase)}\b", text):
            return True
    return False


def _object_name(text: str, *, default: str, choices: tuple[str, ...]) -> str:
    for choice in choices:
        if choice in text:
            if choice == "box":
                return "blue_box" if "blue" in text else "box"
            if choice == "ball" and "basketball" in text:
                return "basketball"
            if choice == "ball" and "soccer" in text:
                return "soccer_ball"
            return choice
    return default


def _dict_field(payload: dict[str, Any] | None, key: str) -> dict[str, Any]:
    value = payload.get(key) if isinstance(payload, dict) else None
    return value if isinstance(value, dict) else {}


def _json_text(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    return json.dumps(payload, sort_keys=True, default=str)


def _dedupe_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for relation in relations:
        key = json.dumps(relation, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        output.append(relation)
    return output


def _dedupe_subgoals(subgoals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for subgoal in subgoals:
        key = (
            str(subgoal.get("kind") or ""),
            str(subgoal.get("object") or subgoal.get("target") or ""),
            str(subgoal.get("region") or subgoal.get("boundary") or ""),
            str(subgoal.get("mechanism") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(subgoal)
    return output


def _short_text(value: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."
