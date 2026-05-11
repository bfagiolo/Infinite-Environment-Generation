"""Route-aware spatial planning for Harness Alpha generation.

The planner does not generate final environment code. It creates an abstract,
validator-friendly spatial skeleton so the Architect builds around a known
possible route or mechanism instead of discovering blocked geometry after the
fact.
"""

from __future__ import annotations

import json
import re
from typing import Any


WORLD_WIDTH = 960
WORLD_HEIGHT = 640
CELL_SIZE = 64
GRID_ORIGIN = (80, 80)


def infer_layout_plan(
    prompt: str,
    *,
    simulation_brief: dict[str, Any] | None = None,
    gameplay_profile: dict[str, Any] | None = None,
    physics_relations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Infer a route-aware layout contract from prompt-level meaning."""

    prompt_text = prompt.lower()
    text = _combined_text(prompt, simulation_brief, gameplay_profile, physics_relations)
    world_context = _world_context(simulation_brief, gameplay_profile)
    layout_type = _choose_layout_type(prompt_text, text, world_context, physics_relations)
    plan = {
        "version": "route_layout_plan.v1",
        "source": "deterministic_layout_planner",
        "layout_type": layout_type,
        "world_bounds": {"width": WORLD_WIDTH, "height": WORLD_HEIGHT},
        "route_first": True,
        "world_context": world_context,
        "route_model": _route_model_for(layout_type),
        "implementation_contract": [
            "Build the critical route or mechanism skeleton before adding decorative or challenge geometry.",
            "Never place solid static blockers on protected zones, goal sensors, trigger sensors, or the critical path corridor.",
            "Decorative props may overlap protected zones only when sensor=True or non-colliding/visual-only.",
            "Place start, objective targets, and final goal from this layout plan unless a repair explicitly moves them.",
            "Run the mental BFS/A* check on the abstract plan before writing final Pymunk objects.",
        ],
    }
    plan.update(_plan_for_type(layout_type, text, world_context, physics_relations))
    plan["validation_checks"] = _validation_checks_for(layout_type, plan)
    plan["repair_knobs"] = _repair_knobs_for(layout_type)
    return plan


def format_layout_plan_guidance(plan: dict[str, Any] | None) -> str:
    """Render layout-plan instructions for Architect prompts."""

    if not plan:
        return "- No route-aware layout plan was provided."
    return (
        "- Treat this as the spatial source of truth for start/goal/route/mechanism placement.\n"
        "- Define `self.layout_plan = {...}` in the mandatory registry and export it in ground truth.\n"
        "- Use `self.layout` numeric knobs to implement the plan: start positions, route waypoints, sensor centers, supports, blockers, and hazard lanes.\n"
        "- Build route/mechanism skeleton first, then add obstacles, enemies, hazards, and props around protected zones.\n"
        "- If the prompt allows creativity, vary decoration, branches, visual props, and secondary obstacles, not the guaranteed route contract.\n\n"
        f"{json.dumps(plan, indent=2, sort_keys=True)}"
    )


def _choose_layout_type(
    prompt_text: str,
    combined_text: str,
    world_context: dict[str, Any],
    physics_relations: dict[str, Any] | None,
) -> str:
    relation_types = {
        str(relation.get("type") or "")
        for relation in (physics_relations or {}).get("relations", [])
        if isinstance(relation, dict)
    }
    if "mechanism_activation" in relation_types or _mentions(prompt_text, "pressure plate", "button", "switch", "gate", "door"):
        return "push_gate_corridor"
    if "ballistic_arc_to_region" in relation_types or _mentions(prompt_text, "hoop", "lob", "arc", "over wall", "over a wall", "over barrier"):
        return "ballistic_arc_lane"
    if "support_boundary_exit" in relation_types or _mentions(prompt_text, "off cliff", "off the cliff", "off edge", "off mountain", "ledge"):
        return "support_exit_lane"
    if _mentions(prompt_text, "maze", "pacman"):
        return "maze_escape_route"
    if _mentions(prompt_text, "falling", "raining", "dropping", "lava", "fireball", "fireballs", "meteor"):
        return "hazard_escape_route"
    if _mentions(prompt_text, "escape", "exit") and _mentions(prompt_text, "chase", "chaser", "angry", "enemy", "enemies"):
        return "maze_escape_route"
    if _mentions(prompt_text, "collect", "touch", "crystal", "asteroid", "floating rock", "gem"):
        return "open_collection_route"
    perspective = str(world_context.get("world_perspective") or "")
    if perspective in {"top_down_or_flat_floor", "zero_g_freeflight"}:
        return "open_navigation_route"
    return "ground_navigation_route"


def _plan_for_type(
    layout_type: str,
    text: str,
    world_context: dict[str, Any],
    physics_relations: dict[str, Any] | None,
) -> dict[str, Any]:
    if layout_type == "maze_escape_route":
        return _maze_escape_plan(text)
    if layout_type == "hazard_escape_route":
        return _hazard_escape_plan(text, world_context)
    if layout_type == "push_gate_corridor":
        return _push_gate_plan(text)
    if layout_type == "ballistic_arc_lane":
        return _ballistic_plan(text)
    if layout_type == "support_exit_lane":
        return _support_exit_plan(text)
    if layout_type == "open_collection_route":
        return _collection_plan(text, world_context)
    return _navigation_plan(text, world_context)


def _maze_escape_plan(text: str) -> dict[str, Any]:
    path_cells = [
        [1, 1],
        [2, 1],
        [3, 1],
        [4, 1],
        [5, 1],
        [6, 1],
        [7, 1],
        [8, 1],
        [9, 1],
        [10, 1],
        [11, 1],
        [11, 2],
        [11, 3],
        [10, 3],
        [9, 3],
        [8, 3],
        [7, 3],
        [7, 4],
        [7, 5],
        [8, 5],
        [9, 5],
        [10, 5],
        [11, 5],
        [11, 6],
    ]
    branch_cells = [
        [[2, 3], [2, 4], [2, 5], [2, 6]],
        [[4, 3], [4, 4], [4, 5]],
        [[6, 2], [6, 3], [6, 4], [6, 5]],
        [[9, 2], [10, 2], [10, 3]],
    ]
    enemy_count = _extract_count(text, default=3)
    enemy_spawn_cells = [[2, 6], [4, 5], [10, 2], [6, 5]][: max(1, min(enemy_count, 4))]
    return {
        "objective_critical_structure": "agent reaches exit while avoiding chasers",
        "grid": {"cell_size": CELL_SIZE, "origin": list(GRID_ORIGIN), "cols": 13, "rows": 8},
        "start": {
            "name": "agent",
            "cell": [1, 1],
            "position": _cell_pos([1, 1]),
            "protected_radius": 96,
        },
        "goal": {
            "name": "exit_zone",
            "cell": [11, 6],
            "position": _cell_pos([11, 6]),
            "size": [132, 132],
            "sensor": True,
        },
        "critical_path_cells": path_cells,
        "critical_path_points": [_cell_pos(cell) for cell in path_cells],
        "branch_cells": branch_cells,
        "corridor_width": 82,
        "wall_thickness": 18,
        "enemy_spawns": [
            {
                "name": f"angry_agent_{index + 1}",
                "cell": cell,
                "position": _cell_pos(cell),
                "min_distance_from_start": 240,
                "min_distance_from_goal": 220,
                "max_speed_relative_to_agent": 0.65,
                "not_on_initial_start_zone": True,
                "not_on_critical_path": True,
                "not_inside_goal_sensor": True,
            }
            for index, cell in enumerate(enemy_spawn_cells)
        ],
        "protected_zones": [
            {"name": "start_clearance", "center": _cell_pos([1, 1]), "radius": 96},
            {"name": "exit_sensor", "center": _cell_pos([11, 6]), "size": [132, 132]},
            {"name": "critical_path_corridor", "cells": path_cells, "corridor_width": 82},
        ],
        "construction_rules": [
            "Carve the critical_path_cells first and make them physically connected.",
            "Rasterize maze walls only around cells not used by critical_path_cells or branch_cells.",
            "Every enemy spawn branch must connect back to the critical path through an open mouth; no chaser may start boxed in.",
            "Place chasers on branch cells or late-route cells, never directly on the start cell.",
            "Do not place chasers inside the final goal sensor or as guards on the last two critical path cells.",
            "For first-pass validation, chasers should be slower than the agent and start in side branches so the proof route remains possible.",
            "If semantic_requirements say all chasers pursue, each chaser must move visibly during passive simulation; do not trap one behind sealed walls.",
            "Chasers may patrol or pursue, but a passive start buffer must exist so validation is fair.",
            "The exit must be a large non-blocking sensor overlapping the final route cell.",
        ],
    }


def _hazard_escape_plan(text: str, world_context: dict[str, Any]) -> dict[str, Any]:
    support_waypoints = [[120, 112], [280, 135], [440, 165], [600, 200], [740, 235], [850, 260]]
    goal_center = [850, 365]
    gravity_model = str(world_context.get("gravity_model") or "normal")
    support_segments = [
        {"name": f"route_support_{index}", "a": support_waypoints[index], "b": support_waypoints[index + 1], "radius": 18}
        for index in range(len(support_waypoints) - 1)
    ]
    return {
        "objective_critical_structure": "agent follows a readable safe route to an exit while hazards cross the route",
        "start": {"name": "agent", "position": support_waypoints[0], "protected_radius": 90},
        "goal": {"name": "exit_zone", "position": goal_center, "size": [280, 260], "sensor": True},
        "critical_path_points": support_waypoints + [goal_center],
        "safe_corridor_width": 120,
        "support_plan": {
            "kind": "continuous_ramp_or_full_floor",
            "waypoints": support_waypoints,
            "segments": support_segments,
            "max_slope_degrees": 26,
            "support_radius": 18,
            "exit_overlap_rule": "The final support segment must end inside the bottom half of the exit_zone sensor so reaching the visible platform counts.",
            "rule": "Under normal gravity, create continuous ramp/stair support under every waypoint; under zero_g, leave open flight corridor.",
        },
        "hazard_lanes": [
            {"name": "hazard_lane_1", "x": 310, "spawn_y": 600, "bottom_y": 70, "phase": 0},
            {"name": "hazard_lane_2", "x": 500, "spawn_y": 600, "bottom_y": 70, "phase": 45},
            {"name": "hazard_lane_3", "x": 690, "spawn_y": 600, "bottom_y": 70, "phase": 90},
        ],
        "protected_zones": [
            {"name": "safe_corridor", "points": support_waypoints + [goal_center], "corridor_width": 120},
            {"name": "exit_sensor", "center": goal_center, "size": [280, 260]},
        ],
        "construction_rules": [
            "Make hazards recurring/staggered when the prompt implies ongoing falling/raining/dodging.",
            "Hazards should cross or threaten the route, not block it permanently.",
            "Under explicit zero gravity, keep world gravity zero and give hazards downward velocity or a hazard-only force zone.",
            "Do not place columns, ledges, or decorative lava props inside the safe corridor unless sensor=True.",
            "Implement the support_plan.segments as low-slope create_static_segment terrain named route_support_*; they are floors/ramps, not blockers.",
            "Do not add overhang_*, support_seg_*, ledge_*, or column_* solids between the final route waypoint and exit_zone. Decorative overhangs near the exit must be sensor=True or outside the safe corridor.",
            "Place exit_zone as a large non-blocking sensor overlapping the last route support; check_objective should become true when the agent reaches that visible sensor.",
            "Use self.touch_threshold >= 120 for this elevated exit, or make check_objective use the actual exit sensor half-size so visible overlap counts as success.",
            f"Use gravity model `{gravity_model}` consistently across EnvConfig, capability_profile, and supports.",
        ],
    }


def _push_gate_plan(text: str) -> dict[str, Any]:
    lane_y = 220
    return {
        "objective_critical_structure": "agent pushes movable object onto trigger, mechanism opens, agent reaches goal",
        "lane": {
            "y": lane_y,
            "agent_start": [130, lane_y + 32],
            "object_start": [275, lane_y + 32],
            "trigger_center": [430, lane_y + 30],
            "gate_center": [610, lane_y + 70],
            "goal_center": [790, lane_y + 45],
            "support_start_x": 70,
            "support_end_x": 890,
        },
        "protected_zones": [
            {"name": "push_lane", "from": [110, lane_y + 32], "to": [455, lane_y + 32], "corridor_width": 78},
            {"name": "post_gate_route", "from": [610, lane_y + 45], "to": [790, lane_y + 45], "corridor_width": 110},
            {"name": "trigger_sensor", "center": [430, lane_y + 30], "size": [110, 72]},
            {"name": "goal_sensor", "center": [790, lane_y + 45], "size": [150, 130]},
        ],
        "construction_rules": [
            "Use a straight, stable support lane for agent -> object -> trigger.",
            "The trigger must be sensor=True and not block the object.",
            "The closed gate may visually block the route, but its opened/passable state must unblock the post-gate route.",
            "Do not place rails, walls, or posts through the object circle/box or along the push contact surface.",
        ],
    }


def _ballistic_plan(text: str) -> dict[str, Any]:
    target_name = "hoop" if _mentions(text, "hoop", "basket", "basketball") else "goal_line"
    object_name = "basketball" if target_name == "hoop" else "ball"
    has_barrier = _mentions(text, "wall", "barrier", "defender", "over")
    return {
        "objective_critical_structure": "agent transfers impulse to object, object follows clear arc/shot corridor into sensor target",
        "shot_lane": {
            "agent_start": [150, 170],
            "object_start": [245, 170],
            "target_name": target_name,
            "target_center": [565 if has_barrier else 540, 315 if target_name == "hoop" else 185],
            "target_size": [150, 130] if target_name == "hoop" else [90, 160],
            "barrier_center": [390, 205] if has_barrier else None,
            "barrier_size": [34, 140] if has_barrier else None,
            "support_y": 130,
        },
        "protected_zones": [
            {"name": "contact_zone", "center": [245, 170], "radius": 95},
            {"name": "arc_corridor", "from": [245, 170], "to": [565 if has_barrier else 540, 315 if target_name == "hoop" else 185], "height_clearance": 175 if has_barrier else 90},
            {"name": "target_sensor", "center": [565 if has_barrier else 540, 315 if target_name == "hoop" else 185], "size": [150, 130] if target_name == "hoop" else [90, 160]},
        ],
        "construction_rules": [
            f"Register the dynamic object as `{object_name}` or keep objective/subgoal names consistent if renamed.",
            "The target must be a non-blocking sensor; rim/posts/backboards are visual or placed outside all feasible arcs.",
            "Keep the projectile light, low-friction, and close enough to the agent for visible impulse transfer.",
            "If a barrier is present, leave a clear arc corridor above it and make the barrier shorter than the planned apex.",
        ],
    }


def _support_exit_plan(text: str) -> dict[str, Any]:
    return {
        "objective_critical_structure": "agent pushes object across the real support boundary, then gravity makes it fall",
        "lane": {
            "agent_start": [145, 232],
            "object_start": [300, 232],
            "edge_x": 485,
            "boundary_center": [485, 224],
            "drop_zone_center": [600, 80],
            "support_y": 190,
            "support_start_x": 70,
            "support_end_x": 485,
        },
        "protected_zones": [
            {"name": "push_lane", "from": [130, 232], "to": [485, 232], "corridor_width": 80},
            {"name": "drop_column", "center": [580, 120], "size": [220, 280]},
            {"name": "edge_sensor", "center": [485, 224], "size": [72, 110]},
        ],
        "construction_rules": [
            "Support terrain ends at edge_x; do not create hidden floor beyond the edge.",
            "Use normal gravity unless the prompt explicitly overrides gravity.",
            "Keep the object passively stable before contact and unsupported after crossing edge_x.",
            "Do not place decorative cliffs or snow patches as solid blockers on the push lane.",
        ],
    }


def _collection_plan(text: str, world_context: dict[str, Any]) -> dict[str, Any]:
    target_count = max(3, min(_extract_count(text, default=4), 6))
    target_points = [[260, 170], [510, 130], [730, 250], [620, 465], [330, 440], [800, 500]][:target_count]
    route = [[140, 320], *target_points]
    return {
        "objective_critical_structure": "agent visits multiple reachable dynamic/static targets in an open field",
        "start": {"name": "agent", "position": route[0], "protected_radius": 90},
        "target_points": [
            {"name": f"target_{index + 1}", "position": point, "radius": 22, "sensor_radius": 55}
            for index, point in enumerate(target_points)
        ],
        "critical_path_points": route,
        "open_field_clearance": 95,
        "protected_zones": [
            {"name": "target_visit_route", "points": route, "corridor_width": 110},
            *[
                {"name": f"target_{index + 1}_sensor", "center": point, "radius": 55}
                for index, point in enumerate(target_points)
            ],
        ],
        "construction_rules": [
            "Keep targets separated enough for readable navigation but inside the playable bounds.",
            "Do not wall off individual targets unless a branch route remains open.",
            "Use zero-g/thrust movement when world_context implies floating or drifting.",
        ],
    }


def _navigation_plan(text: str, world_context: dict[str, Any]) -> dict[str, Any]:
    side_view = str(world_context.get("world_perspective") or "") == "side_view_platformer"
    waypoints = [[120, 120], [300, 170], [480, 245], [655, 340], [820, 480]] if side_view else [[120, 320], [320, 320], [540, 410], [820, 500]]
    return {
        "objective_critical_structure": "agent reaches a goal sensor through a guaranteed route",
        "start": {"name": "agent", "position": waypoints[0], "protected_radius": 90},
        "goal": {"name": "goal", "position": waypoints[-1], "size": [145, 145], "sensor": True},
        "critical_path_points": waypoints,
        "safe_corridor_width": 120,
        "support_plan": {
            "kind": "continuous_ramp_or_full_floor" if side_view else "open_floor_or_zero_g_space",
            "waypoints": waypoints,
        },
        "protected_zones": [
            {"name": "critical_route", "points": waypoints, "corridor_width": 120},
            {"name": "goal_sensor", "center": waypoints[-1], "size": [145, 145]},
        ],
        "construction_rules": [
            "Place the goal on the final reachable lane, not behind decorative blockers.",
            "If using normal gravity, provide continuous support under the whole route.",
            "If top-down or zero-g, keep the route open with no static blockers crossing the corridor.",
        ],
    }


def _validation_checks_for(layout_type: str, plan: dict[str, Any]) -> list[dict[str, Any]]:
    checks = [
        {
            "type": "protected_zone_clearance",
            "must_pass": True,
            "description": "No solid blocker may overlap protected route/sensor zones.",
        }
    ]
    if "critical_path_cells" in plan or "critical_path_points" in plan:
        checks.append(
            {
                "type": "route_reachability",
                "algorithm": "BFS/A* on planned cells or waypoint corridor",
                "from": "start",
                "to": "goal",
                "must_pass": True,
            }
        )
    if layout_type == "maze_escape_route":
        checks.extend(
            [
                {
                    "type": "enemy_spawn_fairness",
                    "must_pass": True,
                    "description": "Enemy/chaser spawns are outside start clearance and not inside the final goal sensor.",
                },
                {
                    "type": "route_before_chase",
                    "must_pass": True,
                    "description": "The maze has a route even before enemy/chaser motion is considered.",
                },
            ]
        )
    if layout_type in {"push_gate_corridor", "support_exit_lane"}:
        checks.append(
            {
                "type": "collinear_interaction_affordance",
                "must_pass": True,
                "description": "Agent, movable object, trigger/boundary are staged on a short aligned lane.",
            }
        )
    if layout_type == "ballistic_arc_lane":
        checks.append(
            {
                "type": "clear_arc_corridor",
                "must_pass": True,
                "description": "Projectile has an unobstructed corridor/apex between contact and target sensor.",
            }
        )
    return checks


def _repair_knobs_for(layout_type: str) -> list[str]:
    common = ["agent_start", "goal_center", "goal_size", "critical_path_points", "protected_zones"]
    knobs = {
        "maze_escape_route": common + ["corridor_width", "wall_thickness", "enemy_spawns", "branch_cells"],
        "hazard_escape_route": common + ["safe_corridor_width", "hazard_lanes", "support_plan"],
        "push_gate_corridor": common + ["lane.agent_start", "lane.object_start", "lane.trigger_center", "lane.gate_center", "lane.goal_center"],
        "ballistic_arc_lane": common + ["shot_lane.object_start", "shot_lane.target_center", "shot_lane.target_size", "shot_lane.barrier_center", "shot_lane.barrier_size"],
        "support_exit_lane": common + ["lane.agent_start", "lane.object_start", "lane.edge_x", "lane.drop_zone_center", "lane.support_end_x"],
        "open_collection_route": common + ["target_points", "open_field_clearance"],
    }
    return knobs.get(layout_type, common)


def _route_model_for(layout_type: str) -> str:
    return {
        "maze_escape_route": "grid_carve_then_wall_rasterize",
        "hazard_escape_route": "safe_corridor_then_crossing_hazards",
        "push_gate_corridor": "ordered_mechanism_lane",
        "ballistic_arc_lane": "contact_then_clear_trajectory_corridor",
        "support_exit_lane": "push_lane_then_support_boundary",
        "open_collection_route": "open_visit_route",
    }.get(layout_type, "waypoint_corridor")


def _cell_pos(cell: list[int]) -> list[int]:
    return [GRID_ORIGIN[0] + cell[0] * CELL_SIZE, GRID_ORIGIN[1] + cell[1] * CELL_SIZE]


def _world_context(
    simulation_brief: dict[str, Any] | None,
    gameplay_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    for source in (gameplay_profile, simulation_brief):
        if not isinstance(source, dict):
            continue
        context = source.get("world_context")
        if isinstance(context, dict):
            return dict(context)
    return {
        "world_perspective": "ground_lane_physics",
        "gravity_model": "normal",
        "movement_model": "ground_force",
        "support_assumption": "provide stable support where needed",
        "route_assumption": "stage goals on reachable lanes",
    }


def _combined_text(*parts: Any) -> str:
    chunks: list[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, str):
            chunks.append(part)
        else:
            chunks.append(json.dumps(part, sort_keys=True))
    return " ".join(chunks).lower()


def _mentions(text: str, *phrases: str) -> bool:
    lowered = text.lower()
    for phrase in phrases:
        escaped = re.escape(phrase.lower())
        suffix = r"(?:s|es|ed|ing)?" if re.fullmatch(r"[a-z]+", phrase.lower()) else ""
        if re.search(rf"(?<![a-z0-9]){escaped}{suffix}(?![a-z0-9])", lowered):
            return True
    return False


def _extract_count(text: str, *, default: int) -> int:
    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
    }
    match = re.search(r"\b([1-6])\b", text)
    if match:
        return int(match.group(1))
    for word, value in number_words.items():
        if re.search(rf"\b{word}\b", text):
            return value
    return default
