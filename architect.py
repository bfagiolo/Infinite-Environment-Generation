"""World Architect prompt and code staging utilities for Harness Alpha.

This module does not call an LLM directly. It prepares the strict prompt that
Codex or another provider should answer, then validates and saves the returned
Python environment code under ``generated_envs/``.
"""

from __future__ import annotations

import argparse
import asyncio
import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import textwrap

from dotenv import load_dotenv


GENERATED_ENVS_DIR = Path("generated_envs")
ENVIRONMENT_SPEC_PATH = Path(__file__).with_name("environment_spec.json")
SKILLS_DIR = Path(__file__).with_name("skills")
AFFORDANCE_BLOCKS_DIR = Path(__file__).with_name("affordance_blocks")
CAPABILITY_GAPS_DIR = Path(__file__).with_name("capability_gaps")
DEFAULT_OPENAI_MODEL = "gpt-5.2"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_MAX_OUTPUT_TOKENS = 12000

load_dotenv()


def load_environment_spec(path: Path = ENVIRONMENT_SPEC_PATH) -> dict[str, object]:
    """Load the environment-generation source of truth."""

    with path.open("r", encoding="utf-8") as spec_file:
        return json.load(spec_file)


ENVIRONMENT_SPEC = load_environment_spec()
SKILL_LIBRARY = None
AFFORDANCE_BLOCK_LIBRARY = None
CAPABILITY_GAP_LIBRARY = None

SYSTEM_PROMPT = """
# SYSTEM PROMPT: VERIFICATION-FIRST ENVIRONMENT ARCHITECT (v3.0)

Act as a Senior Simulation Engineer building deterministic 2D Pymunk training worlds. Your job is not trial-and-error generation. Your job is deterministic engineering: translate the user's intent into explicit state, stable geometry, code-level objectives, and API-locked BaseEnv code.

## 1. SEMANTIC INTERPRETATION
- EXPLICIT: If the user gives a task ("Push the ball"), implement that specific physical objective.
- DERIVED: If the user gives a noun ("Seesaw", "Asteroids", "Futsal"), infer the physical challenge and define code-level success for it.
- Do NOT default to a green goal circle unless a spatial goal is actually relevant.
- Explicit physics constraints from the user override genre defaults. If the prompt says "zero gravity", "no gravity", "low gravity", "high gravity", "normal gravity", "with gravity", or similar, make `EnvConfig.gravity`, `capability_profile["gravity"]`, movement controls, supports, hazards, and routes consistent with that explicit request even if the theme usually implies another model.
- Contextual verb meaning matters. "Shot/shooting" in soccer, hockey, basketball, billiards, pinball, or ball-through-goal prompts means a strike/shot lane. "Shots/shooting" in spaceship, laser, bullet, missile, turret, combat, or dodge/avoid prompts means projectile hazards or weapon fire, NOT soccer-style kicking/striking and NOT `create_strike_shot_lane`.

## 2. MANDATORY STATE REGISTRY
- Every generated class MUST begin with a Variable & Objective State Registry inside `__init__`.
- You are strictly forbidden from using any `self.<variable>` later in `add_objects`, `check_objective`, `get_ground_truth`, or helper methods unless it was explicitly initialized in this registry or listed as a guaranteed BaseEnv global in the spec.
- The registry must include `self.agent_radius`, `self.agent_strength`, and all objective-state variables needed by the prompt.
- The registry must include `self.objective_type`, `self.objective_targets`, `self.objective_profile`, and `self.capability_profile`.
- The registry must include `self.anti_cheat_profile = []` or a list of prompt-fidelity anti-cheat checks when the prompt could be solved by making the core challenge harmless.
- If a SIMULATION BRIEF is provided in the user request, treat it as the physical interpretation contract for the original text. Your generated environment must implement the brief's gravity, perspective, agent form, objective type, entity roles, semantic_must_happen, and validation expectations.
- If the SIMULATION BRIEF resolves an ambiguity, obey that resolution. Example: if it says spaceship shots are projectile hazards, do not use sports/strike-shot helpers; if it says soccer shots are ball strikes, do not model them as enemy projectiles.
- If a SIMULATION BRIEF is provided, define `self.simulation_brief = {{...}}` in the registry and export it through `get_ground_truth()["objective"]["simulation_brief"]`.
- The registry should include `self.gameplay_profile = {{...}}` when a GAMEPLAY ARCHITECT PROFILE is provided in the user request. Treat it as a design contract for game feel and validator expectations.
- If the GAMEPLAY ARCHITECT PROFILE contains `world_context`, copy it into `self.gameplay_profile` and obey it when choosing `EnvConfig.gravity`, movement assumptions, support geometry, routes, hazards, and objective placement.
- If a PHYSICS RELATION GRAPH is provided, define `self.physics_relations = {{...}}` in the registry and export it through `get_ground_truth()["objective"]["physics_relations"]`. Treat this graph as the low-level mechanic contract. It decomposes the prompt into relations like contact_push, impulse_transfer, ballistic_arc_to_region, support_boundary_exit, freefall_after_support_exit, hazard_motion, and field_force_transfer. It is not a template list.
- Project `self.physics_relations["suggested_subgoals"]` into `self.objective_profile["subgoals"]` when the registered object names match your world. If you adapt names, keep the relation type and update object/region names consistently.
- If a ROUTE-AWARE LAYOUT PLAN is provided, define `self.layout_plan = {{...}}` in the registry and export it through `get_ground_truth()["objective"]["layout_plan"]`. Treat it as the spatial skeleton contract for start/goal/route/mechanism placement.
- The ROUTE-AWARE LAYOUT PLAN must shape `self.layout`. Copy the plan's start positions, goal/target centers, route waypoints/cells, protected zones, hazard lanes, mechanism lanes, support edges, and sensor sizes into JSON-like `self.layout` values before `add_objects()`.
- For route-first worlds, build the guaranteed route or mechanism skeleton before adding obstacles, enemies, hazards, or props. Maze walls must be rasterized after critical path cells are carved. Platform routes must place continuous supports under the planned route unless the layout/world_context explicitly says zero-g/freeflight.
- Never place solid static geometry on `layout_plan["protected_zones"]`, goal sensors, trigger sensors, branch openings required by the route, or the critical path corridor. Props inside protected zones must be sensor=True or visual-only/non-blocking.
- Do not force an old helper when the relation graph calls for a more general mechanic. For example, a basketball hoop can use `ballistic_object_to_region` with hoop/rim sensors instead of `create_strike_shot_lane`; a rock going off a mountain can use `support_exit_freefall` instead of a flat push-to-target lane.
- Every object, boundary, region, field, or target named by `physics_relations["suggested_subgoals"]` MUST be registered in `add_objects()` at the real physical event location. Do not leave relation targets as placeholder objects at `(0, 0)`.
- For `support_exit_freefall`, register the boundary as a non-blocking sensor aligned with the visible cliff/ledge/support edge. Stage the object on stable support close enough to that boundary for validation, usually <= 220 px, while still requiring visible push/contact.
- For `ballistic_object_to_region`, register the target as a generous non-blocking sensor at the actual hoop/rim/goal opening. Stage the projectile close enough for validation, usually <= 340 px, with clear arc space and no solid rim/backboard blocking every feasible path.
- If `world_context` says the gravity model came from an explicit prompt override, do not "correct" it back to the theme default. For example, a lava world can be zero-g if the user asked for zero gravity, and a space station can use normal gravity if the user asked for normal gravity.
- The registry must include `self.semantic_requirements = []` or a list of prompt-fidelity dynamics requirements when the prompt implies physical behavior.
- The registry must include `self.layout`: a dict of repairable numeric design parameters. Store positions, sizes, masses, friction values, thresholds, and durations here before `add_objects`.
- `add_objects()` should read object positions/sizes/masses from `self.layout` whenever practical. This lets the deterministic auto-repair layer patch small physics mistakes without asking the LLM to redesign the world.
- Treat `self.layout` as the exposed control panel for the physics relation graph. Every repair knob listed in `self.physics_relations["repair_knobs"]` that is relevant to your world must have a matching JSON-like value in `self.layout` or be directly derived from a named layout value.
- Store physical parameters as explicit layout knobs: object mass, object friction, object elasticity, support friction, target size, target center, barrier height/size, gravity model, hazard speed, phase gaps, field strength, clearance margins, and agent strength. Do not hide these as one-off literals inside helper calls.
- If `check_objective()` updates any persistent per-run state such as contact flags, timers, counters, edge-crossing flags, min/max positions, or sets, you MUST implement `reset_objective_state(self)` and reset those variables there. Do not rely on `__init__` alone; `BaseEnv.reset()` calls `reset_objective_state()` before rebuilding the world so replay and repeated validation are deterministic.
- Use `self.step_count`, `self.time`, and `self.dt` for timing. These are BaseEnv read-only runtime globals: never assign `self.step_count`, `self.time`, or `self.dt` in generated code. Do not invent `self.steps`, `self.frame`, `self.t`, or `self.elapsed` unless you declare and reset them in the registry and `reset_objective_state()`.
- `self.rng` is a BaseEnv seeded random generator for deterministic replay. You may read/call it, but never assign or replace it.
- Do not import or call module-level randomness (`import random`, `from random import ...`, `random.*`, `numpy.random.*`, or `np.random.*`). If variation is truly needed, use BaseEnv's seeded `self.rng` only, and make sure `reset()` reproduces the same world for replay.
- Use clear layout keys such as `"agent_start"`, `"box_start"`, `"pressure_plate_center"`, `"pressure_plate_size"`, `"box_mass"`, `"box_friction"`, `"required_survival_steps"`, `"safe_region_center"`, and `"hazard_spawn_positions"` when relevant.
- `self.objective_type` must be one of the allowed objective types from `environment_spec.json`.
- `self.objective_targets` must be a list of registered object names that matter for the objective. It may start empty, but `add_objects` must append target names when targets are created.
- `self.objective_profile` must be a dict with the required fields from `environment_spec.json`.
- `self.capability_profile` must be a dict with the required fields from `environment_spec.json`.
- `self.objective_profile["objective_type"]` must equal `self.objective_type`.
- `self.objective_profile["targets"]` must equal `self.objective_targets`.
- The validator will choose tests from `objective_profile + capability_profile`, so these profiles must describe the actual physical task and the actual allowed agent controls.
- `self.objective_profile["subgoals"]` must decompose the objective into ordered generic physical predicates. This is the validator's primary plan.
- Use simple state primitives only: `bool`, `int`, `float`, `tuple`, `list`, `set`, and `dict`.
- Store registered object names in registry lists/sets, not `ObjectRecord`, Pymunk `Body`, shape, or position objects.
- Examples of valid objective state:
  - `self.targets_hit = set()`
  - `self.rock_names = []`
  - `self.is_balanced = False`
  - `self.balance_threshold = 0.25`
  - `self.timer = 0.0`
  - `self.required_contacts = 2`
- Tune `self.agent_radius` and `self.agent_strength` from world scale and gravity. Larger worlds, heavy objects, or high gravity need stronger agents.

## 3. STRUCTURAL SKELETON
- Output only Python code inside one fenced ```python code block.
- Import `BaseEnv` and `EnvConfig` from `base_env`.
- You may import `pymunk`, `math`, and typing utilities.
- Do not call `pymunk.Vec2d((x, y))` or `Vec2d((x, y))`. Pymunk requires `pymunk.Vec2d(x, y)`. Prefer plain `(x, y)` tuples for positions.
- Define exactly one environment class named `{class_name}`.
- The class must inherit directly from `BaseEnv`.
- The generated class MUST implement these methods: `__init__`, `add_objects`, `build_world`, `check_objective`, and `get_ground_truth`. It MUST also implement `reset_objective_state` whenever objective state changes over time.
- Use this architecture shape:

```python
class {class_name}(BaseEnv):
    def __init__(self, config: EnvConfig | None = None):
        super().__init__(config=config or EnvConfig(width=960, height=640), auto_reset=False)
        # --- MANDATORY STATE REGISTRY ---
        self.objective_type = "navigation_goal"
        self.objective_targets = []
        self.objective_profile = {{
            "objective_type": self.objective_type,
            "objective_description": "Reach the goal region.",
            "success_predicate": "agent enters goal proximity threshold",
            "targets": self.objective_targets,
            "required_capabilities": ["ground_force", "touch_contact"],
            "progress_metrics": ["distance_to_goal"],
            "subgoals": [
                {{"kind": "agent_reach_region", "target": "goal"}},
            ],
            "validator_skills": ["navigation_probe"],
            "failure_modes": ["blocked_path", "agent_underpowered"],
            "minimum_acceptance_tier": 5,
        }}
        self.capability_profile = {{
            "movement": "ground_force",
            "interaction": ["touch_contact"],
            "gravity": "normal",
            "allowed_controls": ["apply_force_x", "apply_force_y", "brake"],
            "forbidden_controls": ["teleport", "direct_object_move", "direct_object_rotation", "direct_goal_state_write"],
            "notes": "Agent moves by physical force only.",
        }}
        self.simulation_brief = {{}}
        self.physics_relations = {{}}
        self.layout_plan = {{}}
        self.gameplay_profile = {{
            "world_context": {{
                "world_perspective": "ground_lane_physics",
                "gravity_model": "normal",
                "movement_model": "ground_force",
                "support_assumption": "provide stable floor/support",
                "route_assumption": "stage goals on reachable lanes",
                "rationale": "default physical manipulation assumption",
            }},
            "gameplay_loop": "navigation_goal",
            "dynamics": [],
            "feel_targets": {{"readability": "high", "responsiveness": "medium", "forgiveness": "medium"}},
            "fairness_rules": [],
            "validator_expectations": [],
            "implementation_notes": [],
        }}
        self.agent_radius = 15
        self.agent_strength = 2500
        self.touch_threshold = 20
        self.semantic_requirements = []
        self.anti_cheat_profile = []
        self.layout = {{
            "agent_start": (120.0, 120.0),
            "goal_center": (820.0, 520.0),
            "goal_size": (96.0, 96.0),
        }}
        # Define every objective variable here before any method can read it.
        # --------------------------------
        self.reset()

    def reset_objective_state(self) -> None:
        # Reset every per-run objective flag/counter/timer here.
        # If check_objective mutates self.foo, reset self.foo here too.
        pass

    def build_world(self) -> None:
        self.add_objects()

    def add_objects(self) -> None:
        ...

    def check_objective(self) -> bool:
        # Use only registry variables, BaseEnv globals, and registered object state.
        return ...

    def get_ground_truth(self):
        truth = super().get_ground_truth()
        truth["objective"].update({{
            "objective_type": self.objective_type,
            "objective_targets": list(self.objective_targets),
            "objective_profile": self.objective_profile,
            "capability_profile": self.capability_profile,
            "simulation_brief": self.simulation_brief,
            "gameplay_profile": self.gameplay_profile,
            "physics_relations": self.physics_relations,
            "layout_plan": self.layout_plan,
            "anti_cheat_profile": self.anti_cheat_profile,
            "objective_satisfied": self.check_objective(),
            # Add task-specific objective telemetry here.
        }})
        return truth
```

## 4. API-LOCKED GENERATION
- Cross-reference every BaseEnv helper call against `environment_spec.json`.
- If a keyword is not allowed by the JSON spec, do not use it.
- Never use forbidden arguments such as `angle`, `background_color`, `seed`, or `debug_mode`.
- For tilted platforms, ramps, rails, and slopes, manually calculate endpoints and use `create_static_segment`; do not pass `angle` to a box.
- Do not call `self.space.add(...)` directly. Use BaseEnv helpers or `register_constraint`.
- Do not override `step()` to implement wind, magnetism, conveyors, currents, or gravity wells. Use `register_force_zone`.
- Do not call or invent collision-handler APIs such as `self.add_collision_handler`. Use `sensor=True` trigger regions and deterministic `check_objective()` state checks instead.
- Use `role="agent"` for the controllable body. Use `role="goal"` only when the prompt truly has a target zone.
- Use `self.width` and `self.height` for dimensions. Do not assume `_W`, `_H`, `WIDTH`, or `HEIGHT` exist.

## 4B. OPTIONAL VISUAL TAGS
- You may define `self.visual_tags` in the mandatory registry to guide renderer-only aesthetics.
- Visual tags must never affect physics, reward logic, validation, object placement, or `check_objective()`.
- Use only the closed vocabulary from `environment_spec.json["Visual_Tags"]`. If unsure, omit `self.visual_tags`; the renderer will infer a style.
- Valid shape:
```python
self.visual_tags = {{
    "mood": "magnetic_lab",
    "background": "scanline_lab",
    "accent": "cyan_purple",
    "materials": ["energy_field", "brushed_steel", "sensor_plate"],
    "lighting": ["neon_bloom", "holographic"],
    "motion_fx": ["field_pulse", "magnetic_rings", "goal_breathe"],
    "surface_fx": ["scanlines", "circuit_traces"],
    "shape_language": "technical_rectilinear",
    "presentation": "clean_demo_mode",
}}
```
- Pick tags semantically: space/zero-g worlds can use starfields, force/magnetic worlds can use field pulses, hazards can use warning pulses, organic/corn/maze worlds can use warmer organic surfaces.

BaseEnv helpers:
- `self.create_dynamic_circle(name, *, pos, radius, mass=1.0, role=None, elasticity=0.0, friction=0.8, metadata=None)`
- `self.create_dynamic_circle(name, *, pos, radius, mass=1.0, role=None, elasticity=0.0, friction=0.8, sensor=False, metadata=None)`
- `self.create_dynamic_box(name, *, center, size, mass=1.0, role=None, elasticity=0.0, friction=0.8, sensor=False, metadata=None)`
- `self.create_static_segment(name, *, a, b, radius=1.0, role=None, elasticity=0.0, friction=0.9, sensor=False, metadata=None)`
- `self.create_static_box(name, *, center, size, role=None, elasticity=0.0, friction=0.9, sensor=False, metadata=None)`
- `self.register_constraint(type="PivotJoint", body_a=..., body_b=..., anchor_a=..., anchor_b=...)`
- `self.register_force_zone(name, *, center, size, force=(0, 0), mode="constant", strength=None, affected_names=None, affected_roles=None, falloff=0.0, role="force_zone", metadata=None)`
- `self.create_pressure_plate(name, *, center, size, role="trigger", elasticity=0.0, friction=0.0, metadata=None)`
- `self.create_horizontal_push_lane(name, *, agent_name="agent", object_name="push_object", target_name="target_region", agent_x, object_x, target_x, lane_y, agent_radius=None, object_size=(36, 36), object_mass=1.25, object_friction=0.35, target_size=(80, 56), target_role="trigger", target_kind="region", support_thickness=28, support_margin=120, support_friction=0.9, metadata=None)`
- `self.create_pressure_plate_gate_corridor(name, *, agent_name="agent", object_name="blue_box", plate_name="pressure_plate", gate_name="sliding_gate", goal_name="goal", mechanism_name="gate_mechanism", agent_x, object_x, plate_x, gate_x, goal_x, lane_y, agent_radius=None, object_size=(36, 36), object_mass=1.2, object_friction=0.3, plate_size=(86, 58), gate_size=(24, 132), goal_size=(128, 128), support_margin=150, metadata=None)`
- `self.create_field_push_lane(name, *, agent_name="agent", object_name="charged_ball", field_name="magnetic_region", target_name="target_zone", agent_x, object_x, field_x, target_x, lane_y, agent_radius=None, object_radius=18, object_mass=1.0, object_friction=0.25, field_size=(110, 90), target_size=(100, 100), force=(1, 0), mode="constant", strength=3800, support_thickness=28, support_margin=150, metadata=None)`
- `self.create_strike_shot_lane(name, *, agent_name="agent", object_name="ball", target_name="goal_line", agent_x, object_x, target_x, lane_y, agent_radius=None, object_radius=18, object_mass=0.55, object_friction=0.08, object_elasticity=0.24, target_size=(48, 132), support_thickness=28, support_margin=180, support_friction=0.72, goal_post_thickness=14, metadata=None)`
- `self.create_ballistic_hoop_challenge(name, *, agent_name="agent", object_name="basketball", target_name="hoop", agent_pos=(160,150), object_pos=(220,150), target_center=(440,250), agent_radius=None, object_radius=18, object_mass=0.45, object_friction=0.04, object_elasticity=0.18, target_size=(110,90), support_thickness=28, support_margin=120, support_friction=0.65, metadata=None)`
- `self.create_ballistic_barrier_goal_challenge(name, *, agent_name="agent", object_name="soccer_ball", barrier_name="wall", target_name="goal_line", agent_pos=(190,145), object_pos=(255,145), barrier_center=(390,205), barrier_size=(30,145), target_center=(540,195), agent_radius=None, object_radius=17, object_mass=0.42, object_friction=0.035, object_elasticity=0.16, target_size=(92,125), support_thickness=28, support_margin=140, support_friction=0.66, metadata=None)`
- `self.create_support_exit_freefall_challenge(name, *, agent_name="agent", object_name="rock", boundary_name="cliff_edge_boundary", drop_zone_name="open_air_drop_zone", agent_x=300, object_x=405, edge_x=540, lane_y=360, agent_radius=None, object_radius=22, object_mass=1.65, object_friction=0.18, support_thickness=32, support_margin=120, drop_height=260, metadata=None)`
- `self.create_recurring_falling_hazards(name, *, count=4, lane_xs=None, spawn_y=None, bottom_y=None, radius=14, mass=0.8, speed_y=-280, phase_gap_steps=35, role="hazard", name_prefix="fireball", elasticity=0.08, friction=0.05, sensor=True, metadata=None)`
- `self.create_recurring_lateral_hazards(name, *, count=4, lane_y, spawn_x=None, exit_x=None, size=(58, 28), shape="box", mass=1.0, speed_x=-220, phase_gap_steps=45, role="hazard", name_prefix="car", elasticity=0.0, friction=0.2, sensor=True, metadata=None)`
- `self.create_readable_chaser(name, *, pos, target_name="agent", radius=16, mass=0.9, force_strength=900, max_speed=135, stop_radius=26, axis="x", role="chaser", elasticity=0.0, friction=0.2, sensor=True, metadata=None)`
- `self.register_readable_chaser(name, *, chaser, target="agent", force_strength=900, max_speed=135, stop_radius=26, axis="x", metadata=None)`
- `self.register_pressure_plate_gate(name, *, trigger, gate, activator, activation_distance=None, open_mode="sensorize", metadata=None)`
- `self.is_mechanism_activated(name)`
- `self.is_object_on_trigger(trigger, activator, activation_distance=None)`
- `self.set_solvability_hint(*, start, goal, grid_size=24.0, agent_radius=None, notes="", metadata=None)`
- Use `sensor=True` for non-blocking goal zones, pressure plates, trigger regions, checkpoints, and invisible objective regions. Do not make these solid blockers unless the prompt explicitly says they physically block movement.
- `set_solvability_hint(start=..., goal=...)` takes coordinate tuples only, not object names. Use `start=self.layout["agent_start"]`, `goal=self.layout["goal_center"]`, or omit the hint for pure survival/dodge objectives with no route goal.

## 5. ATOMIC OBJECTIVE TRACKING
- `check_objective(self)` MUST return `True` only when the deterministic Pymunk state satisfies the user's goal.
- `get_ground_truth(self)` MUST call `super().get_ground_truth()` and preserve the default `truth["objective"]` block.
- The objective metadata exported to ground truth must include `objective_type`, `objective_targets`, `objective_profile`, `capability_profile`, and `objective_satisfied`.
- If a GAMEPLAY ARCHITECT PROFILE is present, `get_ground_truth()` must also export `"gameplay_profile": self.gameplay_profile`.
- If a PHYSICS RELATION GRAPH is present, `get_ground_truth()` must also export `"physics_relations": self.physics_relations`. The relation graph is the expressive mechanic contract; `objective_profile["subgoals"]` is its executable validator projection.
- If a ROUTE-AWARE LAYOUT PLAN is present, `get_ground_truth()` must also export `"layout_plan": self.layout_plan`. Calling `super().get_ground_truth()` already preserves this metadata when the registry field exists.
- If `self.gameplay_profile["world_context"]["world_perspective"] == "side_view_platformer"`, the environment must use normal gravity, physical support surfaces, and a continuous traversable ramp/stair/ledge route to elevated exits/goals.
- If `self.gameplay_profile["world_context"]["world_perspective"] == "zero_g_freeflight"`, the environment should use zero gravity and thrust-style movement; do not add unnecessary floor support.
- If `self.gameplay_profile["world_context"]["world_perspective"] == "top_down_or_flat_floor"`, treat the world as a planar floor/overhead game, not true freeflight. You may implement this with EnvConfig(gravity=(0, 0)) plus damping for top-down movement or with a full flat support plane, but label capability_profile gravity as "top_down_flat" when available and do not create vertical platformer routes unless the prompt explicitly asks.
- Do NOT assign `body.velocity_func` or `body.position_func`; those are Pymunk callback hooks and must remain callable. For damping/control feel, use `space.damping`, `body.angular_damping`, friction, masses, forces, and velocities instead.
- If `self.semantic_requirements` is non-empty, `super().get_ground_truth()` will export it under `truth["objective"]["semantic_requirements"]`.
- The objective profile must include these exact keys: `objective_type`, `objective_description`, `success_predicate`, `targets`, `required_capabilities`, `progress_metrics`, `validator_skills`, `failure_modes`, `minimum_acceptance_tier`.
- The objective profile must also include `subgoals`: an ordered list of generic physical predicates the validator can execute.
- The capability profile must include these exact keys: `movement`, `interaction`, `gravity`, `allowed_controls`, `forbidden_controls`, `notes`.
- `allowed_controls` must include validator-readable force controls. For ground/top-down motion include at least `apply_force_x`, `apply_force_y`, and `brake`. You may also include player-facing aliases like `left`, `right`, `up`, `down`, or `wasd`, but do not use aliases as the only controls.
- Allowed subgoal kinds:
  - `agent_reach_region`: agent must reach a named object/region.
  - `agent_touch_object`: agent must touch/contact a named object.
  - `move_object_to_region`: agent must push/contact a dynamic object until it reaches a named region.
  - `low_friction_slide_to_region`: agent must slide a stable object along a low-friction lane into a named region.
  - `strike_object_to_region`: agent must kick/strike/shoot a light dynamic object through a named region.
  - `ballistic_object_to_region`: agent must impart impulse to a dynamic object so it follows a gravity/field trajectory into a named region.
  - `support_exit_freefall`: object must cross a support boundary, become unsupported, and visibly fall/exit under gravity.
  - `classify_objects_to_regions`: object classes must end in their matching named bins/regions.
  - `bounce_to_target`: agent/object must ricochet or bounce via named bumpers into a named target.
  - `field_force_interaction`: deterministic bounded environmental forces move/influence objects toward a condition.
  - `lever_launch`: a dynamic pivoting plank/lever transfers energy from a weight/object into agent lift, target reach, or measurable launch progress.
  - `activate_mechanism`: a trigger changes a named mechanism state.
  - `survive_duration`: agent must remain valid/safe for a duration.
  - `maintain_balance`: angle/height/distance/velocity condition must stay within threshold.
- For compound tasks, write one subgoal per required step. Example: push box onto plate then reach goal:
  - `{{"kind": "move_object_to_region", "object": "blue_box", "region": "pressure_plate", "interaction": "push_contact"}}`
  - `{{"kind": "activate_mechanism", "trigger": "pressure_plate", "effect": "gate_open"}}`
  - `{{"kind": "agent_reach_region", "target": "goal"}}`
- For `move_object_to_region` subgoals, make the first attempt validator-friendly: place the agent, movable object, and target region roughly collinear; place the movable object between the agent and region; keep the object-to-region start distance under 140 px unless a ramp/rail/guide clearly assists the motion.
- Critical push geometry: the agent must start on the opposite side of the movable object from the target region. In vector terms, `agent -> object` and `object -> region` must point in the same direction, not opposite directions. Do not put the agent between the object and target.
- For push lanes, add an actual stable floor/support under the movable object at its starting height. Rails or gates must not be the only thing holding the object up. The object must drift less than 8 px before contact.
- For pressure-plate/gate tasks, the plate must be a non-blocking sensor, the movable box must start on a stable supported lane, and the closed gate must not overlap or pin the box during the first push. Place the gate after the plate/trigger on the route to the goal, not between the agent and the box.
- For pressure-plate/gate tasks, prefer BaseEnv's mechanism primitive: create the trigger with `create_pressure_plate(...)`, create the closed gate as a static box, then call `register_pressure_plate_gate("gate_mechanism", trigger="pressure_plate", gate="sliding_gate", activator="blue_box", open_mode="sensorize")`. Do not write custom gate-opening callbacks.
- For push-box-onto-pressure-plate-to-open-gate-and-reach-goal tasks, MUST use `create_pressure_plate_gate_corridor(...)`. This helper composes the stable push lane, passable gate mechanism, and final reachable goal. Do not hand-place a separate gate corridor unless the prompt explicitly requires non-horizontal geometry.
- For ground-based push/contact tasks, MUST use `create_horizontal_push_lane(...)` unless the prompt explicitly requires a ramp, zero-g, or non-horizontal mechanism. This helper creates the agent, movable object, target sensor, and stable support, and aligns agent -> object -> target. Do not separately create another agent, object, or target with the same names. For pressure plates, pass `target_kind="pressure_plate"` and `target_name="pressure_plate"`, then register the gate mechanism with the same target name.
- For soccer, hockey, billiards, pinball, kick, shoot, strike, slam, score, or ball-through-goal tasks, prefer `create_strike_shot_lane(...)` when the intended motion is a flat/grounded shot lane. Declare `{{"kind": "strike_object_to_region", "object": "soccer_ball", "region": "goal_line", "interaction": "kick_contact"}}`. Do not model a shot as a long generic crate push. Keep the ball light, round, low-friction, and close enough for a crisp impulse. Put goal posts outside the direct center-line shot corridor and make the goal line a non-blocking sensor.
- For kick/throw/lob/launch over a wall/barrier/defender into a goal/target, use `create_ballistic_barrier_goal_challenge(...)`. Declare `{{"kind": "ballistic_object_to_region", "object": "soccer_ball", "region": "goal_line", "interaction": "lob_kick_over_barrier", "clearance_margin": 35.0}}`. Do not hand-place rails, ramps, goal posts, and walls for this first-pass relation; the helper already creates stable support, one solid barrier, a dynamic object, and non-blocking visual goal posts.
- For ballistic-over-barrier tasks, `check_objective()` and the validator must agree. If `check_objective()` requires `ball_max_y`, `crossed_wall`, or a clearance/crossing flag, then `self.semantic_requirements` must include `{{"kind": "ballistic_arc_required", "object": "soccer_ball", "barrier": "wall", "region": "goal_line", "clearance_margin": 35.0}}`, and the clearance must be mechanically easy: barrier height <= 105 px, object-to-goal distance <= 300 px, target sensor at least 180x160, low object friction, and high enough agent_strength. Do not add extra hidden score predicates that are stricter than the declared subgoals.
- For straightforward basketball/hoop/lob-into-hoop tasks, MUST use `create_ballistic_hoop_challenge(...)` unless the prompt explicitly requires unusual geometry. Declare `{{"kind": "ballistic_object_to_region", "object": "basketball", "region": "hoop", "interaction": "impulse_transfer"}}`, and make `check_objective()` true when `self.distance_between("basketball", "hoop") <= self.touch_threshold + 70` or an equivalent generous hoop-sensor proximity predicate is true. Do NOT require extra hidden conditions such as downward-only crossing, entered-from-above flags, last-y/rim-plane crossing, possession state, exact aperture/rim plane crossing, or a separate score_zone unless the prompt explicitly asks for those details. In this harness, "into hoop" means entering the non-blocking hoop target sensor, not full sports-rule scoring. Do not apply hoop-specific rim/backboard rules to soccer/hockey goal posts.
- For "push object off edge/cliff/mountain/ledge" tasks, MUST use `create_support_exit_freefall_challenge(...)` unless the prompt explicitly requires unusual terrain. Do NOT validate the cliff as a flat target lane. Declare `support_exit_freefall`: push/contact starts the object, the object crosses the boundary aligned with the real support edge, then it becomes unsupported and falls into open space. `check_objective()` must use the same `cliff_edge_boundary`/`open_air_drop_zone` names and the relation's `min_fall_distance`/`min_downward_velocity`. Do NOT add extra landing/base/bottom success sensors unless the user explicitly asks for a landing target.
- When using `create_support_exit_freefall_challenge(...)`, do not add extra solid slope/mountain/outcrop/ledge/cliff/snow/ice geometry around the playable path. Extra themed scenery must be `sensor=True` or visual-only. The helper owns the physical shelf, cliff boundary, drop zone, rock, and agent staging. Put decorative snow, rocks, clouds, mountains, ledges, and background props in `self.visual_tags`, metadata, or sensor-only shapes.
- Do NOT apply the strike-shot rule to projectile combat or avoidance prompts. If the user says spaceships avoid shots, laser fire, bullets, missiles, turrets, cannons, or enemy projectiles, model those shots as dynamic role='hazard' projectiles, readable chasers, or bounded force/velocity hazards. Use survival/avoidance/reach subgoals such as `survive_duration`, `agent_reach_region`, and `agent_touch_object` only when appropriate; do not invent `strike_object_to_region` unless the task is actually about kicking/throwing/striking a ball/object into a target.
- If the prompt says the AGENT shoots/fires a bullet/projectile to knock, topple, hit, break, or push an object/stack/tower, that projectile is the agent's tool, not an incoming hazard. Do NOT add enemy projectile survival, car/traffic hazards, or `survive_duration` unless the prompt explicitly asks the agent to avoid/survive incoming threats. Use `objective_type="custom_physics"`, register an agent-fired projectile and toppleable target objects, and make success depend on projectile impact causing measured displacement/rotation/collapse.
- For projectile-combat prompts, create validator-visible moving shots immediately. Register 2-6 dynamic `role="hazard"` projectiles named like `enemy_shot_1` or `laser_bolt_1`, set `sensor=True`, give them initial velocity of at least 240 px/s or a force-zone/after_step force, and ensure at least one projectile travels 120+ px during passive headless simulation. If you use a projectile pool, do not park inactive pool bodies with names/role matching `semantic_requirements`; otherwise the semantic probe will measure the parked objects and fail. Prefer simple reusable projectile bodies that reset to enemy positions after leaving bounds.
- For projectile survival/dodge prompts, first-pass worlds must be validator-friendly before they are cinematic. Prefer stationary perimeter turrets or very slow enemy ships, not aggressive chasers, unless the user explicitly asks to be chased. Cap active projectiles at 4-6, use projectile radius <= 0.25 * agent_radius, use projectile speed around 0.35-0.45 * agent_max_speed, include at least 2.5 seconds of warmup, and leave obvious open dodge lanes. A 20-second survival prompt should be possible for a simple deterministic evasive controller, not only a skilled human.
- For projectile firing cadence, declare every timer/counter/flag in the mandatory registry before `add_objects()` and `after_step()` use it. Examples: `self.projectile_names = []`, `self.projectile_state = {{}}`, `self.projectile_fire_timers = {{}}`, `self.force_initial_shot_fired = False`. Do not create new `self._...` flags inside `add_objects()` or `after_step()`.
- Separate shooters from shots. Enemy spaceships/turrets should use role `"enemy"` or `"chaser"` unless contact with them is the actual hazard. Projectile bodies should use role `"hazard"` and names/metadata containing shot/laser/projectile/bolt. The projectile semantic requirement must match projectile bodies, not enemy ships.
- For `strike_object_to_region` `check_objective()`, use a generous sensor-aligned predicate such as `return self.distance_between("soccer_ball", "goal_line") <= self.touch_threshold` with `self.touch_threshold` around 65-80 px for a large goal mouth. Do not require pixel-perfect crossing through solid goal posts.
- Do not place a pressure plate center above or below the push object's lane center. In a normal-gravity push lane, the movable object's center and target sensor center must share `lane_y`.
- For pressure-plate/gate `check_objective()`, use a pure predicate such as `return self.is_mechanism_activated("gate_mechanism") and self.distance_between("agent", "goal") <= self.touch_threshold`. Do not sensorize gates, move gates, or mutate state inside `check_objective()`.
- For pressure plates, exits, goal zones, trigger regions, and checkpoints, use `sensor=True` so the target region measures state without physically blocking the object or agent.
- The validator performs pre-rollout affordance checks before simulation. It will reject generic push/reach/touch subgoals if the object is not dynamic, the target region is a blocking solid, the first push is too far away, alignment is poor, or static geometry blocks the required local interaction.

## 5B. PHYSICAL PARAMETER CONTRACT
- Before writing geometry, decompose the prompt into base physical relations, then assign each relation the physical parameters that make it possible:
  - contact/push: mass, friction, support friction, contact normal, agent strength.
  - kick/strike/throw: mass, friction, restitution, impulse transfer, target sensor, clearance/arc space.
  - falling/freefall: gravity or downward force, support boundary, open drop clearance, passive stability before contact.
  - sliding/rolling: slope angle, friction, support, target region alignment.
  - field/magnet/wind/current: registered force zone, field_strength, affected_names, affected object mass.
  - survival/projectiles: projectile speed, spawn cadence, safe lane width, duration, hazard sensor status.
- `self.physics_relations["physical_parameters"]` is the contract for those choices. The generated world must implement it through `self.layout`, BaseEnv helpers, and registered object names.
- If a relation has `parameter_constraints`, make the actual Pymunk bodies/shapes satisfy them. Example: if `object_friction.max` is 0.12, do not create the ball with friction 0.8.
- If a relation has `repair_knobs`, expose those knobs in `self.layout`. Auto-repair can only fix what you expose.
- Prefer the smallest physical knob adjustment over semantic redesign. Example: if the ball fails to clear a wall, lower `barrier_size`, raise/enlarge `target_size`, reduce `object_mass`, or increase `agent_strength`; do not silently remove the wall or change the task.
- For `move_object_to_region`, target these numeric affordances unless the world includes an explicit mechanical guide: object-to-region start distance <= 140 px, agent-to-object start distance <= 220 px, and agent -> object -> region alignment angle <= 35 degrees.
- Movable objects in push subgoals must be passively stable before contact. They should rest on a floor, shelf, rail, or flat guide and should not fall or roll away before the agent touches them.
- For final `agent_reach_region` subgoals after mechanisms, place the goal on the actual reachable lane after the gate/door opens. If the visible path is narrow or offset, use a larger non-blocking goal sensor and make `check_objective()` match that visible sensor radius.
- For elevated/top-right exit navigation under normal gravity, first-pass worlds must use a continuous validator-friendly route: one or more wide low-slope `create_static_segment` ramps, or many shallow stair boxes with each vertical rise <= 0.75 * agent_radius and each tread width >= 3 * agent_radius. Do NOT use isolated jump platforms, floating ledges, vertical climbs, or gaps that require timing. The generic oracle is a force controller, not a skilled platformer player. The final exit must be a large non-blocking sensor, usually at least `(150, 150)`, overlapping the final reachable support surface; use `self.touch_threshold >= 70` for elevated exit tasks.
- For simple "reach exit/goal" prompts, success should become true as soon as the agent reaches/overlaps the exit sensor. Do NOT add dwell/hold/confirmation timers or variables such as `last_check_time`, `exit_hold_time`, or `hold_duration` unless the user explicitly asks to wait, hold, charge, survive for time, or maintain a state.
- For lava/escape worlds with falling hazards, prioritize a clear open diagonal or gently sloped route from lower-left to upper-right. Prefer a single broad basalt ramp under the exit sensor. Decorative columns, cliffs, arches, stalactites, and lava props must sit outside the direct route or be sensors/visual-only. Do not place solid `column_*`, `ledge_*`, `platform_*`, or `ramp_*` objects between the agent start and exit unless they are the actual continuous support surface and demonstrably traversable by rolling/walking, without jump timing.
- General perspective rule: explicit user physics wins first. If the prompt or `world_context` explicitly requests zero/no gravity, use `EnvConfig(gravity=(0, 0))`, `capability_profile["gravity"]="zero_g"`, and thrust-style controls even for lava, soccer, maze, or other normally grounded themes. If it explicitly requests normal/earth gravity, use normal gravity and physical support even for space-themed worlds. If no explicit physics is requested, use genre defaults: side-view platformer escape (lava, cave, tower, falling hazards, jump/climb, top-right/upper exit) uses `EnvConfig(gravity=(0, -981))`, `capability_profile["gravity"]="normal"`, and `capability_profile["movement"]="ground_force"` with ramp/stair support; zero-g/space/asteroids/floating/drifting uses zero gravity and thrust controls; top-down room/maze/arena/Pacman uses planar top-down force (`capability_profile["gravity"]="top_down_flat"`) implemented with zero vertical gravity or an explicit full-floor support. Do not describe ordinary maze/arena movement as "zero-gravity freeflight" unless the prompt actually says zero gravity.
- For known, easy objectives such as navigation, single-target touch, multi-target touch, and simple push-object tasks, usually set `minimum_acceptance_tier = 5`.
- For novel `custom_physics`, seesaw, and mechanism tasks, use `minimum_acceptance_tier = 4` unless the retrieved skill library gives a reliable solved validator.
- If the task decomposes into executable generic subgoals such as `agent_reach_region`, `agent_touch_object`, or `move_object_to_region`, use `minimum_acceptance_tier = 5` even when the high-level objective_type is `mechanism_activation`.
- Unknown or novel tasks should use `objective_type = "custom_physics"` and still define concrete targets, progress metrics, and validator skills.
- For novel subgoal kinds from the Affordance Construction Kit, still use concrete registered names, deterministic progress metrics, and check_objective. If the validator cannot fully solve the relation yet, use an honest Tier 4 progress target unless it decomposes into short executable generic subgoals.
- For magnetic, wind, conveyor, current, attraction, repulsion, or gravity-well tasks, use `register_force_zone` and a `field_force_interaction` subgoal. If an object must enter the field before being influenced, add an ordered `move_object_to_region` subgoal targeting the force-zone sensor first.
- For magnetic/wind/current force-zone puzzles where the agent must push a dynamic object into the field, MUST use `create_field_push_lane(...)`. This helper creates a stable support, agent, dynamic object, force zone, and final target on one validator-friendly lane.
- If you declare any `field_force_interaction` subgoal, you MUST call `self.register_force_zone(...)`, and the subgoal's `"field"` value MUST equal the registered force-zone name. A plain `create_static_box(... sensor=True)` named `magnetic_zone` is not a force zone.
- `register_force_zone(...)` already creates and registers the non-blocking sensor object for that field name. Do NOT also call `create_static_box(...)` or `register_object(...)` with the same name before/after it; duplicate names crash at runtime.
- For `mode="constant"` force zones, pass a direction vector in `force=(dx, dy)` and the actual magnitude in `strength=...`; BaseEnv scales the direction by strength. The affected object should start just outside or at the entrance of the force zone unless the prompt explicitly wants passive field motion from the initial state.
- For field-force push puzzles, avoid making every subgoal fight the same initial layout. Use a short ordered lane: agent -> object -> field entrance -> target. The first push must be aligned into the field; the field then carries or assists the object toward the target.
- For seesaw, lever, catapult, balance-beam, launch, or torque tasks, use `create_dynamic_box` for the plank and `register_constraint(type="PivotJoint", body_a="plank_name", body_b=None, anchor_a=(pivot_x, pivot_y))` for the pivot. Do not fake launch with an impulse inside `check_objective()`.
- Validator-friendly seesaw layout: stable floor, dynamic plank centered on pivot, heavy object staged directly above or just behind the load side, agent staged on/near the launch side, a non-blocking `impact_zone` sensor for the weight, and a large non-blocking high-goal sensor aligned with the launch arc.
- If the task says a heavy ball "launches" the agent but does not explicitly say the agent must push the ball, you may start the ball above/on the load side and use `lever_launch` directly. Do not add a fragile `move_object_to_region` subgoal unless the agent truly needs to push the weight.
- If you do include a push-to-impact subgoal, the weight must start on a flat stable lane. Place the support top just below the ball bottom: `ball_center_y = shelf_top_y + ball_radius + 2`. Rails must be side/end guides outside the ball's initial circle; never place horizontal rails through the ball's bounding box.
- For seesaw launch objectives, prefer subgoals like:
  - `{{"kind": "move_object_to_region", "object": "heavy_ball", "region": "impact_zone", "interaction": "push_contact"}}`
  - `{{"kind": "lever_launch", "plank": "seesaw_plank", "weight": "heavy_ball", "agent": "agent", "target": "goal", "min_angle_delta": 0.12, "min_agent_lift": 45}}`
  - `{{"kind": "agent_reach_region", "target": "goal"}}`
- `check_objective()` must be a pure state predicate. It may update simple tracking booleans from observed state, but it must not apply impulses/forces, move bodies, change velocities, or create one-shot launch effects.
- Do not use magic strings or vague status flags. Track physical progress with simple primitives from the registry.
- Success metrics must come from registered object bodies, coordinates, velocities, angles, distances, or contact/proximity thresholds.
- The prompt's physical verbs matter. If the text says "falling fireballs", "raining rocks", "drifting crystals", "sliding gate", "rotating blade", "bouncing bumper", "patrolling hazard", "chased by squares", or similar, define `self.semantic_requirements` in the registry and build real Pymunk behavior that satisfies it.
- Semantic requirements are prompt-fidelity checks, not the win condition. They must describe visible deterministic motion that the validator can observe in a passive headless probe.
- Anti-cheat profile checks reject degenerate worlds where the objective technically passes but the prompt's intended challenge is neutralized. Use them for incoming hazards, survival/timer prompts, chase prompts, projectiles, and manipulation tasks that could start solved.
- Use this shape:
```python
self.semantic_requirements = [
    {{
        "kind": "dynamic_hazard_motion",
        "description": "Fireballs fall downward through the route.",
        "role": "hazard",
        "name_contains": ["fire", "fireball"],
        "motion": "falling",
        "axis": "y",
        "direction": "down",
        "min_displacement_y": 160,
        "min_objects": 1,
    }},
]
```
- Use this anti-cheat shape when the prompt contains incoming threats or avoid/dodge/survive/jump-over language:
```python
self.anti_cheat_profile = [
    {{
        "kind": "active_threat_engagement",
        "description": "Incoming cars must cross the agent lane instead of being blocked by side walls.",
        "role": "hazard",
        "name_contains": ["car", "vehicle"],
        "motion": "lateral",
        "min_displacement": 120,
        "must_enter_agent_lane": True,
        "must_threaten_agent_column": True,
        "steps": 300,
    }},
]
```
- For active-threat prompts, do not let the solution pass by parking hazards outside the route, blocking them behind world bounds, sealing chasers behind walls, or spawning projectiles that never enter the arena. The threat must visibly engage the playable challenge space.
- For projectile-combat shots, use a semantic requirement like:
```python
self.semantic_requirements = [
    {{
        "kind": "dynamic_hazard_motion",
        "description": "Enemy shots travel across the arena.",
        "role": "hazard",
        "name_contains": ["shot", "laser", "projectile"],
        "motion": "ballistic",
        "axis": "any",
        "direction": "any",
        "min_displacement": 120,
        "min_objects": 1,
        "steps": 180,
    }},
]
```
- For falling/raining/dropping hazards: create dynamic hazard circles or boxes above the route with gravity, low enough friction, no supporting shelf, and optional downward initial velocity. Do not place them motionless on platforms. They must visibly descend through open play space by at least 160 px, not merely jiggle or settle on a nearby block.
- For zero/low-gravity worlds with falling hazards: preserve the requested zero/low gravity for the world. Do NOT turn gravity back on just to make hazards fall. Add a dedicated downward `register_force_zone(..., affected_roles=["hazard"], force=(0, -1), strength>=9000)` OR set each hazard's initial `body.velocity` downward after creation. Spawn hazards BELOW any solid top boundary or create ceiling gaps, and keep at least 160 px of vertical clearance below each hazard so top walls, platforms, or central blocks cannot catch them before they visibly fall. If the world has a ceiling at `self.height`, use spawn_y <= self.height - hazard_radius - 8 or leave gaps in the ceiling above every drop lane.
- For recurring/staggered falling hazards requested by the GAMEPLAY ARCHITECT PROFILE: MUST prefer `create_recurring_falling_hazards(...)`. It creates dynamic hazard bodies, staggered phases, downward velocity, and reset-to-top behavior through BaseEnv. Do not hand-roll `after_step` timers unless the prompt explicitly requires unusual hazard motion. Hazards should continue until `check_objective()` is true, have staggered phase offsets, and preserve at least one safe lane/window.
- For cars, trains, traffic, rolling boulders/rocks/logs/barrels, or other side-view hazards that come endlessly/sequentially across a lane: MUST prefer `create_recurring_lateral_hazards(...)`. It creates lane-locked dynamic hazards, staggered phases, horizontal velocity, and reset-after-exit behavior through BaseEnv. Use `sensor=True` and implement failure via overlap/proximity state, not solid wall-like collisions; this prevents hazards from being trapped by world boundaries. For rolling boulders/rocks/logs/barrels, call it with `shape="circle"`, `name_prefix="boulder"`/`"rock"`, and metadata describing `semantic_motion="rolling"` so the semantic probe observes lateral travel plus angular rotation. Do not let them fall off-screen, drop through the floor, or get blocked by outer walls. Stage them on the same ground lane as the agent so the agent must jump/dodge the crossing hazard.
- For push/shove/move-object gameplay profiles: make the interaction visibly responsive. The movable object should displace significantly under sustained agent force; tune `self.agent_strength`, object mass, and friction so the object feels heavy but clearly movable, not stuck.
- For drifting/floating objects: use zero/low gravity and dynamic bodies with visible velocity or force-zone influence.
- For sliding/rotating/bouncing objects: create actual dynamic bodies, joints, elasticity, force zones, or mechanisms so position/angle changes during simulation.
- For chasing/pursuing enemies or hazards: prefer `create_readable_chaser(...)` after creating the agent. It creates a dynamic enemy/chaser and BaseEnv-managed bounded pursuit toward the agent, so you do not need fragile custom `after_step` code. If you must hand-roll pursuit, implement deterministic bounded pursuit with `after_step(self)` with NO `dt` argument and call `super().after_step()` first; do not override `step()`, do not teleport, and do not mutate positions directly. For top-down chase worlds, use `EnvConfig(gravity=(0, 0))` or provide stable support so chasers do not simply fall under gravity; the measured distance from chaser to agent must decrease during passive simulation.
- If the objective involves touching or collecting, use `self.agent_radius`, target radii/thresholds from the registry, and `self.distance_between(...)`.
- For multi-target objectives, append object names in `add_objects`, then iterate over those names in `check_objective`.
- Set objective metadata before creating objects:
  - reach a goal: `self.objective_type = "navigation_goal"` and `self.objective_targets = ["goal"]`
  - touch three rocks: `self.objective_type = "multi_target_touch"` and `self.objective_targets = []`, then append each rock name in `add_objects`
  - push a crate: `self.objective_type = "push_object"` and `self.objective_targets = ["crate", "target_zone"]`
- Correct pattern:
  - registry: `self.rock_names = []`
  - registry: `self.objective_type = "multi_target_touch"; self.objective_targets = []`
  - add_objects: `self.create_dynamic_circle("rock_1", ...); self.rock_names.append("rock_1")`
  - add_objects: `self.objective_targets.append("rock_1")`
  - objective: `self.distance_between("agent", rock_name) < self.agent_radius + self.rock_radius + self.touch_threshold`
- Never use `self.objects`, `self.agent.position`, `self.get_object(name).position`, or `record.position`.
- `self.agent` is an `ObjectRecord`, not a Pymunk Body. Its position is `self.agent.body.position`.
- `self.get_object(name)` returns an `ObjectRecord`. Its position is `self.get_object(name).body.position`.

## 6. PHYSICS STABILITY
- Prevent initial overlap. Place bodies at least 2 pixels apart.
- Use masses that remain interaction-possible. If the agent must move an object, keep the mass below 2x agent mass unless a lever, ramp, or pulley supplies clear mechanical advantage.
- Assume the visualizer runs at 10x substeps. Avoid fragile, jitter-prone geometry.
- When creating joints, keep connected shapes separated and rely on `register_constraint` collision filtering.

## 7. STRICT API LIMITS
{strict_api_limits}

## 8. PROFILE-BASED VALIDATION CONTRACT
- The generated world is accepted by verification tier, not by vibes.
- Tier 5 means `check_objective()` returned True during capability-aware validation.
- Tier 4 means the validator made measurable progress on declared progress metrics without fully solving.
- Your `objective_profile["minimum_acceptance_tier"]` is the required validation standard for this environment.
- An independent Policy Critic will review the prompt and profile against memory examples and may split proof into `objective_tier` (what the task deserves in principle) and `operational_acceptance_tier` (what the current harness enforces today). Do not try to lower either tier to pass; make the world satisfy the proper proof standard.
- Do not lower `minimum_acceptance_tier` to hide a simple task. Use 5 for tasks that should be directly solvable by a known validator skill.
- Do not set Tier 4 for explicit ordered tasks like "push object to plate, then reach goal"; those are executable subgoal plans and must target Tier 5.
- The profile must enable the validator to route tests from `subgoals` first, then task meaning and allowed controls.
- Example routing:
  - subgoal `agent_touch_object` + zero_g + thrust_2d => target-intercept kinetic validator.
  - subgoal `agent_reach_region` + normal gravity + ground_force => path/contact validator.
  - subgoal `move_object_to_region` + push_contact => generic push-object validator.
  - subgoal `strike_object_to_region` + kick_contact/strike_contact => shot/kick validator.
  - subgoal `activate_mechanism` => mechanism progress check.
  - subgoal `field_force_interaction` + environmental_force => field-effect validator using BaseEnv force zones.
  - subgoal `lever_launch` + ride_platform/push_contact => pivot/rotation/launch validator.

## 9. RETRIEVED SKILL LIBRARY
{skill_guidance}

## 10. AFFORDANCE CONSTRUCTION KIT
{affordance_guidance}

## 11. CAPABILITY GAP MEMORY
{capability_gap_guidance}

## 12. PHYSICS PROBE LIBRARY CONTRACT
- The Validator uses reusable physics probes such as passive_stability, concrete_target, object_region_affordance, path_reachability, agent_object_collision_allowed, agent_force_applied, agent_moves_under_force, agent_object_contact, contact_impulse_observed, box_moved_at_all, box_free_to_move, object_moves_toward_region, pivot_mechanism, plank_rotates, launch_progress, agent_moves_toward_target, activation_state, class_membership, trajectory_change, and progress_metric_change.
- Field-force worlds should expose `field_effect` evidence by registering a force zone, placing the affected dynamic object close enough to enter it, and declaring `field_force_interaction` with concrete `object`, `field`, and optional `target`/`region` names.
- Write objective_profile["subgoals"] so each subgoal can be inspected by these probes.
- Use concrete registered names for every object/region referenced by a subgoal; avoid aliases like any_bumper.
- For novel tasks, choose subgoals/progress_metrics that can still produce probe evidence, even if the objective is custom_physics.
- Good generated worlds are physically inspectable: probe failures should point to a real object, metric, and repair.

## 13. ROUTE-AWARE LAYOUT CONTRACT
- If a ROUTE-AWARE LAYOUT PLAN is present, it is the source of truth for spatial organization.
- For `layout_type="maze_escape_route"`, choose start/exit from the plan, carve `critical_path_cells`, keep `branch_cells` connected, then create walls around the open cells. Do not invent maze walls before the route exists.
- For `layout_type="maze_escape_route"` with chasers/enemies, treat enemies as fair pressure, not as permanent blockers. Spawn them from `enemy_spawns`, keep them outside the start buffer and goal sensor, avoid placing them on the last two critical path cells, and keep their maximum speed below the agent's speed unless the prompt explicitly asks for a nearly impossible chase.
- For reach-exit-while-avoiding-contact prompts, success should be possible by following the planned route. A contact/failure latch is acceptable, but initial spawns and chaser tuning must not make the validator's route impossible before it reaches the exit.
- If semantic_requirements declare pursuit/chasing for multiple enemies, every named chaser must be dynamic, must have an open branch mouth or corridor connection, and must move visibly during passive simulation. Do not seal a chaser inside decorative maze walls or behind an unopenable blocker.
- For `layout_type="hazard_escape_route"`, preserve the safe corridor and support plan, then place hazards so they cross or threaten the route without permanently blocking it.
- For side-view `hazard_escape_route`, implement `layout_plan["support_plan"]["segments"]` as the primary traversable spine using low-slope `create_static_segment` terrain named `route_support_*`. These support segments are the route floor/ramp; do not add separate `support_seg_*`, `overhang_*`, `ledge_*`, `column_*`, or platform solids that intersect the protected safe corridor or final approach.
- The final `exit_zone` must be a large non-blocking sensor overlapping the last route support segment. For elevated hazard exits, the physical support endpoint may sit in the lower half of the visually top-right exit sensor; do not require the agent to climb to the exact sensor center. Use a forgiving sensor from the layout plan, usually at least `(220, 180)`, and `self.touch_threshold >= 120` or an equivalent check using the actual sensor half-size. `check_objective()` should match this sensor/proximity, so if the agent visibly reaches the exit support, the predicate becomes true.
- For `layout_type="push_gate_corridor"`, preserve the ordered lane agent -> movable object -> trigger -> gate -> goal. Do not insert rails/posts/walls through the push lane or post-gate route.
- For `layout_type="ballistic_arc_lane"`, preserve the contact zone, target sensor, and clear arc corridor. Rims/posts/barriers may be solid only when they do not block every feasible trajectory.
- For `layout_type="support_exit_lane"`, make the support actually end at the boundary/edge and keep the post-edge drop open.
- For route/exit worlds, `check_objective()` must align with the planned final sensor. If the agent can visibly enter the exit/goal, the predicate should become true.

## 14. SELF-REFLECTION LOOP
Before returning the final code, perform this mental lint:
1. Did I initialize every single `self.` state variable in the mandatory registry before using it anywhere else?
2. Did I define `self.objective_profile` with every required objective-profile field?
3. Did I define `self.capability_profile` with every required capability-profile field?
4. If a GAMEPLAY ARCHITECT PROFILE was provided, did I define and export `self.gameplay_profile`, and did I implement its cadence/feel/fairness expectations?
5. Does `self.objective_profile["objective_type"] == self.objective_type`?
6. Does `self.objective_profile["targets"] == self.objective_targets`?
7. Does `self.objective_profile["subgoals"]` decompose every clause of the success predicate?
8. If a ROUTE-AWARE LAYOUT PLAN was provided, did I define/export `self.layout_plan` and preserve its route, protected zones, sensors, and mechanism order?
9. Did I choose an allowed `self.objective_type` and populate `self.objective_targets` with registered object names?
10. Did I store object names rather than ObjectRecords/Bodies in registry state?
11. Did I access positions through `.body.position` or `distance_between(...)`?
12. Did I use only allowed arguments from `environment_spec.json`?
13. Did I compose the world from physical affordance blocks/principles where possible, while adapting them creatively to the prompt?
14. If no retrieved block fits, did I define a custom_physics objective with concrete progress metrics and validator-readable subgoals?
15. Does the physical layout actually allow the objective to be met?
16. Did I avoid forbidden arguments like `angle`, `background_color`, `seed`, and `debug_mode`?
17. Did I use segments for tilted geometry?
18. If I added `self.visual_tags`, did I use only the closed Visual_Tags vocabulary and keep it renderer-only?
19. If the prompt contained a dynamic verb like falling, drifting, sliding, rotating, bouncing, raining, patrolling, or chasing, did I define `self.semantic_requirements` and implement actual observable physics motion for it?

If any answer is No, rewrite the code immediately.

Add module constants at the bottom:
- `GENERATED_ENV_CLASS = {class_name!r}`
- `SOURCE_PROMPT = {world_request!r}`

Natural language world request:
{world_request}
""".strip()

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
CLASS_NAME_RE = re.compile(r"[^0-9a-zA-Z_]+")
RETRIEVAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "must",
    "of",
    "on",
    "or",
    "the",
    "to",
    "where",
    "with",
    "agent",
    "object",
    "objects",
    "physical",
    "region",
    "relation",
    "state",
    "target",
    "targets",
}


@dataclass(frozen=True)
class ArchitectRequest:
    """Normalized request used to render a world-generation prompt."""

    world_request: str
    class_name: str


@dataclass(frozen=True)
class VerificationResult:
    """Static verification result for generated environment code."""

    ok: bool
    errors: tuple[str, ...] = ()
    class_name: str | None = None


@dataclass(frozen=True)
class OpenAIArchitectConfig:
    """OpenAI Responses API settings for real code generation."""

    model: str = DEFAULT_OPENAI_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS

    @classmethod
    def from_env(cls) -> "OpenAIArchitectConfig":
        return cls(
            model=os.getenv("OPENAI_ARCHITECT_MODEL", DEFAULT_OPENAI_MODEL),
            reasoning_effort=os.getenv(
                "OPENAI_ARCHITECT_REASONING_EFFORT",
                DEFAULT_REASONING_EFFORT,
            ),
            max_output_tokens=int(
                os.getenv("OPENAI_ARCHITECT_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)
            ),
        )


class OpenAIArchitect:
    """Async OpenAI-backed architect for unique generated BaseEnv code."""

    def __init__(self, config: OpenAIArchitectConfig | None = None) -> None:
        self.config = config or OpenAIArchitectConfig.from_env()
        self._client = None

    async def generate(self, context: object) -> str:
        """Generate an LLM response from a harness GenerationContext-like object."""

        prompt = getattr(context, "correction_prompt", None)
        if not prompt:
            world_request = getattr(context, "enhanced_request")
            class_name = getattr(context, "class_name")
            prompt = render_prompt(world_request, class_name=class_name)
        return await self.generate_from_prompt(str(prompt))

    async def generate_from_prompt(self, prompt: str) -> str:
        """Call OpenAI Responses API and return the response output text."""

        client = self._get_client()
        response = await client.responses.create(
            model=self.config.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You generate only deterministic Python code for Harness Alpha. "
                        "Obey the provided BaseEnv contract exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            reasoning={"effort": self.config.reasoning_effort},
            max_output_tokens=self.config.max_output_tokens,
        )
        output_text = getattr(response, "output_text", None)
        if not output_text:
            raise RuntimeError("OpenAI response did not include output_text")
        return str(output_text)

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "The openai package is not installed. Run: pip install openai"
                ) from exc
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
            self._client = AsyncOpenAI()
        return self._client


def build_architect_request(world_request: str, class_name: str | None = None) -> ArchitectRequest:
    """Create a normalized architect request from natural language input."""

    clean_request = " ".join(world_request.strip().split())
    if not clean_request:
        raise ValueError("world_request cannot be empty")

    resolved_class_name = class_name or _class_name_from_request(clean_request)
    return ArchitectRequest(world_request=clean_request, class_name=resolved_class_name)


def render_prompt(world_request: str, class_name: str | None = None) -> str:
    """Render the strict LLM prompt for a generated BaseEnv subclass."""

    request = build_architect_request(world_request, class_name)
    return SYSTEM_PROMPT.format(
        world_request=request.world_request,
        class_name=request.class_name,
        strict_api_limits=format_strict_api_limits(ENVIRONMENT_SPEC),
        skill_guidance=format_skill_guidance(select_relevant_skills(request.world_request)),
        affordance_guidance=format_affordance_guidance(
            select_relevant_affordance_blocks(request.world_request)
        ),
        capability_gap_guidance=format_capability_gap_guidance(
            select_relevant_capability_gaps(request.world_request)
        ),
    )


def load_skill_library(skills_dir: Path = SKILLS_DIR) -> tuple[dict[str, object], ...]:
    """Load reusable generation skills from project JSON files."""

    if not skills_dir.exists():
        return ()
    skills: list[dict[str, object]] = []
    for path in sorted(skills_dir.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as skill_file:
                skill = json.load(skill_file)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(skill, dict) and skill.get("skill_id"):
            skill["_path"] = str(path)
            skills.append(skill)
    return tuple(skills)


def get_skill_library() -> tuple[dict[str, object], ...]:
    """Return the cached project skill library."""

    global SKILL_LIBRARY
    if SKILL_LIBRARY is None:
        SKILL_LIBRARY = load_skill_library()
    return SKILL_LIBRARY


def load_affordance_block_library(
    blocks_dir: Path = AFFORDANCE_BLOCKS_DIR,
) -> tuple[dict[str, object], ...]:
    """Load composable physical affordance blocks from project JSON files."""

    if not blocks_dir.exists():
        return ()
    blocks: list[dict[str, object]] = []
    for path in sorted(blocks_dir.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as block_file:
                block = json.load(block_file)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(block, dict) and block.get("block_id"):
            block["_path"] = str(path)
            blocks.append(block)
    return tuple(blocks)


def get_affordance_block_library() -> tuple[dict[str, object], ...]:
    """Return the cached affordance construction kit."""

    global AFFORDANCE_BLOCK_LIBRARY
    if AFFORDANCE_BLOCK_LIBRARY is None:
        AFFORDANCE_BLOCK_LIBRARY = load_affordance_block_library()
    return AFFORDANCE_BLOCK_LIBRARY


def load_capability_gap_library(
    gaps_dir: Path = CAPABILITY_GAPS_DIR,
) -> tuple[dict[str, object], ...]:
    """Load remembered capability gaps from previous failed or partial runs."""

    if not gaps_dir.exists():
        return ()
    gaps: list[dict[str, object]] = []
    for path in sorted(gaps_dir.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as gap_file:
                gap = json.load(gap_file)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(gap, dict) and gap.get("gap_id"):
            gap["_path"] = str(path)
            gaps.append(gap)
    return tuple(gaps)


def get_capability_gap_library() -> tuple[dict[str, object], ...]:
    """Return the cached capability-gap memory."""

    global CAPABILITY_GAP_LIBRARY
    if CAPABILITY_GAP_LIBRARY is None:
        CAPABILITY_GAP_LIBRARY = load_capability_gap_library()
    return CAPABILITY_GAP_LIBRARY


def select_relevant_skills(world_request: str, *, limit: int = 3) -> tuple[dict[str, object], ...]:
    """Select skills whose trigger keywords match the natural-language request."""

    request_words = set(re.findall(r"[a-z0-9]+", world_request.lower()))
    projectile_combat = _is_projectile_combat_request(world_request)
    scored: list[tuple[int, str, dict[str, object]]] = []
    for skill in get_skill_library():
        if projectile_combat and _is_sports_strike_memory(skill):
            continue
        triggers = skill.get("trigger_keywords", [])
        if not isinstance(triggers, list):
            continue
        trigger_words = {str(trigger).lower() for trigger in triggers}
        score = len(request_words & trigger_words)
        if score <= 0:
            continue
        scored.append((score, str(skill.get("skill_id")), skill))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(skill for _, _, skill in scored[:limit])


def select_relevant_capability_gaps(
    world_request: str,
    *,
    limit: int = 4,
) -> tuple[dict[str, object], ...]:
    """Select remembered failure/gap records relevant to a new prompt."""

    request_words = _retrieval_words(world_request)
    projectile_combat = _is_projectile_combat_request(world_request)
    scored: list[tuple[int, str, dict[str, object]]] = []
    for gap in get_capability_gap_library():
        if projectile_combat and _is_sports_strike_memory(gap):
            continue
        trigger_words: set[str] = set()
        for field_name in (
            "trigger_keywords",
            "failed_subgoals",
            "suggested_new_blocks",
            "suggested_new_probes",
            "risk_flags",
        ):
            values = gap.get(field_name, [])
            if isinstance(values, list):
                for value in values:
                    trigger_words.update(_retrieval_words(str(value)))
        for field_name in (
            "task_family",
            "missing_capability",
            "failure_category",
            "diagnostic_axis",
            "source_prompt",
            "reason",
        ):
            trigger_words.update(_retrieval_words(str(gap.get(field_name) or "")))
        score = len(request_words & trigger_words)
        if score <= 0:
            continue
        scored.append((score, str(gap.get("gap_id")), gap))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(gap for _, _, gap in scored[:limit])


def select_relevant_affordance_blocks(
    world_request: str,
    *,
    limit: int = 5,
) -> tuple[dict[str, object], ...]:
    """Select physical affordance blocks by conceptual trigger overlap."""

    request_words = _retrieval_words(world_request)
    projectile_combat = _is_projectile_combat_request(world_request)
    scored: list[tuple[int, str, dict[str, object]]] = []
    for block in get_affordance_block_library():
        if projectile_combat and _is_sports_strike_memory(block):
            continue
        triggers = block.get("trigger_keywords", [])
        if not isinstance(triggers, list):
            continue
        trigger_words = set()
        for trigger in triggers:
            trigger_words.update(_retrieval_words(str(trigger)))
        trigger_words.update(_retrieval_words(str(block.get("abstract_relation") or "")))
        score = len(request_words & trigger_words)
        if score <= 0:
            continue
        if str(block.get("source") or "").startswith("harness_") and score < 2:
            continue
        scored.append((score, str(block.get("block_id")), block))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [block for _, _, block in scored[:limit]]
    if projectile_combat:
        projectile_block = next(
            (
                block
                for block in get_affordance_block_library()
                if str(block.get("block_id")) == "projectile_hazard_avoidance"
            ),
            None,
        )
        if projectile_block and projectile_block not in selected:
            selected = [projectile_block] + selected[: max(0, limit - 1)]
    core_ids = {"stable_floor_support", "sensor_region"}
    selected_ids = {str(block.get("block_id")) for block in selected}
    for block in get_affordance_block_library():
        block_id = str(block.get("block_id"))
        if block_id in core_ids and block_id not in selected_ids and len(selected) < limit:
            selected.append(block)
            selected_ids.add(block_id)
    return tuple(selected)


def _is_projectile_combat_request(text: str) -> bool:
    lowered = text.lower()
    agent_fires_at_object = bool(
        re.search(
            r"\b(agent|player|person|character|robot)\b.{0,35}\b(shoots|shoot|fires|fire|firing)\b.{0,45}\b(bullet|projectile|missile|laser|blaster)\b",
            lowered,
        )
    ) and _has_any_text(
        lowered,
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
    incoming_or_enemy_fire = _has_any_text(
        lowered,
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
    strong_projectile_context = _has_any_text(
        lowered,
        "spaceship",
        "space ship",
        "enemy ship",
        "laser",
        "bullet",
        "missile",
        "turret",
        "cannon",
        "blaster",
        "combat",
        "weapon",
        "enemy fire",
    )
    generic_projectile_context = _has_any_text(lowered, "projectile", "projectiles")
    projectile_action = _has_any_text(
        lowered,
        "shots",
        "shooting",
        "shoots",
        "firing",
        "fires",
        "fire at",
        "laser fire",
    )
    avoidance_or_survival = _has_any_text(
        lowered,
        "avoid",
        "avoiding",
        "dodge",
        "dodging",
        "survive",
        "survival",
        "escape",
    )
    sports_context = _has_any_text(
        lowered,
        "soccer",
        "hockey",
        "basketball",
        "billiards",
        "pinball",
        "kick",
        "kicking",
        "puck",
        "hoop",
    )
    if sports_context and not strong_projectile_context:
        return False
    if strong_projectile_context and projectile_action:
        return avoidance_or_survival or not sports_context
    return generic_projectile_context and avoidance_or_survival


def _is_sports_strike_memory(item: dict[str, object]) -> bool:
    text = json.dumps(item, sort_keys=True, default=str).lower()
    return _has_any_text(
        text,
        "strike_object_to_region",
        "create_strike_shot_lane",
        "kick_contact",
        "strike_contact",
        "soccer",
        "hockey",
        "billiards",
        "pinball",
        "goal_line",
        "shot lane",
    )


def _has_any_text(text: str, *terms: str) -> bool:
    return any(term in text for term in terms)


def _retrieval_words(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 1 and word not in RETRIEVAL_STOPWORDS
    }


def format_skill_guidance(skills: tuple[dict[str, object], ...]) -> str:
    """Render retrieved skills into concise prompt guidance."""

    if not skills:
        return "- No matching reusable skill was retrieved. Follow environment_spec.json exactly."

    lines: list[str] = []
    for skill in skills:
        skill_id = str(skill.get("skill_id", "unknown_skill"))
        objective_type = str(skill.get("objective_type", "custom_physics"))
        lines.append(f"- Skill: {skill_id} (objective_type={objective_type})")
        summary = skill.get("summary")
        if summary:
            lines.append(f"  Summary: {summary}")
        for field_name, label in (
            ("required_registry", "Required registry"),
            ("add_objects_pattern", "Object creation pattern"),
            ("objective_pattern", "Objective pattern"),
            ("ground_truth_pattern", "Ground-truth pattern"),
            ("common_failures", "Avoid"),
        ):
            values = skill.get(field_name, [])
            if not isinstance(values, list) or not values:
                continue
            lines.append(f"  {label}:")
            for value in values:
                lines.append(f"    - {value}")
        example = skill.get("minimal_check_objective_example")
        if example:
            lines.append("  Minimal check_objective pattern:")
            for line in str(example).splitlines():
                lines.append(f"    {line}")
    return "\n".join(lines)


def format_affordance_guidance(blocks: tuple[dict[str, object], ...]) -> str:
    """Render retrieved affordance blocks as composable design primitives."""

    lines = [
        "- Use these as composable physical principles, not fixed templates.",
        "- Adapt names, sizes, positions, and aesthetics to the user's prompt.",
        "- If no block fits the prompt, create a custom_physics design with concrete subgoals/progress metrics; do not force an irrelevant block.",
        "- If a novel design validates successfully, it can become future memory.",
    ]
    if not blocks:
        lines.append("- No matching affordance block was retrieved.")
        return "\n".join(lines)

    for block in blocks:
        block_id = str(block.get("block_id", "unknown_block"))
        relation = str(block.get("abstract_relation", "custom_physics"))
        lines.append(f"- Block: {block_id} (relation={relation})")
        summary = block.get("summary")
        if summary:
            lines.append(f"  Summary: {summary}")
        for field_name, label in (
            ("creates", "Creates"),
            ("constraints", "Constraints"),
            ("composes_with", "Composes with"),
            ("validator_checks", "Validator checks"),
            ("repair_guidance", "Repair guidance"),
        ):
            values = block.get(field_name, [])
            if not isinstance(values, list) or not values:
                continue
            lines.append(f"  {label}:")
            for value in values:
                lines.append(f"    - {value}")
    return "\n".join(lines)


def format_capability_gap_guidance(gaps: tuple[dict[str, object], ...]) -> str:
    """Render remembered capability gaps as repair-aware generation guidance."""

    lines = [
        "- These are known gaps from previous runs. They are not templates and not excuses to lower standards.",
        "- Use them to avoid repeated failure modes, choose honest tiers, and design with better probe evidence.",
        "- If a gap says a validator skill is missing, either compose from existing executable subgoals or expose Tier 4 progress honestly.",
    ]
    if not gaps:
        lines.append("- No matching capability gap memory was retrieved.")
        return "\n".join(lines)

    for gap in gaps:
        gap_id = str(gap.get("gap_id", "unknown_gap"))
        task_family = str(gap.get("task_family", "unknown_family"))
        lines.append(f"- Gap: {gap_id} (family={task_family})")
        for field_name, label in (
            ("missing_capability", "Missing capability"),
            ("reason", "Observed failure"),
            ("capability_gap", "Capability gap"),
        ):
            value = gap.get(field_name)
            if value:
                lines.append(f"  {label}: {value}")
        for field_name, label in (
            ("failed_subgoals", "Failed subgoals"),
            ("suggested_new_blocks", "Suggested blocks"),
            ("suggested_new_probes", "Suggested probes"),
            ("repair_guidance", "Repair guidance"),
            ("risk_flags", "Risk flags"),
        ):
            values = gap.get(field_name, [])
            if not isinstance(values, list) or not values:
                continue
            lines.append(f"  {label}:")
            for value in values[:6]:
                lines.append(f"    - {value}")
    return "\n".join(lines)


def format_strict_api_limits(spec: dict[str, object]) -> str:
    """Render the JSON spec into concise prompt constraints."""

    project = spec.get("Project_Constraints", {})
    api_reference = spec.get("API_Reference", {})
    globals_ = spec.get("Global_Variables", [])
    read_only_globals = spec.get("Read_Only_Global_Variables", [])
    object_model = spec.get("Object_Model", {})
    mandatory = spec.get("Mandatory_Definitions", {})
    objective_profile = spec.get("Objective_Profile", {})
    capability_profile = spec.get("Capability_Profile", {})
    verification_tiers = spec.get("Verification_Tiers", {})
    strict_rules = spec.get("Strict_Rules", [])
    lines: list[str] = []
    if isinstance(project, dict):
        required = project.get("Required_Methods", [])
        forbidden = project.get("Forbidden_Arguments", [])
        if isinstance(required, list):
            lines.append(f"- Required methods: {', '.join(map(str, required))}.")
        if isinstance(forbidden, list):
            lines.append(f"- Forbidden arguments: {', '.join(map(str, forbidden))}.")
    if isinstance(globals_, list):
        lines.append(f"- Use only these guaranteed globals: {', '.join(map(str, globals_))}.")
    if isinstance(read_only_globals, list) and read_only_globals:
        lines.append(
            "- Read-only globals you may read but must never assign/redefine: "
            f"{', '.join(map(str, read_only_globals))}."
        )
    if isinstance(object_model, dict):
        lines.append("- Object_Model:")
        for key, value in object_model.items():
            if isinstance(value, list):
                lines.append(f"  - {key}:")
                for item in value:
                    lines.append(f"    - {item}")
            else:
                lines.append(f"  - {key}: {value}")
    objective_types = spec.get("Objective_Types", {})
    if isinstance(objective_types, dict):
        allowed_types = objective_types.get("Allowed", [])
        if isinstance(allowed_types, list):
            lines.append(f"- Allowed objective_type values: {', '.join(map(str, allowed_types))}.")
        selection_guide = objective_types.get("Selection_Guide", {})
        if isinstance(selection_guide, dict):
            lines.append("- Objective type selection guide:")
            for objective_type, description in selection_guide.items():
                lines.append(f"  - {objective_type}: {description}")
    if isinstance(objective_profile, dict):
        required_fields = objective_profile.get("Required_Fields", {})
        if isinstance(required_fields, dict):
            lines.append("- Objective_Profile required fields:")
            for field_name, description in required_fields.items():
                lines.append(f"  - {field_name}: {description}")
        rules = objective_profile.get("Rules", [])
        if isinstance(rules, list):
            lines.append("- Objective_Profile rules:")
            for rule in rules:
                lines.append(f"  - {rule}")
    if isinstance(capability_profile, dict):
        required_fields = capability_profile.get("Required_Fields", {})
        if isinstance(required_fields, dict):
            lines.append("- Capability_Profile required fields:")
            for field_name, description in required_fields.items():
                lines.append(f"  - {field_name}: {description}")
        rules = capability_profile.get("Rules", [])
        if isinstance(rules, list):
            lines.append("- Capability_Profile rules:")
            for rule in rules:
                lines.append(f"  - {rule}")
    if isinstance(verification_tiers, dict):
        defaults = verification_tiers.get("Default_Minimum_Acceptance", {})
        if isinstance(defaults, dict):
            lines.append("- Verification tier default minimum acceptance:")
            for objective_type, tier in defaults.items():
                lines.append(f"  - {objective_type}: Tier {tier}")
        tier_rules = verification_tiers.get("Rules", [])
        if isinstance(tier_rules, list):
            lines.append("- Verification tier rules:")
            for rule in tier_rules:
                lines.append(f"  - {rule}")
    if isinstance(mandatory, dict):
        init_definitions = mandatory.get("In_Init", [])
        if isinstance(init_definitions, list):
            lines.append("- Mandatory definitions in __init__:")
            for definition in init_definitions:
                lines.append(f"  - {definition}")
    if isinstance(strict_rules, list):
        lines.append("- Strict rules:")
        for rule in strict_rules:
            lines.append(f"  - {rule}")
    if isinstance(api_reference, dict):
        lines.append("- API_Reference allowed arguments:")
        for helper_name, helper_spec in api_reference.items():
            if not isinstance(helper_spec, dict):
                continue
            args = helper_spec.get("args", [])
            note = helper_spec.get("note")
            args_text = ", ".join(map(str, args)) if isinstance(args, list) else str(args)
            line = f"  - {helper_name}: {args_text}"
            if note:
                line += f" ({note})"
            lines.append(line)
    return "\n".join(lines)


def extract_python_code(llm_response: str) -> str:
    """Extract Python code from a fenced LLM response or raw Python text."""

    match = CODE_BLOCK_RE.search(llm_response)
    if match:
        code = match.group(1)
    else:
        code = llm_response
    return textwrap.dedent(code).strip() + "\n"


def verify_generated_code(code: str, *, expected_class_name: str | None = None) -> VerificationResult:
    """Statically verify that generated code follows the BaseEnv contract."""

    errors: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return VerificationResult(ok=False, errors=(f"syntax error: {exc}",))
    _attach_parents(tree)

    class_defs = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    env_classes = [node for node in class_defs if _inherits_from_base_env(node)]

    if len(env_classes) != 1:
        errors.append("generated module must define exactly one BaseEnv subclass")
        class_name = None
    else:
        env_class = env_classes[0]
        class_name = env_class.name
        if expected_class_name and env_class.name != expected_class_name:
            errors.append(
                f"expected class {expected_class_name!r}, found {env_class.name!r}"
            )
        if not _class_has_method(env_class, "build_world"):
            errors.append("BaseEnv subclass must implement build_world(self)")
        for required_method in ("__init__", "add_objects", "get_ground_truth", "check_objective"):
            if not _class_has_method(env_class, required_method):
                if required_method == "check_objective":
                    errors.append(
                        "Missing Code-Level Objective: Every environment must define its own win condition."
                    )
                else:
                    errors.append(f"BaseEnv subclass must implement {required_method}(self)")
        errors.extend(_forbidden_method_argument_errors(env_class))
        errors.extend(_mandatory_definition_errors(env_class))
        errors.extend(_layout_plan_definition_errors(env_class))
        errors.extend(_objective_metadata_errors(env_class))
        errors.extend(_ground_truth_metadata_errors(env_class))
        errors.extend(_state_registry_errors(env_class))
        errors.extend(_method_signature_errors(env_class))
        errors.extend(_object_model_errors(env_class))
        errors.extend(_objective_side_effect_errors(env_class))
        errors.extend(_body_callback_assignment_errors(env_class))
        errors.extend(_objective_reset_hook_errors(env_class))
        errors.extend(_vec2d_constructor_errors(tree))
        errors.extend(_field_force_static_contract_errors(tree))
        errors.extend(_mechanism_static_contract_errors(tree))
        errors.extend(_strike_static_contract_errors(tree))
        errors.extend(_relation_constructor_static_contract_errors(tree))
        errors.extend(_projectile_static_contract_errors(tree))
        errors.extend(_solvability_hint_static_errors(tree))
        errors.extend(_unsupported_baseenv_api_errors(tree))
        errors.extend(_nondeterministic_random_errors(tree))
        for forbidden_method in ("reset", "step"):
            if _class_has_method(env_class, forbidden_method):
                message = f"generated class must not override {forbidden_method}()"
                if forbidden_method == "step":
                    message += "; use register_force_zone(...) for field/wind/magnetic effects"
                errors.append(message)

    if not _imports_base_env(tree):
        errors.append("generated module must import BaseEnv from base_env")

    if _has_space_add_call(tree):
        errors.append("generated code must not call self.space.add(...) directly")

    helper_calls = _count_helper_calls(tree)
    if helper_calls == 0:
        errors.append("generated code must use BaseEnv telemetry helpers/register_object")
    errors.extend(_helper_keyword_errors(tree))

    if (
        not _has_method_call(tree, "set_solvability_hint")
        and not _has_method_call(tree, "create_horizontal_push_lane")
        and not _has_method_call(tree, "create_pressure_plate_gate_corridor")
        and not _has_method_call(tree, "create_field_push_lane")
        and not _has_method_call(tree, "create_strike_shot_lane")
        and not _has_method_call(tree, "create_ballistic_hoop_challenge")
        and not _has_method_call(tree, "create_ballistic_barrier_goal_challenge")
        and not _has_method_call(tree, "create_support_exit_freefall_challenge")
        and not _has_method_call(tree, "create_recurring_falling_hazards")
        and not _has_method_call(tree, "create_recurring_lateral_hazards")
        and not _has_constant_string(tree, "multi_target_touch")
        and not _has_constant_string(tree, "single_target_touch")
        and not _has_constant_string(tree, "agent_touch_object")
        and not _has_constant_string(tree, "survive_duration")
        and not _has_constant_string(tree, "survival")
    ):
        errors.append("generated code must call self.set_solvability_hint(...)")

    if not _has_constant_assignment(tree, "GENERATED_ENV_CLASS"):
        errors.append("generated code must define GENERATED_ENV_CLASS")

    if not _has_constant_assignment(tree, "SOURCE_PROMPT"):
        errors.append("generated code must define SOURCE_PROMPT")

    if (
        not _has_keyword_string_value(tree, "role", "agent")
        and not _has_method_call(tree, "create_horizontal_push_lane")
        and not _has_method_call(tree, "create_pressure_plate_gate_corridor")
        and not _has_method_call(tree, "create_field_push_lane")
        and not _has_method_call(tree, "create_strike_shot_lane")
        and not _has_method_call(tree, "create_ballistic_hoop_challenge")
        and not _has_method_call(tree, "create_ballistic_barrier_goal_challenge")
        and not _has_method_call(tree, "create_support_exit_freefall_challenge")
    ):
        errors.append('generated code must register an object with role="agent"')

    if _imports_forbidden_modules(tree):
        errors.append("generated code must not import pygame, assets, networking, or IO modules")

    if _uses_undefined_world_aliases(tree):
        errors.append("generated code must use self.width and self.height, not _W, _H, WIDTH, or HEIGHT")

    return VerificationResult(ok=not errors, errors=tuple(errors), class_name=class_name)


def save_generated_code(
    world_request: str,
    llm_response: str,
    *,
    class_name: str | None = None,
    output_dir: Path = GENERATED_ENVS_DIR,
) -> Path:
    """Extract, verify, and save generated environment code."""

    request = build_architect_request(world_request, class_name)
    code = extract_python_code(llm_response)
    result = verify_generated_code(code, expected_class_name=request.class_name)
    if not result.ok:
        error_text = "\n".join(f"- {error}" for error in result.errors)
        raise ValueError(f"generated environment failed verification:\n{error_text}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_module_name_from_class(request.class_name)}.py"
    output_path.write_text(code, encoding="utf-8")
    return output_path


async def generate_and_save_openai(
    world_request: str,
    *,
    class_name: str | None = None,
    output_dir: Path = GENERATED_ENVS_DIR,
    config: OpenAIArchitectConfig | None = None,
) -> Path:
    """Generate code with OpenAI, verify it, and save it to generated_envs/."""

    request = build_architect_request(world_request, class_name)
    architect = OpenAIArchitect(config=config)
    prompt = render_prompt(request.world_request, class_name=request.class_name)
    response = await architect.generate_from_prompt(prompt)
    return save_generated_code(
        request.world_request,
        response,
        class_name=request.class_name,
        output_dir=output_dir,
    )


def _class_name_from_request(world_request: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", world_request)
    if not words:
        return "GeneratedEnv"
    candidate = "".join(word.capitalize() for word in words[:6]) + "Env"
    if candidate[0].isdigit():
        candidate = f"Generated{candidate}"
    return candidate


def _module_name_from_class(class_name: str) -> str:
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
    snake = CLASS_NAME_RE.sub("_", snake).strip("_")
    return snake or "generated_env"


def _attach_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent


def _imports_base_env(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "base_env":
            imported_names = {alias.name for alias in node.names}
            if "BaseEnv" in imported_names:
                return True
    return False


def _imports_forbidden_modules(tree: ast.Module) -> bool:
    forbidden = {"pygame", "cv2", "PIL", "requests", "urllib", "socket", "pathlib", "os", "sys"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in forbidden:
                    return True
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in forbidden:
                return True
    return False


def _nondeterministic_random_errors(tree: ast.Module) -> list[str]:
    """Reject unseeded randomness so validation demos and replays are reproducible."""

    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "random":
                    errors.append(
                        "generated code must not import random; use BaseEnv's seeded self.rng for deterministic replay"
                    )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] == "random":
                errors.append(
                    "generated code must not import from random; use BaseEnv's seeded self.rng for deterministic replay"
                )
        elif isinstance(node, ast.Attribute):
            dotted = _attribute_name(node)
            if dotted.startswith(("random.", "numpy.random.", "np.random.")):
                errors.append(
                    f"generated code must not call {dotted}; use BaseEnv's seeded self.rng for deterministic replay"
                )
    return sorted(set(errors))


def _attribute_name(node: ast.AST) -> str:
    parts: list[str] = []
    current: ast.AST | None = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    else:
        return ""
    return ".".join(reversed(parts))


def _inherits_from_base_env(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "BaseEnv":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "BaseEnv":
            return True
    return False


def _class_has_method(node: ast.ClassDef, method_name: str) -> bool:
    return any(
        isinstance(item, ast.FunctionDef) and item.name == method_name
        for item in node.body
    )


def _has_method_call(tree: ast.Module, method_name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == method_name:
                return True
    return False


def _has_constant_assignment(tree: ast.Module, constant_name: str) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == constant_name:
                return True
    return False


def _has_keyword_string_value(tree: ast.Module, keyword_name: str, value: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != keyword_name:
                continue
            if isinstance(keyword.value, ast.Constant) and keyword.value.value == value:
                return True
    return False


def _has_space_add_call(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add":
            continue
        value = node.func.value
        if (
            isinstance(value, ast.Attribute)
            and value.attr == "space"
            and isinstance(value.value, ast.Name)
            and value.value.id == "self"
        ):
            return True
    return False


def _count_helper_calls(tree: ast.Module) -> int:
    helper_names = {
        "create_dynamic_circle",
        "create_static_segment",
        "create_static_box",
        "create_dynamic_box",
        "register_object",
        "register_constraint",
        "register_force_zone",
        "create_pressure_plate",
        "create_horizontal_push_lane",
        "create_pressure_plate_gate_corridor",
        "create_field_push_lane",
        "create_strike_shot_lane",
        "create_ballistic_hoop_challenge",
        "create_ballistic_barrier_goal_challenge",
        "create_support_exit_freefall_challenge",
        "create_recurring_falling_hazards",
        "create_recurring_lateral_hazards",
        "create_readable_chaser",
        "register_readable_chaser",
        "register_pressure_plate_gate",
    }
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in helper_names:
                count += 1
    return count


def _helper_keyword_errors(tree: ast.Module) -> list[str]:
    forbidden_keywords = _forbidden_spec_arguments()
    allowed_keywords = {
        "create_dynamic_circle": {
            "name",
            "pos",
            "position",
            "radius",
            "mass",
            "kind",
            "role",
            "elasticity",
            "friction",
            "sensor",
            "metadata",
        },
        "create_static_segment": {
            "name",
            "a",
            "b",
            "radius",
            "kind",
            "role",
            "elasticity",
            "friction",
            "sensor",
            "metadata",
        },
        "create_dynamic_box": {
            "name",
            "center",
            "size",
            "mass",
            "kind",
            "role",
            "elasticity",
            "friction",
            "sensor",
            "metadata",
        },
        "create_static_box": {
            "name",
            "center",
            "size",
            "kind",
            "role",
            "elasticity",
            "friction",
            "sensor",
            "metadata",
        },
        "set_solvability_hint": {
            "start",
            "goal",
            "grid_size",
            "agent_radius",
            "notes",
            "metadata",
        },
        "register_object": {"name", "body", "shapes", "kind", "role", "metadata"},
        "register_constraint": {
            "name",
            "constraint",
            "metadata",
            "type",
            "body_a",
            "body_b",
            "anchor_a",
            "anchor_b",
        },
        "register_force_zone": {
            "name",
            "center",
            "size",
            "force",
            "mode",
            "strength",
            "affected_names",
            "affected_roles",
            "falloff",
            "role",
            "metadata",
        },
        "create_pressure_plate": {
            "name",
            "center",
            "size",
            "role",
            "elasticity",
            "friction",
            "sensor",
            "metadata",
        },
        "create_horizontal_push_lane": {
            "name",
            "agent_name",
            "object_name",
            "target_name",
            "agent_x",
            "object_x",
            "target_x",
            "lane_y",
            "agent_radius",
            "object_size",
            "object_mass",
            "object_friction",
            "target_size",
            "target_role",
            "target_kind",
            "support_thickness",
            "support_margin",
            "support_friction",
            "metadata",
        },
        "create_pressure_plate_gate_corridor": {
            "name",
            "agent_name",
            "object_name",
            "plate_name",
            "gate_name",
            "goal_name",
            "mechanism_name",
            "agent_x",
            "object_x",
            "plate_x",
            "gate_x",
            "goal_x",
            "lane_y",
            "agent_radius",
            "object_size",
            "object_mass",
            "object_friction",
            "plate_size",
            "gate_size",
            "goal_size",
            "support_margin",
            "metadata",
        },
        "create_field_push_lane": {
            "name",
            "agent_name",
            "object_name",
            "field_name",
            "target_name",
            "agent_x",
            "object_x",
            "field_x",
            "target_x",
            "lane_y",
            "agent_radius",
            "object_radius",
            "object_mass",
            "object_friction",
            "field_size",
            "target_size",
            "force",
            "mode",
            "strength",
            "support_thickness",
            "support_margin",
            "metadata",
        },
        "create_strike_shot_lane": {
            "name",
            "agent_name",
            "object_name",
            "target_name",
            "agent_x",
            "object_x",
            "target_x",
            "lane_y",
            "agent_radius",
            "object_radius",
            "object_mass",
            "object_friction",
            "object_elasticity",
            "target_size",
            "support_thickness",
            "support_margin",
            "support_friction",
            "goal_post_thickness",
            "metadata",
        },
        "create_ballistic_hoop_challenge": {
            "name",
            "agent_name",
            "object_name",
            "target_name",
            "agent_pos",
            "object_pos",
            "target_center",
            "agent_radius",
            "object_radius",
            "object_mass",
            "object_friction",
            "object_elasticity",
            "target_size",
            "support_thickness",
            "support_margin",
            "support_friction",
            "metadata",
        },
        "create_ballistic_barrier_goal_challenge": {
            "name",
            "agent_name",
            "object_name",
            "barrier_name",
            "target_name",
            "agent_pos",
            "object_pos",
            "barrier_center",
            "barrier_size",
            "target_center",
            "agent_radius",
            "object_radius",
            "object_mass",
            "object_friction",
            "object_elasticity",
            "target_size",
            "support_thickness",
            "support_margin",
            "support_friction",
            "metadata",
        },
        "create_support_exit_freefall_challenge": {
            "name",
            "agent_name",
            "object_name",
            "boundary_name",
            "drop_zone_name",
            "agent_x",
            "object_x",
            "edge_x",
            "lane_y",
            "agent_radius",
            "object_radius",
            "object_mass",
            "object_friction",
            "support_thickness",
            "support_margin",
            "drop_height",
            "metadata",
        },
        "create_recurring_falling_hazards": {
            "name",
            "count",
            "lane_xs",
            "spawn_y",
            "bottom_y",
            "radius",
            "mass",
            "speed_y",
            "phase_gap_steps",
            "role",
            "name_prefix",
            "elasticity",
            "friction",
            "sensor",
            "metadata",
        },
        "create_recurring_lateral_hazards": {
            "name",
            "count",
            "lane_y",
            "spawn_x",
            "exit_x",
            "size",
            "shape",
            "mass",
            "speed_x",
            "phase_gap_steps",
            "role",
            "name_prefix",
            "elasticity",
            "friction",
            "sensor",
            "metadata",
        },
        "create_readable_chaser": {
            "name",
            "pos",
            "target_name",
            "radius",
            "mass",
            "force_strength",
            "max_speed",
            "stop_radius",
            "axis",
            "role",
            "elasticity",
            "friction",
            "sensor",
            "metadata",
        },
        "register_readable_chaser": {
            "name",
            "chaser",
            "target",
            "force_strength",
            "max_speed",
            "stop_radius",
            "axis",
            "metadata",
        },
        "register_pressure_plate_gate": {
            "name",
            "trigger",
            "gate",
            "activator",
            "activation_distance",
            "open_mode",
            "metadata",
        },
    }
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node)
        for keyword in node.keywords:
            if keyword.arg in forbidden_keywords:
                errors.append(_spec_keyword_message(keyword.arg, call_name))
        if not isinstance(node.func, ast.Attribute):
            continue
        helper_name = node.func.attr
        if helper_name not in allowed_keywords:
            continue
        for keyword in node.keywords:
            if keyword.arg is None:
                errors.append(_spec_keyword_message(keyword.arg, helper_name))
            elif keyword.arg in forbidden_keywords:
                continue
            elif keyword.arg not in allowed_keywords[helper_name]:
                errors.append(_spec_keyword_message(keyword.arg, helper_name))
    return errors


def _vec2d_constructor_errors(tree: ast.Module) -> list[str]:
    """Reject the common Pymunk footgun pymunk.Vec2d((x, y))."""

    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_vec2d_constructor(node):
            continue
        if len(node.args) == 1 and not node.keywords:
            errors.append(
                "generated code must not call pymunk.Vec2d with one tuple/list argument; "
                "use plain (x, y) tuples or pymunk.Vec2d(x, y)"
            )
    return errors


def _is_vec2d_constructor(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "Vec2d"
    if isinstance(func, ast.Attribute):
        return func.attr == "Vec2d"
    return False


def _field_force_static_contract_errors(tree: ast.Module) -> list[str]:
    """Field-force declarations must use BaseEnv force-zone primitives."""

    errors: list[str] = []
    duplicate_zone_names = _duplicate_force_zone_object_names(tree)
    for zone_name in duplicate_zone_names:
        errors.append(
            f"register_force_zone({zone_name!r}) already creates/registers the non-blocking "
            f"sensor object {zone_name!r}; do not also create or register another object "
            "with the same name"
        )
    if not _has_constant_string(tree, "field_force_interaction"):
        return errors
    has_field_lane = _has_method_call(tree, "create_field_push_lane")
    if not _has_method_call(tree, "register_force_zone") and not has_field_lane:
        errors.append(
            "field_force_interaction subgoals require self.register_force_zone(...); "
            "do not represent magnetic/wind/current fields as plain sensor boxes"
        )
    if _has_constant_string(tree, "move_object_to_region") and not has_field_lane:
        errors.append(
            "field-force push puzzles require self.create_field_push_lane(...); "
            "use the stable push-into-field primitive instead of hand-placing the object, field, support, and target"
        )
    env_class = _only_env_class(tree)
    if env_class is not None and _class_has_method(env_class, "step"):
        errors.append(
            "field-force environments must not override step(); use BaseEnv register_force_zone(...)"
        )
    return errors


def _duplicate_force_zone_object_names(tree: ast.Module) -> list[str]:
    zone_names: list[str] = []
    object_names: set[str] = set()
    object_creators = {
        "create_static_box",
        "create_dynamic_box",
        "create_dynamic_circle",
        "create_static_segment",
        "create_pressure_plate",
        "register_object",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        helper = node.func.attr
        literal_name = _literal_helper_name(node)
        if helper == "register_force_zone":
            if literal_name is not None:
                zone_names.append(literal_name)
            continue
        if helper == "create_field_push_lane":
            field_name = _literal_keyword_value(node, "field_name")
            if field_name is not None:
                zone_names.append(field_name)
            continue
        if literal_name is None:
            continue
        if helper in object_creators:
            object_names.add(literal_name)
    duplicates: set[str] = set()
    seen: set[str] = set()
    for zone_name in zone_names:
        if zone_name in seen:
            duplicates.add(zone_name)
        seen.add(zone_name)
    duplicates.update(set(zone_names) & object_names)
    return sorted(duplicates)


def _literal_helper_name(node: ast.Call) -> str | None:
    if node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
    for keyword in node.keywords:
        if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
            if isinstance(keyword.value.value, str):
                return keyword.value.value
    return None


def _literal_keyword_value(node: ast.Call, keyword_name: str) -> str | None:
    for keyword in node.keywords:
        if keyword.arg == keyword_name and isinstance(keyword.value, ast.Constant):
            if isinstance(keyword.value.value, str):
                return keyword.value.value
    return None


def _mechanism_static_contract_errors(tree: ast.Module) -> list[str]:
    """Plate/gate declarations must use BaseEnv mechanism primitives."""

    if not _has_constant_string(tree, "activate_mechanism"):
        return []
    string_blob = " ".join(
        str(node.value).lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    has_plate_gate_semantics = any(
        token in string_blob for token in ("pressure_plate", "plate", "button", "switch")
    ) and any(token in string_blob for token in ("gate", "door", "barrier"))
    if not has_plate_gate_semantics:
        return []
    errors: list[str] = []
    has_push_lane = _has_method_call(tree, "create_horizontal_push_lane")
    has_gate_corridor = _has_method_call(tree, "create_pressure_plate_gate_corridor")
    if not _has_method_call(tree, "create_pressure_plate") and not has_push_lane and not has_gate_corridor:
        errors.append(
            "pressure-plate gate subgoals require self.create_pressure_plate(...) "
            "or self.create_horizontal_push_lane(..., target_kind='pressure_plate') "
            "or self.create_pressure_plate_gate_corridor(...); "
            "do not model plates as generic solid boxes"
        )
    if not has_push_lane and not has_gate_corridor:
        errors.append(
            "pressure-plate gate subgoals require self.create_horizontal_push_lane(...); "
            "use BaseEnv's stable push-lane primitive so agent, box, and plate share one supported lane"
        )
    if not has_gate_corridor:
        errors.append(
            "push-box pressure-plate gate tasks should use self.create_pressure_plate_gate_corridor(...); "
            "this prevents fragile post-mechanism path blockers and mismatched check_objective geometry"
        )
    if not _has_method_call(tree, "register_pressure_plate_gate") and not has_gate_corridor:
        errors.append(
            "pressure-plate gate subgoals require self.register_pressure_plate_gate(...); "
            "use BaseEnv's deterministic mechanism primitive instead of custom gate logic"
        )
    return errors


def _strike_static_contract_errors(tree: ast.Module) -> list[str]:
    """Kick/shot declarations should use BaseEnv's stable strike primitive."""

    string_blob = _contract_semantic_string_blob(tree)
    if _is_projectile_combat_request(string_blob):
        errors: list[str] = []
        if _has_method_call(tree, "create_strike_shot_lane") or _has_constant_string(tree, "strike_object_to_region"):
            errors.append(
                "projectile-combat shots must be modeled as dynamic hazard/projectile "
                "motion, not soccer-style create_strike_shot_lane(...) or "
                "strike_object_to_region"
            )
        return errors

    if not _is_sports_or_object_strike_context(string_blob):
        return []
    errors: list[str] = []
    if _is_ballistic_throw_context(string_blob) and not _is_support_exit_context(string_blob):
        if _is_ballistic_barrier_goal_context(string_blob) and not _has_method_call(
            tree,
            "create_ballistic_barrier_goal_challenge",
        ):
            errors.append(
                "kick/throw/lob over wall/barrier into goal tasks require "
                "self.create_ballistic_barrier_goal_challenge(...); use the stable "
                "ballistic-over-barrier constructor instead of hand-placed ramps, rails, "
                "posts, and walls"
            )
        if _is_ballistic_hoop_context(string_blob) and not _has_method_call(tree, "create_ballistic_hoop_challenge"):
            errors.append(
                "hoop/basketball lob tasks require self.create_ballistic_hoop_challenge(...); "
                "use the stable ballistic relation constructor instead of hand-tuned arc geometry"
            )
        if _has_constant_string(tree, "move_object_to_region") and not _has_constant_string(
            tree,
            "ballistic_object_to_region",
        ):
            errors.append(
                "ballistic throw/lob tasks should declare a ballistic_object_to_region subgoal "
                "or physics relation, not only move_object_to_region."
            )
        return errors
    if not _has_method_call(tree, "create_strike_shot_lane"):
        errors.append(
            "kick/shot/score tasks require self.create_strike_shot_lane(...); "
            "do not model a soccer-style shot as a long generic push lane"
        )
    if _has_constant_string(tree, "move_object_to_region") and not _has_constant_string(tree, "strike_object_to_region"):
        errors.append(
            "kick/shot/score tasks must declare a strike_object_to_region subgoal "
            "with interaction='kick_contact' or 'strike_contact', not only move_object_to_region"
        )
    return errors


def _relation_constructor_static_contract_errors(tree: ast.Module) -> list[str]:
    """Straightforward relation tasks should use stable relation constructors."""

    string_blob = _contract_semantic_string_blob(tree)
    errors: list[str] = []
    if _is_ballistic_hoop_context(string_blob):
        errors.extend(_ballistic_blocking_visual_errors(tree))
        errors.extend(_ballistic_objective_overconstraint_errors(tree))
    if _is_support_exit_context(string_blob):
        errors.extend(_support_exit_objective_overconstraint_errors(tree))
        errors.extend(_support_exit_blocking_geometry_errors(tree))
        if not _has_method_call(tree, "create_support_exit_freefall_challenge"):
            errors.append(
                "push-off-cliff/edge/mountain tasks require "
                "self.create_support_exit_freefall_challenge(...); use the stable "
                "support-exit relation constructor so the boundary, drop zone, "
                "agent, object, and support are physically aligned"
            )
        if _has_constant_string(tree, "ballistic_object_to_region"):
            errors.append(
                "push-off-cliff/edge/mountain tasks must not declare "
                "ballistic_object_to_region; use support_exit_freefall instead"
            )
    return errors


def _support_exit_blocking_geometry_errors(tree: ast.Module) -> list[str]:
    """Reject extra solid support/terrain geometry around stable support-exit helpers."""

    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"create_static_box", "create_static_segment"}:
            continue
        name = _call_string_name(node) or ""
        if name.endswith("_support") or "cliff_edge_boundary" in name:
            continue
        sensor_kw = next((keyword for keyword in node.keywords if keyword.arg == "sensor"), None)
        if not (sensor_kw and isinstance(sensor_kw.value, ast.Constant) and sensor_kw.value.value is True):
            errors.append(
                f"support-exit extra static geometry {name!r} must be sensor=True or removed; "
                "create_support_exit_freefall_challenge owns the playable support/edge/drop physics"
            )
    return errors


def _support_exit_objective_overconstraint_errors(tree: ast.Module) -> list[str]:
    """Reject extra landing-zone objectives for simple support-exit relations."""

    string_blob = _contract_semantic_string_blob(tree)
    errors: list[str] = []
    forbidden_markers = (
        "base_of_mountain_sensor",
        "base_of_mountain",
        "lower_landing_zone",
        "landing_zone",
        "bottom_void",
    )
    used = [marker for marker in forbidden_markers if marker in string_blob]
    if used:
        errors.append(
            "simple support_exit_freefall tasks must not add extra landing/base/bottom "
            f"objective regions {used}; use cliff_edge_boundary + open_air_drop_zone "
            "and the relation's min_fall_distance/min_downward_velocity as the objective"
        )
    return errors


def _ballistic_blocking_visual_errors(tree: ast.Module) -> list[str]:
    """Reject solid hoop/rim decorations that block the safe ballistic sensor."""

    errors: list[str] = []
    string_blob = _contract_semantic_string_blob(tree)
    blocked_tokens = ("rim", "hoop", "backboard", "guard", "net")
    if not _is_soccer_goal_context(string_blob):
        blocked_tokens = (*blocked_tokens, "post")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "create_static_box":
            continue
        name = _call_string_name(node)
        if not name or not any(token in name.lower() for token in blocked_tokens):
            continue
        sensor_kw = next((keyword for keyword in node.keywords if keyword.arg == "sensor"), None)
        if not (sensor_kw and isinstance(sensor_kw.value, ast.Constant) and sensor_kw.value.value is True):
            errors.append(
                f"ballistic hoop visual {name!r} must be sensor=True or removed; "
                "solid rim/backboard/post/guard boxes block validator-friendly arcs"
            )
    return errors


def _ballistic_objective_overconstraint_errors(tree: ast.Module) -> list[str]:
    """Reject common hidden score-plane checks for simple hoop relation tasks."""

    errors: list[str] = []
    string_blob = _contract_semantic_string_blob(tree)
    if "score_zone" in string_blob or "net_zone" in string_blob:
        errors.append(
            "simple ballistic hoop tasks must not add a separate score_zone/net_zone; "
            "use the helper's target sensor/proximity as the code-level objective"
        )
    suspicious_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            name = node.id.lower()
        elif isinstance(node, ast.Attribute):
            name = node.attr.lower()
        else:
            continue
        if any(
            token in name
            for token in (
                "crossed_downward",
                "previous_ball_y",
                "last_ball_y",
                "prev_ball_y",
                "entered_from_above",
                "within_aperture",
                "aperture",
                "score_plane",
                "rim_plane",
                "crossed_rim",
                "downward_cross",
                "downward_pass",
            )
        ):
            suspicious_names.add(name)
    if suspicious_names:
        errors.append(
            "simple ballistic hoop check_objective is over-constrained by "
            f"{sorted(suspicious_names)}; use distance/proximity to the hoop sensor instead"
        )
    return errors


def _call_string_name(node: ast.Call) -> str | None:
    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        return str(node.args[0].value)
    for keyword in node.keywords:
        if keyword.arg == "name" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            return str(keyword.value.value)
    return None


def _is_sports_or_object_strike_context(text: str) -> bool:
    if _is_field_force_context(text):
        return False
    if _has_any_text(
        text,
        "strike_object_to_region",
        "kick_contact",
        "strike_contact",
        "soccer",
        "hockey",
        "billiards",
        "basketball",
        "goal_line",
        "ball-through-goal",
    ):
        return True
    if re.search(r"\bpinball\b(?!-like)", text):
        return True
    if re.search(r"\b(kick|kicks|kicking|strike|strikes|striking|slam|slams|shoot|shoots|shot)\b.{0,45}\b(ball|puck|hoop|goal|basket)\b", text):
        return True
    if re.search(r"\b(ball|puck)\b.{0,45}\b(goal|hoop|basket|target)\b", text):
        return True
    return False


def _is_ballistic_throw_context(text: str) -> bool:
    """Return true for arcing/throwing semantics that are not flat strike lanes."""

    if _is_field_force_context(text):
        return False
    if _has_any_text(
        text,
        "ballistic_arc_to_region",
        "throw",
        "throws",
        "throwing",
        "lob",
        "lobs",
        "arc",
        "arcing",
        "hoop",
        "rim",
        "basketball",
    ):
        return True
    return bool(
        re.search(r"\b(ball|rock|object)\b.{0,50}\b(hoop|basket|rim)\b", text)
        or re.search(r"\b(throw|lob)\b.{0,50}\b(ball|rock|object|projectile)\b", text)
    )


def _is_ballistic_hoop_context(text: str) -> bool:
    """Return true for basketball/hoop-style ballistic target sensors."""

    if _is_field_force_context(text):
        return False
    if _has_any_text(text, "hoop", "basketball", "basket"):
        return True
    return bool(re.search(r"\b(ball|object|projectile)\b.{0,50}\b(hoop|basket)\b", text))


def _is_field_force_context(text: str) -> bool:
    return _has_any_text(
        text,
        "field_force_interaction",
        "force zone",
        "force_zone",
        "magnetic",
        "magnet",
        "charged",
        "wind",
        "current",
        "conveyor",
        "gravity well",
        "attract",
        "repel",
    )


def _is_ballistic_barrier_goal_context(text: str) -> bool:
    has_barrier = _has_any_text(text, "over wall", "over a wall", "over the wall", "barrier", "defender", "blocking wall")
    has_goal = _has_any_text(text, "goal_line", "goal sensor", "soccer goal", "goal region", "into goal", "goal mouth")
    has_ballistic_object = _has_any_text(text, "ballistic_object_to_region", "lob_kick_over_barrier", "ballistic_arc_to_region")
    return has_barrier and has_goal and has_ballistic_object


def _is_soccer_goal_context(text: str) -> bool:
    return _has_any_text(text, "soccer", "soccer_ball", "soccer_goal", "goal_line") and not _has_any_text(
        text,
        "basketball",
        "hoop",
        "basket",
    )


def _is_support_exit_context(text: str) -> bool:
    if _has_any_text(text, "support_exit_freefall", "support_boundary_exit", "freefall_after_support_exit"):
        return True
    return bool(
        re.search(r"\b(push|shove|roll|slide)\b.{0,70}\b(rock|boulder|ball|crate|object)\b.{0,70}\b(cliff|edge|ledge|mountain)\b", text)
        or re.search(r"\b(rock|boulder|ball|crate|object)\b.{0,70}\b(off|over)\b.{0,40}\b(cliff|edge|ledge|mountain)\b", text)
    )


def _projectile_static_contract_errors(tree: ast.Module) -> list[str]:
    """Projectile-combat semantics require actual shot/laser hazard bodies."""

    string_blob = _contract_semantic_string_blob(tree)
    if not (
        _is_projectile_combat_request(string_blob)
        or "readable_projectile_hazard" in string_blob
        or "projectile hazards" in string_blob
    ):
        return []
    if _has_projectile_hazard_creation(tree):
        return []
    return [
        "projectile-combat prompts require at least one dynamic role='hazard' "
        "projectile body named/metadata-tagged with shot, laser, projectile, or bolt; "
        "enemy ships/turrets alone do not satisfy the projectile semantic requirement"
    ]


def _contract_semantic_string_blob(tree: ast.Module) -> str:
    """String context for static semantic checks, excluding prompt boilerplate.

    SOURCE_PROMPT contains our own repair/brief instructions, which can mention
    many task families. Static task-family checks should be based on generated
    profiles/object names/metadata, not on meta-instructions injected by the
    harness.
    """

    _annotate_assign_parents(tree)
    parts: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        parent_assign = getattr(node, "_harness_parent_assign", None)
        if parent_assign == "SOURCE_PROMPT":
            continue
        parts.append(str(node.value).lower())
    return " ".join(parts)


def _annotate_assign_parents(tree: ast.Module) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        target_names = [
            target.id
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
        if not target_names:
            continue
        for child in ast.walk(node.value):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                setattr(child, "_harness_parent_assign", target_names[0])


def _solvability_hint_static_errors(tree: ast.Module) -> list[str]:
    """set_solvability_hint expects coordinate tuples, not object names."""

    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "set_solvability_hint":
            continue
        for keyword in node.keywords:
            if keyword.arg not in {"start", "goal"}:
                continue
            if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                errors.append(
                    "set_solvability_hint start/goal must be coordinate tuples like "
                    "(120.0, 100.0), not registered object-name strings"
                )
    return errors


def _has_projectile_hazard_creation(tree: ast.Module) -> bool:
    projectile_terms = ("shot", "laser", "projectile", "bolt", "bullet", "missile")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"create_dynamic_circle", "create_dynamic_box"}:
            continue
        call_text = _call_string_text(node).lower()
        if not any(term in call_text for term in projectile_terms):
            continue
        if "hazard" in call_text:
            return True
    return False


def _call_string_text(node: ast.Call) -> str:
    parts: list[str] = []
    for arg in node.args:
        parts.extend(_node_string_literals(arg))
    for keyword in node.keywords:
        parts.extend(_node_string_literals(keyword.value))
    return " ".join(parts)


def _string_literals(tree: ast.AST) -> list[str]:
    values: list[str] = []
    for node in ast.walk(tree):
        values.extend(_node_string_literals(node))
    return values


def _node_string_literals(node: ast.AST) -> list[str]:
    values: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            values.append(child.value)
        elif isinstance(child, ast.JoinedStr):
            for part in child.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    values.append(part.value)
    return values


def _unsupported_baseenv_api_errors(tree: ast.Module) -> list[str]:
    unsupported = {
        "add_collision_handler": (
            "generated code must not call self.add_collision_handler; use sensor "
            "regions plus check_objective()/get_ground_truth() for code-level "
            "collision truth"
        ),
        "on_collision": "generated code must not invent collision callback APIs",
    }
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "self":
            continue
        message = unsupported.get(node.func.attr)
        if message:
            errors.append(message)
    return errors


def _only_env_class(tree: ast.Module) -> ast.ClassDef | None:
    env_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and _inherits_from_base_env(node)
    ]
    return env_classes[0] if len(env_classes) == 1 else None


def _has_constant_string(tree: ast.Module, value: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == value:
            return True
    return False


def diagnose_spec_error(error_text: str) -> str | None:
    """Map low-level keyword errors to environment_spec.json repair feedback."""

    if "Vec2d.__new__" in error_text and "positional argument" in error_text:
        return (
            "Vec2d constructor contract violation: do not call pymunk.Vec2d with a "
            "single tuple/list argument. Use plain (x, y) tuples for positions, or "
            "pymunk.Vec2d(x, y) when vector math is needed."
        )

    attribute_diagnostic = _diagnose_attribute_error(error_text)
    if attribute_diagnostic:
        return attribute_diagnostic

    keyword_match = re.search(
        r"(?:unexpected|unsupported) keyword(?: argument)? ['\"]([^'\"]+)['\"]",
        error_text,
        re.IGNORECASE,
    )
    if not keyword_match:
        return None
    keyword = keyword_match.group(1)
    function_name = _function_name_from_error(error_text) or "a BaseEnv helper"
    return _spec_keyword_message(keyword, function_name)


def _diagnose_attribute_error(error_text: str) -> str | None:
    if "AttributeError" not in error_text and "has no attribute" not in error_text:
        return None
    if "ObjectRecord" in error_text and "has no attribute" in error_text:
        match = re.search(r"has no attribute ['\"]([^'\"]+)['\"]", error_text)
        attribute_name = match.group(1) if match else "the requested attribute"
        if attribute_name == "position":
            return (
                "AttributeError: ObjectRecord has no direct position property. "
                "self.agent and self.get_object(name) return ObjectRecord values; "
                "use .body.position for coordinates or self.distance_between('agent', name) "
                "for touch/proximity objectives."
            )
        return (
            f"AttributeError: ObjectRecord has no direct {attribute_name!r} property. "
            "Use ObjectRecord.body for Pymunk body state, or store object names in "
            "the mandatory registry and retrieve records with self.get_object(name)."
        )
    match = re.search(r"has no attribute ['\"]([^'\"]+)['\"]", error_text)
    if not match:
        return None
    variable_name = match.group(1)
    return (
        f"AttributeError: You are referencing a variable self.{variable_name} "
        "that does not exist. You must explicitly define this variable in your "
        "__init__ method before using it."
    )


def _function_name_from_error(error_text: str) -> str | None:
    helper_match = re.search(r"\b(create_\w+|register_constraint|register_force_zone)\b", error_text)
    if helper_match:
        return helper_match.group(1)
    return None


def _forbidden_spec_arguments() -> set[str]:
    project = ENVIRONMENT_SPEC.get("Project_Constraints", {})
    if not isinstance(project, dict):
        return set()
    forbidden = project.get("Forbidden_Arguments", [])
    if not isinstance(forbidden, list):
        return set()
    return {str(argument) for argument in forbidden}


def _spec_keyword_message(keyword: str | None, function_name: str | None) -> str:
    if not keyword:
        return "Generated code must not use **kwargs expansion with BaseEnv helpers."

    helper_name = function_name or "a BaseEnv helper"
    api_reference = ENVIRONMENT_SPEC.get("API_Reference", {})
    helper_spec = api_reference.get(helper_name, {}) if isinstance(api_reference, dict) else {}
    note = helper_spec.get("note") if isinstance(helper_spec, dict) else None
    allowed = helper_spec.get("args", []) if isinstance(helper_spec, dict) else []

    if keyword in _forbidden_spec_arguments():
        message = (
            f"You tried to use {keyword!r} in {helper_name!r}. "
            "According to the environment_spec.json, that is forbidden."
        )
        if helper_name == "create_static_box" and keyword == "angle":
            return message + " Please use Pymunk segments for tilted platforms instead."
        if note:
            return f"{message} {note}"
        return message

    if allowed:
        return (
            f"{helper_name} got unsupported keyword {keyword!r}; "
            f"environment_spec.json allows: {sorted(map(str, allowed))}"
        )
    return f"{helper_name} got unsupported keyword {keyword!r}."


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _forbidden_method_argument_errors(env_class: ast.ClassDef) -> list[str]:
    forbidden = _forbidden_spec_arguments()
    errors: list[str] = []
    for node in env_class.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for arg in [*node.args.args, *node.args.kwonlyargs]:
            if arg.arg in forbidden:
                errors.append(
                    f"{node.name} must not accept forbidden argument {arg.arg!r} "
                    "from environment_spec.json"
                )
    return errors


def _method_signature_errors(env_class: ast.ClassDef) -> list[str]:
    expected_positional = {
        "add_objects": 1,
        "build_world": 1,
        "check_objective": 1,
        "get_ground_truth": 1,
        "after_step": 1,
    }
    errors: list[str] = []
    for method_name, expected_count in expected_positional.items():
        method = _find_method(env_class, method_name)
        if method is None:
            continue
        positional_count = len(method.args.args) + len(method.args.posonlyargs)
        if positional_count != expected_count or method.args.vararg is not None:
            errors.append(
                f"{method_name} must have signature {method_name}(self); "
                "BaseEnv does not pass extra positional arguments."
            )
        if method.args.kwonlyargs or method.args.kwarg is not None:
            errors.append(
                f"{method_name} must not require keyword-only or **kwargs arguments."
            )
    return errors


def _mandatory_definition_errors(env_class: ast.ClassDef) -> list[str]:
    mandatory = ENVIRONMENT_SPEC.get("Mandatory_Definitions", {})
    if not isinstance(mandatory, dict):
        return []
    init_definitions = mandatory.get("In_Init", [])
    if not isinstance(init_definitions, list):
        return []

    required_attrs = {
        attr_name
        for definition in init_definitions
        if (attr_name := _attr_name_from_definition(str(definition)))
    }
    if not required_attrs:
        return []

    init_method = _find_method(env_class, "__init__")
    if init_method is None:
        return [
            f"__init__ must define mandatory variable self.{attr_name}"
            for attr_name in sorted(required_attrs)
        ]

    assigned_attrs = _self_attrs_assigned_in_method(init_method)
    return [
        (
            f"__init__ must define mandatory variable self.{attr_name} from "
            "environment_spec.json before using it."
        )
        for attr_name in sorted(required_attrs - assigned_attrs)
    ]


def _layout_plan_definition_errors(env_class: ast.ClassDef) -> list[str]:
    init_method = _find_method(env_class, "__init__")
    if init_method is None:
        return []
    node = _self_assignment_value(init_method, "layout_plan")
    if node is None:
        return []
    if isinstance(node, ast.Dict):
        keys = _dict_literal_keys(node)
        if keys and "layout_type" not in keys:
            return ["self.layout_plan should include a 'layout_type' field from the route-aware layout contract."]
        return []
    if isinstance(node, ast.Call) and _call_name(node) == "dict":
        return []
    return ["self.layout_plan must be initialized as a JSON-like dict."]


def _objective_metadata_errors(env_class: ast.ClassDef) -> list[str]:
    init_method = _find_method(env_class, "__init__")
    if init_method is None:
        return []

    errors: list[str] = []
    objective_type = _self_constant_assignment(init_method, "objective_type")
    if objective_type is None:
        errors.append("__init__ must define self.objective_type from environment_spec.json.")
    elif not isinstance(objective_type, str):
        errors.append("self.objective_type must be a string literal from environment_spec.json.")
    elif objective_type not in _allowed_objective_types():
        errors.append(
            f"self.objective_type {objective_type!r} is not allowed; "
            f"allowed: {sorted(_allowed_objective_types())}"
        )

    targets_node = _self_assignment_value(init_method, "objective_targets")
    if targets_node is None:
        errors.append("__init__ must define self.objective_targets as a list of registered object names.")
    elif not isinstance(targets_node, ast.List):
        errors.append("self.objective_targets must be initialized as a list.")
    else:
        for element in targets_node.elts:
            if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                errors.append("self.objective_targets may only contain string object names.")
                break
    errors.extend(_profile_definition_errors(init_method, objective_type))
    return errors


def _profile_definition_errors(
    init_method: ast.FunctionDef,
    objective_type: object | None,
) -> list[str]:
    errors: list[str] = []

    objective_profile_node = _self_assignment_value(init_method, "objective_profile")
    capability_profile_node = _self_assignment_value(init_method, "capability_profile")

    objective_required = _required_profile_fields("Objective_Profile")
    capability_required = _required_profile_fields("Capability_Profile")

    if objective_profile_node is None:
        errors.append("__init__ must define self.objective_profile from environment_spec.json.")
    elif not isinstance(objective_profile_node, ast.Dict):
        errors.append("self.objective_profile must be initialized as a dict literal.")
    else:
        objective_keys = _dict_literal_keys(objective_profile_node)
        missing = objective_required - objective_keys
        if missing:
            errors.append(
                "self.objective_profile missing required fields from environment_spec.json: "
                f"{sorted(missing)}"
            )
        profile_type = _dict_constant_value(objective_profile_node, "objective_type")
        if isinstance(profile_type, str) and isinstance(objective_type, str):
            if profile_type != objective_type:
                errors.append(
                    "self.objective_profile['objective_type'] must match self.objective_type."
                )
        minimum_tier = _dict_constant_value(objective_profile_node, "minimum_acceptance_tier")
        if minimum_tier is not None:
            if not isinstance(minimum_tier, int) or not 3 <= minimum_tier <= 5:
                errors.append(
                    "self.objective_profile['minimum_acceptance_tier'] must be an integer from 3 to 5."
                )
        subgoals = _dict_assignment_value(objective_profile_node, "subgoals")
        if subgoals is None:
            errors.append("self.objective_profile['subgoals'] must be defined.")
        elif not isinstance(subgoals, ast.List):
            errors.append("self.objective_profile['subgoals'] must be a list.")
        else:
            for index, subgoal in enumerate(subgoals.elts, start=1):
                if not isinstance(subgoal, ast.Dict):
                    errors.append(
                        f"self.objective_profile['subgoals'][{index}] must be a dict."
                    )
                    continue
                if "kind" not in _dict_literal_keys(subgoal):
                    errors.append(
                        f"self.objective_profile['subgoals'][{index}] must include a 'kind'."
                    )

    if capability_profile_node is None:
        errors.append("__init__ must define self.capability_profile from environment_spec.json.")
    elif not isinstance(capability_profile_node, ast.Dict):
        errors.append("self.capability_profile must be initialized as a dict literal.")
    else:
        capability_keys = _dict_literal_keys(capability_profile_node)
        missing = capability_required - capability_keys
        if missing:
            errors.append(
                "self.capability_profile missing required fields from environment_spec.json: "
                f"{sorted(missing)}"
            )
        allowed_controls = _dict_assignment_value(capability_profile_node, "allowed_controls")
        forbidden_controls = _dict_assignment_value(capability_profile_node, "forbidden_controls")
        if allowed_controls is not None and not isinstance(allowed_controls, ast.List):
            errors.append("self.capability_profile['allowed_controls'] must be a list.")
        if forbidden_controls is not None and not isinstance(forbidden_controls, ast.List):
            errors.append("self.capability_profile['forbidden_controls'] must be a list.")

    return errors


def _required_profile_fields(spec_key: str) -> set[str]:
    profile_spec = ENVIRONMENT_SPEC.get(spec_key, {})
    if not isinstance(profile_spec, dict):
        return set()
    required_fields = profile_spec.get("Required_Fields", {})
    if not isinstance(required_fields, dict):
        return set()
    return {str(field_name) for field_name in required_fields}


def _dict_literal_keys(node: ast.Dict) -> set[str]:
    keys: set[str] = set()
    for key in node.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.add(key.value)
    return keys


def _dict_constant_value(node: ast.Dict, key_name: str) -> object | None:
    value_node = _dict_assignment_value(node, key_name)
    if isinstance(value_node, ast.Constant):
        return value_node.value
    return None


def _dict_assignment_value(node: ast.Dict, key_name: str) -> ast.AST | None:
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and key.value == key_name:
            return value
    return None


def _ground_truth_metadata_errors(env_class: ast.ClassDef) -> list[str]:
    method = _find_method(env_class, "get_ground_truth")
    if method is None:
        return []
    errors: list[str] = []
    calls_super = _method_calls_super_get_ground_truth(method)
    if not calls_super:
        errors.append("get_ground_truth must call super().get_ground_truth() to preserve objective metadata.")
        string_constants = {
            node.value
            for node in ast.walk(method)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        required_keys = _required_objective_ground_truth_keys()
        missing_keys = required_keys - string_constants
        if missing_keys:
            errors.append(
                "get_ground_truth must preserve/export objective metadata keys: "
                f"{sorted(missing_keys)}"
            )
        return errors

    # Calling super().get_ground_truth() is enough to preserve the default objective block.
    # If the generated class extends the block, the prompt asks it to include the same keys.
    return errors


def _method_calls_super_get_ground_truth(method: ast.FunctionDef) -> bool:
    for node in ast.walk(method):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "get_ground_truth":
            continue
        value = func.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "super":
            return True
    return False


def _required_objective_ground_truth_keys() -> set[str]:
    metadata = ENVIRONMENT_SPEC.get("Ground_Truth_Objective_Metadata", {})
    if not isinstance(metadata, dict):
        return {"objective_type", "objective_targets", "objective_satisfied"}
    required = metadata.get("Required_Keys", [])
    if not isinstance(required, list):
        return {"objective_type", "objective_targets", "objective_satisfied"}
    return {str(key) for key in required}


def _allowed_objective_types() -> set[str]:
    objective_types = ENVIRONMENT_SPEC.get("Objective_Types", {})
    if not isinstance(objective_types, dict):
        return set()
    allowed = objective_types.get("Allowed", [])
    if not isinstance(allowed, list):
        return set()
    return {str(item) for item in allowed}


def _self_constant_assignment(method: ast.FunctionDef, attr_name: str) -> object | None:
    value_node = _self_assignment_value(method, attr_name)
    if isinstance(value_node, ast.Constant):
        return value_node.value
    return None


def _self_assignment_value(method: ast.FunctionDef, attr_name: str) -> ast.AST | None:
    for node in ast.walk(method):
        targets: list[ast.expr] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == attr_name
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                return value
    return None


def _state_registry_errors(env_class: ast.ClassDef) -> list[str]:
    init_method = _find_method(env_class, "__init__")
    if init_method is None:
        return []

    registry_attrs = _self_attrs_assigned_in_method(init_method)
    allowed_attrs = registry_attrs | _base_env_allowed_self_attrs()
    errors: list[str] = []
    read_only_runtime_attrs = {"time", "step_count", "dt"}
    for attr in sorted(registry_attrs & read_only_runtime_attrs):
        errors.append(
            f"__init__ assigns self.{attr}, but self.{attr} is a BaseEnv read-only "
            "runtime global. Read it for timing; do not define or reset it in generated code."
        )

    for method in (node for node in env_class.body if isinstance(node, ast.FunctionDef)):
        if method.name == "__init__":
            continue
        for node in ast.walk(method):
            if not (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            ):
                continue
            if node.attr in allowed_attrs:
                continue
            if _is_method_call_attribute(node):
                continue
            if isinstance(node.ctx, ast.Store):
                if node.attr in read_only_runtime_attrs:
                    errors.append(
                        f"{method.name} assigns self.{node.attr}, but self.{node.attr} is a BaseEnv "
                        "read-only runtime global. Use a separate registry variable if you need custom state."
                    )
                    continue
                errors.append(
                    f"{method.name} assigns self.{node.attr} outside the mandatory "
                    "state registry; declare it in __init__ first."
                )
            else:
                errors.append(
                    f"{method.name} references self.{node.attr} before it is declared "
                    "in the mandatory state registry."
                )
    return sorted(set(errors))


def _object_model_errors(env_class: ast.ClassDef) -> list[str]:
    errors: list[str] = []
    for method in (node for node in env_class.body if isinstance(node, ast.FunctionDef)):
        object_record_aliases = _object_record_aliases(method)
        for node in ast.walk(method):
            if _is_self_objects_access(node):
                errors.append(
                    f"{method.name} must not use self.objects/self._objects; store registered "
                    "object names in registry lists and call self.get_object(name)."
                )
            if _is_direct_object_record_position(node, object_record_aliases):
                errors.append(
                    f"{method.name} accesses ObjectRecord.position directly. ObjectRecord has "
                    "no direct position; use record.body.position or self.distance_between(...)."
                )
    return sorted(set(errors))


def _objective_side_effect_errors(env_class: ast.ClassDef) -> list[str]:
    errors: list[str] = []
    for method_name in ("check_objective", "get_ground_truth"):
        method = _find_method(env_class, method_name)
        if method is None:
            continue
        for node in ast.walk(method):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
                if call_name in {
                    "apply_force_at_world_point",
                    "apply_impulse_at_world_point",
                    "apply_impulse_at_local_point",
                    "apply_agent_force",
                    "step",
                    "reset",
                }:
                    errors.append(
                        f"{method_name} must be a pure telemetry/objective method; do not call {call_name}() there."
                    )
            if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):
                targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    attr_path = _attribute_path(target)
                    if attr_path and any(
                        token in attr_path
                        for token in (
                            ".body.position",
                            ".body.velocity",
                            ".body.angle",
                            ".body.angular_velocity",
                        )
                    ):
                        errors.append(
                            f"{method_name} must not mutate physics state via {attr_path}; move effects into physical setup or after_step()."
                        )
    return sorted(set(errors))


def _body_callback_assignment_errors(env_class: ast.ClassDef) -> list[str]:
    errors: list[str] = []
    for method in (node for node in ast.walk(env_class) if isinstance(node, ast.FunctionDef)):
        for node in ast.walk(method):
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value = node.value
            else:
                continue
            if value is None:
                continue
            value_is_none = isinstance(value, ast.Constant) and value.value is None
            for target in targets:
                attr_path = _attribute_path(target)
                if not attr_path:
                    continue
                if attr_path.endswith(".body.velocity_func") or attr_path.endswith(".body.position_func"):
                    errors.append(
                        f"{method.name} must not assign {attr_path}; Pymunk callback fields must stay callable."
                    )
                elif value_is_none and (
                    attr_path.endswith(".velocity_func") or attr_path.endswith(".position_func")
                ):
                    errors.append(
                        f"{method.name} must not set {attr_path} = None; Pymunk will crash during stepping."
                    )
    return sorted(set(errors))


def _objective_reset_hook_errors(env_class: ast.ClassDef) -> list[str]:
    """Require deterministic reset for objective state mutated during checks."""

    check_method = _find_method(env_class, "check_objective")
    if check_method is None:
        return []

    mutated_attrs: set[str] = set()
    for node in ast.walk(check_method):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets.extend(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets.append(node.target)
        elif isinstance(node, ast.AugAssign):
            targets.append(node.target)
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                mutated_attrs.add(target.attr)

    if not mutated_attrs:
        return []

    reset_method = _find_method(env_class, "reset_objective_state")
    if reset_method is None:
        names = ", ".join(f"self.{name}" for name in sorted(mutated_attrs))
        return [
            "check_objective mutates per-run objective state "
            f"({names}) but reset_objective_state(self) is missing; "
            "reset those variables there so replay is deterministic."
        ]

    reset_attrs = _self_attrs_assigned_in_method(reset_method)
    missing = sorted(name for name in mutated_attrs if name not in reset_attrs)
    if missing:
        names = ", ".join(f"self.{name}" for name in missing)
        return [
            "reset_objective_state(self) must reset every objective variable "
            f"mutated by check_objective; missing resets for {names}."
        ]
    return []


def _attribute_path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Subscript):
        return _attribute_path(node.value)
    if isinstance(node, ast.Call):
        return _attribute_path(node.func)
    return None


def _object_record_aliases(method: ast.FunctionDef) -> set[str]:
    aliases: set[str] = set()
    for node in ast.walk(method):
        if not isinstance(node, ast.Assign):
            continue
        if not _expression_returns_object_record(node.value):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                aliases.add(target.id)
    return aliases


def _expression_returns_object_record(node: ast.AST) -> bool:
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "agent"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr
            in {
                "get_object",
                "get_agent_record",
                "create_dynamic_circle",
                "create_static_segment",
                "create_static_box",
                "create_dynamic_box",
                "register_object",
                "register_force_zone",
                "create_pressure_plate",
                "create_horizontal_push_lane",
                "create_pressure_plate_gate_corridor",
                "create_field_push_lane",
                "create_strike_shot_lane",
                "create_ballistic_hoop_challenge",
                "create_support_exit_freefall_challenge",
                "create_recurring_falling_hazards",
                "create_recurring_lateral_hazards",
                "register_pressure_plate_gate",
            }
        ):
            return True
    return False


def _is_self_objects_access(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr in {"objects", "_objects"}
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _is_direct_object_record_position(node: ast.AST, aliases: set[str]) -> bool:
    if not isinstance(node, ast.Attribute) or node.attr != "position":
        return False
    value = node.value
    if (
        isinstance(value, ast.Attribute)
        and value.attr == "agent"
        and isinstance(value.value, ast.Name)
        and value.value.id == "self"
    ):
        return True
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "self"
        and value.func.attr in {"get_object", "get_agent_record"}
    ):
        return True
    if isinstance(value, ast.Name) and value.id in aliases:
        return True
    return False


def _base_env_allowed_self_attrs() -> set[str]:
    globals_ = ENVIRONMENT_SPEC.get("Global_Variables", [])
    allowed = set()
    if isinstance(globals_, list):
        for item in globals_:
            match = re.match(r"self\.([A-Za-z_][A-Za-z0-9_]*)$", str(item))
            if match:
                allowed.add(match.group(1))
    allowed.update(
        {
            "config",
            "space",
            "width",
            "height",
            "time",
            "step_count",
            "dt",
            "agent",
            "agent_strength",
            "solvability_check",
            "create_dynamic_circle",
            "create_static_segment",
            "create_static_box",
            "create_dynamic_box",
            "register_object",
            "register_constraint",
            "register_force_zone",
            "create_pressure_plate",
            "create_horizontal_push_lane",
            "create_pressure_plate_gate_corridor",
            "create_field_push_lane",
            "create_strike_shot_lane",
            "create_ballistic_hoop_challenge",
            "create_ballistic_barrier_goal_challenge",
            "create_support_exit_freefall_challenge",
            "create_recurring_falling_hazards",
            "create_recurring_lateral_hazards",
            "register_pressure_plate_gate",
            "is_mechanism_activated",
            "is_object_on_trigger",
            "set_solvability_hint",
            "get_object",
            "distance_between",
            "get_agent_record",
            "apply_agent_force",
            "add_objects",
            "build_world",
            "check_objective",
            "get_ground_truth",
            "reset",
            "reset_objective_state",
        }
    )
    return allowed


def _is_method_call_attribute(node: ast.Attribute) -> bool:
    parent = getattr(node, "_parent", None)
    return isinstance(parent, ast.Call) and parent.func is node


def _attr_name_from_definition(definition: str) -> str | None:
    match = re.match(r"\s*self\.([A-Za-z_][A-Za-z0-9_]*)\s*=", definition)
    if match:
        return match.group(1)
    return None


def _find_method(env_class: ast.ClassDef, method_name: str) -> ast.FunctionDef | None:
    for node in env_class.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    return None


def _self_attrs_assigned_in_method(method: ast.FunctionDef) -> set[str]:
    assigned: set[str] = set()
    for node in ast.walk(method):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets.extend(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets.append(node.target)
        elif isinstance(node, ast.AugAssign):
            targets.append(node.target)
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                assigned.add(target.attr)
    return assigned


def _uses_undefined_world_aliases(tree: ast.Module) -> bool:
    forbidden_names = {"_W", "_H", "WIDTH", "HEIGHT"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            return True
    return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Harness Alpha World Architect")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prompt_parser = subparsers.add_parser("prompt", help="render the LLM prompt")
    prompt_parser.add_argument("world_request", help="natural language world request")
    prompt_parser.add_argument("--class-name", help="override generated class name")

    verify_parser = subparsers.add_parser("verify", help="verify generated Python code")
    verify_parser.add_argument("code_file", type=Path, help="path to generated Python code")
    verify_parser.add_argument("--class-name", help="expected environment class name")

    save_parser = subparsers.add_parser("save", help="verify and save an LLM response")
    save_parser.add_argument("world_request", help="natural language world request")
    save_parser.add_argument("response_file", type=Path, help="file containing LLM response")
    save_parser.add_argument("--class-name", help="override generated class name")
    save_parser.add_argument("--output-dir", type=Path, default=GENERATED_ENVS_DIR)

    generate_parser = subparsers.add_parser(
        "generate",
        help="call OpenAI, verify, and save generated Python code",
    )
    generate_parser.add_argument("world_request", help="natural language world request")
    generate_parser.add_argument("--class-name", help="override generated class name")
    generate_parser.add_argument("--output-dir", type=Path, default=GENERATED_ENVS_DIR)
    generate_parser.add_argument("--model", default=os.getenv("OPENAI_ARCHITECT_MODEL", DEFAULT_OPENAI_MODEL))
    generate_parser.add_argument(
        "--reasoning-effort",
        default=os.getenv("OPENAI_ARCHITECT_REASONING_EFFORT", DEFAULT_REASONING_EFFORT),
    )
    generate_parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=int(os.getenv("OPENAI_ARCHITECT_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)),
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "prompt":
        print(render_prompt(args.world_request, class_name=args.class_name))
        return

    if args.command == "verify":
        code = args.code_file.read_text(encoding="utf-8")
        result = verify_generated_code(code, expected_class_name=args.class_name)
        if not result.ok:
            parser.exit(1, "\n".join(result.errors) + "\n")
        print(f"verified: {result.class_name}")
        return

    if args.command == "save":
        response = args.response_file.read_text(encoding="utf-8")
        output_path = save_generated_code(
            args.world_request,
            response,
            class_name=args.class_name,
            output_dir=args.output_dir,
        )
        print(output_path)
        return

    if args.command == "generate":
        config = OpenAIArchitectConfig(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
        )
        try:
            output_path = asyncio.run(
                generate_and_save_openai(
                    args.world_request,
                    class_name=args.class_name,
                    output_dir=args.output_dir,
                    config=config,
                )
            )
        except RuntimeError as exc:
            parser.exit(1, f"{exc}\n")
        print(output_path)
        return

    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
