"""Gameplay Architect agent for high-level game-feel contracts.

This agent turns a raw user prompt into a structured gameplay_profile. The
profile captures the "unspoken game designer" expectations: cadence, fairness,
readability, responsiveness, and ongoing interaction loops. It is a design
contract, not executable physics code.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

from prompt_cache import load_cached_json, save_cached_json


DEFAULT_GAMEPLAY_MODEL = os.getenv("OPENAI_GAMEPLAY_MODEL", "gpt-5.2")
DEFAULT_GAMEPLAY_REASONING = os.getenv("OPENAI_GAMEPLAY_REASONING_EFFORT", "low")
DEFAULT_GAMEPLAY_MAX_OUTPUT = int(os.getenv("OPENAI_GAMEPLAY_MAX_OUTPUT_TOKENS", "2400"))
GAMEPLAY_PROFILE_CACHE_VERSION = "gameplay_profile.v7"

load_dotenv()


class GameplayArchitect:
    """LLM-backed gameplay-profile designer with deterministic fallback."""

    def __init__(self) -> None:
        self._client = None

    async def design(self, prompt: str, simulation_brief: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return a structured gameplay_profile for a prompt."""

        cache_key = _cache_key(prompt, simulation_brief)
        cached = load_cached_json("gameplay_profile", cache_key)
        if cached is not None:
            return _normalize_profile(cached, prompt, simulation_brief=simulation_brief)

        if os.getenv("OPENAI_API_KEY"):
            try:
                profile = _normalize_profile(
                    await self._design_with_openai(prompt, simulation_brief),
                    prompt,
                    simulation_brief=simulation_brief,
                )
                save_cached_json("gameplay_profile", cache_key, profile)
                return profile
            except Exception:
                # Keep the generation path robust. The profile is helpful, not
                # allowed to block the proven physics pipeline.
                return _fallback_profile(prompt, simulation_brief=simulation_brief)
        profile = _fallback_profile(prompt, simulation_brief=simulation_brief)
        save_cached_json("gameplay_profile", cache_key, profile)
        return profile

    async def _design_with_openai(
        self,
        prompt: str,
        simulation_brief: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        client = self._get_client()
        response = await client.responses.create(
            model=DEFAULT_GAMEPLAY_MODEL,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are the Gameplay Architect for a deterministic 2D physics world factory. "
                        "Output only compact JSON. Do not write Python code. Describe gameplay feel, "
                        "cadence, fairness, readability, and validator expectations."
                    ),
                },
                {
                    "role": "user",
                    "content": _profile_prompt(prompt, simulation_brief=simulation_brief),
                },
            ],
            reasoning={"effort": DEFAULT_GAMEPLAY_REASONING},
            max_output_tokens=DEFAULT_GAMEPLAY_MAX_OUTPUT,
        )
        text = str(getattr(response, "output_text", "") or "").strip()
        if not text:
            raise RuntimeError("empty gameplay profile response")
        return _extract_json(text)

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI()
        return self._client


def _profile_prompt(prompt: str, simulation_brief: dict[str, Any] | None = None) -> str:
    brief_block = ""
    if simulation_brief:
        brief_block = (
            "\nSIMULATION BRIEF (treat as the physical interpretation contract):\n"
            f"{json.dumps(simulation_brief, indent=2, sort_keys=True)}\n"
        )
    return f"""
Create a gameplay_profile JSON object for this 2D Pymunk training-world prompt:

{prompt}
{brief_block}

The profile must be a design contract for the Physics Architect and Validator.
It must not include Python code.

Required top-level keys:
- gameplay_loop: short snake_case string
- world_context: dict with world_perspective, gravity_model, movement_model, support_assumption, route_assumption, rationale
- difficulty: easy, medium, hard
- dynamics: list of gameplay dynamic objects
- feel_targets: dict with readability, responsiveness, forgiveness, feedback
- fairness_rules: list of concrete rules
- validator_expectations: list of measurable checks
- implementation_notes: list of concrete implementation hints

Important design principles:
- Do not invent a major mechanic the user did not ask for. You may enrich cadence/readability, but if the prompt only asks for chasers, do not add falling rocks; if it asks for pushing, do not add hazards.
- User-stated physics constraints override genre defaults. If the prompt says "zero gravity", "no gravity", "low gravity", "high gravity", or "normal gravity", preserve that gravity model even if the theme usually implies something else.
- Interpret physical verbs by context. "Shots/shooting" from spaceships, lasers, turrets, cannons, bullets, missiles, or enemies means projectile hazards/weapon fire only when the prompt describes incoming/enemy/avoid/survival pressure. If the AGENT shoots/fires a bullet/projectile to knock/topple/hit/break a stack, target, or object, that projectile is the agent's tool and the loop is projectile-impact manipulation, not survival against projectile hazards.
- Infer the game perspective from contextual cues before specifying mechanics. Lava/cave/platform/exit-at-top/right/falling-hazard prompts usually imply side-view platformer physics with normal gravity and ramps/stairs. Maze/room/arena/Pacman prompts usually imply top-down or flat-floor navigation. Zero-g/space/asteroid/orbit prompts imply zero-gravity thrust.
- If the world_context says side_view_platformer, require normal gravity, reachable support surfaces, and a continuous ramp or shallow stair route. Do not use isolated jump platforms for first-pass validation; the oracle is a force controller, not a skilled platformer player.
- If the world_context says top_down_arena, avoid vertical platformer assumptions and use zero gravity or full-floor support so objects do not fall out of the arena.
- Falling/raining/dropping hazards should usually be staggered and recurring, not all fired once.
- Throw/lob basketball/ball into a hoop should be validator-friendly by default: design it as a ballistic object entering a generous non-blocking hoop sensor. Do not require strict real-sports rim-plane/downward-crossing/aperture scoring unless the user explicitly asks for regulation basketball rules.
- Push/shove/move-object tasks need heavy-but-visibly-movable interaction; the object should move clearly under agent force.
- Chase tasks need pursuit that is readable and escapable; enemies should be slower than agent max control.
- Avoidance tasks need safe windows or safe lanes.
- The world should feel like a small playable 2D game, not just a static physics diagram.

Preferred dynamic type names when applicable:
- recurring_falling_hazard
- heavy_but_movable_push
- readable_chaser
- readable_projectile_hazard
""".strip()


def _cache_key(prompt: str, simulation_brief: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "version": GAMEPLAY_PROFILE_CACHE_VERSION,
        "prompt": prompt,
        "simulation_brief": simulation_brief or {},
        "model": DEFAULT_GAMEPLAY_MODEL,
        "reasoning_effort": DEFAULT_GAMEPLAY_REASONING,
        "max_output_tokens": DEFAULT_GAMEPLAY_MAX_OUTPUT,
    }


def _fallback_profile(prompt: str, simulation_brief: dict[str, Any] | None = None) -> dict[str, Any]:
    text = prompt.lower()
    brief_world = {}
    if isinstance(simulation_brief, dict) and isinstance(simulation_brief.get("world_context"), dict):
        brief_world = simulation_brief["world_context"]
    profile: dict[str, Any] = {
        "source": "deterministic_fallback",
        "gameplay_loop": "physics_objective",
        "world_context": _world_context_from_brief(brief_world) or _infer_world_context(prompt),
        "difficulty": "medium",
        "dynamics": [],
        "feel_targets": {
            "readability": "high",
            "responsiveness": "medium_high",
            "forgiveness": "medium",
            "feedback": "visible_state_change",
        },
        "fairness_rules": [
            "avoid pixel-perfect success requirements",
            "make the intended interaction visually readable",
            "preserve deterministic code-level objective truth",
        ],
        "validator_expectations": ["objective_can_be_checked_from_pymunk_state"],
        "implementation_notes": [
            "store tuning values in self.layout so deterministic repair can adjust them",
            "export gameplay_profile in get_ground_truth objective metadata",
        ],
    }
    if _prompt_requests_projectile_hazards(prompt):
        profile.update(
            {
                "gameplay_loop": "navigate_while_avoiding_projectile_hazards",
                "dynamics": [
                    {
                        "type": "readable_projectile_hazard",
                        "hazard_name_pattern": "enemy_shot_*",
                        "cadence": "staggered_continuous",
                        "projectile_speed": "slow_readable_validator_friendly",
                        "max_simultaneous_projectiles": 6,
                        "projectile_speed_relative_to_agent_max": 0.4,
                        "projectile_radius_relative_to_agent_radius": 0.25,
                        "preferred_shooter_motion": "stationary_turrets_or_very_slow_enemies",
                        "minimum_visible_travel_px": 120,
                        "telegraph_seconds": 0.35,
                        "continues_until_objective": True,
                        "safe_lane_policy": "at_least_one_escape_vector",
                    }
                ],
                "fairness_rules": [
                    "projectile shots should be visible hazards, not soccer balls or scoring objects",
                    "do not spawn all shots at once",
                    "prefer stationary perimeter shooters for first-pass survival reliability",
                    "use slow readable projectiles with clear gaps rather than dense crossfire",
                    "projectiles should start away from immediate agent overlap",
                    "always leave at least one dodge route or timing window",
                ],
                "validator_expectations": [
                    "projectile_hazards_exist",
                    "projectile_hazards_move_through_play_area",
                    "at_least_one_projectile_travels_120_px_in_headless_probe",
                    "agent_has_reachable_safe_path_or_survival_window",
                ],
                "implementation_notes": [
                    "define enemy_shot or projectile hazard names in the registry",
                    "register shots as dynamic role='hazard' objects with nonzero velocity or bounded after_step force",
                    "make at least one projectile active immediately or within the first 30 simulation steps",
                    "do not name inactive parked projectile pool bodies in a way that matches semantic_requirements",
                    "declare every projectile timer/counter/initial-shot flag in __init__ before after_step uses it",
                    "use survival/avoidance/reach subgoals; do not use strike_object_to_region for enemy shots",
                    "if the prompt is zero-g/space, keep zero gravity and move projectiles with initial velocity or force zones",
                ],
            }
        )
    elif _has_any(text, "falling", "raining", "dropping", "fireball", "meteor", "from sky"):
        profile.update(
            {
                "gameplay_loop": "navigate_while_avoiding_recurring_hazards",
                "dynamics": [
                    {
                        "type": "recurring_falling_hazard",
                        "hazard_name_pattern": "fireball_*",
                        "cadence": "staggered_continuous",
                        "spawn_lanes": 4,
                        "phase_offsets_seconds": [0.0, 0.7, 1.4, 2.1],
                        "reset_when_below_world": True,
                        "continues_until_objective": True,
                        "minimum_visible_fall_px": 180,
                        "safe_lane_policy": "at_least_one_lane_open",
                    }
                ],
                "fairness_rules": [
                    "do not drop every hazard at the same time",
                    "always leave at least one safe lane or safe timing window",
                    "do not spawn hazards directly on the agent",
                    "hazards should continue until the objective is complete",
                ],
                "validator_expectations": [
                    "falling_hazards_recur_over_time",
                    "hazards_have_staggered_phase_offsets",
                    "hazards_visibly_fall_inside_world_bounds",
                    "agent_has_reachable_safe_path_to_goal",
                ],
                "implementation_notes": [
                    "define self.gameplay_profile in __init__",
                    "use self.hazard_timers or self.hazard_phase_offsets in the registry",
                    "use after_step() to reset falling hazards to top when below the world",
                    "set each hazard's phase offset so they do not all begin falling simultaneously",
                    "keep at least one lane open by spacing hazard lanes away from the main route",
                ],
            }
        )
    elif _has_any(text, "push", "shove", "move", "rock", "box", "crate", "pressure plate", "platform"):
        profile.update(
            {
                "gameplay_loop": "push_object_to_target",
                "dynamics": [
                    {
                        "type": "heavy_but_movable_push",
                        "visible_displacement_required_px": 80,
                        "agent_force_margin": 2.5,
                        "alignment_tolerance": "forgiving",
                        "target_feedback": "clear_success_region",
                    }
                ],
                "fairness_rules": [
                    "the movable object must visibly move under sustained agent force",
                    "the push target should be visually marked and non-blocking",
                    "the agent should start behind the object on the push axis",
                    "friction and mass should feel heavy but not stuck",
                ],
                "validator_expectations": [
                    "agent_contacts_push_object",
                    "object_displaces_visibly_under_force",
                    "object_can_reach_target_region",
                ],
                "implementation_notes": [
                    "set self.agent_strength from object mass with a clear force margin",
                    "keep object friction moderate and target region sensor=True",
                    "use guide rails only if they do not pin the object",
                ],
            }
        )
    elif _has_any(text, "chase", "chased", "pursue", "enemy", "ghost"):
        profile.update(
            {
                "gameplay_loop": "escape_readable_pursuers",
                "dynamics": [
                    {
                        "type": "readable_chaser",
                        "pursuit_speed": "below_agent_max",
                        "reaction_delay_seconds": 0.35,
                        "safe_route_policy": "multiple_escape_routes",
                    }
                ],
                "fairness_rules": [
                    "chasers should pursue over time, not start in immediate collision",
                    "agent must have at least one escape route",
                    "pursuit speed should be dangerous but escapable",
                ],
                "validator_expectations": [
                    "chasers_reduce_distance_to_agent",
                    "agent_can_make_progress_toward_objective",
                ],
                "implementation_notes": [
                    "use after_step() or force zones for deterministic pursuit",
                    "cap chaser velocity below agent effective speed",
                ],
            }
        )
    return _normalize_profile(profile, prompt, simulation_brief=simulation_brief)


def _normalize_profile(
    profile: dict[str, Any],
    prompt: str,
    *,
    simulation_brief: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(profile, dict):
        profile = {}
    brief_world = {}
    if isinstance(simulation_brief, dict) and isinstance(simulation_brief.get("world_context"), dict):
        brief_world = simulation_brief["world_context"]
    brief_context = _world_context_from_brief(brief_world)
    normalized = {
        "source": str(profile.get("source") or "llm_gameplay_architect"),
        "source_prompt": prompt,
        "gameplay_loop": _snake(str(profile.get("gameplay_loop") or "physics_objective")),
        "world_context": brief_context or _normalize_world_context(profile.get("world_context"), prompt),
        "difficulty": str(profile.get("difficulty") or "medium"),
        "dynamics": _normalize_dynamics(profile.get("dynamics"), prompt),
        "feel_targets": profile.get("feel_targets") if isinstance(profile.get("feel_targets"), dict) else {},
        "fairness_rules": _string_list(profile.get("fairness_rules")),
        "validator_expectations": _string_list(profile.get("validator_expectations")),
        "implementation_notes": _string_list(profile.get("implementation_notes")),
    }
    if not normalized["dynamics"]:
        prompt_defaults = _default_dynamics_for_prompt(prompt)
        if prompt_defaults:
            normalized["dynamics"] = prompt_defaults
            if _prompt_requests_projectile_hazards(prompt):
                normalized["gameplay_loop"] = "navigate_while_avoiding_projectile_hazards"
    _sanitize_profile_for_prompt(normalized, prompt)
    if isinstance(simulation_brief, dict):
        normalized["simulation_brief_alignment"] = {
            "objective_type": (simulation_brief.get("objective") or {}).get("type")
            if isinstance(simulation_brief.get("objective"), dict)
            else None,
            "semantic_must_happen": simulation_brief.get("semantic_must_happen", []),
            "agent_form": (simulation_brief.get("agent") or {}).get("form")
            if isinstance(simulation_brief.get("agent"), dict)
            else None,
        }
    _apply_simulation_brief_safety_governor(normalized, prompt, simulation_brief)
    normalized["feel_targets"].setdefault("readability", "high")
    normalized["feel_targets"].setdefault("responsiveness", "medium")
    normalized["feel_targets"].setdefault("forgiveness", "medium")
    normalized["feel_targets"].setdefault("feedback", "visible")
    return normalized


def _apply_simulation_brief_safety_governor(
    profile: dict[str, Any],
    prompt: str,
    simulation_brief: dict[str, Any] | None,
) -> None:
    """Keep LLM game-feel ideas inside validator-friendly physics bounds."""

    objective = {}
    world_context = {}
    if isinstance(simulation_brief, dict) and isinstance(simulation_brief.get("objective"), dict):
        objective = simulation_brief["objective"]
    if isinstance(simulation_brief, dict) and isinstance(simulation_brief.get("world_context"), dict):
        world_context = simulation_brief["world_context"]
    objective_type = str(objective.get("type") or "").lower()
    world_gravity = str(world_context.get("gravity") or world_context.get("gravity_model") or "").lower()
    explicit_zero_g = _prompt_explicit_gravity_model(prompt) == "zero_g" or world_gravity in {
        "zero_g",
        "zero gravity",
        "zero-gravity",
        "no gravity",
        "weightless",
        "microgravity",
    }

    if explicit_zero_g and not (objective_type == "ballistic_object_to_region" or _is_ballistic_hoop_request(prompt)):
        profile["world_context"] = _zero_g_context(
            "Explicit zero/no-gravity interpretation from prompt or Simulation Brief overrides genre defaults."
        )
        rules = _string_list(profile.get("fairness_rules"))
        rules.extend(
            [
                "Preserve zero gravity throughout the world; do not add platformer support just because the theme is lava/cave/escape.",
                "If hazards must fall in zero gravity, move them with downward velocity or a hazard-only force zone instead of changing global gravity.",
            ]
        )
        profile["fairness_rules"] = _dedupe(rules)
        notes = _string_list(profile.get("implementation_notes"))
        notes.extend(
            [
                "Use EnvConfig(gravity=(0, 0)) for explicit zero-gravity prompts.",
                "Use thrust_2d controls for the agent and bounded arena containment instead of ground-force ramps.",
                "For falling hazards under zero gravity, use initial downward velocity or register_force_zone affected_roles=['hazard'].",
            ]
        )
        profile["implementation_notes"] = _dedupe(notes)

    if objective_type == "ballistic_object_to_region" or _is_ballistic_hoop_request(prompt):
        profile["difficulty"] = "easy"
        profile["gameplay_loop"] = "contact_launch_to_sensor"
        profile["world_context"] = _side_view_platformer_context(
            "Validator-friendly ballistic hoop task: use normal gravity, stable floor, and clear arc space."
        )
        profile["dynamics"] = [
            {
                "type": "ballistic_object_to_region",
                "name": "basketball_to_hoop",
                "cadence": "single_clear_impulse_then_arc",
                "intent": "agent contacts the ball and it enters the generous hoop sensor under gravity",
                "parameters": {
                    "target_sensor": "generous_non_blocking",
                    "strict_rim_plane_scoring": False,
                    "solid_backboard_required": False,
                },
            }
        ]
        rules = _string_list(profile.get("fairness_rules"))
        rules.extend(
            [
                "Success is ball entering the non-blocking hoop sensor; do not require downward rim-plane crossing unless explicitly requested.",
                "Hoop/backboard/rim visuals must be sensor-only or outside the arc path.",
                "Keep ball-to-hoop distance short enough for the generic ballistic validator.",
            ]
        )
        profile["fairness_rules"] = _dedupe(rules)
        notes = _string_list(profile.get("implementation_notes"))
        notes.extend(
            [
                "Use create_ballistic_hoop_challenge(...) and a ballistic_object_to_region subgoal.",
                "check_objective should be a generous proximity/sensor predicate such as distance_between('basketball', 'hoop') <= touch_threshold + 70.",
                "Do not add previous-y, rim-plane, entered-from-above, aperture, or downward-only scoring state for simple hoop prompts.",
            ]
        )
        profile["implementation_notes"] = _dedupe(notes)
        expectations = _string_list(profile.get("validator_expectations"))
        expectations.extend(
            [
                "basketball is dynamic",
                "hoop is non-blocking sensor",
                "generic ballistic_object_to_region subgoal can satisfy check_objective",
            ]
        )
        profile["validator_expectations"] = _dedupe(expectations)
        return

    if _prompt_implies_side_view_platformer(prompt) and not explicit_zero_g:
        rules = _string_list(profile.get("fairness_rules"))
        rules.extend(
            [
                "First-pass elevated routes should use continuous ramps or shallow stairs, not isolated jump platforms.",
                "Every elevated exit must sit directly on a reachable support surface with no final jump requirement.",
                "Elevated exit sensors should be large and forgiving, usually at least 150x150 px, and overlap the final support.",
                "Do not require holding/dwelling in the exit unless the user explicitly asks for a hold/wait/charge/survive duration.",
                "Hazard spawns should be separated from the main ramp enough to leave a clear validator-safe lane.",
            ]
        )
        profile["fairness_rules"] = _dedupe(rules)
        notes = _string_list(profile.get("implementation_notes"))
        notes.extend(
            [
                "For top-right exits, build one broad diagonal static-segment ramp or shallow stair chain from start to exit.",
                "Avoid platform gaps requiring jump timing; use ramps/stairs the generic force controller can traverse.",
                "Use a large non-blocking exit sensor and match check_objective to that visible sensor radius.",
                "For plain reach-exit objectives, check_objective should be immediate proximity/overlap, not a dwell timer.",
                "Place falling hazards above/around the route but preserve one open traversal lane.",
            ]
        )
        profile["implementation_notes"] = _dedupe(notes)
        profile["world_context"] = _side_view_platformer_context(
            "Side-view elevated route must be traversable by a generic force controller using ramps or shallow stairs."
        )

    is_projectile_survival = objective_type == "survive_duration" and (
        _prompt_requests_projectile_hazards(prompt)
        or any("projectile" in str(item).lower() for item in _string_list((simulation_brief or {}).get("semantic_must_happen")))
    )
    if not is_projectile_survival:
        return

    profile["difficulty"] = "easy_medium"
    rules = profile.setdefault("fairness_rules", [])
    if isinstance(rules, list):
        rules.extend(
            [
                "Validator-friendly projectile survival: cap active projectile hazards at 4-6 unless the prompt explicitly requests bullet-hell difficulty.",
                "Projectile speed should be around 0.35-0.45 * agent max speed for first-pass solvability.",
                "Use at least 2.0 seconds of warmup and visible gaps/safe lanes; do not start with unavoidable crossfire.",
                "Prefer stationary perimeter shooters or very slow shooter drift unless the prompt explicitly asks for chasing enemies.",
            ]
        )
        profile["fairness_rules"] = _string_list(rules)

    notes = profile.setdefault("implementation_notes", [])
    if isinstance(notes, list):
        notes.extend(
            [
                "For survive_duration + projectile hazards, implement a validator-friendly baseline: max_simultaneous_projectiles <= 6, projectile radius <= 0.25 * agent_radius, projectile speed <= 0.45 * agent_max_speed, warmup >= 2 seconds.",
                "Shots should be readable lane hazards with clear gaps; avoid dense crossfire or perfect lead aim in the first generated version.",
            ]
        )
        profile["implementation_notes"] = _string_list(notes)

    for dynamic in profile.get("dynamics", []):
        if not isinstance(dynamic, dict):
            continue
        dynamic_text = json.dumps(dynamic, sort_keys=True, default=str).lower()
        if "projectile" not in dynamic_text and "shot" not in dynamic_text and "laser" not in dynamic_text:
            continue
        dynamic["validator_friendly_baseline"] = {
            "max_simultaneous_projectiles": 6,
            "projectile_speed_relative_to_agent_max": 0.4,
            "projectile_radius_relative_to_agent_radius": 0.25,
            "warmup_seconds": 2.0,
            "preferred_shooter_motion": "stationary_perimeter_or_very_slow_drift",
            "aim_model": "straight_or_low_lead_with_visible_gaps",
        }
        projectile_behavior = dynamic.get("projectile_behavior")
        if isinstance(projectile_behavior, dict):
            projectile_behavior["max_simultaneous_projectiles"] = min(
                int(float(projectile_behavior.get("max_simultaneous_projectiles", 6) or 6)),
                6,
            )
            projectile_behavior["speed_relative_to_agent_max"] = min(
                float(projectile_behavior.get("speed_relative_to_agent_max", 0.4) or 0.4),
                0.45,
            )
            projectile_behavior["projectile_radius_relative_to_agent_radius"] = min(
                float(projectile_behavior.get("projectile_radius_relative_to_agent_radius", 0.25) or 0.25),
                0.25,
            )
        cadence = dynamic.get("cadence")
        if isinstance(cadence, dict):
            cadence["warmup_seconds"] = max(float(cadence.get("warmup_seconds", 2.0) or 2.0), 2.0)
            if isinstance(cadence.get("fire_interval_seconds_range"), list):
                cadence["fire_interval_seconds_range"] = [
                    max(float(cadence["fire_interval_seconds_range"][0]), 0.9),
                    max(float(cadence["fire_interval_seconds_range"][-1]), 1.3),
                ]


def _world_context_from_brief(value: dict[str, Any]) -> dict[str, str]:
    if not value:
        return {}
    perspective = str(value.get("perspective") or value.get("world_perspective") or "")
    gravity = str(value.get("gravity") or value.get("gravity_model") or "")
    movement = str(value.get("movement_model") or "")
    support = str(value.get("support_model") or value.get("support_assumption") or "")
    rationale = str(value.get("rationale") or "Simulation brief physical interpretation.")
    route = str(value.get("route_assumption") or "")
    if not any((perspective, gravity, movement, support)):
        return {}
    return {
        "world_perspective": perspective or "ground_lane_physics",
        "gravity_model": gravity or "normal",
        "movement_model": movement or "ground_force",
        "support_assumption": support or "stable_floor_support",
        "route_assumption": route or "stage objectives on reachable routes",
        "rationale": rationale,
    }


def _default_dynamics_for_prompt(prompt: str) -> list[dict[str, Any]]:
    if _is_ballistic_hoop_request(prompt):
        return [
            {
                "type": "ballistic_object_to_region",
                "name": "basketball_to_hoop",
                "cadence": "single_clear_impulse_then_arc",
                "target_sensor": "generous_non_blocking",
            }
        ]
    if _prompt_requests_projectile_hazards(prompt):
        return [
            {
                "type": "readable_projectile_hazard",
                "hazard_name_pattern": "enemy_shot_*",
                "cadence": "staggered_continuous",
                "projectile_speed": "slow_readable_validator_friendly",
                "max_simultaneous_projectiles": 6,
                "projectile_speed_relative_to_agent_max": 0.4,
                "projectile_radius_relative_to_agent_radius": 0.25,
                "preferred_shooter_motion": "stationary_turrets_or_very_slow_enemies",
                "minimum_visible_travel_px": 120,
                "telegraph_seconds": 0.35,
                "continues_until_objective": True,
                "safe_lane_policy": "at_least_one_escape_vector",
            }
        ]
    if _prompt_requests_lateral_hazards(prompt):
        return [
            {
                "type": "readable_hazard_stream",
                "hazard_name_pattern": "car_*",
                "cadence": "staggered_continuous",
                "motion": "ground_locked_lateral",
                "minimum_lateral_travel_px": 160,
                "max_ground_y_drift_px": 42,
                "target_hazards_per_episode": 4,
                "continues_until_objective": True,
                "implementation_primitive": "create_recurring_lateral_hazards",
            }
        ]
    return []


def _normalize_world_context(value: Any, prompt: str) -> dict[str, Any]:
    inferred = _infer_world_context(prompt)
    supplied = value if isinstance(value, dict) else {}
    normalized = {
        "world_perspective": str(supplied.get("world_perspective") or inferred["world_perspective"]),
        "gravity_model": str(supplied.get("gravity_model") or inferred["gravity_model"]),
        "movement_model": str(supplied.get("movement_model") or inferred["movement_model"]),
        "support_assumption": str(supplied.get("support_assumption") or inferred["support_assumption"]),
        "route_assumption": str(supplied.get("route_assumption") or inferred["route_assumption"]),
        "rationale": str(supplied.get("rationale") or inferred["rationale"]),
    }
    # Deterministic vetoes preserve prompt-faithfulness when an LLM profile
    # under-specifies or contradicts strong context cues.
    if _prompt_implies_zero_g(prompt):
        normalized.update(_infer_world_context(prompt))
    elif _prompt_implies_side_view_platformer(prompt):
        inferred_side = _infer_world_context(prompt)
        for key in (
            "world_perspective",
            "gravity_model",
            "movement_model",
            "support_assumption",
            "route_assumption",
        ):
            normalized[key] = inferred_side[key]
        normalized["rationale"] = inferred_side["rationale"]
    elif _prompt_implies_top_down_or_flat_floor(prompt):
        inferred_flat = _infer_world_context(prompt)
        for key in (
            "world_perspective",
            "gravity_model",
            "movement_model",
            "support_assumption",
            "route_assumption",
        ):
            normalized[key] = inferred_flat[key]
        normalized["rationale"] = inferred_flat["rationale"]
    return normalized


def _infer_world_context(prompt: str) -> dict[str, str]:
    explicit_gravity = _prompt_explicit_gravity_model(prompt)
    if explicit_gravity == "zero_g":
        return _zero_g_context("Prompt explicitly requests zero/no gravity; this overrides genre defaults.")

    context = _infer_genre_world_context(prompt, allow_implicit_zero_g=explicit_gravity is None)
    if explicit_gravity:
        return _apply_explicit_gravity(context, explicit_gravity)
    return context


def _infer_genre_world_context(prompt: str, *, allow_implicit_zero_g: bool = True) -> dict[str, str]:
    if allow_implicit_zero_g and _prompt_implies_zero_g(prompt):
        return _zero_g_context("Prompt implies space/zero-gravity/floating motion.")
    if _is_ballistic_hoop_request(prompt):
        return _side_view_platformer_context(
            "Throw/lob into hoop implies side-view gravity and clear ballistic arc space."
        )
    if _prompt_implies_side_view_platformer(prompt):
        return _side_view_platformer_context("Prompt implies vertical side-view escape/platformer play with gravity and falling hazards.")
    if _prompt_implies_top_down_or_flat_floor(prompt):
        return _top_down_or_flat_floor_context()
    return _ground_lane_context()


def _zero_g_context(rationale: str) -> dict[str, str]:
    return {
        "world_perspective": "zero_g_freeflight",
        "gravity_model": "zero_g",
        "movement_model": "thrust_2d",
        "support_assumption": "no floor support required; use bounded arena walls or soft containment",
        "route_assumption": "agent navigates freely through open 2D space using thrust",
        "rationale": rationale,
    }


def _side_view_platformer_context(rationale: str) -> dict[str, str]:
    return {
        "world_perspective": "side_view_platformer",
        "gravity_model": "normal",
        "movement_model": "ground_force",
        "support_assumption": "agent and movable bodies need solid traversable floors, broad ramps, or shallow stairs; avoid isolated jump platforms for first-pass validation",
        "route_assumption": "provide a continuous wide traversable ramp/stair route from start to exit/goal; elevated goals sit on reachable support",
        "rationale": rationale,
    }


def _top_down_or_flat_floor_context() -> dict[str, str]:
    return {
        "world_perspective": "top_down_or_flat_floor",
        "gravity_model": "top_down_flat",
        "movement_model": "ground_force",
        "support_assumption": "treat the play area as a full flat navigable floor/support plane unless explicit zero-g is requested",
        "route_assumption": "route is planar; walls block laterally, not as vertical platforms",
        "rationale": "Prompt reads like a planar room/maze/arena navigation task.",
    }


def _ground_lane_context() -> dict[str, str]:
    return {
        "world_perspective": "ground_lane_physics",
        "gravity_model": "normal",
        "movement_model": "ground_force",
        "support_assumption": "provide stable floor/support under dynamic objects unless prompt says zero-g",
        "route_assumption": "stage start, interactions, and goals on reachable lanes",
        "rationale": "Default 2D Pymunk assumption for physical manipulation tasks.",
    }


def _apply_explicit_gravity(context: dict[str, str], gravity_model: str) -> dict[str, str]:
    updated = dict(context)
    updated["gravity_model"] = gravity_model
    if gravity_model == "normal":
        updated["movement_model"] = "ground_force"
        updated["support_assumption"] = (
            "explicit normal gravity requested; provide stable floor/support for agent and movable bodies"
        )
    elif gravity_model == "low_g":
        updated["movement_model"] = "low_g_ground_force"
        updated["support_assumption"] = (
            "explicit low gravity requested; provide supports where needed and tune motion for slow readable arcs"
        )
    elif gravity_model == "high_g":
        updated["movement_model"] = "high_g_ground_force"
        updated["support_assumption"] = (
            "explicit high gravity requested; provide strong support, short routes, and extra agent strength"
        )
    updated["rationale"] = f"{updated['rationale']} Explicit prompt gravity override: {gravity_model}."
    return updated


def _normalize_dynamics(value: Any, prompt: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    prompt_text = prompt.lower()
    normalized: list[dict[str, Any]] = []
    prompt_allows_falling = _prompt_requests_falling_hazards(prompt)
    for item in value:
        if not isinstance(item, dict):
            continue
        dynamic = dict(item)
        text = json.dumps(dynamic, sort_keys=True, default=str).lower() + " " + prompt_text
        dynamic_type = _snake(str(dynamic.get("type") or ""))
        if dynamic_type in {"recurring_hazard", "hazard_rain", "falling_hazard"} and _has_any(
            text, "fall", "falling", "rain", "raining", "drop", "fireball", "meteor"
        ):
            dynamic_type = "recurring_falling_hazard"
        elif dynamic_type in {"push", "push_object", "movable_object"} and _has_any(
            text, "push", "shove", "box", "crate", "rock", "pressure plate"
        ):
            dynamic_type = "heavy_but_movable_push"
        elif dynamic_type in {"chaser", "pursuer", "enemy_pursuit"}:
            dynamic_type = "readable_chaser"
        elif dynamic_type in {"projectile", "projectile_hazard", "enemy_shot", "laser_fire", "weapon_fire"}:
            dynamic_type = "readable_projectile_hazard"
        elif dynamic_type in {"readable_hazard_stream", "lateral_hazard", "lateral_hazard_stream", "vehicle_stream", "traffic_stream", "rolling_hazard"}:
            dynamic_type = "readable_hazard_stream"
        if dynamic_type == "readable_projectile_hazard" and not _prompt_requests_projectile_hazards(prompt):
            continue
        if dynamic_type == "recurring_falling_hazard" and not prompt_allows_falling:
            continue
        if dynamic_type == "readable_hazard_stream" and not _prompt_requests_lateral_hazards(prompt):
            continue
        dynamic["type"] = dynamic_type or "custom_dynamic"
        normalized.append(dynamic)
    return normalized


def _sanitize_profile_for_prompt(profile: dict[str, Any], prompt: str) -> None:
    if not _prompt_requests_projectile_hazards(prompt):
        profile["dynamics"] = [
            dynamic
            for dynamic in profile.get("dynamics", [])
            if not _dynamic_is_unrequested_projectile_hazard(dynamic)
        ]
        if _prompt_requests_agent_projectile_impact(prompt):
            if _has_any(str(profile.get("gameplay_loop") or ""), "avoid_projectile", "projectile_hazard", "survive_projectile"):
                profile["gameplay_loop"] = "agent_projectile_impact_manipulation"
        blocked_projectile_hazard_terms = (
            "projectile hazards",
            "projectile hazard",
            "hazard projectiles",
            "hazard projectile",
            "enemy shots",
            "enemy shot",
            "enemy projectiles",
            "enemy projectile",
            "incoming projectiles",
            "incoming projectile",
            "avoid projectiles",
            "dodge projectiles",
            "survive projectiles",
            "survive projectile",
            "readable_projectile_hazard",
            "projectile_hazard",
            "hazard_projectile",
            "enemy_shot",
        )
        for key in ("fairness_rules", "validator_expectations", "implementation_notes"):
            value = profile.get(key)
            if isinstance(value, list):
                profile[key] = [
                    item
                    for item in value
                    if not _has_any(str(item).lower(), *blocked_projectile_hazard_terms)
                ]

    if _is_ballistic_hoop_request(prompt):
        blocked_terms = (
            "rim plane",
            "rim-plane",
            "downward crossing",
            "downward pass",
            "entered from above",
            "aperture",
            "regulation",
        )
        for key in ("fairness_rules", "validator_expectations", "implementation_notes"):
            value = profile.get(key)
            if isinstance(value, list):
                profile[key] = [
                    item
                    for item in value
                    if not _has_any(str(item).lower(), *blocked_terms)
                ]
        return
    if _prompt_requests_falling_hazards(prompt):
        return
    blocked_terms = ("falling", "raining", "fireball", "meteor", "ceiling rock", "hazard rain")
    for key in ("fairness_rules", "validator_expectations", "implementation_notes"):
        value = profile.get(key)
        if isinstance(value, list):
            profile[key] = [
                item
                for item in value
                if not _has_any(str(item).lower(), *blocked_terms)
            ]


def _dynamic_is_unrequested_projectile_hazard(dynamic: object) -> bool:
    if not isinstance(dynamic, dict):
        return False
    dynamic_type = str(dynamic.get("type") or "").lower()
    dynamic_text = json.dumps(dynamic, sort_keys=True, default=str).lower()
    return dynamic_type == "readable_projectile_hazard" or _has_any(
        dynamic_text,
        "enemy_shot",
        "hazard_projectile",
        "projectile_hazard",
        "enemy projectile",
        "enemy shot",
        "incoming projectile",
        "projectile hazards",
    )


def _prompt_requests_falling_hazards(prompt: str) -> bool:
    text = prompt.lower()
    return _has_any(text, "falling", "raining", "rain down", "dropping", "drop down", "from sky", "from the sky", "fall from")


def _prompt_requests_lateral_hazards(prompt: str) -> bool:
    text = prompt.lower()
    vehicle_or_crossing = _has_any(
        text,
        "car",
        "cars",
        "truck",
        "trucks",
        "train",
        "trains",
        "traffic",
        "vehicle",
        "vehicles",
        "rolling log",
        "rolling logs",
        "rolling barrel",
        "rolling barrels",
    )
    lateral_action = _has_any(
        text,
        "come",
        "coming",
        "incoming",
        "endless",
        "endlessly",
        "sequential",
        "sequentially",
        "cross",
        "crossing",
        "drive",
        "driving",
        "roll",
        "rolling",
        "jump over",
        "avoid",
        "dodge",
        "survive",
    )
    return vehicle_or_crossing and lateral_action and not _prompt_requests_falling_hazards(prompt)


def _prompt_requests_projectile_hazards(prompt: str) -> bool:
    text = prompt.lower()
    agent_fires_at_object = _prompt_requests_agent_projectile_impact(prompt)
    incoming_or_enemy_fire = _has_any(
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

    strong_projectile_context = _has_any(
        text,
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
    generic_projectile_context = _has_any(text, "projectile", "projectiles")
    projectile_action = _has_any(text, "shots", "shooting", "shoots", "firing", "fires", "fire at", "laser fire")
    avoidance = _has_any(text, "avoid", "avoiding", "dodge", "dodging", "escape", "survive")
    sports_context = _has_any(text, "soccer", "hockey", "basketball", "billiards", "pinball", "kick", "goal", "hoop")
    if sports_context and not strong_projectile_context:
        return False
    if strong_projectile_context and projectile_action:
        return avoidance or not sports_context
    return generic_projectile_context and avoidance


def _prompt_requests_agent_projectile_impact(prompt: str) -> bool:
    text = prompt.lower()
    return bool(
        re.search(
            r"\b(agent|player|person|character|robot)\b.{0,35}\b(shoots|shoot|fires|fire|firing)\b.{0,45}\b(bullet|projectile|missile|laser|blaster)\b",
            text,
        )
    ) and _has_any(
        text,
        "knock",
        "knocks",
        "knock over",
        "topple",
        "break",
        "hit",
        "hits",
        "push",
        "pile",
        "stack",
        "tower",
        "target",
        "button",
        "squares",
        "blocks",
    )


def _is_ballistic_hoop_request(prompt: str) -> bool:
    text = prompt.lower()
    if _prompt_requests_projectile_hazards(prompt):
        return False
    return _has_any(text, "hoop", "basket") and _has_any(
        text,
        "throw",
        "throws",
        "toss",
        "lob",
        "arc",
        "basketball",
    )


def _prompt_implies_zero_g(prompt: str) -> bool:
    explicit_gravity = _prompt_explicit_gravity_model(prompt)
    if explicit_gravity == "zero_g":
        return True
    if explicit_gravity in {"normal", "low_g", "high_g"}:
        return False
    text = prompt.lower()
    return _has_any(
        text,
        "zero gravity",
        "zero-gravity",
        "zero_g",
        "space",
        "spaceship",
        "asteroid",
        "orbit",
        "orbital",
        "floating",
        "drifting",
    )


def _prompt_explicit_gravity_model(prompt: str) -> str | None:
    text = prompt.lower()
    if re.search(r"\b(?:not|no)\s+zero[-\s]?g(?:ravity)?\b", text):
        return "normal"
    if re.search(r"\b(?:normal|earth|standard)\s+gravity\b", text):
        return "normal"
    if re.search(r"\bgravity\s+(?:on|enabled)\b", text):
        return "normal"
    if re.search(r"\bwith\s+gravity\b", text) and not re.search(r"\b(?:zero|low|high)\s+gravity\b", text):
        return "normal"
    if re.search(r"\b(?:low|moon|lunar)\s+gravity\b|\blow[-\s]?g\b", text):
        return "low_g"
    if re.search(r"\b(?:high|heavy|strong)\s+gravity\b|\bhigh[-\s]?g\b", text):
        return "high_g"
    if re.search(r"\bzero[-\s]?g(?:ravity)?\b|\bzero\s+gravity\b|\bno\s+gravity\b|\bwithout\s+gravity\b", text):
        return "zero_g"
    if re.search(r"\bgravity\s+(?:off|disabled)\b|\bdisable\s+gravity\b", text):
        return "zero_g"
    return None


def _prompt_implies_side_view_platformer(prompt: str) -> bool:
    text = prompt.lower()
    vertical_goal = _has_any(text, "top right", "top-right", "upper right", "upper-right", "high goal", "climb", "upward", "escape")
    platform_terms = _has_any(
        text,
        "lava",
        "platform",
        "platformer",
        "cave",
        "tower",
        "ledge",
        "ramp",
        "stairs",
        "jump",
        "falling",
        "fireball",
        "meteor",
    )
    top_down_terms = _has_any(text, "maze", "pacman", "top-down", "overhead", "arena")
    if top_down_terms and not _has_any(text, "lava", "platformer", "falling", "fireball", "jump"):
        return False
    return platform_terms and (vertical_goal or _prompt_requests_falling_hazards(prompt))


def _prompt_implies_top_down_or_flat_floor(prompt: str) -> bool:
    text = prompt.lower()
    if _prompt_implies_side_view_platformer(prompt) or _prompt_implies_zero_g(prompt):
        return False
    return _has_any(text, "maze", "pacman", "arena", "room", "chamber", "dungeon", "labyrinth", "corridor")


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("gameplay profile must be a JSON object")
    return data


def _has_any(text: str, *terms: str) -> bool:
    return any(term in text for term in terms)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _dedupe(items: list[Any]) -> list[Any]:
    seen = set()
    output = []
    for item in items:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _snake(value: str) -> str:
    words = re.findall(r"[a-z0-9]+", value.lower())
    return "_".join(words) or "physics_objective"
