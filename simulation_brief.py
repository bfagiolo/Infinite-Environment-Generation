"""Simulation Brief agent for prompt-to-physics interpretation.

The simulation brief is the first structured interpretation layer. It states
what the prompt means physically before the gameplay/profile/code generators
start writing implementation details.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

from prompt_cache import load_cached_json, save_cached_json


DEFAULT_BRIEF_MODEL = os.getenv("OPENAI_SIMULATION_BRIEF_MODEL", "gpt-5.2")
DEFAULT_BRIEF_REASONING = os.getenv("OPENAI_SIMULATION_BRIEF_REASONING_EFFORT", "low")
DEFAULT_BRIEF_MAX_OUTPUT = int(os.getenv("OPENAI_SIMULATION_BRIEF_MAX_OUTPUT_TOKENS", "1800"))
SIMULATION_BRIEF_CACHE_VERSION = "simulation_brief.v6"

load_dotenv()


class SimulationBriefArchitect:
    """LLM-backed prompt interpreter with deterministic guardrails."""

    def __init__(self) -> None:
        self._client = None

    async def design(self, prompt: str) -> dict[str, Any]:
        cache_key = _cache_key(prompt)
        cached = load_cached_json("simulation_brief", cache_key)
        if cached is not None:
            return _normalize_brief(cached, prompt)

        if os.getenv("OPENAI_API_KEY"):
            try:
                brief = _normalize_brief(await self._design_with_openai(prompt), prompt)
                save_cached_json("simulation_brief", cache_key, brief)
                return brief
            except Exception as exc:
                brief = _fallback_brief(prompt)
                brief.setdefault("warnings", []).append(
                    f"Simulation brief LLM failed: {type(exc).__name__}: {exc}"
                )
                brief["source"] = "deterministic_fallback_after_llm_error"
                return brief
        brief = _fallback_brief(prompt)
        save_cached_json("simulation_brief", cache_key, brief)
        return brief

    async def _design_with_openai(self, prompt: str) -> dict[str, Any]:
        client = self._get_client()
        response = await client.responses.create(
            model=DEFAULT_BRIEF_MODEL,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are the Simulation Brief Architect for a deterministic "
                        "2D physics world factory. Output compact JSON only. Do not "
                        "write code. Your job is to interpret the prompt physically."
                    ),
                },
                {"role": "user", "content": _brief_prompt(prompt)},
            ],
            reasoning={"effort": DEFAULT_BRIEF_REASONING},
            max_output_tokens=DEFAULT_BRIEF_MAX_OUTPUT,
        )
        text = str(getattr(response, "output_text", "") or "").strip()
        if not text:
            raise RuntimeError("empty simulation brief response")
        return _extract_json(text)

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI()
        return self._client


def _brief_prompt(prompt: str) -> str:
    return f"""
Create a simulation_brief JSON object for this 2D Pymunk training-world prompt:

{prompt}

The brief must state what the text means physically before code generation.
Do not include Python code.

Required top-level keys:
- intent_summary: one sentence
- world_context: dict with theme, perspective, gravity, movement_model, support_model, rationale
- agent: dict with form, role, controls, capability_assumptions
- objective: dict with type, success_condition, failure_condition, duration_seconds if relevant, target_names_hint
- entities: list of important objects/hazards/mechanisms with role and expected_motion
- semantic_must_happen: list of visible physical behaviors that must occur in simulation
- validation: dict with required_tier, semantic_checks, objective_checks, capability_checks
- visuals: dict with style_intent, agent_avatar, important_props, effect_notes
- ambiguity_notes: list of possible ambiguities and the chosen interpretation

Critical context rules:
- Explicit physics words override genre defaults. If prompt says zero/no/low/high/normal gravity, preserve that.
- "shots/shooting" with spaceships, enemies, lasers, bullets, missiles, turrets, or combat means projectile hazards, not soccer/kick/score.
- If the AGENT shoots/fires a bullet/projectile to knock/topple/hit/break a stack, tower, pile, target, squares, or blocks, that projectile is the agent's tool. Do not add enemy projectile hazards, survival duration, or avoidance pressure unless the prompt explicitly says enemy/incoming/avoid/dodge/survive.
- "shoot/kick/score" with soccer, basketball, hockey, billiards, puck, hoop, goal line, or ball means strike/shot-object task.
- "throw/lob basketball into hoop" means a validator-friendly ballistic object-to-region task: success is the ball entering a generous non-blocking hoop sensor. Do not require strict real-basketball rim-plane/downward-crossing/aperture rules unless the user explicitly asks for regulation scoring.
- Falling/raining/dropping hazards must visibly move downward or be recurring/staggered if implied by the prompt.
- Top-right/elevated exits under normal gravity need a continuous broad ramp or shallow stair route, not isolated jump platforms. The validator controls a force-driven body and should not need skilled platformer timing. The exit should be a large non-blocking sensor overlapping the final reachable support. For a plain reach-exit prompt, success should be immediate on reaching the exit; do not add dwell/hold timers unless the user asks for waiting/holding/survival time.
- Push/pressure-plate/gate means object manipulation plus mechanism, not pure navigation.
- Survival "for N seconds" means a finite code-level duration objective.
- The brief may be creative visually, but physics assumptions must remain validator-friendly.
""".strip()


def _cache_key(prompt: str) -> dict[str, Any]:
    return {
        "version": SIMULATION_BRIEF_CACHE_VERSION,
        "prompt": prompt,
        "model": DEFAULT_BRIEF_MODEL,
        "reasoning_effort": DEFAULT_BRIEF_REASONING,
        "max_output_tokens": DEFAULT_BRIEF_MAX_OUTPUT,
    }


def _fallback_brief(prompt: str) -> dict[str, Any]:
    text = prompt.lower()
    world_context = _infer_world_context(text)
    objective = _infer_objective(text)
    agent = _infer_agent(text, world_context)
    entities = _infer_entities(text, objective)
    semantic = _infer_semantic_requirements(text, objective, entities)
    return _normalize_brief(
        {
            "source": "deterministic_fallback",
            "intent_summary": _intent_summary(prompt, objective),
            "world_context": world_context,
            "agent": agent,
            "objective": objective,
            "entities": entities,
            "semantic_must_happen": semantic,
            "validation": _validation_plan(objective, semantic),
            "visuals": _visual_plan(text, agent, entities),
            "ambiguity_notes": _ambiguity_notes(text, objective),
            "warnings": [],
        },
        prompt,
    )


def _normalize_brief(raw: dict[str, Any], prompt: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    text = prompt.lower()
    fallback = _fallback_skeleton(prompt)
    brief = {**fallback, **raw}
    brief["source"] = str(brief.get("source") or "llm_simulation_brief")
    brief["intent_summary"] = str(brief.get("intent_summary") or fallback["intent_summary"])
    brief["world_context"] = _merge_dict(fallback["world_context"], brief.get("world_context"))
    brief["agent"] = _merge_dict(fallback["agent"], brief.get("agent"))
    brief["objective"] = _merge_dict(fallback["objective"], brief.get("objective"))
    brief["visuals"] = _merge_dict(fallback["visuals"], brief.get("visuals"))
    brief["validation"] = _merge_dict(fallback["validation"], brief.get("validation"))
    brief["entities"] = _list_of_dicts(brief.get("entities")) or fallback["entities"]
    brief["semantic_must_happen"] = _string_list(brief.get("semantic_must_happen")) or fallback[
        "semantic_must_happen"
    ]
    brief["ambiguity_notes"] = _string_list(brief.get("ambiguity_notes"))
    brief["warnings"] = _string_list(brief.get("warnings"))

    _apply_hard_context_vetoes(brief, text)
    _coerce_brief_schema(brief, text)
    brief["validation"] = _validation_plan(brief["objective"], brief["semantic_must_happen"], brief["validation"])
    return brief


def _coerce_brief_schema(brief: dict[str, Any], text: str) -> None:
    """Keep the brief creative in content but stable in shape."""

    world = brief["world_context"]
    gravity_raw = world.get("gravity")
    if isinstance(gravity_raw, dict):
        gravity_text = json.dumps(gravity_raw, sort_keys=True, default=str).lower()
        if "zero" in gravity_text or "none" in gravity_text or "[0, 0]" in gravity_text:
            world["gravity"] = "zero_g"
        elif "low" in gravity_text:
            world["gravity"] = "low_gravity"
        elif "high" in gravity_text:
            world["gravity"] = "high_gravity"
        else:
            world["gravity"] = "normal"
    else:
        gravity = str(gravity_raw or "normal").lower()
        if "top_down_flat" in gravity or "top-down-flat" in gravity:
            world["gravity"] = "top_down_flat"
        elif "zero" in gravity or "none" in gravity or "no gravity" in gravity:
            world["gravity"] = "zero_g"
        elif "low" in gravity:
            world["gravity"] = "low_gravity"
        elif "high" in gravity:
            world["gravity"] = "high_gravity"
        else:
            world["gravity"] = "normal"

    perspective = str(world.get("perspective") or "")
    if "top" in perspective and "down" in perspective:
        world["perspective"] = "top_down_or_flat_floor"
    elif "side" in perspective or "platform" in perspective:
        world["perspective"] = "side_view_platformer"
    elif "zero" in perspective or "freeflight" in perspective:
        world["perspective"] = "zero_g_freeflight"
    else:
        world["perspective"] = str(world.get("perspective") or "ground_lane_physics")

    for key, default in (
        ("theme", "research_simulation"),
        ("movement_model", "ground_force"),
        ("support_model", "stable_floor_support"),
        ("rationale", "Simulation brief physical interpretation."),
    ):
        value = world.get(key)
        if isinstance(value, dict):
            world[key] = str(value.get("type") or value.get("mode") or value.get("rationale") or default)
        elif value is None:
            world[key] = default
        else:
            world[key] = str(value)

    agent = brief["agent"]
    form = agent.get("form")
    if isinstance(form, dict):
        agent["form"] = str(form.get("shape") or form.get("type") or "simple_agent")
    elif form is None:
        agent["form"] = "simple_agent"
    else:
        agent["form"] = str(form)
    controls = agent.get("controls")
    if isinstance(controls, dict):
        raw_controls = controls.get("inputs") or controls.get("controls") or controls.get("allowed") or []
        agent["controls"] = _string_list(raw_controls)
    else:
        agent["controls"] = _string_list(controls)
    if not agent["controls"]:
        agent["controls"] = ["apply_force_x", "apply_force_y", "brake"]

    objective = brief["objective"]
    if _duration_seconds(text) is None and str(objective.get("type") or "") != "survive_duration":
        objective.pop("duration_seconds", None)
    elif objective.get("duration_seconds") is not None:
        try:
            objective["duration_seconds"] = float(objective["duration_seconds"])
        except (TypeError, ValueError):
            objective.pop("duration_seconds", None)


def _fallback_skeleton(prompt: str) -> dict[str, Any]:
    text = prompt.lower()
    world_context = _infer_world_context(text)
    objective = _infer_objective(text)
    agent = _infer_agent(text, world_context)
    entities = _infer_entities(text, objective)
    semantic = _infer_semantic_requirements(text, objective, entities)
    return {
        "source": "fallback_skeleton",
        "intent_summary": _intent_summary(prompt, objective),
        "world_context": world_context,
        "agent": agent,
        "objective": objective,
        "entities": entities,
        "semantic_must_happen": semantic,
        "validation": _validation_plan(objective, semantic),
        "visuals": _visual_plan(text, agent, entities),
        "ambiguity_notes": _ambiguity_notes(text, objective),
        "warnings": [],
    }


def _apply_hard_context_vetoes(brief: dict[str, Any], text: str) -> None:
    world = brief["world_context"]
    objective = brief["objective"]
    agent = brief["agent"]
    semantic = set(_string_list(brief.get("semantic_must_happen")))
    if _is_agent_projectile_impact_request(text):
        objective["type"] = "custom_physics"
        objective["success_condition"] = "agent-fired projectile contacts the target structure and causes measurable displacement/topple/collapse"
        objective["failure_condition"] = "projectile never contacts target or target remains effectively unchanged"
        objective["target_names_hint"] = ["agent_bullet", "target_stack"]
        objective.pop("duration_seconds", None)
        validation = brief.get("validation")
        if isinstance(validation, dict):
            validation["required_tier"] = 4
        brief["entities"] = [
            entity
            for entity in _list_of_dicts(brief.get("entities"))
            if not _entity_is_unrequested_projectile_hazard(entity)
        ]
        _ensure_entity(
            brief,
            {
                "name_hint": "agent_bullet",
                "role": "projectile",
                "kind": "agent_fired_projectile",
                "expected_motion": "travels from agent toward target structure",
            },
        )
        _ensure_entity(
            brief,
            {
                "name_hint": "target_stack",
                "role": "target",
                "kind": "toppleable_dynamic_structure",
                "expected_motion": "stable initially, then visibly topples after projectile impact",
            },
        )
        blocked_semantic = (
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
        )
        semantic = {
            item
            for item in semantic
            if not _has_any(str(item).lower(), *blocked_semantic)
        }
        semantic.update(
            {
                "agent projectile visibly travels across open space",
                "projectile contacts target structure",
                "target structure visibly displaces or topples after impact",
            }
        )
        brief["ambiguity_notes"].append(
            "Interpreted agent-fired bullet as a projectile-impact manipulation tool, not an incoming hazard."
        )

    if _prompt_implies_zero_g(text):
        world.update(
            {
                "gravity": "zero_g",
                "perspective": "zero_g_freeflight",
                "movement_model": "thrust_2d",
                "support_model": "bounded_arena_no_floor_required",
                "rationale": "Prompt explicitly requested zero/no gravity; this overrides genre defaults.",
            }
        )
        agent["controls"] = _dedupe(["thrust_forward", "rotate_left", "rotate_right", "apply_force_x", "apply_force_y", "brake"])
    elif _prompt_implies_normal_gravity(text):
        world["gravity"] = "normal"
        if world.get("perspective") == "zero_g_freeflight":
            world["perspective"] = "side_view_or_supported_world"
        world.setdefault("support_model", "stable_floor_or_platform_support")
    elif _has_any(text, "maze", "room", "arena", "pacman", "top down", "top-down", "labyrinth"):
        world["perspective"] = "top_down_or_flat_floor"
        world["gravity"] = "top_down_flat"
        world["movement_model"] = "ground_force"
        world["support_model"] = "flat_floor_or_top_down_no_fall"
        world["rationale"] = (
            "Planar maze/room/arena cue: use top-down flat-plane physics, not "
            "true zero-gravity freeflight, unless the prompt explicitly says zero gravity."
        )

    if _is_projectile_combat_request(text):
        objective.setdefault("type", "survive_duration")
        if "avoid" in text or "survive" in text:
            objective["type"] = "survive_duration"
        objective["failure_condition"] = "agent hit by projectile hazard"
        seconds = _duration_seconds(text)
        if seconds:
            objective["duration_seconds"] = seconds
            objective["success_condition"] = f"agent remains unhit for {seconds:g} seconds"
        agent["form"] = "spaceship" if "ship" in text or "space" in text else agent.get("form", "agent")
        agent["controls"] = _dedupe(["thrust_forward", "rotate_left", "rotate_right", "apply_force_x", "apply_force_y", "brake"])
        _ensure_entity(
            brief,
            {
                "name_hint": "enemy_projectile",
                "role": "hazard",
                "kind": "projectile",
                "expected_motion": "visible ballistic travel across arena",
            },
        )
        semantic.update(
            {
                "projectile hazards exist",
                "projectile hazards visibly travel across the play area",
                "projectiles are hazards, not kickable sports balls",
            }
        )
        brief["ambiguity_notes"].append(
            "Interpreted shots as enemy projectile fire because the prompt contains space/combat context."
        )

    if _is_ballistic_hoop_request(text):
        objective["type"] = "ballistic_object_to_region"
        objective["success_condition"] = "ball/object enters a generous non-blocking hoop target sensor after physical impulse"
        objective["failure_condition"] = "object fails to reach the hoop target region"
        semantic.update({"ball/object moves after contact", "object enters target/goal region", "ball follows a gravity arc"})
        brief["ambiguity_notes"].append(
            "Interpreted hoop scoring as validator-friendly target-sensor entry, not strict rim-plane basketball rules."
        )
    elif _is_sports_strike_request(text):
        objective["type"] = "strike_object_to_region"
        objective["success_condition"] = "agent physically strikes/kicks a ball-like object into a target region"
        objective["failure_condition"] = "object fails to enter target region"
        semantic.update({"ball/object moves after contact", "object enters target/goal region"})

    if _prompt_requests_falling_hazards(text):
        _ensure_entity(
            brief,
            {
                "name_hint": "falling_hazard",
                "role": "hazard",
                "kind": "falling_or_raining_hazard",
                "expected_motion": "staggered downward recurring motion",
            },
        )
        semantic.update({"hazards visibly fall downward", "hazards are staggered or recurring"})

    if _prompt_requests_rolling_lane_hazards(text):
        _ensure_entity(
            brief,
            {
                "name_hint": "boulder",
                "role": "hazard",
                "kind": "rolling_lateral_hazard",
                "expected_motion": "ground-locked lateral rolling through the agent route",
            },
        )
        semantic.update(
            {
                "rolling hazards visibly travel laterally",
                "rolling hazards enter the agent route",
                "rolling hazards do not fall off-screen or remain blocked",
            }
        )

    if _has_any(text, "pressure plate", "sliding gate", "gate", "door") and _has_any(
        text, "push", "box", "crate", "rock", "barrel"
    ):
        objective["type"] = "mechanism_sequence"
        objective["success_condition"] = "agent moves object onto trigger, mechanism opens, agent reaches final region"
        objective["failure_condition"] = "object cannot activate mechanism or route remains blocked"
        semantic.update({"movable object visibly moves", "trigger/plate activates mechanism", "gate/door changes state"})

    brief["semantic_must_happen"] = sorted(semantic)


def _infer_world_context(text: str) -> dict[str, str]:
    if _prompt_implies_zero_g(text):
        return {
            "theme": _theme(text),
            "perspective": "zero_g_freeflight",
            "gravity": "zero_g",
            "movement_model": "thrust_2d",
            "support_model": "bounded_arena_no_floor_required",
            "rationale": "Explicit zero/no-gravity prompt cue.",
        }
    if _is_ballistic_hoop_request(text):
        return {
            "theme": _theme(text),
            "perspective": "side_view_platformer",
            "gravity": "normal",
            "movement_model": "ground_force",
            "support_model": "stable_floor_support_with_clear_ballistic_arc_space",
            "rationale": "Throw/lob into a hoop is best represented as side-view gravity with a generous target sensor.",
        }
    if _prompt_implies_normal_gravity(text) or _has_any(
        text, "lava", "cave", "platform", "jump", "climb", "top right", "falling", "from sky"
    ):
        return {
            "theme": _theme(text),
            "perspective": "side_view_platformer",
            "gravity": "normal",
            "movement_model": "ground_force",
            "support_model": "stable_floor_broad_ramp_or_shallow_stairs",
            "rationale": "Platform/vertical/falling-hazard cues imply side-view gravity with a validator-friendly continuous route unless overridden.",
        }
    if _has_any(text, "soccer", "basketball", "hockey", "billiards", "pinball"):
        return {
            "theme": _theme(text),
            "perspective": "top_down_or_flat_floor",
            "gravity": "top_down_flat",
            "movement_model": "ground_force",
            "support_model": "flat_sports_surface_no_vertical_fall",
            "rationale": "Sports-object strike prompts are best treated as planar 2D contact dynamics unless explicit gravity is requested.",
        }
    if _has_any(text, "maze", "room", "arena", "pacman", "top down"):
        return {
            "theme": _theme(text),
            "perspective": "top_down_or_flat_floor",
            "gravity": "top_down_flat",
            "movement_model": "ground_force",
            "support_model": "flat_floor_or_top_down_no_fall",
            "rationale": "Maze/room/arena cues imply top-down navigation unless overridden.",
        }
    return {
        "theme": _theme(text),
        "perspective": "ground_lane_physics",
        "gravity": "normal",
        "movement_model": "ground_force",
        "support_model": "stable_floor_support",
        "rationale": "Default 2D physical manipulation assumption.",
    }


def _infer_agent(text: str, world: dict[str, str]) -> dict[str, Any]:
    if _has_any(text, "spaceship", "space ship", "ship", "rocket"):
        form = "spaceship"
    elif _has_any(text, "person", "human", "runner", "kick", "throw", "jump"):
        form = "stick_figure"
    elif _has_any(text, "pacman"):
        form = "arcade_player"
    else:
        form = "simple_agent"
    movement = world.get("movement_model", "ground_force")
    controls = ["apply_force_x", "apply_force_y", "brake"]
    if movement == "thrust_2d":
        controls = ["thrust_forward", "rotate_left", "rotate_right", "apply_force_x", "apply_force_y", "brake"]
    if "jump" in text:
        controls.append("jump")
    return {
        "form": form,
        "role": "agent",
        "controls": _dedupe(controls),
        "capability_assumptions": [movement, world.get("gravity", "normal")],
    }


def _infer_objective(text: str) -> dict[str, Any]:
    seconds = _duration_seconds(text)
    if _is_agent_projectile_impact_request(text):
        return {
            "type": "custom_physics",
            "success_condition": "agent-fired projectile contacts the target structure and causes measurable displacement/topple/collapse",
            "failure_condition": "projectile never contacts target or target remains effectively unchanged",
            "target_names_hint": ["agent_bullet", "target_stack"],
        }
    if seconds and _has_any(text, "survive", "avoid", "dodge", "escape from"):
        return {
            "type": "survive_duration",
            "duration_seconds": seconds,
            "success_condition": f"agent remains safe for {seconds:g} seconds",
            "failure_condition": "agent contacts a hazard",
            "target_names_hint": [],
        }
    if _has_any(text, "pressure plate", "gate", "door") and _has_any(text, "push", "box", "crate", "rock"):
        return {
            "type": "mechanism_sequence",
            "success_condition": "object activates trigger and agent reaches opened path/goal",
            "failure_condition": "mechanism cannot activate or route remains blocked",
            "target_names_hint": ["push_object", "pressure_plate", "gate", "goal"],
        }
    if _is_ballistic_hoop_request(text):
        return {
            "type": "ballistic_object_to_region",
            "success_condition": "ball/object enters a generous non-blocking hoop target sensor after physical impulse",
            "failure_condition": "object fails to reach the hoop target",
            "target_names_hint": ["basketball", "hoop"],
        }
    if _is_sports_strike_request(text):
        return {
            "type": "strike_object_to_region",
            "success_condition": "ball/object enters target/goal region after physical contact",
            "failure_condition": "object fails to enter target",
            "target_names_hint": ["ball", "goal"],
        }
    if _has_any(text, "touch", "collect", "crystal", "rock"):
        return {
            "type": "touch_collection",
            "success_condition": "agent touches all required targets",
            "failure_condition": "targets remain untouched",
            "target_names_hint": ["target"],
        }
    return {
        "type": "navigation_goal",
        "success_condition": "agent reaches target/exit/goal region",
        "failure_condition": "agent cannot reach target region",
        "target_names_hint": ["goal"],
    }


def _infer_entities(text: str, objective: dict[str, Any]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = [
        {"name_hint": "agent", "role": "agent", "kind": "controllable_body", "expected_motion": "agent-controlled motion"}
    ]
    if _is_projectile_combat_request(text):
        entities.extend(
            [
                {"name_hint": "enemy_ship", "role": "enemy", "kind": "projectile_source", "expected_motion": "stationary or slow readable shooter"},
                {"name_hint": "laser_bolt", "role": "hazard", "kind": "projectile", "expected_motion": "visible ballistic travel"},
            ]
        )
    if _is_agent_projectile_impact_request(text):
        entities.extend(
            [
                {"name_hint": "agent_bullet", "role": "projectile", "kind": "agent_fired_projectile", "expected_motion": "travels from agent toward target structure"},
                {"name_hint": "target_stack", "role": "target", "kind": "toppleable_dynamic_structure", "expected_motion": "stable initially, then visibly topples after projectile impact"},
            ]
        )
    if _prompt_requests_falling_hazards(text):
        entities.append(
            {"name_hint": "falling_hazard", "role": "hazard", "kind": "falling_hazard", "expected_motion": "recurring downward motion"}
        )
    if _prompt_requests_rolling_lane_hazards(text):
        entities.append(
            {
                "name_hint": "boulder",
                "role": "hazard",
                "kind": "rolling_lateral_hazard",
                "expected_motion": "ground-locked lateral rolling through the agent route",
            }
        )
    if objective.get("type") == "mechanism_sequence":
        entities.extend(
            [
                {"name_hint": "push_object", "role": "movable", "kind": "dynamic_object", "expected_motion": "visible push displacement"},
                {"name_hint": "pressure_plate", "role": "trigger", "kind": "sensor", "expected_motion": "activation state changes"},
                {"name_hint": "gate", "role": "mechanism", "kind": "door_or_gate", "expected_motion": "opens or becomes non-blocking"},
            ]
        )
    return entities


def _infer_semantic_requirements(text: str, objective: dict[str, Any], entities: list[dict[str, Any]]) -> list[str]:
    requirements: list[str] = []
    if _is_agent_projectile_impact_request(text):
        requirements.extend(
            [
                "agent projectile visibly travels across open space",
                "projectile contacts target structure",
                "target structure visibly displaces or topples after impact",
            ]
        )
    if _is_projectile_combat_request(text):
        requirements.extend(["projectile hazards exist", "projectile hazards visibly travel across the play area"])
    if _prompt_requests_falling_hazards(text):
        requirements.extend(["hazards visibly fall downward", "hazards are staggered or recurring"])
    if _prompt_requests_rolling_lane_hazards(text):
        requirements.extend(["rolling hazards visibly travel laterally", "rolling hazards enter the agent route"])
    if objective.get("type") == "mechanism_sequence":
        requirements.extend(["movable object visibly moves", "trigger/plate activates mechanism"])
    if objective.get("type") == "ballistic_object_to_region":
        requirements.extend(["ball/object moves after contact", "object enters target/goal region", "ball follows a gravity arc"])
    if objective.get("type") == "strike_object_to_region":
        requirements.extend(["ball/object moves after contact", "object enters target/goal region"])
    return _dedupe(requirements)


def _validation_plan(
    objective: dict[str, Any],
    semantic: list[str],
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    objective_type = str(objective.get("type") or "navigation_goal")
    required_tier = 5 if objective_type in {"survive_duration", "mechanism_sequence", "strike_object_to_region", "ballistic_object_to_region"} else 4
    plan = {
        "required_tier": required_tier,
        "semantic_checks": list(semantic),
        "objective_checks": [str(objective.get("success_condition") or "objective state becomes true")],
        "capability_checks": ["agent controls match declared movement model"],
    }
    if isinstance(existing, dict):
        merged = {**plan, **existing}
        merged["semantic_checks"] = _dedupe(_string_list(existing.get("semantic_checks")) + plan["semantic_checks"])
        merged["objective_checks"] = _dedupe(_string_list(existing.get("objective_checks")) + plan["objective_checks"])
        merged["capability_checks"] = _dedupe(_string_list(existing.get("capability_checks")) + plan["capability_checks"])
        if not isinstance(merged.get("required_tier"), int):
            merged["required_tier"] = required_tier
        return merged
    return plan


def _visual_plan(text: str, agent: dict[str, Any], entities: list[dict[str, Any]]) -> dict[str, Any]:
    theme = _theme(text)
    return {
        "style_intent": theme,
        "agent_avatar": agent.get("form", "simple_agent"),
        "important_props": [str(entity.get("kind") or entity.get("name_hint")) for entity in entities],
        "effect_notes": _visual_effects(text),
    }


def _intent_summary(prompt: str, objective: dict[str, Any]) -> str:
    return f"Build a 2D physics world where {prompt.strip()} (objective: {objective.get('type')})."


def _ambiguity_notes(text: str, objective: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if "shot" in text or "shoot" in text:
        if _is_projectile_combat_request(text):
            notes.append("Resolved 'shot/shooting' as projectile weapon fire from context.")
        elif _is_sports_strike_request(text):
            notes.append("Resolved 'shot/shooting' as a sports/object strike from context.")
    if _prompt_implies_zero_g(text):
        notes.append("Explicit zero-gravity wording overrides any normal-gravity genre assumptions.")
    return notes


def _extract_json(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("simulation brief response must be a JSON object")
    return value


def _merge_dict(base: dict[str, Any], value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {**base, **value}
    return dict(base)


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _ensure_entity(brief: dict[str, Any], entity: dict[str, Any]) -> None:
    entities = _list_of_dicts(brief.get("entities"))
    name_hint = str(entity.get("name_hint") or "")
    if name_hint and any(str(item.get("name_hint") or "") == name_hint for item in entities):
        return
    entities.append(entity)
    brief["entities"] = entities


def _entity_is_unrequested_projectile_hazard(entity: dict[str, Any]) -> bool:
    text = json.dumps(entity, sort_keys=True, default=str).lower()
    role = str(entity.get("role") or "").lower()
    kind = str(entity.get("kind") or "").lower()
    return (
        role == "hazard"
        and _has_any(kind + " " + text, "projectile", "shot", "laser", "missile", "bullet", "bolt")
    ) or _has_any(
        text,
        "enemy_projectile",
        "enemy shot",
        "hazard projectile",
        "projectile hazard",
        "incoming projectile",
    )


def _has_any(text: str, *tokens: str) -> bool:
    return any(token in text for token in tokens)


def _prompt_requests_rolling_lane_hazards(text: str) -> bool:
    return _has_any(text, "rolling", "rolls", "roll ") and _has_any(
        text,
        "boulder",
        "boulders",
        "rock",
        "rocks",
        "stone",
        "stones",
        "barrel",
        "barrels",
        "log",
        "logs",
        "obstacle",
        "obstacles",
    ) and not _prompt_requests_falling_hazards(text)


def _dedupe(items: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for item in items:
        key = json.dumps(item, sort_keys=True, default=str) if not isinstance(item, str) else item
        if key not in seen:
            result.append(item)
            seen.add(key)
    return result


def _prompt_implies_zero_g(text: str) -> bool:
    return bool(re.search(r"\b(zero[- ]?g|zero gravity|no gravity|weightless|microgravity)\b", text))


def _prompt_implies_normal_gravity(text: str) -> bool:
    return bool(re.search(r"\b(normal gravity|earth gravity|with gravity|platformer gravity)\b", text))


def _prompt_requests_falling_hazards(text: str) -> bool:
    return _has_any(text, "falling", "raining", "rain down", "dropping", "drop down", "from sky", "from the sky")


def _is_projectile_combat_request(text: str) -> bool:
    agent_fires_at_object = _is_agent_projectile_impact_request(text)
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
    avoidance = _has_any(text, "avoid", "avoiding", "dodge", "dodging", "escape", "survive", "survival")
    sports_context = _has_any(text, "soccer", "hockey", "basketball", "billiards", "pinball", "kick", "puck", "hoop")
    if sports_context and not strong_projectile_context:
        return False
    if strong_projectile_context and projectile_action:
        return avoidance or not sports_context
    return generic_projectile_context and avoidance


def _is_agent_projectile_impact_request(text: str) -> bool:
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


def _is_sports_strike_request(text: str) -> bool:
    if _is_projectile_combat_request(text):
        return False
    if _is_ballistic_hoop_request(text):
        return False
    return bool(
        re.search(r"\b(kick|kicks|kicking|strike|strikes|shoot|shot|score)\b.{0,50}\b(ball|puck|goal|hoop|basket)\b", text)
        or _has_any(text, "soccer", "hockey", "basketball", "billiards")
    )


def _is_ballistic_hoop_request(text: str) -> bool:
    if _is_projectile_combat_request(text):
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


def _duration_seconds(text: str) -> float | None:
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*(seconds?|secs?|s)\b", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _theme(text: str) -> str:
    if _has_any(text, "space", "asteroid", "spaceship", "laser"):
        return "space_combat" if _has_any(text, "shot", "laser", "enemy") else "space"
    if _has_any(text, "lava", "fire", "volcano"):
        return "lava_world"
    if _has_any(text, "forest", "jungle", "garden"):
        return "forest"
    if _has_any(text, "soccer", "basketball", "hockey"):
        return "sports_arena"
    if _has_any(text, "magnetic", "field", "lab"):
        return "physics_lab"
    return "research_simulation"


def _visual_effects(text: str) -> list[str]:
    effects: list[str] = []
    if _has_any(text, "space", "asteroid", "spaceship"):
        effects.extend(["starfield", "engine_glow", "projectile_trails"])
    if _has_any(text, "lava", "fire"):
        effects.extend(["lava_glow", "heat_haze", "ember_particles"])
    if _has_any(text, "magnetic", "field"):
        effects.extend(["field_rings", "electric_arcs"])
    if _has_any(text, "forest", "jungle"):
        effects.extend(["foliage_props", "soft_parallax"])
    return effects or ["neon_grid", "telemetry_overlay"]
