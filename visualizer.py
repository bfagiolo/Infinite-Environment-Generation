"""Telemetry-rich Pygame visualizer for verified Harness Alpha worlds."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping

import pygame
import pymunk

from base_env import BaseEnv
from validator import (
    ValidatorConfig,
    _step_subgoal,
    _subgoal_satisfied,
    load_env_class,
    validate_generated_env,
    validate_ground_truth,
)
from visual_grammar import VisualGrammar, color_for_record, infer_visual_grammar


BACKGROUND = (18, 18, 18)
GRID = (28, 105, 112)
GRID_MAJOR = (44, 170, 180)
TEXT = (205, 242, 246)
MUTED_TEXT = (108, 145, 150)
AGENT = (38, 169, 255)
GOAL = (42, 232, 143)
OBSTACLE = (255, 55, 93)
TERRAIN = (92, 110, 126)
HAZARD = (255, 171, 64)
PATH = (120, 238, 255)
HITBOX = (111, 244, 255)
VELOCITY = (45, 245, 196)
SUCCESS = (64, 255, 166)
SUCCESS_GOLD = (255, 221, 91)
OVERLAY_DARK = (3, 9, 13, 205)

GRID_SIZE = 32
CONTROL_ACCELERATION = 5200.0
BOOST_MULTIPLIER = 6.0
IMPULSE_MULTIPLIER = 18.0
LIFT_IMPULSE_HEIGHT = 140.0
MAX_VECTOR_LENGTH = 90.0
VISUALIZER_SUBSTEPS = 10
ACTION_ASSIST_COOLDOWN_MS = 260
SELECTED = (255, 245, 128)
_ACTION_ASSIST_LAST_MS: dict[tuple[int, str], int] = {}


@dataclass(frozen=True)
class AgentAction:
    pose: str
    nearest: Any | None
    facing: int
    intensity: float = 0.0


class Camera:
    """World-to-screen transform with Pymunk y-up coordinates."""

    def __init__(self, env_width: float, env_height: float, screen_size: tuple[int, int]) -> None:
        screen_width, screen_height = screen_size
        margin = 46
        self.env_width = env_width
        self.env_height = env_height
        self.scale = min(
            (screen_width - margin * 2) / max(env_width, 1.0),
            (screen_height - margin * 2) / max(env_height, 1.0),
        )
        self.left = (screen_width - env_width * self.scale) / 2.0
        self.top = (screen_height - env_height * self.scale) / 2.0

    def point(self, world: pymunk.Vec2d | tuple[float, float] | list[float]) -> tuple[int, int]:
        if hasattr(world, "x") and hasattr(world, "y"):
            x, y = float(world.x), float(world.y)
        else:
            x, y = float(world[0]), float(world[1])
        return (
            int(self.left + x * self.scale),
            int(self.top + (self.env_height - y) * self.scale),
        )

    def length(self, value: float) -> int:
        return max(1, int(value * self.scale))


def run_visualizer(
    env_path: str | Path,
    *,
    width: int = 1280,
    height: int = 800,
    smoke_frames: int | None = None,
    autoplay: bool = False,
    auto_close_after_solve: float | None = None,
) -> None:
    """Load and render a generated BaseEnv module in an interactive Pygame loop."""

    pygame.init()
    pygame.display.set_caption("Harness Alpha // Telemetry Visualizer")
    fullscreen = False
    logical_size = (width, height)

    def set_display_mode() -> pygame.Surface:
        flags = pygame.SCALED
        if fullscreen:
            flags |= pygame.FULLSCREEN
        return pygame.display.set_mode(logical_size, flags)

    screen = set_display_mode()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 15)
    small_font = pygame.font.SysFont("consolas", 13)
    success_font = pygame.font.SysFont("consolas", 70, bold=True)
    success_small_font = pygame.font.SysFont("consolas", 22, bold=True)
    button_font = pygame.font.SysFont("consolas", 24, bold=True)

    env = _load_env(env_path)
    grammar = infer_visual_grammar(env)
    camera = _camera_for_env(env, (width, height))
    path_points = _validator_path(env)
    validation_summary = _validation_summary(env_path)
    overlay_enabled = True
    selected_name = _initial_selected_name(env)
    autoplay_state = _AutoplayState.from_env(env) if autoplay else None
    running = True
    frame_count = 0

    while running:
        frame_count += 1
        dt = min(max(clock.tick(60) / 1000.0, 1.0 / 240.0), 1.0 / 30.0)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif (
                autoplay_state is not None
                and autoplay_state.menu_visible
                and event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
            ):
                replay_rect, quit_rect = _solve_menu_button_rects(screen)
                if replay_rect.collidepoint(event.pos):
                    _ACTION_ASSIST_LAST_MS.clear()
                    env.reset()
                    path_points = _validator_path(env)
                    autoplay_state = _AutoplayState.from_env(env)
                    if selected_name not in env._objects:
                        selected_name = _initial_selected_name(env)
                elif quit_rect.collidepoint(event.pos):
                    running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11 or (
                    event.key == pygame.K_RETURN and (pygame.key.get_mods() & pygame.KMOD_ALT)
                ):
                    fullscreen = not fullscreen
                    screen = set_display_mode()
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    _ACTION_ASSIST_LAST_MS.clear()
                    env.reset()
                    path_points = _validator_path(env)
                    autoplay_state = _AutoplayState.from_env(env) if autoplay else None
                    if selected_name not in env._objects:
                        selected_name = _initial_selected_name(env)
                elif event.key == pygame.K_SPACE:
                    overlay_enabled = not overlay_enabled
                elif event.key == pygame.K_TAB:
                    selected_name = _next_dynamic_name(env, selected_name)
                elif event.key == pygame.K_l:
                    _lift_selected_body(env, selected_name)
                elif event.key == pygame.K_i:
                    _impulse_selected_body(env, selected_name, pymunk.Vec2d(0, 1))
                elif event.key == pygame.K_j:
                    _impulse_selected_body(env, selected_name, pymunk.Vec2d(-1, 0))
                elif event.key == pygame.K_k:
                    _impulse_selected_body(env, selected_name, pymunk.Vec2d(0, -1))
                elif event.key == pygame.K_SEMICOLON:
                    _impulse_selected_body(env, selected_name, pymunk.Vec2d(1, 0))

        if autoplay_state is not None:
            autoplay_state.step(env)
        else:
            _apply_selected_controls(env, selected_name)
            _apply_visual_action_assist(env, selected_name)
            env.step(dt=dt, substeps=VISUALIZER_SUBSTEPS)

        tick = pygame.time.get_ticks()
        _draw_background(screen, camera, env, grammar, tick)
        _draw_grid(screen, camera, env, grammar, tick)
        _draw_path(screen, camera, path_points, grammar)
        _draw_world(screen, camera, env, overlay_enabled, selected_name, grammar, tick)
        if overlay_enabled:
            _draw_velocity_vectors(screen, camera, env, grammar)
            _draw_hud(
                screen,
                env,
                font,
                small_font,
                path_points,
                selected_name,
                validation_summary,
                grammar,
            )
        _draw_footer(screen, small_font, overlay_enabled, selected_name, autoplay_state)
        if autoplay_state is not None:
            _draw_autoplay_countdown(screen, success_small_font, env, autoplay_state, grammar)
            _draw_autoplay_success_layer(
                screen,
                success_font,
                success_small_font,
                button_font,
                autoplay_state,
                grammar,
                tick,
            )
            if (
                auto_close_after_solve is not None
                and autoplay_state.solved_at_ms is not None
                and tick - autoplay_state.solved_at_ms >= int(max(0.0, auto_close_after_solve) * 1000.0)
            ):
                running = False
        pygame.display.flip()
        if smoke_frames is not None and frame_count >= smoke_frames:
            running = False

    pygame.quit()


class _AutoplayState:
    """Small live wrapper around the validator's generic subgoal controller."""

    def __init__(self, subgoals: list[dict[str, Any]]) -> None:
        self.subgoals = subgoals
        self.index = 0
        self.config = ValidatorConfig(simulation_steps=4)
        self.solved_at_ms: int | None = None
        self.solve_step_frames = 0
        self.duration_seconds = _duration_seconds_from_subgoals(subgoals)

    @classmethod
    def from_env(cls, env: BaseEnv) -> "_AutoplayState":
        objective = env.get_ground_truth().get("objective", {})
        profile = objective.get("objective_profile") if isinstance(objective, dict) else {}
        subgoals = profile.get("subgoals") if isinstance(profile, dict) else []
        if not isinstance(subgoals, list):
            subgoals = []
        return cls([dict(item) for item in subgoals if isinstance(item, dict)])

    @property
    def label(self) -> str:
        if self.solved_at_ms is not None:
            return "AI SOLVER: COMPLETE"
        if self.index >= len(self.subgoals):
            return "AI SOLVER: VERIFYING"
        kind = self.subgoals[self.index].get("kind", "subgoal")
        return f"AI SOLVER: {self.index + 1}/{len(self.subgoals)} {kind}"

    @property
    def success_visible(self) -> bool:
        if self.solved_at_ms is None:
            return False
        elapsed = pygame.time.get_ticks() - self.solved_at_ms
        return 1000 <= elapsed < 3000

    @property
    def menu_visible(self) -> bool:
        if self.solved_at_ms is None:
            return False
        return pygame.time.get_ticks() - self.solved_at_ms >= 3000

    def step(self, env: BaseEnv) -> None:
        if self.solved_at_ms is not None:
            return
        self.solve_step_frames += 1
        try:
            if callable(getattr(env, "check_objective", None)) and bool(env.check_objective()):
                self.solved_at_ms = pygame.time.get_ticks()
                return
        except Exception:
            pass
        agent = env.get_agent_record()
        if agent is None or self.index >= len(self.subgoals):
            env.step(substeps=VISUALIZER_SUBSTEPS)
            try:
                if callable(getattr(env, "check_objective", None)) and bool(env.check_objective()):
                    self.solved_at_ms = pygame.time.get_ticks()
            except Exception:
                pass
            return
        subgoal = self.subgoals[self.index]
        if _subgoal_satisfied(env, subgoal, self.config):
            self.index += 1
            env.step(substeps=VISUALIZER_SUBSTEPS)
            try:
                if callable(getattr(env, "check_objective", None)) and bool(env.check_objective()):
                    self.solved_at_ms = pygame.time.get_ticks()
            except Exception:
                pass
            return
        _step_subgoal(env, agent, subgoal, self.config)

    def countdown_seconds(self, env: BaseEnv) -> float | None:
        if self.duration_seconds <= 0.0 or self.solved_at_ms is not None:
            return 0.0 if self.solved_at_ms is not None else None
        elapsed = float(getattr(env, "survival_steps", 0.0) or 0.0) / 60.0
        if elapsed <= 0.0:
            elapsed = float(getattr(env, "_time", 0.0) or 0.0)
        return max(0.0, self.duration_seconds - elapsed)


def _duration_seconds_from_subgoals(subgoals: list[dict[str, Any]]) -> float:
    for subgoal in subgoals:
        kind = str(subgoal.get("kind", "") or "").lower()
        if "survive" not in kind and "duration" not in kind:
            continue
        for key in ("duration_seconds", "duration_s", "duration", "seconds"):
            value = subgoal.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
        steps = subgoal.get("duration_steps")
        if isinstance(steps, (int, float)) and steps > 0:
            return float(steps) / 60.0
    return 0.0


def _load_env(env_path: str | Path) -> BaseEnv:
    env_class = load_env_class(env_path)
    return env_class()


def _camera_for_env(env: BaseEnv, screen_size: tuple[int, int]) -> Camera:
    ground_truth = env.get_ground_truth()
    config = ground_truth.get("config", {})
    return Camera(
        env_width=float(config.get("width") or 1024),
        env_height=float(config.get("height") or 768),
        screen_size=screen_size,
    )


def _validator_path(env: BaseEnv) -> tuple[tuple[float, float], ...]:
    result = validate_ground_truth(
        env.get_ground_truth(),
        config=ValidatorConfig(simulation_steps=1, include_dynamic_blockers=False),
    )
    return result.path


def _validation_summary(env_path: str | Path) -> dict[str, Any]:
    """Return a compact validator summary for the visual overlay."""

    try:
        result = validate_generated_env(
            env_path,
            config=ValidatorConfig(simulation_steps=1, include_dynamic_blockers=False),
        )
    except Exception as exc:
        return {
            "accepted": False,
            "achieved_tier": 0,
            "tier_name": "validation_error",
            "minimum_acceptance_tier": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {
        "accepted": result.accepted,
        "achieved_tier": result.achieved_tier,
        "tier_name": result.tier_name,
        "minimum_acceptance_tier": result.minimum_acceptance_tier,
        "reason": result.reason,
    }


def _apply_selected_controls(env: BaseEnv, selected_name: str | None) -> None:
    if not selected_name:
        return
    selected_record = env._objects.get(selected_name)
    if selected_record is None or selected_record.body.body_type != pymunk.Body.DYNAMIC:
        return

    keys = pygame.key.get_pressed()
    multiplier = BOOST_MULTIPLIER if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 1.0
    if selected_record.role == "agent":
        control_force = env.agent_strength * multiplier
    else:
        control_force = max(env.agent_strength, CONTROL_ACCELERATION * selected_record.body.mass) * multiplier
    force = pymunk.Vec2d(0.0, 0.0)
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        force += pymunk.Vec2d(-control_force, 0.0)
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        force += pymunk.Vec2d(control_force, 0.0)
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        force += pymunk.Vec2d(0.0, control_force)
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        force += pymunk.Vec2d(0.0, -control_force)

    if force.length > 0.0:
        if selected_record.role == "agent":
            env.apply_agent_force(force, strength=force.length)
        else:
            selected_record.body.apply_force_at_world_point(force, selected_record.body.position)


def _apply_visual_action_assist(env: BaseEnv, selected_name: str | None) -> None:
    """Make interactive human actions read clearly without changing env generation."""

    if not selected_name:
        return
    agent = env._objects.get(selected_name)
    if agent is None or agent.role != "agent" or agent.body.body_type != pymunk.Body.DYNAMIC:
        return
    action = _infer_agent_action(env, agent)
    target = action.nearest
    if target is None or target.body.body_type != pymunk.Body.DYNAMIC:
        return

    offset = target.body.position - agent.body.position
    if offset.length <= 1.0:
        return
    axis = offset.normalized()
    toward = float(agent.body.velocity.dot(axis))
    if action.pose not in {"kick", "push", "throw"} and toward <= 35.0:
        return

    now = pygame.time.get_ticks()
    key = (id(target.body), action.pose)
    last = _ACTION_ASSIST_LAST_MS.get(key, -ACTION_ASSIST_COOLDOWN_MS)
    if now - last < ACTION_ASSIST_COOLDOWN_MS:
        return

    strength = max(float(getattr(env, "agent_strength", 1.0)), 1.0)
    mass = max(float(getattr(target.body, "mass", 1.0) or 1.0), 0.1)
    if action.pose == "kick":
        impulse = axis * max(mass * 260.0, strength * 0.015)
        target.body.apply_impulse_at_world_point(impulse, target.body.position)
    elif action.pose == "throw":
        goal_axis = _axis_to_nearest_goal(env, target) or axis
        throw_axis = (goal_axis + pymunk.Vec2d(0.0, 0.55)).normalized()
        impulse = throw_axis * max(mass * 230.0, strength * 0.013)
        target.body.apply_impulse_at_world_point(impulse, target.body.position)
    elif action.pose == "push":
        force = axis * max(strength * 0.42, mass * 1200.0)
        target.body.apply_force_at_world_point(force, target.body.position)
    _ACTION_ASSIST_LAST_MS[key] = now


def _lift_selected_body(env: BaseEnv, selected_name: str | None) -> None:
    if not selected_name:
        return
    selected_record = env._objects.get(selected_name)
    if selected_record is None or selected_record.body.body_type != pymunk.Body.DYNAMIC:
        return
    selected_record.body.position = selected_record.body.position + pymunk.Vec2d(0, LIFT_IMPULSE_HEIGHT)
    selected_record.body.velocity = (0, 0)
    selected_record.body.angular_velocity = 0


def _impulse_selected_body(
    env: BaseEnv,
    selected_name: str | None,
    direction: pymunk.Vec2d,
) -> None:
    if not selected_name:
        return
    selected_record = env._objects.get(selected_name)
    if selected_record is None or selected_record.body.body_type != pymunk.Body.DYNAMIC:
        return
    impulse = direction.normalized() * selected_record.body.mass * CONTROL_ACCELERATION * IMPULSE_MULTIPLIER
    selected_record.body.apply_impulse_at_world_point(impulse, selected_record.body.position)


def _draw_background(
    screen: pygame.Surface,
    camera: Camera,
    env: BaseEnv,
    grammar: VisualGrammar,
    tick: int,
) -> None:
    screen.fill(grammar.background_color)
    if grammar.visual_recipe:
        _draw_recipe_background(screen, grammar, tick)
        return
    if grammar.background in {"parallax_starfield", "dense_starfield", "orbital_dust"}:
        _draw_starfield(screen, grammar, tick, dense=grammar.background == "dense_starfield")
    elif grammar.background in {"scanline_lab", "oscilloscope"}:
        _draw_scanlines(screen, grammar, tick)
    elif grammar.background in {"hazard_warning", "ember_drift", "smoke_haze"}:
        _draw_hazard_backdrop(screen, grammar, tick)
    elif grammar.background in {"corn_rows", "organic_noise", "leaf_shadow"}:
        _draw_organic_backdrop(screen, grammar, tick)
    elif grammar.background in {"circuit_board", "blueprint_grid", "radar_sweep"}:
        _draw_circuit_backdrop(screen, grammar, tick)
    elif grammar.background in {"crystal_speckle", "ice_frost", "desert_grain", "toxic_bubbles", "underwater_caustics", "retro_vignette"}:
        _draw_particle_backdrop(screen, grammar, tick)


def _draw_recipe_background(screen: pygame.Surface, grammar: VisualGrammar, tick: int) -> None:
    recipe = grammar.visual_recipe
    layers = recipe.get("background_layers")
    if not isinstance(layers, list):
        return
    rng = random.Random(grammar.seed)
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        layer_type = str(layer.get("type") or "")
        if layer_type in {"basalt_noise", "brass_plate_noise"}:
            density = float(layer.get("density") or 0.5)
            count = int(90 * density)
            for index in range(count):
                x = rng.randrange(screen.get_width())
                y = rng.randrange(screen.get_height())
                radius = 1 + (index + grammar.seed) % 3
                color = grammar.hot if layer_type == "basalt_noise" and index % 17 == 0 else grammar.primary
                pygame.draw.circle(overlay, (*color, 12 + index % 18), (x, y), radius)
        elif layer_type == "molten_cracks":
            count = int(layer.get("count") or 16)
            for index in range(count):
                y = int(screen.get_height() * (0.16 + 0.72 * ((index * 37 + grammar.seed % 97) % 100) / 100))
                x0 = int((index * 113 + grammar.seed) % screen.get_width())
                points = []
                for step in range(6):
                    x = (x0 + step * 46 + int(math.sin(tick * 0.001 + index + step) * 12)) % screen.get_width()
                    points.append((x, y + int(math.sin(step * 1.7 + index) * 18)))
                pygame.draw.lines(overlay, (*grammar.hot, 70), False, points, 2)
                pygame.draw.lines(overlay, (255, 207, 92, 42), False, points, 5)
        elif layer_type in {"ember_field", "workshop_dust", "dust_motes", "cold_mist", "acid_vapor", "orbital_dust"}:
            count = int(layer.get("count") or 80)
            for index in range(count):
                speed = 0.018 + (index % 7) * 0.004
                x = (rng.randrange(screen.get_width()) + int(math.sin(tick * 0.001 + index) * 12)) % screen.get_width()
                direction = -1 if layer_type in {"ember_field", "cold_mist"} else 1
                y = (rng.randrange(screen.get_height()) + direction * int(tick * speed)) % screen.get_height()
                color = grammar.hot if layer_type == "ember_field" else grammar.secondary
                pygame.draw.circle(overlay, (*color, 36 + index % 48), (x, y), 1 + index % 2)
        elif layer_type in {"heat_shimmer", "nebula_wash"}:
            for band in range(5):
                y = int((band + 1) * screen.get_height() / 6 + math.sin(tick * 0.0015 + band) * 18)
                pygame.draw.line(overlay, (*grammar.hot, 18), (0, y), (screen.get_width(), y + int(math.sin(band) * 14)), 3)
        elif layer_type in {"crt_scanlines", "arcade_grid", "oscilloscope_grid", "simulation_grid"}:
            cell = int(layer.get("cell") or 32)
            for x in range((tick // 160) % cell, screen.get_width(), cell):
                pygame.draw.line(overlay, (*grammar.primary, 24), (x, 0), (x, screen.get_height()), 1)
            for y in range((tick // 190) % cell, screen.get_height(), cell):
                pygame.draw.line(overlay, (*grammar.secondary, 18), (0, y), (screen.get_width(), y), 1)
        elif layer_type in {"parallax_starfield", "star_dust"}:
            count = int(layer.get("count") or 150)
            for index in range(count):
                x = (rng.randrange(screen.get_width()) + int(tick * (0.006 + index % 4 * 0.001))) % screen.get_width()
                y = rng.randrange(screen.get_height())
                color = grammar.secondary if index % 9 == 0 else grammar.primary
                pygame.draw.circle(overlay, (*color, 80 + index % 90), (x, y), 1 if index % 13 else 2)
        elif layer_type in {"organic_canopy", "leaf_shadow"}:
            count = int(layer.get("count") or 60)
            for index in range(count):
                x = (index * 31 + grammar.seed) % max(screen.get_width(), 1)
                sway = int(math.sin(tick * 0.0018 + index) * 9)
                pygame.draw.ellipse(overlay, (*grammar.secondary, 24), (x + sway, (index * 47) % screen.get_height(), 30, 10))
        elif layer_type == "atmospheric_vignette":
            strength = float(layer.get("strength") or 0.3)
            pygame.draw.rect(overlay, (0, 0, 0, int(70 * strength)), screen.get_rect(), 18)
        elif layer_type == "set_dressing_props":
            _draw_recipe_set_dressing(overlay, grammar, layer, tick)
    screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_ADD)


def _draw_recipe_set_dressing(
    overlay: pygame.Surface,
    grammar: VisualGrammar,
    layer: Mapping[str, Any],
    tick: int,
) -> None:
    """Draw non-colliding scenic landmarks so fast variants read as different places."""

    width, height = overlay.get_size()
    theme = str(layer.get("theme") or "").lower()
    count = int(layer.get("count") or 6)
    label = str(grammar.visual_recipe.get("variant_label", "")).lower()
    mood = grammar.mood.lower()
    seed = grammar.seed + (17 if theme == "cinematic" else 37 if theme == "alternate" else 5)

    if "water" in mood or label in {"lagoon", "moonpool", "stormwater"}:
        if label == "moonpool":
            pygame.draw.circle(overlay, (*grammar.secondary, 58), (width - 130, 108), 46)
            pygame.draw.circle(overlay, (*grammar.primary, 24), (width - 154, 96), 64, 2)
            for index in range(5):
                y = int(height * (0.32 + index * 0.105) + math.sin(tick * 0.001 + index) * 10)
                pygame.draw.line(overlay, (*grammar.primary, 38), (0, y), (width, y + 18), 3)
        elif label == "stormwater":
            for index in range(count + 8):
                x = (index * 91 + seed) % max(width, 1)
                y = (index * 47 + tick // 12) % max(height, 1)
                pygame.draw.line(overlay, (*grammar.secondary, 72), (x, y), (x + 12, y + 42), 2)
            for index in range(4):
                x = 45 + index * 130
                pygame.draw.rect(overlay, (*grammar.primary, 34), (x, height - 190 - index * 7, 58, 190), border_radius=12)
                pygame.draw.rect(overlay, (*grammar.secondary, 52), (x + 8, height - 182 - index * 7, 42, 18), 2, border_radius=8)
        else:
            pygame.draw.circle(overlay, (*grammar.secondary, 46), (118, 94), 54)
            for index in range(count):
                x = (index * 153 + seed) % max(width, 1)
                y = int(height * (0.66 + 0.18 * math.sin(index)))
                pygame.draw.ellipse(overlay, (*grammar.secondary, 44), (x - 46, y - 16, 92, 28))
                pygame.draw.ellipse(overlay, (*grammar.primary, 54), (x - 34, y - 11, 68, 20), 2)
        return

    if "forest" in mood or "woods" in mood or "grove" in mood:
        for index in range(count + 5):
            x = (index * 97 + seed) % max(width, 1)
            trunk_h = 54 + (index % 4) * 17
            base_y = height - 16
            pygame.draw.line(overlay, (*grammar.hot, 30), (x, base_y), (x + 8, base_y - trunk_h), 5)
            pygame.draw.circle(overlay, (*grammar.secondary, 42), (x + 8, base_y - trunk_h - 18), 24 + index % 3 * 8)
        return

    if "space" in mood or "orbital" in mood:
        for index in range(count):
            x = (index * 167 + seed) % max(width, 1)
            y = (index * 83 + seed // 3) % max(height, 1)
            radius = 18 + (index % 4) * 9
            pygame.draw.circle(overlay, (*grammar.primary, 25), (x, y), radius)
            pygame.draw.circle(overlay, (*grammar.secondary, 42), (x, y), radius, 2)
        return

    if label in {"court", "arena", "street"}:
        horizon = int(height * 0.70)
        pygame.draw.line(overlay, (*grammar.secondary, 52), (0, horizon), (width, horizon), 3)
        for index in range(6):
            x = (index * 170 + seed) % max(width, 1)
            pygame.draw.arc(overlay, (*grammar.primary, 42), (x - 42, horizon - 42, 84, 84), 0, math.pi, 2)
            pygame.draw.line(overlay, (*grammar.hot, 58), (x, horizon - 70), (x, horizon - 112), 3)
            pygame.draw.circle(overlay, (*grammar.hot, 80), (x, horizon - 116), 12, 2)
        return

    if "mechanical" in mood or "clockwork" in mood or "electric" in mood:
        for index in range(count + 4):
            x = (index * 127 + seed) % max(width, 1)
            y = (index * 79 + seed // 5) % max(height, 1)
            radius = 16 + (index % 4) * 7
            pygame.draw.circle(overlay, (*grammar.secondary, 34), (x, y), radius, 2)
            for spoke in range(6):
                angle = spoke * math.tau / 6 + tick * 0.0008
                pygame.draw.line(
                    overlay,
                    (*grammar.primary, 32),
                    (x, y),
                    (int(x + math.cos(angle) * radius), int(y + math.sin(angle) * radius)),
                    1,
                )
        return

    if label in {"arcade", "hologrid", "crystal"} or "maze" in mood or "arcade" in mood:
        cell = 42 if label != "arcade" else 34
        for x in range((seed % cell), width, cell):
            pygame.draw.line(overlay, (*grammar.primary, 40), (x, 0), (x, height), 2 if label == "arcade" else 1)
        for y in range(((seed // 3) % cell), height, cell):
            pygame.draw.line(overlay, (*grammar.secondary, 34), (0, y), (width, y), 1)
        if label == "crystal":
            for index in range(count + 5):
                x = (index * 113 + seed) % max(width, 1)
                y = (index * 67 + seed // 2) % max(height, 1)
                size = 14 + (index % 3) * 8
                pygame.draw.polygon(overlay, (*grammar.secondary, 42), [(x, y - size), (x + size, y), (x, y + size), (x - size, y)], 2)
        return

    for index in range(count):
        x = (index * 137 + seed) % max(width, 1)
        y = height - 45 - (index % 3) * 42
        size = 24 + (index % 4) * 10
        points = [(x, y - size), (x + size, y), (x, y + size), (x - size, y)]
        pygame.draw.polygon(overlay, (*grammar.secondary, 32), points)
        pygame.draw.polygon(overlay, (*grammar.primary, 50), points, 2)


def _draw_starfield(screen: pygame.Surface, grammar: VisualGrammar, tick: int, *, dense: bool) -> None:
    rng = random.Random(grammar.seed)
    count = 165 if dense else 92
    width, height = screen.get_size()
    nebula = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    for index in range(5 if dense else 3):
        x = (rng.randrange(width) + int(tick * (0.004 + index * 0.002))) % width
        y = (rng.randrange(height) + int(math.sin(tick * 0.0008 + index) * 18)) % height
        radius = 120 + index * 38
        color = grammar.primary if index % 2 else grammar.secondary
        pygame.draw.circle(nebula, (*color, 10 + index * 3), (x, y), radius)
    screen.blit(nebula, (0, 0), special_flags=pygame.BLEND_ADD)
    for index in range(count):
        x = (rng.randrange(width) + int(tick * (0.006 + (index % 5) * 0.002))) % width
        y = rng.randrange(height)
        radius = 1 if index % 9 else 2
        alpha = 90 + (index * 37 + tick // 12) % 120
        color = grammar.primary if index % 7 else grammar.secondary
        star = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
        pygame.draw.circle(star, (*color, alpha), (radius * 2, radius * 2), radius)
        screen.blit(star, (x, y), special_flags=pygame.BLEND_ADD)


def _draw_scanlines(screen: pygame.Surface, grammar: VisualGrammar, tick: int) -> None:
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    for y in range((tick // 80) % 6, screen.get_height(), 6):
        pygame.draw.line(overlay, (*grammar.primary, 22), (0, y), (screen.get_width(), y), 1)
    for x in range(0, screen.get_width(), 96):
        pygame.draw.line(overlay, (*grammar.secondary, 14), (x, 0), (x, screen.get_height()), 1)
    screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_ADD)


def _draw_hazard_backdrop(screen: pygame.Surface, grammar: VisualGrammar, tick: int) -> None:
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    pulse = 22 + int(18 * (0.5 + 0.5 * math.sin(tick * 0.006)))
    for x in range(-screen.get_height(), screen.get_width(), 120):
        pygame.draw.line(overlay, (*grammar.hot, pulse), (x, screen.get_height()), (x + screen.get_height(), 0), 3)
    screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_ADD)


def _draw_organic_backdrop(screen: pygame.Surface, grammar: VisualGrammar, tick: int) -> None:
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    for x in range(-80, screen.get_width() + 80, 54):
        sway = int(math.sin(tick * 0.0015 + x * 0.04) * 7)
        pygame.draw.line(overlay, (*grammar.secondary, 24), (x + sway, 0), (x - 32 + sway, screen.get_height()), 2)
    screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_ADD)


def _draw_circuit_backdrop(screen: pygame.Surface, grammar: VisualGrammar, tick: int) -> None:
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    for y in range(44, screen.get_height(), 88):
        pygame.draw.line(overlay, (*grammar.primary, 22), (0, y), (screen.get_width(), y), 1)
        dot_x = (tick // 18 + y * 3) % screen.get_width()
        pygame.draw.circle(overlay, (*grammar.secondary, 85), (dot_x, y), 3)
    screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_ADD)


def _draw_particle_backdrop(screen: pygame.Surface, grammar: VisualGrammar, tick: int) -> None:
    rng = random.Random(grammar.seed ^ 0xA51CE)
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    for index in range(46):
        x = rng.randrange(screen.get_width())
        y = (rng.randrange(screen.get_height()) + int(tick * (0.01 + index * 0.0008))) % screen.get_height()
        color = grammar.secondary if index % 3 else grammar.primary
        pygame.draw.circle(overlay, (*color, 34), (x, y), 1 + index % 3)
    screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_ADD)


def _draw_grid(screen: pygame.Surface, camera: Camera, env: BaseEnv, grammar: VisualGrammar, tick: int) -> None:
    env_width = int(camera.env_width)
    env_height = int(camera.env_height)
    for x in range(0, env_width + GRID_SIZE, GRID_SIZE):
        color = _dim(grammar.primary, 0.42 if x % (GRID_SIZE * 4) == 0 else 0.22)
        start = camera.point((x, 0))
        end = camera.point((x, env_height))
        pygame.draw.line(screen, color, start, end, 1)

    for y in range(0, env_height + GRID_SIZE, GRID_SIZE):
        color = _dim(grammar.primary, 0.42 if y % (GRID_SIZE * 4) == 0 else 0.22)
        start = camera.point((0, y))
        end = camera.point((env_width, y))
        pygame.draw.line(screen, color, start, end, 1)

    bounds = pygame.Rect(
        int(camera.left),
        int(camera.top),
        int(camera.env_width * camera.scale),
        int(camera.env_height * camera.scale),
    )
    pygame.draw.rect(screen, grammar.primary, bounds, 1)


def _draw_world(
    screen: pygame.Surface,
    camera: Camera,
    env: BaseEnv,
    overlay_enabled: bool,
    selected_name: str | None,
    grammar: VisualGrammar,
    tick: int,
) -> None:
    glow = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    agent_records: dict[str, Any] = {}
    for shape in env.space.shapes:
        role = getattr(shape, "harness_role", None)
        object_name = getattr(shape, "harness_object_name", None)
        record = env._objects.get(object_name) if object_name else None
        if record is not None and str(record.role or "").lower() == "agent":
            agent_records[record.name] = record
        color = color_for_record(record, grammar, sensor=bool(getattr(shape, "sensor", False))) if record else _color_for_role(role)
        if overlay_enabled:
            _draw_shape(glow, camera, shape, HITBOX + (42,), outline=True, width=5)
            _draw_shape(glow, camera, shape, color + (46,), outline=True, width=9)
            if object_name == selected_name:
                _draw_shape(glow, camera, shape, SELECTED + (95,), outline=True, width=14)
        if record is not None and str(record.role or "").lower() == "agent":
            if overlay_enabled or object_name == selected_name:
                _draw_shape(screen, camera, shape, (54, 104, 122), outline=True, width=1)
                if object_name == selected_name:
                    _draw_shape(screen, camera, shape, SELECTED, outline=True, width=2)
            continue
        _draw_animated_accent(screen, glow, camera, shape, record, grammar, tick)
        if record is not None and _draw_recipe_stylized_shape(screen, glow, camera, shape, record, grammar, tick):
            if object_name == selected_name:
                _draw_shape(screen, camera, shape, SELECTED, outline=True, width=3)
            continue
        _draw_shape(screen, camera, shape, color, outline=False)
        _draw_shape(screen, camera, shape, _brighten(color), outline=True, width=1)
        if object_name == selected_name:
            _draw_shape(screen, camera, shape, SELECTED, outline=True, width=3)
    for record in agent_records.values():
        _draw_agent_avatar(
            screen,
            glow,
            camera,
            env,
            record,
            grammar,
            tick,
            selected=record.name == selected_name,
        )
    if overlay_enabled:
        screen.blit(glow, (0, 0), special_flags=pygame.BLEND_ADD)
    for record in agent_records.values():
        _draw_agent_action_foreground(screen, camera, env, record, grammar, tick)


def _draw_recipe_stylized_shape(
    screen: pygame.Surface,
    glow: pygame.Surface,
    camera: Camera,
    shape: pymunk.Shape,
    record: Any,
    grammar: VisualGrammar,
    tick: int,
) -> bool:
    skin = _recipe_skin_for_record(record, grammar)
    material = str(skin.get("material", "") or "").lower()
    if not material:
        return False
    if "enemy_fighter" in material and isinstance(shape, pymunk.Poly):
        _draw_enemy_fighter_object(screen, glow, camera, shape, record, skin, grammar, tick)
        return True
    if "laser_bolt" in material and isinstance(shape, pymunk.Circle):
        _draw_laser_bolt_object(screen, glow, camera, shape, record, skin, grammar, tick)
        return True
    if "capital_ship_shadow" in material and isinstance(shape, pymunk.Poly):
        _draw_capital_ship_shadow_object(screen, glow, camera, shape, record, skin, grammar, tick)
        return True
    if ("lava_wave" in material or "molten_surface" in material) and isinstance(shape, pymunk.Poly):
        _draw_lava_pool_object(screen, glow, camera, shape, record, skin, grammar, tick)
        return True
    if "obsidian_exit_gate" in material and isinstance(shape, pymunk.Poly):
        _draw_obsidian_exit_object(screen, glow, camera, shape, record, skin, grammar, tick)
        return True
    if "bear_avatar" in material and isinstance(shape, pymunk.Circle):
        _draw_bear_avatar_object(screen, glow, camera, shape, record, skin, grammar, tick)
        return True
    if "heavy_iron_ball" in material and isinstance(shape, pymunk.Circle):
        _draw_heavy_weight_object(screen, glow, camera, shape, record, skin, grammar, tick)
        return True
    if "seesaw_plank" in material and isinstance(shape, pymunk.Poly):
        _draw_seesaw_plank_object(screen, glow, camera, shape, record, skin, grammar, tick)
        return True
    if "seesaw_pivot" in material and isinstance(shape, pymunk.Poly):
        _draw_seesaw_pivot_object(screen, glow, camera, shape, record, skin, grammar, tick)
        return True
    if "tropical_water" in material and isinstance(shape, pymunk.Poly):
        _draw_water_pool_object(screen, glow, camera, shape, record, skin, grammar, tick)
        return True
    if "water_current" in material:
        _draw_water_current_object(screen, glow, camera, shape, record, skin, grammar, tick)
        return True
    if "tropical_beacon" in material and isinstance(shape, pymunk.Poly):
        _draw_tropical_beacon_object(screen, glow, camera, shape, record, skin, grammar, tick)
        return True
    if "sand_bank" in material:
        _draw_shape(screen, camera, shape, _skin_tuple(skin, "fill", (214, 183, 114)), outline=False)
        _draw_shape(screen, camera, shape, _skin_tuple(skin, "outline", (255, 230, 155)), outline=True, width=2)
        return True
    if "evergreen_tree" in material and isinstance(shape, pymunk.Poly):
        _draw_evergreen_tree_object(screen, glow, camera, shape, record, skin, grammar, tick)
        return True
    if "wood_cabin" in material and isinstance(shape, pymunk.Poly):
        _draw_cabin_object(screen, glow, camera, shape, record, skin, grammar, tick)
        return True
    if "moss_path" in material:
        _draw_shape(screen, camera, shape, _skin_tuple(skin, "fill", grammar.secondary) + (80,), outline=False)
        _draw_shape(screen, camera, shape, _skin_tuple(skin, "outline", grammar.primary) + (92,), outline=True, width=2)
        return True
    return False


def _draw_enemy_fighter_object(
    screen: pygame.Surface,
    glow: pygame.Surface,
    camera: Camera,
    shape: pymunk.Poly,
    record: Any,
    skin: dict[str, Any],
    grammar: VisualGrammar,
    tick: int,
) -> None:
    points = [camera.point(shape.body.local_to_world(vertex)) for vertex in shape.get_vertices()]
    if len(points) < 3:
        return
    bb = shape.bb
    center = camera.point(record.body.position)
    unit = max(12, camera.length(max(bb.right - bb.left, bb.top - bb.bottom) * 0.24))
    fill = _skin_tuple(skin, "fill", (30, 20, 42))
    outline = _skin_tuple(skin, "outline", grammar.hot)
    glow_color = _skin_tuple(skin, "glow", grammar.hot)
    nose = (center[0] - int(unit * 1.35), center[1])
    tail_top = (center[0] + int(unit * 0.95), center[1] - int(unit * 0.62))
    tail_mid = (center[0] + int(unit * 0.40), center[1])
    tail_bottom = (center[0] + int(unit * 0.95), center[1] + int(unit * 0.62))
    wing_top = (center[0] - int(unit * 0.10), center[1] - int(unit * 0.95))
    wing_bottom = (center[0] - int(unit * 0.10), center[1] + int(unit * 0.95))
    hull = [nose, wing_top, tail_top, tail_mid, tail_bottom, wing_bottom]
    pulse = 0.55 + 0.45 * math.sin(tick * 0.055 + center[1] * 0.02)
    pygame.draw.polygon(glow, (*glow_color, int(70 * pulse)), hull, max(4, unit // 2))
    pygame.draw.polygon(screen, fill, hull)
    pygame.draw.polygon(screen, outline, hull, max(2, unit // 8))
    pygame.draw.circle(screen, _brighten(outline), (center[0] - int(unit * 0.45), center[1]), max(2, unit // 7))
    for sign in (-1, 1):
        emitter = (center[0] - int(unit * 1.08), center[1] + sign * int(unit * 0.24))
        pygame.draw.circle(glow, (*outline, int(90 * pulse)), emitter, max(3, unit // 6))


def _draw_laser_bolt_object(
    screen: pygame.Surface,
    glow: pygame.Surface,
    camera: Camera,
    shape: pymunk.Circle,
    record: Any,
    skin: dict[str, Any],
    grammar: VisualGrammar,
    tick: int,
) -> None:
    center = camera.point(shape.body.position)
    velocity = record.body.velocity
    if velocity.length > 1.0:
        direction = velocity.normalized()
    else:
        direction = pymunk.Vec2d(-1, 0)
    length = max(36, camera.length(shape.radius * 7.5))
    radius = max(3, camera.length(shape.radius))
    start = camera.point(shape.body.position - direction * length * 0.42)
    end = camera.point(shape.body.position + direction * length * 0.58)
    fill = _skin_tuple(skin, "fill", grammar.hot)
    outline = _skin_tuple(skin, "outline", (255, 235, 250))
    glow_color = _skin_tuple(skin, "glow", grammar.hot)
    pulse = 0.72 + 0.28 * math.sin(tick * 0.16 + center[0] * 0.03)
    pygame.draw.line(glow, (*glow_color, int(120 * pulse)), start, end, radius * 4)
    pygame.draw.line(screen, fill, start, end, radius * 2)
    pygame.draw.line(screen, outline, start, end, max(2, radius // 2))
    pygame.draw.circle(glow, (*outline, 115), end, radius * 2)


def _draw_capital_ship_shadow_object(
    screen: pygame.Surface,
    glow: pygame.Surface,
    camera: Camera,
    shape: pymunk.Poly,
    record: Any,
    skin: dict[str, Any],
    grammar: VisualGrammar,
    tick: int,
) -> None:
    points = [camera.point(shape.body.local_to_world(vertex)) for vertex in shape.get_vertices()]
    if len(points) < 3:
        return
    fill = _skin_tuple(skin, "fill", (20, 22, 42))
    outline = _skin_tuple(skin, "outline", (88, 115, 172))
    glow_color = _skin_tuple(skin, "glow", grammar.secondary)
    pygame.draw.polygon(glow, (*glow_color, 28), points, 12)
    pygame.draw.polygon(screen, (*fill, 125), points)
    pygame.draw.polygon(screen, (*outline, 130), points, 2)
    bb = shape.bb
    left, top = camera.point((bb.left, bb.top))
    right, bottom = camera.point((bb.right, bb.bottom))
    for index in range(6):
        y = min(top, bottom) + int((index + 1) * abs(bottom - top) / 7)
        pygame.draw.line(screen, (*outline, 74), (min(left, right) + 12, y), (max(left, right) - 12, y - 6), 1)


def _draw_lava_pool_object(
    screen: pygame.Surface,
    glow: pygame.Surface,
    camera: Camera,
    shape: pymunk.Poly,
    record: Any,
    skin: dict[str, Any],
    grammar: VisualGrammar,
    tick: int,
) -> None:
    points = [camera.point(shape.body.local_to_world(vertex)) for vertex in shape.get_vertices()]
    if len(points) < 3:
        return
    fill = _skin_tuple(skin, "fill", (255, 54, 14))
    outline = _skin_tuple(skin, "outline", (255, 184, 52))
    glow_color = _skin_tuple(skin, "glow", grammar.hot)
    pygame.draw.polygon(glow, (*glow_color, 82), points, 0)
    pygame.draw.polygon(screen, fill, points, 0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    for index in range(5):
        y = top + int((index + 0.55) * (bottom - top) / 5.8)
        wave_points: list[tuple[int, int]] = []
        for x in range(left, right + 8, 8):
            wave = int(math.sin(x * 0.045 + tick * 0.018 + index * 1.4) * (4 + index))
            wave_points.append((x, y + wave))
        if len(wave_points) > 1:
            pygame.draw.lines(glow, (*outline, 96), False, wave_points, 3 if index == 0 else 2)
            pygame.draw.lines(screen, outline if index == 0 else _dim(outline, 0.78), False, wave_points, 1)
    for index in range(12):
        bx = left + (index * 41 + int(tick * 0.33)) % max(1, right - left)
        by = bottom - ((index * 17 + int(tick * 0.52)) % max(1, bottom - top))
        pygame.draw.circle(glow, (255, 220, 86, 80), (bx, by), 2 + index % 3, 1)


def _draw_obsidian_exit_object(
    screen: pygame.Surface,
    glow: pygame.Surface,
    camera: Camera,
    shape: pymunk.Poly,
    record: Any,
    skin: dict[str, Any],
    grammar: VisualGrammar,
    tick: int,
) -> None:
    points = [camera.point(shape.body.local_to_world(vertex)) for vertex in shape.get_vertices()]
    if len(points) < 3:
        return
    fill = _skin_tuple(skin, "fill", (34, 20, 31))
    outline = _skin_tuple(skin, "outline", (255, 218, 100))
    glow_color = _skin_tuple(skin, "glow", grammar.hot)
    pulse = 0.55 + 0.45 * math.sin(tick * 0.008)
    pygame.draw.polygon(glow, (*glow_color, int(72 * pulse)), points, 10)
    pygame.draw.polygon(screen, fill, points, 0)
    pygame.draw.polygon(screen, outline, points, 3)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    rect = pygame.Rect(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
    pygame.draw.arc(glow, (*glow_color, int(118 * pulse)), rect.inflate(20, 20), math.pi * 0.08, math.pi * 1.92, 4)
    pygame.draw.circle(screen, _brighten(outline), rect.center, max(5, min(rect.size) // 8), 1)


def _draw_water_pool_object(
    screen: pygame.Surface,
    glow: pygame.Surface,
    camera: Camera,
    shape: pymunk.Poly,
    record: Any,
    skin: dict[str, Any],
    grammar: VisualGrammar,
    tick: int,
) -> None:
    points = [camera.point(shape.body.local_to_world(vertex)) for vertex in shape.get_vertices()]
    if len(points) < 3:
        return
    fill = _skin_tuple(skin, "fill", (27, 144, 196))
    outline = _skin_tuple(skin, "outline", (105, 247, 255))
    foam = _skin_tuple(skin, "foam", (221, 255, 248))
    alpha = int(float(skin.get("alpha", 0.56)) * 255) if isinstance(skin.get("alpha"), (int, float)) else 142
    pygame.draw.polygon(screen, fill + (alpha,), points, 0)
    pygame.draw.polygon(glow, outline + (42,), points, 0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    for line_index in range(5):
        y = top + int((line_index + 0.45) * (bottom - top) / 5.7)
        wave_points: list[tuple[int, int]] = []
        for x in range(left, right + 8, 8):
            wave = int(math.sin(x * 0.035 + tick * 0.014 + line_index * 1.7) * (4 + line_index * 0.7))
            wave_points.append((x, y + wave))
        if len(wave_points) > 1:
            color = foam if line_index == 0 else outline
            pygame.draw.lines(glow, color + (70 if line_index == 0 else 38,), False, wave_points, 2 if line_index == 0 else 1)
            pygame.draw.lines(screen, color, False, wave_points, 1)
    for index in range(10):
        bx = left + (index * 37 + int(tick * 0.3)) % max(1, right - left)
        by = bottom - ((index * 29 + int(tick * 0.45)) % max(1, bottom - top))
        pygame.draw.circle(glow, foam + (42,), (bx, by), 2 + index % 3, 1)


def _draw_water_current_object(
    screen: pygame.Surface,
    glow: pygame.Surface,
    camera: Camera,
    shape: pymunk.Shape,
    record: Any,
    skin: dict[str, Any],
    grammar: VisualGrammar,
    tick: int,
) -> None:
    color = _skin_tuple(skin, "outline", grammar.secondary)
    center = camera.point(record.body.position)
    width = max(60, camera.length(250))
    for index in range(4):
        y = center[1] + int((index - 1.5) * 18)
        phase = int((tick * 0.8 + index * 47) % width)
        start = (center[0] - width // 2 + phase, y)
        end = (start[0] + 32, y - 5)
        pygame.draw.line(glow, color + (38,), start, end, 2)
        pygame.draw.circle(glow, color + (45,), end, 3, 1)


def _draw_tropical_beacon_object(
    screen: pygame.Surface,
    glow: pygame.Surface,
    camera: Camera,
    shape: pymunk.Poly,
    record: Any,
    skin: dict[str, Any],
    grammar: VisualGrammar,
    tick: int,
) -> None:
    bb = shape.bb
    left = camera.point((bb.left, bb.bottom))[0]
    right = camera.point((bb.right, bb.top))[0]
    bottom = camera.point((bb.left, bb.bottom))[1]
    top = camera.point((bb.right, bb.top))[1]
    rect = pygame.Rect(min(left, right), min(top, bottom), abs(right - left), abs(bottom - top))
    fill = _skin_tuple(skin, "fill", grammar.secondary)
    outline = _skin_tuple(skin, "outline", (241, 255, 248))
    glow_color = _skin_tuple(skin, "glow", grammar.secondary)
    pulse = 0.62 + 0.38 * math.sin(tick * 0.01)
    pygame.draw.circle(glow, glow_color + (int(86 * pulse),), rect.center, max(rect.width, rect.height), 3)
    pygame.draw.rect(screen, _dim(fill, 0.55), rect, border_radius=4)
    pygame.draw.rect(screen, outline, rect, max(2, rect.width // 16), border_radius=4)
    pygame.draw.circle(screen, fill, (rect.centerx, rect.top + rect.height // 3), max(5, rect.width // 4), 0)
    for sign in (-1, 1):
        pygame.draw.line(glow, glow_color + (62,), rect.center, (rect.centerx + sign * rect.width, rect.top - rect.height // 2), 2)


def _draw_heavy_weight_object(
    screen: pygame.Surface,
    glow: pygame.Surface,
    camera: Camera,
    shape: pymunk.Circle,
    record: Any,
    skin: dict[str, Any],
    grammar: VisualGrammar,
    tick: int,
) -> None:
    center = camera.point(record.body.position)
    radius = max(10, camera.length(shape.radius))
    fill = _skin_tuple(skin, "fill", (88, 94, 100))
    outline = _skin_tuple(skin, "outline", (225, 235, 232))
    glow_color = _skin_tuple(skin, "glow", grammar.hot)
    pygame.draw.circle(glow, glow_color + (62,), center, int(radius * 1.45), 3)
    pygame.draw.circle(screen, fill, center, radius)
    pygame.draw.circle(screen, outline, center, radius, max(2, radius // 8))
    pygame.draw.circle(screen, _brighten(fill), (center[0] - radius // 4, center[1] - radius // 4), max(3, radius // 4), 0)
    pygame.draw.circle(screen, _dim(fill, 0.55), (center[0] + radius // 4, center[1] + radius // 5), max(4, radius // 3), 0)
    if getattr(record.body, "velocity", pymunk.Vec2d(0, 0)).length > 18:
        pygame.draw.circle(glow, glow_color + (45,), center, int(radius * (1.65 + 0.12 * math.sin(tick * 0.04))), 2)


def _draw_seesaw_plank_object(
    screen: pygame.Surface,
    glow: pygame.Surface,
    camera: Camera,
    shape: pymunk.Poly,
    record: Any,
    skin: dict[str, Any],
    grammar: VisualGrammar,
    tick: int,
) -> None:
    points = [camera.point(shape.body.local_to_world(vertex)) for vertex in shape.get_vertices()]
    if len(points) < 3:
        return
    fill = _skin_tuple(skin, "fill", (164, 96, 42))
    outline = _skin_tuple(skin, "outline", (255, 213, 115))
    stripe = _skin_tuple(skin, "stripe", (92, 50, 25))
    glow_color = _skin_tuple(skin, "glow", grammar.primary)
    pygame.draw.polygon(glow, glow_color + (54,), points, 0)
    pygame.draw.polygon(screen, fill, points, 0)
    pygame.draw.polygon(screen, outline, points, max(2, camera.length(1.5)))
    center = camera.point(record.body.position)
    angle = float(record.body.angle)
    axis = pymunk.Vec2d(math.cos(angle), math.sin(angle))
    normal = pymunk.Vec2d(-axis.y, axis.x)
    half_len = max(40, camera.length(175))
    for offset in (-0.22, 0.22):
        a = (int(center[0] - axis.x * half_len + normal.x * offset * 18), int(center[1] + axis.y * half_len - normal.y * offset * 18))
        b = (int(center[0] + axis.x * half_len + normal.x * offset * 18), int(center[1] - axis.y * half_len - normal.y * offset * 18))
        pygame.draw.line(screen, stripe, a, b, max(1, camera.length(1.0)))
    for sign, accent in ((-1, grammar.hot), (1, grammar.secondary)):
        cap = (int(center[0] + sign * axis.x * half_len), int(center[1] - sign * axis.y * half_len))
        pygame.draw.circle(glow, accent + (88,), cap, max(5, camera.length(8)), 2)
        pygame.draw.circle(screen, accent, cap, max(4, camera.length(5)), 0)


def _draw_seesaw_pivot_object(
    screen: pygame.Surface,
    glow: pygame.Surface,
    camera: Camera,
    shape: pymunk.Poly,
    record: Any,
    skin: dict[str, Any],
    grammar: VisualGrammar,
    tick: int,
) -> None:
    bb = shape.bb
    left = camera.point((bb.left, bb.bottom))[0]
    right = camera.point((bb.right, bb.top))[0]
    bottom = camera.point((bb.left, bb.bottom))[1]
    top = camera.point((bb.right, bb.top))[1]
    rect = pygame.Rect(min(left, right), min(top, bottom), abs(right - left), abs(bottom - top))
    fill = _skin_tuple(skin, "fill", (92, 79, 67))
    outline = _skin_tuple(skin, "outline", (255, 220, 138))
    glow_color = _skin_tuple(skin, "glow", grammar.primary)
    triangle = [(rect.centerx, rect.top - 4), (rect.left - 10, rect.bottom), (rect.right + 10, rect.bottom)]
    pygame.draw.polygon(glow, glow_color + (54,), triangle, 0)
    pygame.draw.polygon(screen, fill, triangle, 0)
    pygame.draw.polygon(screen, outline, triangle, max(2, rect.width // 12))
    pygame.draw.circle(screen, outline, (rect.centerx, rect.top + rect.height // 6), max(4, rect.width // 6), 0)


def _draw_bear_avatar_object(
    screen: pygame.Surface,
    glow: pygame.Surface,
    camera: Camera,
    shape: pymunk.Circle,
    record: Any,
    skin: dict[str, Any],
    grammar: VisualGrammar,
    tick: int,
) -> None:
    center = camera.point(record.body.position)
    radius = max(12, camera.length(shape.radius))
    velocity = getattr(record.body, "velocity", pymunk.Vec2d(0, 0))
    facing = 1 if velocity.x >= -2 else -1
    stride = math.sin(tick * 0.018 + record.body.position.x * 0.03)
    fill = _skin_tuple(skin, "fill", (112, 68, 36))
    outline = _skin_tuple(skin, "outline", (248, 196, 122))
    glow_color = _skin_tuple(skin, "glow", grammar.hot)
    body_rect = pygame.Rect(0, 0, int(radius * 2.45), int(radius * 1.45))
    body_rect.center = (center[0] - facing * radius // 5, center[1] + radius // 5)
    head_center = (center[0] + facing * int(radius * 0.98), center[1] - int(radius * 0.18))
    pygame.draw.ellipse(glow, glow_color + (85,), body_rect.inflate(radius, radius // 2), 3)
    pygame.draw.ellipse(screen, fill, body_rect)
    pygame.draw.ellipse(screen, outline, body_rect, max(2, radius // 7))
    pygame.draw.circle(screen, fill, head_center, int(radius * 0.72))
    pygame.draw.circle(screen, outline, head_center, int(radius * 0.72), max(2, radius // 8))
    for ear_side in (-1, 1):
        ear = (head_center[0] - facing * int(radius * 0.18), head_center[1] - int(radius * (0.48 + ear_side * 0.04)))
        pygame.draw.circle(screen, _dim(fill, 0.82), ear, max(3, int(radius * 0.23)))
    eye = (head_center[0] + facing * int(radius * 0.24), head_center[1] - int(radius * 0.12))
    pygame.draw.circle(screen, (5, 8, 4), eye, max(2, radius // 10))
    nose = (head_center[0] + facing * int(radius * 0.56), head_center[1] + int(radius * 0.12))
    pygame.draw.circle(screen, (8, 7, 5), nose, max(2, radius // 8))
    for leg_index, xmul in enumerate((-0.65, -0.22, 0.28, 0.68)):
        phase = stride if leg_index % 2 == 0 else -stride
        hip = (center[0] + int(radius * xmul), center[1] + int(radius * 0.72))
        paw = (hip[0] + int(facing * phase * radius * 0.18), hip[1] + int(radius * 0.62))
        pygame.draw.line(screen, _dim(fill, 0.72), hip, paw, max(4, radius // 4))
        pygame.draw.circle(screen, outline, paw, max(3, radius // 7))
    for index in range(2):
        trail = (center[0] - facing * int(radius * (1.65 + index * 0.55)), center[1] + int(stride * radius * 0.14))
        pygame.draw.circle(glow, glow_color + (44 - index * 12,), trail, max(3, radius // (3 + index)), 1)


def _draw_evergreen_tree_object(
    screen: pygame.Surface,
    glow: pygame.Surface,
    camera: Camera,
    shape: pymunk.Poly,
    record: Any,
    skin: dict[str, Any],
    grammar: VisualGrammar,
    tick: int,
) -> None:
    bb = shape.bb
    left, right = camera.point((bb.left, bb.bottom))[0], camera.point((bb.right, bb.top))[0]
    bottom = camera.point((bb.left, bb.bottom))[1]
    top = camera.point((bb.right, bb.top))[1]
    cx = (left + right) // 2
    width = max(18, abs(right - left))
    height = max(30, abs(bottom - top))
    fill = _skin_tuple(skin, "fill", (32, 122, 61))
    outline = _skin_tuple(skin, "outline", (130, 230, 122))
    trunk = _skin_tuple(skin, "trunk", (91, 57, 31))
    sway = int(math.sin(tick * 0.003 + record.body.position.x * 0.02) * max(1, width * 0.08))
    trunk_rect = pygame.Rect(cx - width // 8, bottom - int(height * 0.34), max(4, width // 4), int(height * 0.34))
    pygame.draw.rect(screen, trunk, trunk_rect)
    for layer in range(3):
        y = bottom - int(height * (0.24 + layer * 0.25))
        half = int(width * (0.75 - layer * 0.12))
        points = [(cx + sway, y - int(height * 0.34)), (cx - half, y + int(height * 0.16)), (cx + half, y + int(height * 0.16))]
        pygame.draw.polygon(glow, fill + (32,), points, 0)
        pygame.draw.polygon(screen, _brighten(fill) if layer == 2 else fill, points, 0)
        pygame.draw.polygon(screen, outline, points, 1)


def _draw_cabin_object(
    screen: pygame.Surface,
    glow: pygame.Surface,
    camera: Camera,
    shape: pymunk.Poly,
    record: Any,
    skin: dict[str, Any],
    grammar: VisualGrammar,
    tick: int,
) -> None:
    bb = shape.bb
    left = camera.point((bb.left, bb.bottom))[0]
    right = camera.point((bb.right, bb.top))[0]
    bottom = camera.point((bb.left, bb.bottom))[1]
    top = camera.point((bb.right, bb.top))[1]
    rect = pygame.Rect(min(left, right), min(top, bottom), abs(right - left), abs(bottom - top))
    fill = _skin_tuple(skin, "fill", (121, 73, 40))
    outline = _skin_tuple(skin, "outline", (255, 218, 150))
    roof = _skin_tuple(skin, "roof", (70, 35, 26))
    glow_color = _skin_tuple(skin, "glow", grammar.secondary)
    roof_points = [(rect.left - 8, rect.top + rect.height // 3), (rect.centerx, rect.top - rect.height // 4), (rect.right + 8, rect.top + rect.height // 3)]
    pygame.draw.rect(glow, glow_color + (44,), rect.inflate(18, 14), border_radius=4)
    pygame.draw.rect(screen, fill, rect, border_radius=3)
    pygame.draw.rect(screen, outline, rect, max(2, rect.width // 40), border_radius=3)
    pygame.draw.polygon(screen, roof, roof_points)
    pygame.draw.polygon(screen, outline, roof_points, 2)
    window = pygame.Rect(0, 0, max(10, rect.width // 4), max(10, rect.height // 4))
    window.center = (rect.centerx + rect.width // 5, rect.centery)
    pulse = 0.65 + 0.35 * math.sin(tick * 0.008)
    pygame.draw.rect(glow, glow_color + (int(120 * pulse),), window.inflate(18, 18), border_radius=3)
    pygame.draw.rect(screen, glow_color, window, border_radius=2)
    door = pygame.Rect(rect.left + rect.width // 5, rect.bottom - rect.height // 2, max(8, rect.width // 5), rect.height // 2)
    pygame.draw.rect(screen, _dim(fill, 0.72), door)


def _skin_tuple(skin: dict[str, Any], key: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    value = skin.get(key)
    if isinstance(value, str) and value.startswith("#") and len(value) == 7:
        try:
            return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))
        except ValueError:
            return fallback
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return tuple(max(0, min(255, int(float(channel)))) for channel in value[:3])  # type: ignore[return-value]
        except (TypeError, ValueError):
            return fallback
    return fallback


def _draw_animated_accent(
    screen: pygame.Surface,
    glow: pygame.Surface,
    camera: Camera,
    shape: pymunk.Shape,
    record: Any,
    grammar: VisualGrammar,
    tick: int,
) -> None:
    if record is None:
        return
    role = str(record.role or "").lower()
    name = str(record.name or "").lower()
    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    text = " ".join([role, name, json.dumps(metadata, default=str).lower()])
    skin = _recipe_skin_for_record(record, grammar)
    skin_text = json.dumps(skin, sort_keys=True, default=str).lower() if skin else ""
    pulse = 0.5 + 0.5 * math.sin(tick * 0.005 + len(name))
    if any(token in skin_text for token in ("flame", "fire", "ember", "molten")):
        center = camera.point(record.body.position)
        visual_scale = float(skin.get("visual_radius_scale", 1.0)) if isinstance(skin.get("visual_radius_scale"), (int, float)) else 1.0
        radius = max(6, int((16 + int(9 * pulse)) * visual_scale))
        pygame.draw.circle(glow, grammar.hot + (118,), center, radius, 2)
        pygame.draw.circle(glow, (255, 221, 111, 76), center, max(4, radius // 2), 1)
        velocity = getattr(record.body, "velocity", pymunk.Vec2d(0, 0))
        if velocity.length > 12:
            direction = -velocity.normalized()
            for index in range(5):
                ember_pos = record.body.position + direction * (10 + index * 9) + pymunk.Vec2d(
                    math.sin(tick * 0.01 + index) * 3,
                    math.cos(tick * 0.011 + index) * 3,
                )
                pygame.draw.circle(glow, (255, 145, 48, max(28, 90 - index * 12)), camera.point(ember_pos), max(1, 4 - index // 2))
    if any(token in skin_text for token in ("lava_wave", "molten_surface", "crack_glow")) and isinstance(shape, pymunk.Poly):
        _draw_shape(glow, camera, shape, grammar.hot + (64 + int(45 * pulse),), outline=True, width=6)
    if role == "goal" or "goal" in text or "target" in text:
        if "goal_breathe" in grammar.motion_fx:
            _draw_shape(glow, camera, shape, grammar.secondary + (55 + int(70 * pulse),), outline=True, width=12 + int(5 * pulse))
    if "force_zone" in role or "force_zone" in text or "magnetic" in text or "field" in text:
        if "field_pulse" in grammar.motion_fx or "magnetic_rings" in grammar.motion_fx:
            center = camera.point(record.body.position)
            radius = 18 + int(22 * pulse)
            pygame.draw.circle(glow, grammar.primary + (72,), center, radius, 2)
            pygame.draw.circle(glow, grammar.secondary + (42,), center, max(5, radius // 2), 1)
    if role == "hazard" or "hazard" in text or "danger" in text or "falling" in text:
        if "hazard_flash" in grammar.motion_fx:
            _draw_shape(glow, camera, shape, grammar.hot + (70 + int(90 * pulse),), outline=True, width=8 + int(4 * pulse))
    if "gate" in text or "door" in text:
        open_gate = bool(metadata.get("mechanism_open") or metadata.get("passable"))
        accent = grammar.secondary if open_gate else grammar.hot
        _draw_shape(glow, camera, shape, accent + (56,), outline=True, width=8)


def _draw_agent_avatar(
    screen: pygame.Surface,
    glow: pygame.Surface,
    camera: Camera,
    env: BaseEnv,
    record: Any,
    grammar: VisualGrammar,
    tick: int,
    *,
    selected: bool = False,
) -> None:
    """Renderer-only agent avatar; physics remains the simple agent body."""

    body = record.body
    center = camera.point(body.position)
    radius_world = _agent_body_radius(record)
    skin = _recipe_skin_for_record(record, grammar)
    visual_scale = float(skin.get("scale", 1.0)) if isinstance(skin.get("scale"), (int, float)) else 1.0
    unit = max(18, int(camera.length(radius_world * 2.15) * visual_scale))
    velocity = getattr(body, "velocity", pymunk.Vec2d(0, 0))
    action = _infer_agent_action(env, record)
    nearest = action.nearest
    pose = action.pose
    facing = action.facing
    phase = tick * 0.009 + body.position.x * 0.03
    avatar = str(getattr(grammar, "agent_avatar", "human") or "human")
    if avatar != "human":
        _draw_nonhuman_agent_avatar(
            screen,
            glow,
            center=center,
            unit=unit,
            facing=facing,
            phase=phase,
            pose=pose,
            style=avatar,
            grammar=grammar,
            selected=selected,
            velocity=velocity,
        )
        return

    primary = _avatar_visible_color(grammar.primary)
    secondary = _avatar_visible_color(grammar.secondary)
    hot = (255, 116, 46)
    hot_bright = (255, 218, 128)
    stroke = max(6, int(unit * 0.30))
    glow_width = stroke + 8

    skeleton = _agent_skeleton_points(center, unit, facing, phase, pose)
    shadow = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    pygame.draw.ellipse(
        shadow,
        (*primary, 35),
        (
            int(center[0] - unit * 0.95),
            int(center[1] + unit * 0.92),
            int(unit * 1.9),
            max(3, int(unit * 0.34)),
        ),
    )
    screen.blit(shadow, (0, 0), special_flags=pygame.BLEND_ADD)

    for a, b in _agent_limb_pairs():
        _draw_round_limb(glow, (*primary, 58), skeleton[a], skeleton[b], glow_width)
    pygame.draw.circle(glow, (*secondary, 92), skeleton["head"], max(7, int(unit * 0.56)), 5)
    for a, b in _agent_limb_pairs():
        _draw_round_limb(screen, primary, skeleton[a], skeleton[b], stroke)
    _draw_round_limb(screen, _brighten(primary), skeleton["neck"], skeleton["hip"], max(5, stroke + 1))
    head_radius = max(8, int(unit * 0.55))
    pygame.draw.circle(screen, secondary, skeleton["head"], head_radius, 0)
    pygame.draw.circle(screen, _brighten(secondary), skeleton["head"], head_radius, max(2, stroke // 4))

    if pose == "kick" and nearest is not None:
        target = camera.point(nearest.body.position)
        foot = skeleton["foot_front"]
        pygame.draw.line(glow, (*hot, 150), foot, target, max(3, stroke))
        pygame.draw.circle(glow, (*hot, 160), target, max(10, unit // 2), 3)
        pygame.draw.circle(glow, (*secondary, 84), foot, max(5, unit // 4), 0)
        pygame.draw.line(screen, hot_bright, foot, target, max(3, stroke // 2))
        pygame.draw.circle(screen, hot_bright, foot, max(5, unit // 5), 0)
        pygame.draw.circle(screen, hot, target, max(8, unit // 3), 3)
        arc_rect = pygame.Rect(0, 0, int(unit * 2.6), int(unit * 1.9))
        arc_rect.center = (target[0] - facing * unit // 2, target[1] - unit // 2)
        start = -0.6 if facing > 0 else math.pi - 0.6
        end = 0.75 if facing > 0 else math.pi + 0.75
        pygame.draw.arc(glow, (*hot, 170), arc_rect, start, end, max(4, stroke))
        pygame.draw.arc(screen, hot_bright, arc_rect, start, end, max(3, stroke // 2))
    elif pose == "throw" and nearest is not None:
        target = camera.point(nearest.body.position)
        hand = skeleton["hand_front"]
        pygame.draw.line(glow, (*hot, 140), hand, target, max(3, stroke // 2))
        pygame.draw.circle(glow, (*hot, 140), target, max(7, unit // 3), 3)
        pygame.draw.line(screen, hot_bright, hand, target, max(3, stroke // 2))
        pygame.draw.circle(screen, hot_bright, hand, max(5, unit // 5), 0)
        pygame.draw.circle(screen, hot, target, max(6, unit // 4), 3)
        for index in range(3):
            trail_point = (
                target[0] - facing * int(unit * (0.45 + index * 0.32)),
                target[1] + int(unit * (0.18 + index * 0.18)),
            )
            pygame.draw.circle(glow, (*secondary, 62 - index * 12), trail_point, max(3, unit // (4 + index)), 1)
    elif pose == "push" and nearest is not None:
        target = camera.point(nearest.body.position)
        hand_mid = (
            int((skeleton["hand_front"][0] + skeleton["hand_back"][0]) * 0.5),
            int((skeleton["hand_front"][1] + skeleton["hand_back"][1]) * 0.5),
        )
        pygame.draw.line(glow, (*hot, 92), hand_mid, target, max(1, stroke))
        pygame.draw.circle(glow, (*hot, 96), target, max(4, unit // 3), 1)
    elif pose == "float":
        for index in range(3):
            offset = int(math.sin(phase + index) * unit * 0.7)
            pygame.draw.circle(glow, (*secondary, 36), (center[0] - facing * unit * (index + 1), center[1] + offset), max(3, unit // (3 + index)), 1)
    elif pose == "swim":
        water = _mix_tuple(grammar.secondary, (220, 255, 248), 0.38)
        for index in range(6):
            bubble = (
                int(center[0] - facing * unit * (0.35 + index * 0.24)),
                int(center[1] + math.sin(phase * 1.8 + index) * unit * 0.28 - index * 3),
            )
            pygame.draw.circle(glow, water + (54 - index * 5,), bubble, max(2, unit // (6 + index)), 1)
        crest: list[tuple[int, int]] = []
        for step in range(-5, 6):
            x = int(center[0] + step * unit * 0.18)
            y = int(center[1] - unit * 0.62 + math.sin(phase * 2.2 + step) * unit * 0.07)
            crest.append((x, y))
        pygame.draw.lines(glow, water + (92,), False, crest, max(2, unit // 12))
        pygame.draw.lines(screen, water, False, crest, max(1, unit // 18))
    elif pose == "run":
        trail = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        for index in range(3):
            p = (
                int(center[0] - facing * unit * (0.8 + index * 0.45)),
                int(center[1] + math.sin(phase - index) * unit * 0.15),
            )
            pygame.draw.circle(trail, (*primary, 46 - index * 10), p, max(2, unit // (3 + index)))
        screen.blit(trail, (0, 0), special_flags=pygame.BLEND_ADD)
    elif pose == "jump":
        for index in range(3):
            puff = (
                int(center[0] - facing * unit * (0.2 + index * 0.25)),
                int(center[1] + unit * (0.95 + index * 0.08)),
            )
            pygame.draw.circle(glow, (*secondary, 42 - index * 8), puff, max(3, unit // (3 + index)), 1)


def _draw_nonhuman_agent_avatar(
    screen: pygame.Surface,
    glow: pygame.Surface,
    *,
    center: tuple[int, int],
    unit: int,
    facing: int,
    phase: float,
    pose: str,
    style: str,
    grammar: VisualGrammar,
    selected: bool,
    velocity: pymunk.Vec2d,
) -> None:
    if style == "ship":
        _draw_ship_avatar(screen, glow, center, unit, facing, phase, grammar, selected, velocity)
    elif style == "robot":
        _draw_robot_avatar(screen, glow, center, unit, facing, phase, pose, grammar, selected)
    elif style == "arcade_disc":
        _draw_arcade_disc_avatar(screen, glow, center, unit, facing, phase, grammar, selected)
    elif style == "marble":
        _draw_marble_avatar(screen, glow, center, unit, phase, grammar, selected)
    elif style == "drone":
        _draw_drone_avatar(screen, glow, center, unit, phase, grammar, selected)
    elif style == "creature":
        _draw_creature_avatar(screen, glow, center, unit, facing, phase, pose, grammar, selected)
    else:
        _draw_orb_avatar(screen, glow, center, unit, phase, grammar, selected)


def _draw_agent_action_foreground(
    screen: pygame.Surface,
    camera: Camera,
    env: BaseEnv,
    record: Any,
    grammar: VisualGrammar,
    tick: int,
) -> None:
    action = _infer_agent_action(env, record)
    if action.pose not in {"kick", "throw"} or action.nearest is None:
        return
    center = camera.point(record.body.position)
    unit = max(22, camera.length(_agent_body_radius(record) * 2.15))
    phase = tick * 0.009 + record.body.position.x * 0.03
    skeleton = _agent_skeleton_points(center, unit, action.facing, phase, action.pose)
    target = camera.point(action.nearest.body.position)
    hot = (255, 116, 46)
    bright = (255, 230, 150)
    if action.pose == "kick":
        source = skeleton["foot_front"]
        pygame.draw.line(screen, bright, source, target, max(4, unit // 5))
        pygame.draw.circle(screen, hot, target, max(10, unit // 2), 3)
        pygame.draw.circle(screen, bright, source, max(6, unit // 4), 0)
        burst_dirs = [
            (action.facing, 0.0),
            (action.facing * 0.82, -0.52),
            (action.facing * 0.82, 0.52),
        ]
        for dx, dy in burst_dirs:
            end = (int(target[0] + dx * unit * 0.95), int(target[1] + dy * unit * 0.95))
            pygame.draw.line(screen, hot, target, end, max(2, unit // 10))
    else:
        source = skeleton["hand_front"]
        pygame.draw.line(screen, bright, source, target, max(4, unit // 6))
        pygame.draw.circle(screen, hot, target, max(8, unit // 3), 3)
        for index in range(3):
            trail = (
                int(target[0] - action.facing * unit * (0.28 + index * 0.32)),
                int(target[1] + unit * (0.18 + index * 0.12)),
            )
            pygame.draw.circle(screen, bright, trail, max(3, unit // (5 + index)), 1)


def _draw_ship_avatar(
    screen: pygame.Surface,
    glow: pygame.Surface,
    center: tuple[int, int],
    unit: int,
    facing: int,
    phase: float,
    grammar: VisualGrammar,
    selected: bool,
    velocity: pymunk.Vec2d,
) -> None:
    cx, cy = center
    bank = int(max(-unit * 0.22, min(unit * 0.22, float(velocity.y) * 0.025)))
    nose = (cx + facing * int(unit * 1.15), cy + bank)
    tail_top = (cx - facing * int(unit * 0.78), cy - int(unit * 0.62))
    tail_bottom = (cx - facing * int(unit * 0.78), cy + int(unit * 0.62))
    wing = (cx - facing * int(unit * 0.16), cy + int(math.sin(phase) * unit * 0.12))
    flame = (cx - facing * int(unit * (1.18 + 0.18 * math.sin(phase * 3.0))), cy)
    pygame.draw.polygon(glow, (*grammar.primary, 92), [nose, tail_top, flame, tail_bottom], max(3, unit // 3))
    pygame.draw.circle(glow, (*grammar.secondary, 70), center, int(unit * 1.25), 2)
    pygame.draw.polygon(screen, (8, 16, 24), [nose, tail_top, wing, tail_bottom])
    pygame.draw.polygon(screen, grammar.primary, [nose, tail_top, wing, tail_bottom], max(2, unit // 6))
    pygame.draw.circle(screen, _brighten(grammar.secondary), (cx + facing * unit // 4, cy - unit // 8), max(2, unit // 6))
    pygame.draw.polygon(glow, (*grammar.hot, 120), [tail_top, flame, tail_bottom])
    if selected:
        pygame.draw.circle(glow, (*SELECTED, 96), center, int(unit * 1.55), 2)


def _draw_robot_avatar(
    screen: pygame.Surface,
    glow: pygame.Surface,
    center: tuple[int, int],
    unit: int,
    facing: int,
    phase: float,
    pose: str,
    grammar: VisualGrammar,
    selected: bool,
) -> None:
    cx, cy = center
    bob = int(math.sin(phase * 1.8) * unit * 0.08)
    body = pygame.Rect(0, 0, int(unit * 1.08), int(unit * 1.18))
    body.center = (cx, cy - int(unit * 0.42) + bob)
    head = pygame.Rect(0, 0, int(unit * 0.88), int(unit * 0.62))
    head.center = (cx + facing * unit // 8, body.top - int(unit * 0.38))
    leg_stride = int(math.sin(phase * 2.3) * unit * 0.25) if pose == "run" else 0
    pygame.draw.rect(glow, (*grammar.primary, 88), body.inflate(unit, unit), border_radius=max(3, unit // 5))
    pygame.draw.rect(screen, (10, 18, 22), body, border_radius=max(3, unit // 6))
    pygame.draw.rect(screen, grammar.primary, body, max(2, unit // 7), border_radius=max(3, unit // 6))
    pygame.draw.rect(screen, (8, 14, 18), head, border_radius=max(3, unit // 7))
    pygame.draw.rect(screen, grammar.secondary, head, max(2, unit // 8), border_radius=max(3, unit // 7))
    eye_y = head.centery
    pygame.draw.circle(screen, _brighten(grammar.secondary), (head.centerx + facing * unit // 7, eye_y), max(2, unit // 8))
    for side in (-1, 1):
        arm_start = (body.centerx + side * body.width // 2, body.centery - unit // 5)
        arm_end = (
            arm_start[0] + side * int(unit * (0.58 if pose != "push" else 0.9)),
            arm_start[1] + int(unit * (0.38 if pose != "push" else 0.02)),
        )
        pygame.draw.line(screen, grammar.primary, arm_start, arm_end, max(2, unit // 6))
        pygame.draw.circle(screen, grammar.hot if pose == "push" else grammar.secondary, arm_end, max(2, unit // 6))
    for side in (-1, 1):
        hip = (body.centerx + side * unit // 4, body.bottom)
        foot = (hip[0] + side * leg_stride, hip[1] + int(unit * 0.55))
        pygame.draw.line(screen, grammar.primary, hip, foot, max(2, unit // 6))
    if selected:
        pygame.draw.circle(glow, (*SELECTED, 96), center, int(unit * 1.55), 2)


def _draw_arcade_disc_avatar(
    screen: pygame.Surface,
    glow: pygame.Surface,
    center: tuple[int, int],
    unit: int,
    facing: int,
    phase: float,
    grammar: VisualGrammar,
    selected: bool,
) -> None:
    radius = int(unit * 0.9)
    mouth = 0.34 + 0.2 * abs(math.sin(phase * 3.0))
    cx, cy = center
    points = [center]
    for index in range(25):
        angle = -mouth + (2 * math.pi - 2 * mouth) * index / 24.0
        if facing < 0:
            angle = math.pi - angle
        points.append((int(cx + math.cos(angle) * radius), int(cy + math.sin(angle) * radius)))
    pygame.draw.circle(glow, (*grammar.secondary, 96), center, int(radius * 1.25), 2)
    pygame.draw.polygon(screen, grammar.secondary, points)
    pygame.draw.polygon(screen, _brighten(grammar.primary), points, max(2, unit // 7))
    eye = (cx + facing * unit // 5, cy - unit // 3)
    pygame.draw.circle(screen, (8, 10, 18), eye, max(2, unit // 8))
    if selected:
        pygame.draw.circle(glow, (*SELECTED, 96), center, int(unit * 1.55), 2)


def _draw_marble_avatar(
    screen: pygame.Surface,
    glow: pygame.Surface,
    center: tuple[int, int],
    unit: int,
    phase: float,
    grammar: VisualGrammar,
    selected: bool,
) -> None:
    radius = int(unit * 0.82)
    pygame.draw.circle(glow, (*grammar.primary, 72), center, int(radius * 1.35), 3)
    pygame.draw.circle(screen, _dim(grammar.primary, 0.62), center, radius)
    pygame.draw.circle(screen, _brighten(grammar.secondary), center, radius, max(2, unit // 7))
    for index in range(3):
        angle = phase * (1.1 + index * 0.25) + index * 2.1
        offset = (int(math.cos(angle) * radius * 0.42), int(math.sin(angle) * radius * 0.42))
        pygame.draw.arc(
            screen,
            _brighten(grammar.primary if index % 2 else grammar.secondary),
            pygame.Rect(center[0] - radius + offset[0] // 3, center[1] - radius + offset[1] // 3, radius * 2, radius * 2),
            angle,
            angle + math.pi * 0.95,
            max(1, unit // 10),
        )
    pygame.draw.circle(screen, (245, 255, 255), (center[0] - radius // 3, center[1] - radius // 3), max(2, radius // 5))
    if selected:
        pygame.draw.circle(glow, (*SELECTED, 96), center, int(unit * 1.55), 2)


def _draw_drone_avatar(
    screen: pygame.Surface,
    glow: pygame.Surface,
    center: tuple[int, int],
    unit: int,
    phase: float,
    grammar: VisualGrammar,
    selected: bool,
) -> None:
    cx, cy = center
    bob = int(math.sin(phase * 2.0) * unit * 0.12)
    core = pygame.Rect(0, 0, int(unit * 1.05), int(unit * 0.46))
    core.center = (cx, cy + bob)
    pygame.draw.rect(glow, (*grammar.primary, 80), core.inflate(unit, unit // 2), border_radius=max(4, unit // 4))
    pygame.draw.rect(screen, (8, 16, 20), core, border_radius=max(3, unit // 5))
    pygame.draw.rect(screen, grammar.primary, core, max(2, unit // 8), border_radius=max(3, unit // 5))
    for side in (-1, 1):
        arm_end = (cx + side * int(unit * 0.95), cy + bob)
        pygame.draw.line(screen, grammar.primary, core.center, arm_end, max(2, unit // 8))
        rotor_r = max(4, unit // 4)
        pygame.draw.circle(glow, (*grammar.secondary, 70), arm_end, rotor_r + int(2 * abs(math.sin(phase * 5))), 1)
        pygame.draw.line(screen, grammar.secondary, (arm_end[0] - rotor_r, arm_end[1]), (arm_end[0] + rotor_r, arm_end[1]), max(1, unit // 12))
        pygame.draw.line(screen, grammar.secondary, (arm_end[0], arm_end[1] - rotor_r), (arm_end[0], arm_end[1] + rotor_r), max(1, unit // 12))
    if selected:
        pygame.draw.circle(glow, (*SELECTED, 96), center, int(unit * 1.55), 2)


def _draw_creature_avatar(
    screen: pygame.Surface,
    glow: pygame.Surface,
    center: tuple[int, int],
    unit: int,
    facing: int,
    phase: float,
    pose: str,
    grammar: VisualGrammar,
    selected: bool,
) -> None:
    cx, cy = center
    squash = 1.0 + (0.14 * math.sin(phase * 2.3) if pose == "run" else 0.05 * math.sin(phase))
    body_rect = pygame.Rect(0, 0, int(unit * 1.55), int(unit * 1.0 / squash))
    body_rect.center = (cx, cy)
    pygame.draw.ellipse(glow, (*grammar.secondary, 78), body_rect.inflate(unit, unit // 2), 2)
    pygame.draw.ellipse(screen, _dim(grammar.secondary, 0.75), body_rect)
    pygame.draw.ellipse(screen, grammar.primary, body_rect, max(2, unit // 7))
    eye = (cx + facing * unit // 3, cy - unit // 6)
    pygame.draw.circle(screen, (4, 12, 14), eye, max(2, unit // 7))
    for side in (-1, 1):
        base = (cx + side * unit // 3, cy - unit // 2)
        tip = (base[0] + side * int(math.sin(phase + side) * unit * 0.18), base[1] - int(unit * 0.5))
        pygame.draw.line(screen, grammar.secondary, base, tip, max(1, unit // 10))
        pygame.draw.circle(screen, grammar.hot, tip, max(1, unit // 8))
    if selected:
        pygame.draw.circle(glow, (*SELECTED, 96), center, int(unit * 1.55), 2)


def _draw_orb_avatar(
    screen: pygame.Surface,
    glow: pygame.Surface,
    center: tuple[int, int],
    unit: int,
    phase: float,
    grammar: VisualGrammar,
    selected: bool,
) -> None:
    radius = int(unit * 0.78)
    pygame.draw.circle(glow, (*grammar.primary, 96), center, int(radius * 1.45), 3)
    pygame.draw.circle(screen, (8, 17, 22), center, radius)
    pygame.draw.circle(screen, grammar.primary, center, radius, max(2, unit // 6))
    pygame.draw.circle(screen, _brighten(grammar.secondary), center, max(2, radius // 3))
    for index in range(3):
        angle = phase * (0.8 + index * 0.25) + index * 2.0
        dot = (center[0] + int(math.cos(angle) * radius * 1.15), center[1] + int(math.sin(angle) * radius * 0.65))
        pygame.draw.circle(glow, (*grammar.secondary, 90), dot, max(2, unit // 8))
    if selected:
        pygame.draw.circle(glow, (*SELECTED, 96), center, int(unit * 1.55), 2)


def _agent_body_radius(record: Any) -> float:
    radii: list[float] = []
    for shape in getattr(record, "shapes", ()) or ():
        if isinstance(shape, pymunk.Circle):
            radii.append(float(shape.radius))
        elif isinstance(shape, pymunk.Poly):
            bb = shape.bb
            radii.append(max(float(bb.right - bb.left), float(bb.top - bb.bottom)) * 0.42)
    return max(radii or [15.0])


def _draw_round_limb(
    surface: pygame.Surface,
    color: tuple[int, ...],
    start: tuple[int, int],
    end: tuple[int, int],
    width: int,
) -> None:
    pygame.draw.line(surface, color, start, end, width)
    radius = max(2, width // 2)
    pygame.draw.circle(surface, color, start, radius)
    pygame.draw.circle(surface, color, end, radius)


def _avatar_visible_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    brightness = sum(color) / 3.0
    if brightness >= 150:
        return color
    boosted = color
    while sum(boosted) / 3.0 < 150:
        boosted = _brighten(boosted)
        if boosted == (255, 255, 255):
            break
    return boosted


def _agent_skeleton_points(
    center: tuple[int, int],
    unit: int,
    facing: int,
    phase: float,
    pose: str,
) -> dict[str, tuple[int, int]]:
    cx, cy = center
    sway = math.sin(phase)
    bob = int(math.sin(phase * 1.7) * unit * 0.08)
    hip = (cx - int(facing * unit * 0.06), cy - int(unit * 0.12) + bob)
    neck = (cx + int(facing * unit * 0.06), cy - int(unit * 1.18) + bob)
    head = (cx + int(facing * unit * 0.14), cy - int(unit * 1.72) + bob)

    if pose == "throw":
        windup = math.sin(phase * 3.0)
        lean = int(facing * unit * 0.18)
        neck = (neck[0] + lean, neck[1] - int(unit * 0.05))
        head = (head[0] + lean, head[1] - int(unit * 0.05))
        shoulder = (neck[0], neck[1] + int(unit * 0.10))
        hand_front = (neck[0] + int(facing * unit * (1.10 + 0.12 * windup)), neck[1] - int(unit * 0.42))
        hand_back = (neck[0] - int(facing * unit * 0.60), neck[1] + int(unit * 0.42))
        knee_front = (hip[0] + int(facing * unit * 0.44), hip[1] + int(unit * 0.62))
        foot_front = (hip[0] + int(facing * unit * 0.78), hip[1] + int(unit * 1.12))
        knee_back = (hip[0] - int(facing * unit * 0.42), hip[1] + int(unit * 0.62))
        foot_back = (hip[0] - int(facing * unit * 0.74), hip[1] + int(unit * 1.12))
    elif pose == "swim":
        stroke = math.sin(phase * 3.4)
        hip = (cx - int(facing * unit * 0.10), cy + int(unit * 0.18) + bob)
        neck = (cx + int(facing * unit * 0.22), cy - int(unit * 0.30) + bob)
        head = (cx + int(facing * unit * 0.58), cy - int(unit * 0.52) + bob)
        shoulder = (neck[0] - int(facing * unit * 0.10), neck[1] + int(unit * 0.14))
        hand_front = (shoulder[0] + int(facing * unit * (0.95 + 0.25 * stroke)), shoulder[1] + int(unit * (0.08 - 0.20 * abs(stroke))))
        hand_back = (shoulder[0] - int(facing * unit * (0.72 - 0.15 * stroke)), shoulder[1] + int(unit * (0.18 + 0.18 * abs(stroke))))
        knee_front = (hip[0] + int(facing * unit * 0.42), hip[1] + int(unit * (0.22 + 0.16 * stroke)))
        foot_front = (hip[0] + int(facing * unit * 0.95), hip[1] + int(unit * (0.28 - 0.18 * stroke)))
        knee_back = (hip[0] - int(facing * unit * 0.38), hip[1] + int(unit * (0.20 - 0.14 * stroke)))
        foot_back = (hip[0] - int(facing * unit * 0.88), hip[1] + int(unit * (0.30 + 0.16 * stroke)))
    elif pose == "kick":
        windup = math.sin(phase * 4.2)
        lean = int(facing * unit * 0.24)
        hip = (hip[0] - int(facing * unit * 0.16), hip[1] + int(unit * 0.04))
        neck = (neck[0] + lean, neck[1] + int(unit * 0.02))
        head = (head[0] + lean, head[1] + int(unit * 0.02))
        shoulder = (neck[0], neck[1] + int(unit * 0.12))
        hand_front = (neck[0] - int(facing * unit * 0.22), neck[1] + int(unit * 0.48))
        hand_back = (neck[0] - int(facing * unit * 0.72), neck[1] + int(unit * 0.36))
        knee_front = (hip[0] + int(facing * unit * 0.68), hip[1] + int(unit * (0.50 - 0.08 * windup)))
        foot_front = (hip[0] + int(facing * unit * 1.58), hip[1] + int(unit * (0.62 - 0.10 * abs(windup))))
        knee_back = (hip[0] - int(facing * unit * 0.42), hip[1] + int(unit * 0.68))
        foot_back = (hip[0] - int(facing * unit * 0.94), hip[1] + int(unit * 1.18))
    elif pose == "push":
        lean = int(facing * unit * 0.38)
        hip = (hip[0] - int(facing * unit * 0.24), hip[1] + int(unit * 0.16))
        neck = (neck[0] + lean, neck[1] + int(unit * 0.18))
        head = (head[0] + lean, head[1] + int(unit * 0.16))
        shoulder = (neck[0], neck[1] + int(unit * 0.12))
        hand_front = (neck[0] + int(facing * unit * 1.28), neck[1] + int(unit * 0.20))
        hand_back = (neck[0] + int(facing * unit * 1.05), neck[1] + int(unit * 0.42))
        knee_front = (hip[0] + int(facing * unit * 0.25), hip[1] + int(unit * 0.72))
        foot_front = (hip[0] + int(facing * unit * 0.55), hip[1] + int(unit * 1.22))
        knee_back = (hip[0] - int(facing * unit * 0.62), hip[1] + int(unit * 0.72))
        foot_back = (hip[0] - int(facing * unit * 1.05), hip[1] + int(unit * 1.1))
    elif pose == "float":
        shoulder = (neck[0], neck[1] + int(unit * 0.08))
        hand_front = (neck[0] + int(facing * unit * (0.65 + 0.18 * sway)), neck[1] - int(unit * (0.18 + 0.08 * sway)))
        hand_back = (neck[0] - int(facing * unit * (0.56 - 0.12 * sway)), neck[1] + int(unit * (0.35 + 0.1 * sway)))
        knee_front = (hip[0] + int(facing * unit * 0.36), hip[1] + int(unit * (0.55 + 0.08 * sway)))
        foot_front = (hip[0] + int(facing * unit * 0.8), hip[1] + int(unit * (0.78 - 0.12 * sway)))
        knee_back = (hip[0] - int(facing * unit * 0.35), hip[1] + int(unit * (0.56 - 0.1 * sway)))
        foot_back = (hip[0] - int(facing * unit * 0.78), hip[1] + int(unit * (0.9 + 0.1 * sway)))
    elif pose == "fall":
        shoulder = (neck[0], neck[1] + int(unit * 0.1))
        hand_front = (neck[0] + int(facing * unit * 0.65), neck[1] - int(unit * 0.38))
        hand_back = (neck[0] - int(facing * unit * 0.62), neck[1] - int(unit * 0.24))
        knee_front = (hip[0] + int(facing * unit * 0.36), hip[1] + int(unit * 0.52))
        foot_front = (hip[0] + int(facing * unit * 0.44), hip[1] + int(unit * 1.08))
        knee_back = (hip[0] - int(facing * unit * 0.4), hip[1] + int(unit * 0.42))
        foot_back = (hip[0] - int(facing * unit * 0.78), hip[1] + int(unit * 0.86))
    elif pose == "jump":
        tuck = math.sin(phase * 2.8)
        shoulder = (neck[0], neck[1] + int(unit * 0.08))
        hand_front = (neck[0] + int(facing * unit * 0.56), neck[1] - int(unit * (0.40 + 0.08 * tuck)))
        hand_back = (neck[0] - int(facing * unit * 0.50), neck[1] - int(unit * 0.28))
        knee_front = (hip[0] + int(facing * unit * 0.46), hip[1] + int(unit * 0.44))
        foot_front = (hip[0] + int(facing * unit * 0.88), hip[1] + int(unit * 0.82))
        knee_back = (hip[0] - int(facing * unit * 0.42), hip[1] + int(unit * 0.46))
        foot_back = (hip[0] - int(facing * unit * 0.84), hip[1] + int(unit * 0.76))
    elif pose == "run":
        stride = math.sin(phase * 2.3)
        shoulder = (neck[0], neck[1] + int(unit * 0.08))
        hand_front = (neck[0] - int(facing * unit * 0.52 * stride), neck[1] + int(unit * (0.42 + 0.08 * abs(stride))))
        hand_back = (neck[0] + int(facing * unit * 0.52 * stride), neck[1] + int(unit * (0.42 - 0.08 * abs(stride))))
        knee_front = (hip[0] + int(facing * unit * 0.46 * stride), hip[1] + int(unit * 0.62))
        foot_front = (hip[0] + int(facing * unit * (0.92 * stride + 0.08)), hip[1] + int(unit * 1.16))
        knee_back = (hip[0] - int(facing * unit * 0.46 * stride), hip[1] + int(unit * 0.62))
        foot_back = (hip[0] - int(facing * unit * (0.92 * stride - 0.08)), hip[1] + int(unit * 1.16))
    else:
        shoulder = (neck[0], neck[1] + int(unit * 0.1))
        hand_front = (neck[0] + int(facing * unit * (0.42 + 0.05 * sway)), neck[1] + int(unit * 0.58))
        hand_back = (neck[0] - int(facing * unit * (0.36 - 0.04 * sway)), neck[1] + int(unit * 0.62))
        knee_front = (hip[0] + int(facing * unit * 0.28), hip[1] + int(unit * 0.7))
        foot_front = (hip[0] + int(facing * unit * 0.4), hip[1] + int(unit * 1.18))
        knee_back = (hip[0] - int(facing * unit * 0.28), hip[1] + int(unit * 0.7))
        foot_back = (hip[0] - int(facing * unit * 0.4), hip[1] + int(unit * 1.18))

    return {
        "head": head,
        "neck": neck,
        "shoulder": shoulder,
        "hip": hip,
        "hand_front": hand_front,
        "hand_back": hand_back,
        "knee_front": knee_front,
        "knee_back": knee_back,
        "foot_front": foot_front,
        "foot_back": foot_back,
    }


def _agent_limb_pairs() -> tuple[tuple[str, str], ...]:
    return (
        ("neck", "hip"),
        ("shoulder", "hand_front"),
        ("shoulder", "hand_back"),
        ("hip", "knee_front"),
        ("knee_front", "foot_front"),
        ("hip", "knee_back"),
        ("knee_back", "foot_back"),
    )


def _nearest_interactable(env: BaseEnv, agent_record: Any) -> Any | None:
    best = None
    best_distance = float("inf")
    for record in env._objects.values():
        if record is agent_record or str(record.role or "").lower() in {"terrain", "goal", "trigger", "region"}:
            continue
        if record.body.body_type != pymunk.Body.DYNAMIC:
            continue
        distance = float(agent_record.body.position.get_distance(record.body.position))
        if distance < best_distance:
            best = record
            best_distance = distance
    if best_distance <= max(80.0, _agent_body_radius(agent_record) * 5.0):
        return best
    return None


def _infer_agent_action(env: BaseEnv, agent_record: Any) -> AgentAction:
    body = agent_record.body
    velocity = getattr(body, "velocity", pymunk.Vec2d(0, 0))
    speed = float(velocity.length)
    nearest = _nearest_interactable(env, agent_record)
    facing = _agent_facing(body, nearest)
    gravity = getattr(env.space, "gravity", pymunk.Vec2d(0, -981)).length
    zero_g = _env_uses_freeflight_controls(env, gravity)
    throwing = _agent_is_throwing(env, agent_record, nearest)
    kicking = _agent_is_kicking(env, agent_record, nearest)
    pushing = _agent_is_pushing(agent_record, nearest) and not kicking and not throwing
    jumping = not zero_g and float(velocity.y) > 95.0
    falling = not zero_g and float(velocity.y) < -95.0 and speed > 110.0
    running = abs(float(velocity.x)) > 45.0 and not pushing and not kicking and not throwing
    swimming = _agent_is_in_water(env, agent_record)

    if throwing:
        pose = "throw"
    elif kicking:
        pose = "kick"
    elif pushing:
        pose = "push"
    elif swimming:
        pose = "swim"
    elif zero_g:
        pose = "float"
    elif jumping:
        pose = "jump"
    elif falling:
        pose = "fall"
    elif running:
        pose = "run"
    else:
        pose = "idle"
    intensity = min(1.0, speed / 420.0)
    return AgentAction(pose=pose, nearest=nearest, facing=facing, intensity=intensity)


def _agent_is_in_water(env: BaseEnv, agent_record: Any) -> bool:
    agent_pos = agent_record.body.position
    for record in env._objects.values():
        role = str(record.role or "").lower()
        metadata = getattr(record, "metadata", None)
        metadata_text = json.dumps(metadata, default=str).lower() if isinstance(metadata, dict) else ""
        if role != "water" and "swim_zone" not in metadata_text and "water" not in metadata_text:
            continue
        for shape in getattr(record, "shapes", ()) or ():
            if not isinstance(shape, pymunk.Poly):
                continue
            bb = shape.bb
            if bb.left <= agent_pos.x <= bb.right and bb.bottom <= agent_pos.y <= bb.top + 24:
                return True
    return False


def _agent_is_pushing(agent_record: Any, target_record: Any | None) -> bool:
    if target_record is None:
        return False
    offset = target_record.body.position - agent_record.body.position
    if offset.length <= 1.0:
        return False
    velocity = agent_record.body.velocity
    target_velocity = target_record.body.velocity
    toward = velocity.dot(offset.normalized())
    object_motion = target_velocity.dot(offset.normalized())
    return toward > 18.0 or object_motion > 10.0 or offset.length < _agent_body_radius(agent_record) * 3.2


def _env_uses_freeflight_controls(env: BaseEnv, gravity_length: float | None = None) -> bool:
    """Mirror BaseEnv's capability governor for pose/control presentation."""

    if gravity_length is None:
        gravity_length = getattr(env.space, "gravity", pymunk.Vec2d(0, -981)).length
    if gravity_length < 30.0:
        return True
    objective = env.get_ground_truth().get("objective", {})
    profile = objective.get("capability_profile") if isinstance(objective, dict) else {}
    if not isinstance(profile, dict):
        profile = {}
    text = " ".join(
        str(profile.get(key, "")).lower()
        for key in ("movement", "gravity", "notes")
    )
    return any(
        token in text
        for token in (
            "thrust",
            "freeflight",
            "free_flight",
            "zero_g",
            "zero-g",
            "flying",
            "flight",
            "spaceship",
            "hover",
        )
    )


def _agent_is_throwing(env: BaseEnv, agent_record: Any, target_record: Any | None) -> bool:
    if target_record is None:
        return False
    text = _action_text(env, target_record)
    if not any(token in text for token in ("throw", "thrown", "toss", "hurl", "lob", "launch projectile", "projectile")):
        return False
    offset = target_record.body.position - agent_record.body.position
    if offset.length <= 1.0 or offset.length > max(90.0, _agent_body_radius(agent_record) * 5.5):
        return False
    axis = offset.normalized()
    toward = float(agent_record.body.velocity.dot(axis))
    upward = float(agent_record.body.velocity.y)
    object_motion = float(target_record.body.velocity.length)
    return toward > 8.0 or upward > 25.0 or object_motion > 8.0 or offset.length < _agent_body_radius(agent_record) * 3.8


def _agent_is_kicking(env: BaseEnv, agent_record: Any, target_record: Any | None) -> bool:
    if target_record is None:
        return False
    text = _action_text(env, target_record)
    if not any(token in text for token in ("ball", "soccer", "puck", "kick", "strike", "shot", "shoot")):
        return False
    offset = target_record.body.position - agent_record.body.position
    if offset.length <= 1.0:
        return False
    radius = _agent_body_radius(agent_record)
    if offset.length > max(95.0, radius * 6.5):
        return False
    toward = agent_record.body.velocity.dot(offset.normalized())
    object_motion = target_record.body.velocity.dot(offset.normalized())
    return toward > 8.0 or object_motion > 6.0 or offset.length < radius * 4.2


def _action_text(env: BaseEnv, record: Any | None = None) -> str:
    pieces = [_objective_text(env)]
    if record is not None:
        pieces.extend(
            [
                str(getattr(record, "name", "") or ""),
                str(getattr(record, "role", "") or ""),
                json.dumps(getattr(record, "metadata", {}) or {}, default=str),
            ]
        )
    return " ".join(pieces).lower()


def _axis_to_nearest_goal(env: BaseEnv, source_record: Any) -> pymunk.Vec2d | None:
    best_axis = None
    best_distance = float("inf")
    for record in env._objects.values():
        role = str(record.role or "").lower()
        if role not in {"goal", "trigger", "region", "force_zone"}:
            continue
        offset = record.body.position - source_record.body.position
        if offset.length <= 1.0:
            continue
        if float(offset.length) < best_distance:
            best_distance = float(offset.length)
            best_axis = offset.normalized()
    return best_axis


def _agent_facing(body: pymunk.Body, nearest: Any | None) -> int:
    if nearest is not None:
        dx = float(nearest.body.position.x - body.position.x)
        if abs(dx) > 1.0:
            return 1 if dx >= 0 else -1
    if abs(float(body.velocity.x)) > 8.0:
        return 1 if body.velocity.x >= 0 else -1
    return 1


def _objective_text(env: BaseEnv) -> str:
    try:
        objective = env.get_ground_truth().get("objective", {})
    except Exception:
        return ""
    return json.dumps(objective, default=str).lower() if isinstance(objective, dict) else ""


def _recipe_skin_for_record(record: Any, grammar: VisualGrammar) -> dict[str, Any]:
    recipe = grammar.visual_recipe
    skins = recipe.get("object_skins") if isinstance(recipe, dict) else None
    if not isinstance(skins, dict):
        return {}
    name = str(getattr(record, "name", "") or "").lower()
    role = str(getattr(record, "role", "") or "").lower()
    metadata = getattr(record, "metadata", None)
    text = " ".join([name, role, json.dumps(metadata, sort_keys=True, default=str).lower()])
    for pattern, skin in skins.items():
        if not isinstance(skin, dict):
            continue
        pattern_text = str(pattern).lower()
        if pattern_text == name or pattern_text == role:
            return skin
        if pattern_text.startswith("*") and pattern_text.endswith("*") and pattern_text.strip("*") in text:
            return skin
        if pattern_text in text:
            return skin
    return {}


def _draw_shape(
    surface: pygame.Surface,
    camera: Camera,
    shape: pymunk.Shape,
    color: tuple[int, ...],
    *,
    outline: bool,
    width: int = 0,
) -> None:
    if isinstance(shape, pymunk.Circle):
        center = camera.point(shape.body.local_to_world(shape.offset))
        radius = camera.length(shape.radius)
        pygame.draw.circle(surface, color, center, radius, width if outline else 0)
    elif isinstance(shape, pymunk.Segment):
        start = camera.point(shape.body.local_to_world(shape.a))
        end = camera.point(shape.body.local_to_world(shape.b))
        line_width = width if outline else max(2, camera.length(shape.radius * 2.0))
        pygame.draw.line(surface, color, start, end, line_width)
        if not outline and shape.radius > 0:
            pygame.draw.circle(surface, color, start, camera.length(shape.radius))
            pygame.draw.circle(surface, color, end, camera.length(shape.radius))
    elif isinstance(shape, pymunk.Poly):
        points = [camera.point(shape.body.local_to_world(vertex)) for vertex in shape.get_vertices()]
        if len(points) >= 3:
            pygame.draw.polygon(surface, color, points, width if outline else 0)


def _draw_velocity_vectors(screen: pygame.Surface, camera: Camera, env: BaseEnv, grammar: VisualGrammar) -> None:
    for record in env._objects.values():
        body = record.body
        if body.body_type == pymunk.Body.STATIC or body.velocity.length < 8.0:
            continue
        start = camera.point(body.position)
        scale = min(MAX_VECTOR_LENGTH, body.velocity.length * 0.08)
        direction = body.velocity.normalized() * scale
        end = camera.point(body.position + direction)
        pygame.draw.line(screen, grammar.secondary, start, end, 2)
        _draw_vector_head(screen, start, end, grammar.secondary)


def _draw_vector_head(
    screen: pygame.Surface,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    size = 8
    points = [
        end,
        (
            int(end[0] - math.cos(angle - 0.45) * size),
            int(end[1] - math.sin(angle - 0.45) * size),
        ),
        (
            int(end[0] - math.cos(angle + 0.45) * size),
            int(end[1] - math.sin(angle + 0.45) * size),
        ),
    ]
    pygame.draw.polygon(screen, color, points)


def _draw_path(
    screen: pygame.Surface,
    camera: Camera,
    path_points: tuple[tuple[float, float], ...],
    grammar: VisualGrammar,
) -> None:
    if len(path_points) < 2:
        return
    projected = [camera.point(point) for point in path_points]
    for index in range(0, len(projected) - 1, 2):
        pygame.draw.line(screen, grammar.primary, projected[index], projected[index + 1], 2)
    for point in projected[::3]:
        pygame.draw.circle(screen, grammar.primary, point, 3, 1)


def _draw_hud(
    screen: pygame.Surface,
    env: BaseEnv,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    path_points: tuple[tuple[float, float], ...],
    selected_name: str | None,
    validation_summary: dict[str, Any],
    grammar: VisualGrammar,
) -> None:
    ground_truth = env.get_ground_truth()
    hud_data = _hud_data(
        env,
        ground_truth,
        path_points,
        selected_name,
        validation_summary,
        grammar,
    )
    lines = json.dumps(hud_data, indent=2).splitlines()
    panel_width = 540
    panel_height = min(screen.get_height() - 72, 374)
    panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel.fill((5, 18, 20, 208))
    pygame.draw.rect(panel, (61, 220, 230, 125), panel.get_rect(), 1)
    screen.blit(panel, (18, 18))

    title = font.render("HARNESS ALPHA // CODE-LEVEL TRUTH", True, TEXT)
    screen.blit(title, (34, 31))
    max_lines = max(1, (panel_height - 48) // 14)
    for index, line in enumerate(lines[:max_lines]):
        color = TEXT if index < 2 else MUTED_TEXT
        rendered = small_font.render(line, True, color)
        screen.blit(rendered, (34, 58 + index * 14))


def _draw_footer(
    screen: pygame.Surface,
    font: pygame.font.Font,
    overlay_enabled: bool,
    selected_name: str | None,
    autoplay_state: _AutoplayState | None = None,
) -> None:
    state = "ON" if overlay_enabled else "OFF"
    selected = selected_name or "none"
    if autoplay_state is not None:
        line = f"{autoplay_state.label}  |  R reset  |  SPACE telemetry {state}  |  ESC quit"
    else:
        line = (
            f"TAB select body ({selected})  |  A/D move, W jumps only when grounded  |  SHIFT boost  |  "
            f"I/J/K/; impulse  |  L lift  |  R reset  |  SPACE telemetry {state}  |  ESC quit"
        )
    rendered = font.render(line, True, MUTED_TEXT)
    screen.blit(rendered, (18, screen.get_height() - 28))


def _solve_menu_button_rects(screen: pygame.Surface) -> tuple[pygame.Rect, pygame.Rect]:
    panel_width = min(560, screen.get_width() - 96)
    button_width = 190
    button_height = 54
    gap = 28
    y = screen.get_height() // 2 + 62
    total_width = button_width * 2 + gap
    start_x = screen.get_width() // 2 - total_width // 2
    replay_rect = pygame.Rect(start_x, y, button_width, button_height)
    quit_rect = pygame.Rect(start_x + button_width + gap, y, button_width, button_height)
    if panel_width < total_width:
        replay_rect.width = max(140, (panel_width - gap) // 2)
        quit_rect.width = replay_rect.width
        replay_rect.x = screen.get_width() // 2 - (replay_rect.width * 2 + gap) // 2
        quit_rect.x = replay_rect.right + gap
    return replay_rect, quit_rect


def _draw_autoplay_success_layer(
    screen: pygame.Surface,
    success_font: pygame.font.Font,
    success_small_font: pygame.font.Font,
    button_font: pygame.font.Font,
    autoplay_state: _AutoplayState,
    grammar: VisualGrammar,
    tick: int,
) -> None:
    if autoplay_state.solved_at_ms is None:
        return

    elapsed = tick - autoplay_state.solved_at_ms
    if autoplay_state.success_visible:
        _draw_success_burst(screen, success_font, success_small_font, grammar, tick, elapsed)
    elif autoplay_state.menu_visible:
        _draw_solve_menu_overlay(screen, success_small_font, button_font, grammar, tick)


def _draw_autoplay_countdown(
    screen: pygame.Surface,
    font: pygame.font.Font,
    env: BaseEnv,
    autoplay_state: _AutoplayState,
    grammar: VisualGrammar,
) -> None:
    remaining = autoplay_state.countdown_seconds(env)
    if remaining is None:
        return
    label = "SURVIVE"
    value = f"{remaining:04.1f}s"
    panel = pygame.Surface((242, 82), pygame.SRCALPHA)
    panel.fill((4, 10, 18, 190))
    pygame.draw.rect(panel, (*grammar.hot, 160), panel.get_rect(), 2, border_radius=8)
    pygame.draw.rect(panel, (*grammar.primary, 76), panel.get_rect().inflate(-8, -8), 1, border_radius=6)
    label_surface = font.render(label, True, grammar.primary)
    value_surface = pygame.font.SysFont("consolas", 38, bold=True).render(value, True, grammar.hot)
    panel.blit(label_surface, (18, 12))
    panel.blit(value_surface, (18, 36))
    x = screen.get_width() - panel.get_width() - 24
    y = 24
    screen.blit(panel, (x, y))


def _draw_success_burst(
    screen: pygame.Surface,
    success_font: pygame.font.Font,
    success_small_font: pygame.font.Font,
    grammar: VisualGrammar,
    tick: int,
    elapsed: int,
) -> None:
    width, height = screen.get_size()
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    alpha = 70 + int(24 * math.sin(tick * 0.012))
    overlay.fill((0, 0, 0, 58))

    center = (width // 2, height // 2)
    pulse = 0.5 + 0.5 * math.sin(tick * 0.018)
    radius = 120 + int(44 * pulse)
    pygame.draw.circle(overlay, (*grammar.primary, 28), center, radius + 80)
    pygame.draw.circle(overlay, (*SUCCESS, 42), center, radius + 34, 6)
    pygame.draw.circle(overlay, (*SUCCESS_GOLD, 34), center, radius + 76, 2)

    for index in range(28):
        angle = tick * 0.0025 + index * math.tau / 28.0
        inner = radius * 0.62
        outer = radius + 82 + 12 * math.sin(tick * 0.01 + index)
        p1 = (int(center[0] + math.cos(angle) * inner), int(center[1] + math.sin(angle) * inner))
        p2 = (int(center[0] + math.cos(angle) * outer), int(center[1] + math.sin(angle) * outer))
        color = SUCCESS_GOLD if index % 3 == 0 else SUCCESS
        pygame.draw.line(overlay, (*color, alpha), p1, p2, 2)

    for index in range(40):
        orbit = 170 + (index % 5) * 28
        angle = tick * (0.0018 + (index % 4) * 0.0003) + index * 0.71
        x = int(center[0] + math.cos(angle) * orbit)
        y = int(center[1] + math.sin(angle * 1.13) * orbit * 0.46)
        color = grammar.secondary if index % 2 else SUCCESS_GOLD
        pygame.draw.rect(overlay, (*color, 120), (x, y, 7, 7), border_radius=2)

    screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_ADD)

    scale_phase = min(1.0, max(0.0, (elapsed - 1000) / 360.0))
    title = success_font.render("SUCCESS!", True, SUCCESS)
    title_glow = success_font.render("SUCCESS!", True, SUCCESS_GOLD)
    title_x = center[0] - title.get_width() // 2
    title_y = center[1] - title.get_height() // 2 - 24 + int((1.0 - scale_phase) * 24)
    for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3)):
        screen.blit(title_glow, (title_x + dx, title_y + dy), special_flags=pygame.BLEND_ADD)
    screen.blit(title, (title_x, title_y))

    subtitle = success_small_font.render("OBJECTIVE VERIFIED BY CODE-LEVEL TRUTH", True, TEXT)
    screen.blit(subtitle, (center[0] - subtitle.get_width() // 2, title_y + title.get_height() + 8))


def _draw_solve_menu_overlay(
    screen: pygame.Surface,
    title_font: pygame.font.Font,
    button_font: pygame.font.Font,
    grammar: VisualGrammar,
    tick: int,
) -> None:
    width, height = screen.get_size()
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    overlay.fill(OVERLAY_DARK)
    screen.blit(overlay, (0, 0))

    panel = pygame.Rect(0, 0, min(620, width - 96), 286)
    panel.center = (width // 2, height // 2)
    shadow = pygame.Surface(panel.size, pygame.SRCALPHA)
    shadow.fill((0, 0, 0, 120))
    screen.blit(shadow, (panel.x + 8, panel.y + 10))
    pygame.draw.rect(screen, (6, 18, 22), panel, border_radius=12)
    pygame.draw.rect(screen, grammar.primary, panel, 2, border_radius=12)
    pygame.draw.rect(screen, (*SUCCESS, 120), panel.inflate(-10, -10), 1, border_radius=10)

    pulse = 0.5 + 0.5 * math.sin(tick * 0.01)
    crown = pygame.Rect(panel.centerx - 128, panel.y + 28, 256, 6)
    pygame.draw.rect(screen, (*SUCCESS_GOLD, int(105 + 80 * pulse)), crown, border_radius=3)

    title = title_font.render("OBJECTIVE COMPLETE", True, SUCCESS)
    subtitle = title_font.render("Replay the verified solve or close this run.", True, TEXT)
    screen.blit(title, (panel.centerx - title.get_width() // 2, panel.y + 58))
    screen.blit(subtitle, (panel.centerx - subtitle.get_width() // 2, panel.y + 96))

    replay_rect, quit_rect = _solve_menu_button_rects(screen)
    _draw_solve_menu_button(screen, replay_rect, "REPLAY", SUCCESS, button_font, tick)
    _draw_solve_menu_button(screen, quit_rect, "QUIT", OBSTACLE, button_font, tick)


def _draw_solve_menu_button(
    screen: pygame.Surface,
    rect: pygame.Rect,
    label: str,
    color: tuple[int, int, int],
    font: pygame.font.Font,
    tick: int,
) -> None:
    mouse_pos = pygame.mouse.get_pos()
    hovered = rect.collidepoint(mouse_pos)
    fill = (8, 28, 29) if hovered else (7, 17, 21)
    glow_alpha = 120 if hovered else 62 + int(24 * math.sin(tick * 0.008))
    glow = pygame.Surface(rect.inflate(20, 20).size, pygame.SRCALPHA)
    pygame.draw.rect(glow, (*color, glow_alpha), glow.get_rect(), border_radius=12)
    screen.blit(glow, (rect.x - 10, rect.y - 10), special_flags=pygame.BLEND_ADD)
    pygame.draw.rect(screen, fill, rect, border_radius=9)
    pygame.draw.rect(screen, color, rect, 2, border_radius=9)
    rendered = font.render(label, True, color)
    screen.blit(
        rendered,
        (
            rect.centerx - rendered.get_width() // 2,
            rect.centery - rendered.get_height() // 2,
        ),
    )


def _hud_data(
    env: BaseEnv,
    ground_truth: dict[str, Any],
    path_points: tuple[tuple[float, float], ...],
    selected_name: str | None,
    validation_summary: dict[str, Any],
    grammar: VisualGrammar,
) -> dict[str, Any]:
    agent = _object_by_role(ground_truth, "agent")
    goal = _object_by_role(ground_truth, "goal")
    agent_pos = agent.get("body", {}).get("position", [None, None]) if agent else [None, None]
    goal_pos = goal.get("body", {}).get("position", [None, None]) if goal else [None, None]
    velocity = agent.get("body", {}).get("velocity", [0.0, 0.0]) if agent else [0.0, 0.0]
    distance = None
    if None not in agent_pos and None not in goal_pos:
        distance = round(math.dist(agent_pos, goal_pos), 3)

    objective = ground_truth.get("objective") or {}
    objective_active = bool(objective.get("objective_active")) or callable(
        getattr(env, "check_objective", None)
    )
    objective_satisfied = bool(objective.get("objective_satisfied"))
    objective_profile = _as_dict(objective.get("objective_profile"))
    capability_profile = _as_dict(objective.get("capability_profile"))
    semantic_requirements = objective.get("semantic_requirements")
    semantic_count = len(semantic_requirements) if isinstance(semantic_requirements, list) else 0
    verification = {
        "accepted": validation_summary.get("accepted"),
        "tier": (
            f"{validation_summary.get('achieved_tier')}/"
            f"{validation_summary.get('minimum_acceptance_tier')}"
        ),
        "tier_name": validation_summary.get("tier_name"),
        "reason": _short_text(str(validation_summary.get("reason") or ""), 72),
    }
    capability = {
        "movement": capability_profile.get("movement"),
        "gravity": capability_profile.get("gravity"),
        "interaction": capability_profile.get("interaction"),
        "allowed_controls": capability_profile.get("allowed_controls"),
    }

    return {
        "env": ground_truth.get("env"),
        "OBJECTIVE": "ACTIVE" if objective_active else "INACTIVE",
        "objective_type": objective.get("objective_type"),
        "objective_description": _short_text(
            str(objective_profile.get("objective_description") or ""),
            72,
        ),
        "objective_satisfied": objective_satisfied,
        "objective_targets": objective.get("objective_targets"),
        "semantic_requirements": semantic_count,
        "verification": verification,
        "visual_grammar": {
            "mood": grammar.mood,
            "background": grammar.background,
            "accent": grammar.accent,
            "source": grammar.source,
        },
        "capability_profile": capability,
        "progress_metrics": objective_profile.get("progress_metrics"),
        "step": ground_truth.get("step_count"),
        "agent_xy": [round(float(agent_pos[0]), 3), round(float(agent_pos[1]), 3)]
        if None not in agent_pos
        else None,
        "goal_distance": distance,
        "linear_velocity": [round(float(velocity[0]), 3), round(float(velocity[1]), 3)],
        "bfs_path_points": len(path_points),
        "selected_body": selected_name,
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _short_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _record_by_role(env: BaseEnv, role: str):
    for record in env._objects.values():
        if record.role == role:
            return record
    return None


def _dynamic_names(env: BaseEnv) -> list[str]:
    names = [
        name
        for name, record in env._objects.items()
        if record.body.body_type == pymunk.Body.DYNAMIC
    ]
    return names


def _initial_selected_name(env: BaseEnv) -> str | None:
    agent = _record_by_role(env, "agent")
    if agent is not None and agent.body.body_type == pymunk.Body.DYNAMIC:
        return agent.name
    names = _dynamic_names(env)
    return names[0] if names else None


def _next_dynamic_name(env: BaseEnv, selected_name: str | None) -> str | None:
    names = _dynamic_names(env)
    if not names:
        return None
    if selected_name not in names:
        return names[0]
    index = names.index(selected_name)
    return names[(index + 1) % len(names)]


def _object_by_role(ground_truth: dict[str, Any], role: str) -> dict[str, Any] | None:
    for data in ground_truth.get("objects", {}).values():
        if data.get("role") == role:
            return data
    return None


def _color_for_role(role: str | None) -> tuple[int, int, int]:
    if role == "agent":
        return AGENT
    if role == "goal":
        return GOAL
    if role == "obstacle":
        return OBSTACLE
    if role == "hazard":
        return HAZARD
    return TERRAIN


def _brighten(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(min(255, channel + 54) for channel in color)


def _dim(color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    factor = max(0.0, min(1.0, amount))
    return tuple(max(0, min(255, int(channel * factor))) for channel in color)


def _mix_tuple(a: tuple[int, int, int], b: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    factor = max(0.0, min(1.0, amount))
    return tuple(max(0, min(255, int(a[index] * (1.0 - factor) + b[index] * factor))) for index in range(3))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Harness Alpha Pygame visualizer")
    parser.add_argument("env_path", type=Path, help="generated environment .py path")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--smoke-frames", type=int, help="render N frames and exit")
    parser.add_argument("--autoplay", action="store_true", help="let the validator policy drive Tier-5 worlds")
    parser.add_argument("--auto-close-after-solve", type=float, help="close autoplay window N seconds after solve")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_visualizer(
        args.env_path,
        width=args.width,
        height=args.height,
        smoke_frames=args.smoke_frames,
        autoplay=args.autoplay,
        auto_close_after_solve=args.auto_close_after_solve,
    )


if __name__ == "__main__":
    main()
