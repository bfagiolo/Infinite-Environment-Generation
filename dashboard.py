"""Pygame front-end for Harness Alpha's prompt-to-world loop."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import tkinter
from typing import Callable

import pygame
import pymunk

from base_env import BaseEnv
from validator import ValidatorConfig, _step_subgoal, _subgoal_satisfied, load_env_class
from visual_grammar import VisualGrammar, color_for_record, infer_visual_grammar


WIDTH = 1320
HEIGHT = 840
BACKGROUND = (14, 16, 18)
PANEL = (22, 27, 31)
PANEL_DARK = (16, 20, 24)
BORDER = (54, 78, 86)
GRID = (20, 68, 74)
TEXT = (219, 246, 248)
MUTED = (112, 145, 150)
CYAN = (72, 221, 236)
GREEN = (52, 232, 154)
RED = (255, 77, 109)
YELLOW = (255, 206, 86)
BLUE = (79, 170, 255)
PURPLE = (177, 130, 255)
BLACK_ALPHA = (4, 10, 14, 175)
MINI_PREVIEW_STEPS_PER_FRAME = 4
MAX_PROMPT_CHARS = 1200

DEFAULT_PROMPT = ""


def _normalize_prompt_match(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
    return re.sub(r"\s+", " ", normalized)


SAVED_DEMOS = [
    {
        "title": "Neon Tiny Maze",
        "prompt": "A tiny neon maze where the agent must reach a glowing exit.",
        "env_path": Path("saved_demos/tiny_maze/env.py"),
        "export_dir": Path("saved_demos/tiny_maze/export"),
        "tier": 5,
    },
    {
        "title": "Zero-G Crystals",
        "prompt": "A zero-gravity salvage field where the agent must touch four drifting crystals.",
        "env_path": Path("saved_demos/zero_gravity_crystals/env.py"),
        "export_dir": Path("saved_demos/zero_gravity_crystals/export"),
        "tier": 5,
    },
    {
        "title": "Weighted Seesaw",
        "prompt": "A weighted seesaw puzzle where a heavy ball must launch the agent to a high goal.",
        "env_path": Path("saved_demos/seesaw_launch/env.py"),
        "export_dir": Path("saved_demos/seesaw_launch/export"),
        "tier": 5,
    },
    {
        "title": "Basketball Hoop",
        "prompt": "A neon basketball court where the agent throws a basketball into a glowing hoop.",
        "env_path": Path("saved_demos/basketball_hoop/env.py"),
        "export_dir": Path("saved_demos/basketball_hoop/export"),
        "tier": 5,
    },
    {
        "title": "Forest Bear Chase",
        "prompt": "A forest clearing where the agent must escape to a cabin while a bear chases from the trees.",
        "env_path": Path("saved_demos/forest_bear_escape/env.py"),
        "export_dir": Path("saved_demos/forest_bear_escape/export"),
        "tier": 5,
    },
    {
        "title": "Water Swim",
        "prompt": "A tropical water channel where the agent jumps in, swims across, and climbs out to reach a beacon.",
        "env_path": Path("saved_demos/water_swim_beacon/env.py"),
        "export_dir": Path("saved_demos/water_swim_beacon/export"),
        "tier": 5,
    },
    {
        "title": "8s Space Battle",
        "prompt": "A space battle where enemy ships fire glowing shots while the agent survives for 8 seconds.",
        "env_path": Path("saved_demos/spaceship_survival/env.py"),
        "export_dir": Path("saved_demos/spaceship_survival/export"),
        "tier": 5,
    },
    {
        "title": "Lava Fireball Run",
        "prompt": "A lava world where the agent reaches the obsidian exit while staggered fireballs rain from above.",
        "env_path": Path("saved_demos/lava_fireball_escape/env.py"),
        "export_dir": Path("saved_demos/lava_fireball_escape/export"),
        "tier": 5,
    },
]
EXAMPLE_CARD_HEIGHT = 54
SAVED_DEMO_LOAD_EVENT = pygame.USEREVENT + 7

FAST_VARIANT_STYLES = [
    {
        "label": "READABLE",
        "mood": "training_sim",
        "background": "blueprint_grid",
        "accent": "cyan_green",
        "palette": {
            "background_color": [10, 19, 22],
            "primary": [72, 221, 236],
            "secondary": [52, 232, 154],
            "hot": [255, 206, 86],
        },
        "motion_fx": ["agent_trail", "goal_breathe", "scan_sweep"],
        "surface_fx": ["grid_crosshairs", "scanlines"],
        "note": "same solved world with cleaner training-sim readability",
    },
    {
        "label": "CINEMATIC",
        "mood": "cyber_grid",
        "background": "radar_sweep",
        "accent": "pink_cyan",
        "palette": {
            "background_color": [13, 10, 22],
            "primary": [255, 116, 215],
            "secondary": [87, 232, 255],
            "hot": [255, 200, 80],
        },
        "motion_fx": ["agent_trail", "particle_sparks", "impact_ripples"],
        "surface_fx": ["crt_vignette", "procedural_noise"],
        "note": "same solved world with higher-energy demo styling",
    },
    {
        "label": "ALT WORLD",
        "mood": "crystal_cavern",
        "background": "crystal_speckle",
        "accent": "violet_lime",
        "palette": {
            "background_color": [14, 13, 24],
            "primary": [170, 115, 255],
            "secondary": [178, 255, 86],
            "hot": [255, 86, 142],
        },
        "motion_fx": ["agent_trail", "crystal_shimmer", "goal_breathe"],
        "surface_fx": ["crystal_facets", "film_grain"],
        "note": "same solved world with alternate palette/material language",
    },
]

SPACE_FAST_VARIANT_STYLES = [
    {
        "label": "STARFIELD",
        "mood": "deep_space",
        "background": "dense_starfield",
        "accent": "deep_space_neon",
        "palette": {
            "background_color": [4, 6, 18],
            "primary": [82, 216, 255],
            "secondary": [190, 96, 255],
            "hot": [255, 76, 132],
        },
        "motion_fx": ["agent_trail", "object_trail", "zero_g_drift", "particle_sparks"],
        "surface_fx": ["crt_vignette", "procedural_noise"],
        "note": "same solved dogfight with bright starfield readability",
    },
    {
        "label": "NEBULA",
        "mood": "cosmic_research",
        "background": "dense_starfield",
        "accent": "cyan_purple",
        "palette": {
            "background_color": [7, 5, 22],
            "primary": [105, 232, 255],
            "secondary": [225, 96, 255],
            "hot": [255, 88, 82],
        },
        "motion_fx": ["agent_trail", "object_trail", "field_pulse", "particle_sparks"],
        "surface_fx": ["film_grain", "procedural_noise"],
        "note": "same solved dogfight with nebula glow and hotter laser contrast",
    },
    {
        "label": "ORBITAL",
        "mood": "orbital_station",
        "background": "orbital_dust",
        "accent": "blue_white",
        "palette": {
            "background_color": [5, 10, 18],
            "primary": [110, 190, 255],
            "secondary": [238, 248, 255],
            "hot": [255, 70, 118],
        },
        "motion_fx": ["agent_trail", "object_trail", "radar_ping", "scan_sweep"],
        "surface_fx": ["metal_panel_lines", "scanlines"],
        "note": "same solved dogfight with orbital HUD styling",
    },
]

LAVA_FAST_VARIANT_STYLES = [
    {
        "label": "INFERNO",
        "mood": "volcanic_foundry",
        "background": "ember_drift",
        "accent": "solar_orange",
        "palette": {
            "background_color": [24, 8, 5],
            "primary": [255, 154, 52],
            "secondary": [255, 221, 92],
            "hot": [255, 50, 18],
        },
        "motion_fx": ["agent_trail", "falling_embers", "hazard_flash", "goal_breathe"],
        "surface_fx": ["hazard_chevrons", "stone_facets", "film_grain"],
        "note": "same solved fireball run with volcanic inferno readability",
    },
    {
        "label": "OBSIDIAN",
        "mood": "industrial_noir",
        "background": "smoke_haze",
        "accent": "gold_crimson",
        "palette": {
            "background_color": [18, 10, 10],
            "primary": [255, 202, 86],
            "secondary": [255, 72, 109],
            "hot": [255, 84, 34],
        },
        "motion_fx": ["agent_trail", "dust_motes", "hazard_flash", "gate_unlock_flash"],
        "surface_fx": ["metal_panel_lines", "stone_facets"],
        "note": "same solved run with darker obsidian-and-smoke styling",
    },
    {
        "label": "MAGMA STORM",
        "mood": "hazard_arena",
        "background": "hazard_warning",
        "accent": "red_warning",
        "palette": {
            "background_color": [26, 9, 12],
            "primary": [255, 96, 64],
            "secondary": [255, 190, 74],
            "hot": [255, 32, 66],
        },
        "motion_fx": ["agent_trail", "falling_embers", "particle_sparks", "hazard_flash"],
        "surface_fx": ["warning_stripes", "hazard_chevrons"],
        "note": "same solved run with high-danger arcade hazard styling",
    },
]

FOREST_FAST_VARIANT_STYLES = [
    {
        "label": "CANOPY",
        "mood": "organic_forest",
        "background": "layered_canopy",
        "accent": "emerald_sunbeam",
        "palette": {
            "background_color": [7, 18, 12],
            "primary": [84, 210, 118],
            "secondary": [238, 214, 116],
            "hot": [255, 92, 92],
        },
        "motion_fx": ["agent_trail", "leaf_drift", "goal_breathe"],
        "surface_fx": ["organic_stripes", "film_grain"],
        "note": "same solved chase with brighter forest canopy readability",
    },
    {
        "label": "TWILIGHT",
        "mood": "twilight_woods",
        "background": "misty_treeline",
        "accent": "violet_amber",
        "palette": {
            "background_color": [12, 11, 22],
            "primary": [144, 118, 255],
            "secondary": [255, 190, 86],
            "hot": [255, 75, 118],
        },
        "motion_fx": ["agent_trail", "dust_motes", "danger_pulse"],
        "surface_fx": ["crt_vignette", "procedural_noise"],
        "note": "same solved chase with twilight contrast and warmer cabin glow",
    },
    {
        "label": "GROVE",
        "mood": "ancient_grove",
        "background": "moss_and_fireflies",
        "accent": "lime_cyan",
        "palette": {
            "background_color": [6, 20, 18],
            "primary": [155, 255, 112],
            "secondary": [90, 230, 255],
            "hot": [255, 106, 82],
        },
        "motion_fx": ["agent_trail", "particle_sparks", "scan_sweep"],
        "surface_fx": ["organic_stripes", "stone_facets"],
        "note": "same solved chase with firefly-like particles and mossy geometry",
    },
]

WATER_FAST_VARIANT_STYLES = [
    {
        "label": "LAGOON",
        "mood": "tropical_water",
        "background": "sunlit_lagoon",
        "accent": "aqua_coral",
        "palette": {
            "background_color": [3, 24, 26],
            "primary": [35, 238, 245],
            "secondary": [255, 168, 92],
            "hot": [255, 72, 138],
        },
        "motion_fx": ["agent_trail", "water_ripples", "goal_breathe"],
        "surface_fx": ["film_grain", "procedural_noise"],
        "note": "same solved swim with warm lagoon highlights",
    },
    {
        "label": "MOONPOOL",
        "mood": "moonlit_water",
        "background": "deep_blue_pool",
        "accent": "blue_white",
        "palette": {
            "background_color": [2, 5, 22],
            "primary": [70, 128, 255],
            "secondary": [238, 246, 255],
            "hot": [190, 90, 255],
        },
        "motion_fx": ["agent_trail", "field_pulse", "scan_sweep"],
        "surface_fx": ["crt_vignette", "scanlines"],
        "note": "same solved swim with cooler moonpool styling",
    },
    {
        "label": "STORMWATER",
        "mood": "stormwater_channel",
        "background": "rain_haze",
        "accent": "cyan_gold",
        "palette": {
            "background_color": [7, 12, 15],
            "primary": [82, 172, 190],
            "secondary": [255, 226, 72],
            "hot": [255, 86, 64],
        },
        "motion_fx": ["agent_trail", "object_trail", "particle_sparks"],
        "surface_fx": ["metal_panel_lines", "procedural_noise"],
        "note": "same solved swim with storm-channel drama",
    },
]

MAZE_FAST_VARIANT_STYLES = [
    {
        "label": "ARCADE",
        "mood": "retro_arcade",
        "background": "retro_vignette",
        "accent": "magenta_orange",
        "palette": {
            "background_color": [16, 7, 24],
            "primary": [255, 78, 205],
            "secondary": [255, 166, 64],
            "hot": [91, 235, 255],
        },
        "motion_fx": ["agent_trail", "goal_breathe", "particle_sparks"],
        "surface_fx": ["crt_vignette", "scanlines"],
        "note": "same solved route with arcade cabinet energy",
    },
    {
        "label": "HOLOGRID",
        "mood": "cyber_grid",
        "background": "radar_sweep",
        "accent": "cyan_purple",
        "palette": {
            "background_color": [4, 13, 18],
            "primary": [66, 238, 255],
            "secondary": [178, 104, 255],
            "hot": [255, 210, 86],
        },
        "motion_fx": ["agent_trail", "scan_sweep", "radar_ping"],
        "surface_fx": ["grid_crosshairs", "procedural_noise"],
        "note": "same solved route with holographic maze traces",
    },
    {
        "label": "CRYSTAL",
        "mood": "crystal_cavern",
        "background": "crystal_speckle",
        "accent": "violet_lime",
        "palette": {
            "background_color": [10, 8, 24],
            "primary": [178, 118, 255],
            "secondary": [170, 255, 90],
            "hot": [255, 92, 150],
        },
        "motion_fx": ["agent_trail", "crystal_shimmer", "goal_breathe"],
        "surface_fx": ["crystal_facets", "film_grain"],
        "note": "same solved route with cavern/crystal styling",
    },
]

ZERO_G_FAST_VARIANT_STYLES = [
    {
        "label": "SALVAGE",
        "mood": "deep_space",
        "background": "parallax_starfield",
        "accent": "deep_space_neon",
        "palette": {
            "background_color": [3, 6, 18],
            "primary": [72, 220, 255],
            "secondary": [174, 98, 255],
            "hot": [255, 86, 170],
        },
        "motion_fx": ["agent_trail", "zero_g_drift", "object_trail"],
        "surface_fx": ["crt_vignette", "procedural_noise"],
        "note": "same solved zero-g task with bright salvage field styling",
    },
    {
        "label": "AURORA",
        "mood": "cosmic_research",
        "background": "dense_starfield",
        "accent": "cyan_purple",
        "palette": {
            "background_color": [4, 3, 22],
            "primary": [87, 248, 255],
            "secondary": [248, 102, 255],
            "hot": [255, 218, 82],
        },
        "motion_fx": ["agent_trail", "field_pulse", "particle_sparks"],
        "surface_fx": ["film_grain", "procedural_noise"],
        "note": "same solved zero-g task with nebula/aurora styling",
    },
    {
        "label": "ASTEROID",
        "mood": "asteroid_belt",
        "background": "orbital_dust",
        "accent": "blue_white",
        "palette": {
            "background_color": [7, 7, 14],
            "primary": [130, 184, 255],
            "secondary": [236, 245, 255],
            "hot": [255, 104, 126],
        },
        "motion_fx": ["agent_trail", "object_trail", "dust_motes"],
        "surface_fx": ["stone_facets", "scanlines"],
        "note": "same solved zero-g task with asteroid-belt silhouettes",
    },
]

MECHANISM_FAST_VARIANT_STYLES = [
    {
        "label": "WORKSHOP",
        "mood": "mechanical_factory",
        "background": "circuit_board",
        "accent": "teal_yellow",
        "palette": {
            "background_color": [13, 18, 19],
            "primary": [52, 220, 206],
            "secondary": [255, 225, 86],
            "hot": [255, 96, 142],
        },
        "motion_fx": ["agent_trail", "impact_ripples", "goal_breathe"],
        "surface_fx": ["metal_panel_lines", "grid_crosshairs"],
        "note": "same solved mechanism with readable workshop staging",
    },
    {
        "label": "CLOCKWORK",
        "mood": "clockwork_chamber",
        "background": "blueprint_grid",
        "accent": "emerald_gold",
        "palette": {
            "background_color": [18, 14, 10],
            "primary": [58, 232, 156],
            "secondary": [255, 202, 74],
            "hot": [255, 92, 88],
        },
        "motion_fx": ["agent_trail", "scan_sweep", "particle_sparks"],
        "surface_fx": ["metal_panel_lines", "film_grain"],
        "note": "same solved mechanism with clockwork brass styling",
    },
    {
        "label": "ELECTRIC",
        "mood": "electric_substation",
        "background": "oscilloscope",
        "accent": "electric_blue",
        "palette": {
            "background_color": [5, 12, 18],
            "primary": [42, 170, 255],
            "secondary": [100, 248, 255],
            "hot": [255, 214, 76],
        },
        "motion_fx": ["agent_trail", "electric_arcs", "field_pulse"],
        "surface_fx": ["circuit_traces", "scanlines"],
        "note": "same solved mechanism with electrical lab styling",
    },
]

SPORT_FAST_VARIANT_STYLES = [
    {
        "label": "COURT",
        "mood": "retro_arcade",
        "background": "retro_vignette",
        "accent": "magenta_orange",
        "palette": {
            "background_color": [18, 8, 18],
            "primary": [255, 134, 54],
            "secondary": [255, 238, 112],
            "hot": [80, 228, 255],
        },
        "motion_fx": ["agent_trail", "impact_ripples", "goal_breathe"],
        "surface_fx": ["crt_vignette", "procedural_noise"],
        "note": "same solved sports objective with warm court energy",
    },
    {
        "label": "ARENA",
        "mood": "cyber_grid",
        "background": "radar_sweep",
        "accent": "pink_cyan",
        "palette": {
            "background_color": [8, 8, 20],
            "primary": [255, 90, 214],
            "secondary": [82, 236, 255],
            "hot": [255, 220, 72],
        },
        "motion_fx": ["agent_trail", "particle_sparks", "scan_sweep"],
        "surface_fx": ["grid_crosshairs", "scanlines"],
        "note": "same solved sports objective with neon arena staging",
    },
    {
        "label": "STREET",
        "mood": "industrial_noir",
        "background": "smoke_haze",
        "accent": "gold_crimson",
        "palette": {
            "background_color": [13, 12, 10],
            "primary": [255, 206, 74],
            "secondary": [255, 80, 116],
            "hot": [82, 225, 255],
        },
        "motion_fx": ["agent_trail", "dust_motes", "impact_ripples"],
        "surface_fx": ["metal_panel_lines", "film_grain"],
        "note": "same solved sports objective with street-court contrast",
    },
]


@dataclass
class ConsoleLine:
    text: str
    color: tuple[int, int, int]
    style: str = "normal"


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    action: Callable[[], None]
    enabled: bool = True
    accent: tuple[int, int, int] = CYAN


@dataclass
class VariantSlot:
    index: int
    label: str
    prompt: str
    status: str = "pending"
    tier: int | None = None
    env_path: Path | None = None
    world_export_dir: Path | None = None
    log_dir: Path | None = None
    preview_env: BaseEnv | None = None
    autoplay: "MiniAutoplayState | None" = None
    grammar: VisualGrammar | None = None
    trail: list[tuple[float, float]] = None

    def __post_init__(self) -> None:
        if self.trail is None:
            self.trail = []


class MiniAutoplayState:
    """Tiny validator-policy driver for live multiverse previews."""

    def __init__(self, env: BaseEnv) -> None:
        objective = env.get_ground_truth().get("objective", {})
        profile = objective.get("objective_profile") if isinstance(objective, dict) else {}
        subgoals = profile.get("subgoals") if isinstance(profile, dict) else []
        if not isinstance(subgoals, list):
            subgoals = []
        self.subgoals = [dict(item) for item in subgoals if isinstance(item, dict)]
        self.config = ValidatorConfig(simulation_steps=4)
        self.index = 0
        self.solved_hold_frames = 0
        self.total_steps = 0

    @property
    def label(self) -> str:
        if self.solved_hold_frames > 0:
            return "SOLVED"
        if self.index >= len(self.subgoals):
            return "VERIFYING"
        return str(self.subgoals[self.index].get("kind") or "acting")

    def step(self, env: BaseEnv) -> None:
        self.total_steps += 1
        if callable(getattr(env, "check_objective", None)):
            try:
                if bool(env.check_objective()):
                    self.solved_hold_frames += 1
                    env.step(substeps=2)
                    if self.solved_hold_frames >= 90:
                        env.reset()
                        self.index = 0
                        self.solved_hold_frames = 0
                    return
            except Exception:
                pass

        agent = env.get_agent_record()
        if agent is None:
            env.step(substeps=2)
            return
        if self.index < len(self.subgoals):
            subgoal = self.subgoals[self.index]
            if _subgoal_satisfied(env, subgoal, self.config):
                self.index += 1
                env.step(substeps=2)
                return
            _step_subgoal(env, agent, subgoal, self.config)
            return
        env.step(substeps=2)


class DashboardApp:
    """Interactive dashboard that streams Harness Alpha subprocess output."""

    def __init__(
        self,
        *,
        width: int = WIDTH,
        height: int = HEIGHT,
        max_seeds: int = 2,
        max_repairs: int = 3,
        smoke_frames: int | None = None,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("Harness Alpha // World Factory")
        self.window_size = (width, height)
        self.fullscreen = False
        self.screen = self._set_display_mode()
        self._clipboard_ready = self._init_clipboard()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)
        self.small_font = pygame.font.SysFont("consolas", 13)
        self.example_font = pygame.font.SysFont("consolas", 16, bold=True)
        self.example_title_font = pygame.font.SysFont("consolas", 18, bold=True)
        self.title_font = pygame.font.SysFont("consolas", 30, bold=True)
        self.label_font = pygame.font.SysFont("consolas", 18, bold=True)
        self.prompt_overlay_font = pygame.font.SysFont("consolas", 27, bold=True)
        self.width = width
        self.height = height
        self.max_seeds = max_seeds
        self.max_repairs = max_repairs
        self.execution_mode = "fast"
        self.smoke_frames = smoke_frames
        self.frame_count = 0

        self.prompt = DEFAULT_PROMPT
        self.input_active = True
        self.prompt_selected = True
        self.console: deque[ConsoleLine] = deque(maxlen=280)
        self.console_scroll = 0
        self.console_follow = True
        self.generate_locked = False
        self.queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.reader_thread: threading.Thread | None = None
        self.state = "idle"
        self.status_text = "READY"
        self.status_color = CYAN
        self.pipeline_stage = "Idle"
        self.pipeline_detail = "Awaiting a text command."
        self.current_seed: int | None = None
        self.current_repair: int | None = None
        self.completed_attempts = 0
        self.last_announced_seed: int | None = None
        self.verified_env: Path | None = None
        self.failed_env: Path | None = None
        self.world_export_dir: Path | None = None
        self.log_dir: Path | None = None
        self.tier: int | None = None
        self.accepted = False
        self.variant_mode = False
        self.variant_total = 0
        self.variant_index = 0
        self.variant_envs: list[Path] = []
        self.variant_slots: list[VariantSlot] = []
        self.variant_styles: list[dict[str, object]] = list(FAST_VARIANT_STYLES)
        self.variant_windows_opened = False
        self.auto_open_variant_windows = True
        self.latest_prompt = self.prompt
        self.example_scroll = 0
        self.prompt_focus_overlay = False
        self.pending_demo_load: dict[str, object] | None = None
        self.pending_demo_load_at_ms: int | None = None

        self._add_console("Harness Alpha dashboard ready.", CYAN)
        self._add_console("Type or paste a prompt, or click an instant saved showcase card below.", MUTED)
        self._add_console("Tip: press F11 or Alt+Enter to toggle fullscreen for recording.", MUTED)
        self._add_console("Mode: FAST races first-attempt seeds in parallel; NORMAL uses the classic sequential loop.", MUTED)

    def _set_display_mode(self) -> pygame.Surface:
        flags = pygame.SCALED
        if self.fullscreen:
            flags |= pygame.FULLSCREEN
        return pygame.display.set_mode(self.window_size, flags)

    def _toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self.screen = self._set_display_mode()
        mode = "FULLSCREEN" if self.fullscreen else "WINDOWED"
        self._add_console(f"Display mode: {mode}", CYAN)

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_keydown(event)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
                elif event.type == pygame.MOUSEWHEEL:
                    self._handle_mousewheel(event)
                elif event.type == SAVED_DEMO_LOAD_EVENT:
                    self._complete_pending_saved_demo_load()

            self._drain_process_events()
            self._check_process_exit()
            self._update_variant_previews()
            self._draw()
            pygame.display.flip()
            self.frame_count += 1
            if self.smoke_frames is not None and self.frame_count >= self.smoke_frames:
                running = False
            self.clock.tick(60)

        if self.process and self.process.poll() is None:
            self.process.terminate()
        pygame.quit()

    def _handle_keydown(self, event: pygame.event.Event) -> None:
        if event.key == pygame.K_F11 or (
            event.key == pygame.K_RETURN and (pygame.key.get_mods() & pygame.KMOD_ALT)
        ):
            self._toggle_fullscreen()
            return
        if event.key == pygame.K_ESCAPE and self.prompt_focus_overlay:
            self.prompt_focus_overlay = False
            return
        if event.key == pygame.K_ESCAPE:
            raise SystemExit
        if event.key == pygame.K_a and (pygame.key.get_mods() & pygame.KMOD_CTRL) and self.input_active:
            self.prompt_selected = True
            return
        if event.key == pygame.K_RETURN:
            if self.input_active and self.prompt.strip() and self._can_generate():
                self.prompt_focus_overlay = False
                self.start_generation(self.prompt)
            return
        if event.key == pygame.K_BACKSPACE and self.input_active:
            if self.prompt_selected:
                self.prompt = ""
                self.prompt_selected = False
            else:
                self.prompt = self.prompt[:-1]
            return
        if event.key == pygame.K_DELETE and self.input_active:
            self.prompt = ""
            self.prompt_selected = False
            return
        if event.key == pygame.K_v and (pygame.key.get_mods() & pygame.KMOD_CTRL):
            pasted = self._read_clipboard_text()
            if pasted:
                self._insert_prompt_text(pasted)
            return
        if self.input_active and event.unicode and event.key != pygame.K_TAB:
            self._insert_prompt_text(event.unicode)

    def _handle_mousewheel(self, event: pygame.event.Event) -> None:
        if self._console_panel_rect().collidepoint(pygame.mouse.get_pos()):
            max_scroll = self._console_max_scroll()
            self.console_scroll = min(
                max(self.console_scroll - int(event.y) * 42, 0),
                max_scroll,
            )
            self.console_follow = self.console_scroll >= max_scroll - 4
            return
        if self.variant_slots:
            return
        if not self._examples_panel_rect().collidepoint(pygame.mouse.get_pos()):
            return
        visible_height = max(1, self._examples_view_rect().height)
        content_height = len(SAVED_DEMOS) * EXAMPLE_CARD_HEIGHT
        max_scroll = max(0, content_height - visible_height)
        self.example_scroll = min(max(self.example_scroll - int(event.y) * 34, 0), max_scroll)

    def _insert_prompt_text(self, text: str) -> None:
        if not self.input_active or not text:
            return
        raw = str(text).replace("\x00", "")
        if raw == " ":
            normalized = " "
        else:
            normalized = raw.replace("\r", " ").replace("\n", " ").replace("\t", " ")
            if len(normalized) > 1:
                normalized = re.sub(r" {2,}", " ", normalized)
        if not normalized:
            return
        base = "" if self.prompt_selected else self.prompt
        self.prompt = (base + normalized)[:MAX_PROMPT_CHARS]
        self.prompt_selected = False
        self.prompt_focus_overlay = True

    def _init_clipboard(self) -> bool:
        try:
            pygame.scrap.init()
            return True
        except Exception:
            return False

    def _read_clipboard_text(self) -> str:
        if self._clipboard_ready:
            try:
                payload = pygame.scrap.get(pygame.SCRAP_TEXT)
                if payload:
                    return payload.decode("utf-8", errors="ignore")
            except Exception:
                pass
        try:
            root = tkinter.Tk()
            root.withdraw()
            try:
                return str(root.clipboard_get())
            finally:
                root.destroy()
        except Exception:
            self._add_console("Clipboard paste failed; try clicking the prompt box first.", YELLOW)
            return ""

    def _handle_click(self, pos: tuple[int, int]) -> None:
        if self.prompt_focus_overlay:
            if self._prompt_overlay_generate_rect().collidepoint(pos) and self._can_generate():
                self.prompt_focus_overlay = False
                self.start_generation(self.prompt)
                return
            self.prompt_focus_overlay = False
            return
        if self._prompt_clear_rect().collidepoint(pos) and not self._busy():
            if self.generate_locked or self.state != "idle" or self.verified_env or self.failed_env:
                self.reset_dashboard()
            else:
                self.clear_prompt()
            return
        self.input_active = self._input_rect().collidepoint(pos)
        if self.input_active:
            self.prompt_selected = False
            self.prompt_focus_overlay = True
        for button in self._buttons():
            if button.enabled and button.rect.collidepoint(pos):
                button.action()
                return
        if not self.variant_slots:
            for demo, rect in self._example_card_rects():
                if rect.collidepoint(pos) and not self._busy():
                    self.load_saved_demo(demo)
                    return

    def start_generation(self, prompt: str, *, variant: bool = False) -> None:
        if self._busy():
            return
        if not variant and self.generate_locked:
            self._add_console("Generate is locked after a run. Press Clear to reset before launching another prompt.", YELLOW)
            return
        self.latest_prompt = prompt.strip()
        if not variant:
            demo = self._matching_saved_demo(self.latest_prompt)
            if demo is not None:
                self._begin_saved_demo_load_from_prompt(demo)
                return
        self.state = "running"
        if not variant:
            self.generate_locked = True
        self.status_text = "GENERATING"
        self.status_color = YELLOW
        self.pipeline_stage = "Launching harness"
        self.pipeline_detail = (
            "Starting Architect, Validator, Healer, and export pipeline "
            f"({self.execution_mode.upper()} mode)."
        )
        self.current_seed = None
        self.current_repair = None
        self.completed_attempts = 0
        self.last_announced_seed = None
        self.verified_env = None
        self.failed_env = None
        self.world_export_dir = None
        self.log_dir = None
        self.tier = None
        self.accepted = False
        if not variant:
            self.variant_mode = False
            self.variant_index = 0
            self.variant_total = 0
            self.variant_envs = []
            self.variant_slots = []
            self.console.clear()
            self.console_scroll = 0
            self.console_follow = True
        prefix = (
            f"Variant {self.variant_index}/{self.variant_total}: "
            if variant
            else "Prompt: "
        )
        self._add_console(prefix + self.latest_prompt, TEXT, "section")
        self._add_console("BOOT // Spawning harness.py and loading project contracts.", MUTED, "step")

        command = [
            sys.executable,
            "-u",
            "harness.py",
            self.latest_prompt,
            "--max-seeds",
            str(self.max_seeds),
            "--max-repairs",
            str(self.max_repairs),
            "--execution-mode",
            self.execution_mode,
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        self.reader_thread = threading.Thread(
            target=self._read_process_output,
            args=(self.process,),
            daemon=True,
        )
        self.reader_thread.start()

    def _matching_saved_demo(self, prompt: str) -> dict[str, object] | None:
        normalized = _normalize_prompt_match(prompt)
        if not normalized:
            return None
        for demo in SAVED_DEMOS:
            if normalized == _normalize_prompt_match(str(demo.get("prompt") or "")):
                return demo
        return None

    def _begin_saved_demo_load_from_prompt(self, demo: dict[str, object]) -> None:
        prompt = str(demo.get("prompt") or self.latest_prompt)
        title = str(demo.get("title") or "Saved Demo")
        self.pending_demo_load = demo
        self.pending_demo_load_at_ms = pygame.time.get_ticks()
        self.generate_locked = True
        self.prompt_focus_overlay = False
        self.state = "running"
        self.status_text = "CACHE HIT"
        self.status_color = CYAN
        self.pipeline_stage = "Showcase Match"
        self.pipeline_detail = "Exact prompt matches a prebuilt verified example; loading without LLM latency."
        self.current_seed = None
        self.current_repair = None
        self.completed_attempts = 0
        self.verified_env = None
        self.failed_env = None
        self.world_export_dir = None
        self.log_dir = None
        self.tier = None
        self.accepted = False
        self.variant_mode = False
        self.variant_index = 0
        self.variant_total = 0
        self.variant_envs = []
        self.variant_slots = []
        self.variant_windows_opened = False
        self.console.clear()
        self.console_scroll = 0
        self.console_follow = True
        self._add_console(f"PROMPT MATCH // {title}", CYAN, "section")
        self._add_console(prompt, TEXT)
        self._add_console("CACHE HIT // Found matching verified showcase environment.", GREEN, "step")
        self._add_console("STAGING // Loading prebuilt world, Godot export, and AI solve metadata...", MUTED, "step")
        pygame.time.set_timer(SAVED_DEMO_LOAD_EVENT, 3000, loops=1)

    def _complete_pending_saved_demo_load(self) -> None:
        if self.pending_demo_load is None:
            return
        demo = self.pending_demo_load
        self.pending_demo_load = None
        self.pending_demo_load_at_ms = None
        self.load_saved_demo(demo, from_prompt_match=True)

    def load_saved_demo(self, demo: dict[str, object], *, from_prompt_match: bool = False) -> None:
        """Load a curated saved demo without running the Architect loop."""

        env_path = Path(str(demo.get("env_path") or ""))
        export_dir = Path(str(demo.get("export_dir") or ""))
        prompt = str(demo.get("prompt") or "")
        title = str(demo.get("title") or env_path.parent.name)
        tier = int(demo.get("tier") or 5)
        self.prompt = prompt
        self.prompt_selected = True
        self.input_active = True
        self.latest_prompt = prompt
        self.variant_mode = False
        self.variant_total = 0
        self.variant_index = 0
        self.variant_envs = []
        self.variant_slots = []
        self.variant_windows_opened = False
        self.failed_env = None
        if from_prompt_match:
            self.generate_locked = True
        self.log_dir = env_path.parent if env_path.exists() else None
        self.verified_env = env_path if env_path.exists() else None
        self.world_export_dir = export_dir if (export_dir / "world_schema.json").exists() else None
        self.tier = tier if self.verified_env is not None else None
        self.accepted = self.verified_env is not None
        self.state = "success" if self.accepted else "failed"
        self.console.clear()
        self.console_scroll = 0
        self.console_follow = True
        if self.accepted:
            self.status_text = f"DEMO TIER {tier}"
            self.status_color = GREEN
            self.pipeline_stage = "Saved Demo Loaded" if not from_prompt_match else "Prompt-Matched Demo Loaded"
            self.pipeline_detail = "This world is prebuilt and ready for Godot, AI solve, or fast variants."
            section = "PROMPT-MATCHED DEMO" if from_prompt_match else "INSTANT DEMO"
            self._add_console(f"{section} // {title}", GREEN, "section")
            self._add_console(prompt, TEXT)
            if from_prompt_match:
                self._add_console("Loaded from verified showcase cache after typed prompt match.", GREEN, "step")
            self._add_console(f"ENV: {env_path}", CYAN)
            if self.world_export_dir is not None:
                self._add_console(f"GODOT EXPORT: {self.world_export_dir / 'world_schema.json'}", CYAN)
            else:
                self._add_console("Godot export missing; Pygame AI solve is still available.", YELLOW)
            self._add_console("No live generation needed. Press Play in Godot, Watch AI Solve, or Fork 3 Fast Variants.", MUTED, "step")
        else:
            self.status_text = "DEMO MISSING"
            self.status_color = RED
            self.pipeline_stage = "Saved Demo Missing"
            self.pipeline_detail = f"Could not find {env_path}."
            self._add_console(f"Saved demo missing: {title}", RED, "section")
            self._add_console(str(env_path), RED)

    def start_variants(self) -> None:
        if not self.verified_env or self._busy():
            return
        self.variant_mode = False
        self.variant_styles = self._fast_variant_styles_for_current_world()
        self.variant_total = len(self.variant_styles)
        self.variant_index = 0
        self.variant_envs = []
        self.variant_windows_opened = False
        self.variant_slots = [
            VariantSlot(
                index=index + 1,
                label=str(style["label"]),
                prompt=f"Fast verified fork of: {self.latest_prompt or self.prompt}",
            )
            for index, style in enumerate(self.variant_styles)
        ]
        self.console.clear()
        self.console_scroll = 0
        self.console_follow = True
        self._add_console("MULTIVERSE // Forking the verified world without rerunning the LLM or repair loop.", CYAN, "section")
        self._add_console("Physics/objective stay locked to the accepted env; variants mutate renderer/export presentation only.", MUTED, "step")
        for slot in self.variant_slots:
            self._materialize_fast_variant(slot)
        self._add_console(f"Variant batch complete: {len(self.variant_envs)}/{self.variant_total} fast verified forks ready.", GREEN)
        if self.auto_open_variant_windows:
            self.open_variant_visualizers()

    def _start_current_variant(self) -> None:
        slot = self.variant_slots[self.variant_index - 1]
        slot.status = "running"
        self.start_generation(slot.prompt, variant=True)

    def _materialize_fast_variant(self, slot: VariantSlot) -> None:
        style = self.variant_styles[slot.index - 1]
        slot.status = "running"
        slot.tier = self.tier or 5
        slot.env_path = self.verified_env
        slot.log_dir = self.log_dir
        try:
            slot.preview_env = self._load_preview_env(self.verified_env) if self.verified_env else None
            if slot.preview_env is None:
                raise RuntimeError("verified environment could not be loaded for preview")
            base_grammar = infer_visual_grammar(slot.preview_env)
            slot.grammar = self._variant_grammar(base_grammar, style, slot.index)
            slot.env_path = self._create_fast_variant_env_file(slot, style)
            slot.autoplay = MiniAutoplayState(slot.preview_env) if slot.tier == 5 else None
            slot.world_export_dir = self._create_fast_variant_export(slot, style)
            slot.status = "success"
            self.variant_envs.append(slot.env_path or self.verified_env)
            self._add_console(
                f"V{slot.index} READY // {slot.label}: {style['note']}.",
                GREEN,
                "step",
            )
        except Exception as exc:
            slot.status = "failed"
            slot.autoplay = None
            self._add_console(f"V{slot.index} fast fork failed: {type(exc).__name__}: {exc}", RED)

    def _fast_variant_styles_for_current_world(self) -> list[dict[str, object]]:
        context = " ".join(
            [
                str(self.latest_prompt or self.prompt),
                str(self.verified_env or ""),
                str(self.world_export_dir or ""),
            ]
        ).lower()
        if any(token in context for token in ["basketball", "hoop", "soccer", "football", "court", "goal line", "kick", "throw", "sports"]):
            return list(SPORT_FAST_VARIANT_STYLES)
        if any(token in context for token in ["seesaw", "see-saw", "lever", "pivot", "plank", "gear", "clockwork", "mechanism", "pressure plate", "sliding gate"]):
            return list(MECHANISM_FAST_VARIANT_STYLES)
        if any(token in context for token in ["zero-g", "zero_g", "zero gravity", "crystal", "salvage", "asteroid"]):
            return list(ZERO_G_FAST_VARIANT_STYLES)
        if any(token in context for token in ["space", "spaceship", "ship", "laser"]):
            return list(SPACE_FAST_VARIANT_STYLES)
        if any(token in context for token in ["lava", "fire", "fireball", "volcanic", "obsidian", "magma"]):
            return list(LAVA_FAST_VARIANT_STYLES)
        if any(token in context for token in ["forest", "bear", "cabin", "tree", "woods", "jungle", "garden"]):
            return list(FOREST_FAST_VARIANT_STYLES)
        if any(token in context for token in ["water", "swim", "pool", "river", "lagoon", "ocean", "beacon"]):
            return list(WATER_FAST_VARIANT_STYLES)
        if any(token in context for token in ["maze", "corridor", "exit", "labyrinth", "door", "room"]):
            return list(MAZE_FAST_VARIANT_STYLES)
        return list(FAST_VARIANT_STYLES)

    def open_visualizer(self, *, autoplay: bool = False) -> None:
        env_path = self.verified_env
        if env_path is None or not env_path.exists():
            return
        command = [sys.executable, "visualizer.py", str(env_path)]
        if autoplay:
            command.append("--autoplay")
        subprocess.Popen(command)

    def _create_fast_variant_env_file(
        self,
        slot: VariantSlot,
        style: dict[str, object],
    ) -> Path:
        if self.verified_env is None:
            raise RuntimeError("no verified environment to fork")
        root = (self.log_dir / "variant_envs") if self.log_dir is not None else (Path("exports") / "variant_envs")
        root.mkdir(parents=True, exist_ok=True)
        stem = str(style.get("label", "variant")).lower().replace(" ", "_")
        target = root / f"variant_{slot.index:02d}_{stem}.py"
        shutil.copyfile(self.verified_env, target)
        self._ensure_variant_env_owns_module_sidecar(target)
        if slot.grammar is not None:
            recipe = self._visual_recipe_from_grammar(slot.grammar, slot, style)
            self._write_json_file(target.with_suffix(".visual.json"), recipe)
        return target

    def _ensure_variant_env_owns_module_sidecar(self, target: Path) -> None:
        """Wrap imported saved-demo classes so sidecar visual recipes resolve per variant."""

        try:
            text = target.read_text(encoding="utf-8")
        except OSError:
            return
        if "class " in text or "from demo_envs import" not in text:
            return
        match = re.search(r"from\s+demo_envs\s+import\s+([A-Za-z_][A-Za-z0-9_]*)", text)
        if not match:
            return
        base_name = match.group(1)
        wrapper_name = f"{base_name}Variant"
        target.write_text(
            "\n".join(
                [
                    f"from demo_envs import {base_name} as _BaseVariantEnv",
                    "",
                    "",
                    f"class {wrapper_name}(_BaseVariantEnv):",
                    "    pass",
                    "",
                    "",
                    f'GENERATED_ENV_CLASS = "{wrapper_name}"',
                    f"SOURCE_PROMPT = {wrapper_name}.prompt",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def _visual_recipe_from_grammar(
        self,
        grammar: VisualGrammar,
        slot: VariantSlot,
        style: dict[str, object],
    ) -> dict[str, object]:
        recipe = dict(grammar.visual_recipe or {})
        recipe.update(
            {
                "mood": grammar.mood,
                "background": grammar.background,
                "accent": grammar.accent,
                "palette": {
                    "background_color": list(grammar.background_color),
                    "primary": list(grammar.primary),
                    "secondary": list(grammar.secondary),
                    "hot": list(grammar.hot),
                },
                "motion_fx": list(grammar.motion_fx),
                "surface_fx": list(grammar.surface_fx),
                "agent_avatar": grammar.agent_avatar,
                "seed": grammar.seed,
                "fast_variant": True,
                "variant_index": slot.index,
                "variant_label": style.get("label"),
                "variant_note": style.get("note"),
                "background_layers": self._variant_recipe_layers(style, slot.index),
                "object_skins": self._variant_object_skins(recipe.get("object_skins"), style),
                "visual_program": self._variant_visual_program(
                    recipe.get("visual_program") if isinstance(recipe.get("visual_program"), dict) else {},
                    style,
                    slot.index,
                ),
            }
        )
        return recipe

    def open_godot_runtime(self) -> None:
        schema_path = self._world_schema_path()
        if schema_path is None:
            self._add_console("Godot runtime unavailable: no exported world_schema.json found.", RED)
            return
        command = [
            sys.executable,
            "godot_bridge.py",
            "--schema",
            str(schema_path),
            "--no-wait",
        ]
        self._add_console(f"Launching Godot runtime: {schema_path}", CYAN)
        subprocess.Popen(command)

    def open_variant_visualizers(self) -> None:
        if self.variant_windows_opened:
            return
        successful_slots = [
            slot
            for slot in self.variant_slots
            if slot.status == "success" and slot.env_path is not None and slot.env_path.exists()
        ]
        if not successful_slots:
            self._add_console("No verified variant windows to open.", MUTED)
            return

        self.variant_windows_opened = True
        self._add_console(
            f"Opening {len(successful_slots)} sequential AI-solve variant window(s).",
            CYAN,
        )
        threading.Thread(
            target=self._run_variant_solve_sequence,
            args=(successful_slots,),
            daemon=True,
        ).start()

    def _run_variant_solve_sequence(self, slots: list[VariantSlot]) -> None:
        for slot in slots:
            if slot.env_path is None:
                continue
            self.queue.put(("line", f"VARIANT {slot.index}: launching AI solve window..."))
            command = [
                sys.executable,
                "visualizer.py",
                str(slot.env_path),
                "--autoplay",
                "--auto-close-after-solve",
                "3.2",
            ]
            subprocess.run(command, check=False)
            self.queue.put(("line", f"VARIANT {slot.index}: AI solve window closed."))

    def stop_generation(self) -> None:
        if self.pending_demo_load is not None:
            self.pending_demo_load = None
            self.pending_demo_load_at_ms = None
            pygame.time.set_timer(SAVED_DEMO_LOAD_EVENT, 0)
            self._add_console("Prebuilt demo load cancelled by user.", RED)
            self.state = "idle"
            self.status_text = "CANCELLED"
            self.status_color = RED
            return
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self._add_console("Generation cancelled by user.", RED)
            self.state = "idle"
            self.status_text = "CANCELLED"
            self.status_color = RED

    def clear_console(self) -> None:
        if not self._busy():
            self.console.clear()
            self.console_scroll = 0
            self.console_follow = True
            self._add_console("Console cleared.", MUTED)

    def reset_dashboard(self) -> None:
        if self._busy():
            return
        self.prompt = ""
        self.input_active = True
        self.prompt_selected = False
        self.prompt_focus_overlay = False
        self.console.clear()
        self.console_scroll = 0
        self.console_follow = True
        self.queue = queue.Queue()
        self.process = None
        self.reader_thread = None
        self.state = "idle"
        self.status_text = "READY"
        self.status_color = CYAN
        self.pipeline_stage = "Idle"
        self.pipeline_detail = "Awaiting a text command."
        self.pending_demo_load = None
        self.pending_demo_load_at_ms = None
        self.generate_locked = False
        pygame.time.set_timer(SAVED_DEMO_LOAD_EVENT, 0)
        self.current_seed = None
        self.current_repair = None
        self.completed_attempts = 0
        self.last_announced_seed = None
        self.verified_env = None
        self.failed_env = None
        self.world_export_dir = None
        self.log_dir = None
        self.tier = None
        self.accepted = False
        self.variant_mode = False
        self.variant_total = 0
        self.variant_index = 0
        self.variant_envs = []
        self.variant_slots = []
        self.variant_styles = list(FAST_VARIANT_STYLES)
        self.variant_windows_opened = False
        self.latest_prompt = ""
        self.example_scroll = 0
        self._add_console("Harness Alpha dashboard reset.", CYAN)
        self._add_console("Type or paste a prompt, or click an instant saved showcase card below.", MUTED)

    def clear_prompt(self) -> None:
        if self._busy():
            return
        self.prompt = ""
        self.input_active = True
        self.prompt_selected = False
        self.prompt_focus_overlay = False

    def toggle_execution_mode(self) -> None:
        if self._busy():
            return
        self.execution_mode = "normal" if self.execution_mode == "fast" else "fast"
        if self.execution_mode == "fast":
            self._add_console("Execution mode: FAST // races first-attempt seeds in parallel.", CYAN)
            self.pipeline_detail = "Fast mode enabled: spend more API calls to reduce wall-clock latency."
        else:
            self._add_console("Execution mode: NORMAL // classic sequential Reflexion loop.", MUTED)
            self.pipeline_detail = "Normal mode enabled: lower API spend, more sequential repair feedback."

    def _read_process_output(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        for raw_line in process.stdout:
            self.queue.put(("line", raw_line.rstrip("\n")))
        self.queue.put(("done", str(process.wait())))

    def _drain_process_events(self) -> None:
        while True:
            try:
                kind, payload = self.queue.get_nowait()
            except queue.Empty:
                return
            if kind == "line":
                self._handle_process_line(payload)
            elif kind == "done":
                pass

    def _handle_process_line(self, line: str) -> None:
        if not line:
            return
        self._update_pipeline_from_line(line)
        color = self._line_color(line)
        style = self._line_style(line)
        self._add_console(line, color, style)
        if "SUCCESS - VERIFIED SOLVABLE" in line:
            self.status_text = "SUCCESS"
            self.status_color = GREEN
            self.pipeline_stage = "Verified"
            self.pipeline_detail = "Objective accepted by deterministic validation."
        elif "AUTO-REPAIRING" in line:
            self.status_text = "AUTO-REPAIR"
            self.status_color = YELLOW
            self.pipeline_stage = "Auto-repair"
            self.pipeline_detail = "Applying measured numeric layout repair before another validation pass."
        elif "CODE ERROR" in line or "FAILED_ENV" in line or "Post-Mortem" in line:
            self.status_text = "FAILURE"
            self.status_color = RED
            if "FAILED_ENV" in line or "Post-Mortem" in line:
                self.pipeline_stage = "Post-mortem"
                self.pipeline_detail = "All attempts are exhausted; showing final diagnostic evidence."
        elif "REGENERATING" in line:
            self.status_text = "GENERATING"
            self.status_color = CYAN
            self.pipeline_stage = "Architect"
            self.pipeline_detail = "Generating a contract-checked environment candidate."

        if line.startswith("VERIFIED_ENV:"):
            self.verified_env = Path(line.split(":", 1)[1].strip())
            self.accepted = True
        elif line.startswith("FAILED_ENV:"):
            self.failed_env = Path(line.split(":", 1)[1].strip())
            self.accepted = False
        elif line.startswith("WORLD_EXPORT:"):
            self.world_export_dir = Path(line.split(":", 1)[1].strip())
        elif line.startswith("LOG_DIR:"):
            self.log_dir = Path(line.split(":", 1)[1].strip())
        elif line.startswith("VERIFICATION:"):
            match = re.search(r"TIER\s+(\d+)", line)
            if match:
                self.tier = int(match.group(1))

    def _update_pipeline_from_line(self, line: str) -> None:
        attempt = re.search(r"\[SEED\s+(\d+)\]\s+\[REPAIR\s+(\d+)/(\d+)\]", line, re.IGNORECASE)
        if attempt:
            seed = int(attempt.group(1))
            repair = int(attempt.group(2))
            self.current_seed = seed
            self.current_repair = repair
            self.completed_attempts = max(self.completed_attempts, (seed - 1) * self.max_repairs + repair - 1)
            if self.last_announced_seed != seed:
                self.last_announced_seed = seed
                seed_label = {
                    1: "baseline layout",
                    2: "inverse/peripheral pivot",
                    3: "high-entropy pivot",
                }.get(seed, "strategic pivot")
                self._add_console(f"SEED {seed} // {seed_label.upper()}", CYAN, "section")

        upper = line.upper()
        if "[FAST MODE]" in upper:
            self.pipeline_stage = "Fast seed race"
            self.pipeline_detail = "Speculative seeds are running in parallel; first accepted world wins."
        elif "FAST SPECULATIVE GENERATION" in upper:
            self.pipeline_stage = "Fast seed race"
            self.pipeline_detail = "Parallel first-attempt candidates are being generated and validated."
        elif "REGENERATING" in upper:
            self.pipeline_stage = "Architect"
            self.pipeline_detail = "Drafting Python environment code from prompt, memory, and specs."
        elif "GENERATION OR CONTRACT VERIFICATION FAILED" in upper or "CODE ERROR" in upper:
            self.pipeline_stage = "Contract check"
            self.pipeline_detail = "Rejected candidate code; preparing narrow repair instructions."
        elif "AUTO-REPAIR" in upper:
            self.pipeline_stage = "Auto-repair"
            self.pipeline_detail = "Measured geometry issue found; applying deterministic numeric patch."
        elif "TIER" in upper and ("FAILED" in upper or "DID NOT COMPLETE" in upper or "BLOCKED" in upper):
            self.pipeline_stage = "Validator"
            self.pipeline_detail = "Running objective, affordance, and rollout probes over deterministic state."
        elif "PIVOTING TO SEED" in upper:
            self.pipeline_stage = "Strategic pivot"
            self.pipeline_detail = "Local repairs exhausted; switching to a more diverse world layout."
        elif "WORLD_EXPORT:" in line:
            self.pipeline_stage = "Export"
            self.pipeline_detail = "Building Godot/Pygame presentation artifacts for the verified world."
        elif "SUCCESS - VERIFIED SOLVABLE" in upper:
            self.pipeline_stage = "Verified"
            self.pipeline_detail = "World passed the required acceptance tier."

    def _check_process_exit(self) -> None:
        if self.process is None or self.process.poll() is None:
            return
        exit_code = self.process.returncode
        self.process = None
        if self.verified_env is not None and self.accepted:
            self.world_export_dir = self._resolved_world_export_dir()
            self.state = "success"
            self.status_text = f"TIER {self.tier or 5} VERIFIED"
            self.status_color = GREEN
            if self.variant_mode:
                slot = self.variant_slots[self.variant_index - 1]
                slot.status = "success"
                slot.tier = self.tier or 5
                slot.env_path = self.verified_env
                slot.world_export_dir = self._resolved_world_export_dir()
                slot.log_dir = self.log_dir
                slot.preview_env = self._load_preview_env(self.verified_env)
                slot.grammar = infer_visual_grammar(slot.preview_env) if slot.preview_env is not None else None
                slot.autoplay = MiniAutoplayState(slot.preview_env) if slot.preview_env is not None and slot.tier == 5 else None
                self.variant_envs.append(self.verified_env)
        else:
            self.state = "failed"
            self.status_text = f"FAILED ({exit_code})"
            self.status_color = RED
            if self.variant_mode:
                slot = self.variant_slots[self.variant_index - 1]
                slot.status = "failed"
                slot.tier = self.tier
                slot.env_path = self.failed_env
                slot.log_dir = self.log_dir

        if self.variant_mode and self.variant_index < self.variant_total:
            self.variant_index += 1
            self._start_current_variant()
        elif self.variant_mode:
            self._add_console(f"Variant batch complete: {len(self.variant_envs)}/{self.variant_total} verified.", GREEN if self.variant_envs else RED)
            self.variant_mode = False
            if self.auto_open_variant_windows:
                self.open_variant_visualizers()

    def _load_preview_env(self, env_path: Path) -> BaseEnv | None:
        try:
            env_class = load_env_class(env_path)
            return env_class()
        except Exception as exc:
            self._add_console(f"Preview load failed for {env_path.name}: {type(exc).__name__}: {exc}", RED)
            return None

    def _resolved_world_export_dir(self) -> Path | None:
        if self.log_dir is not None:
            log_export = self.log_dir / "world_export"
            if (log_export / "world_schema.json").exists():
                return log_export
        if self.world_export_dir is not None and (self.world_export_dir / "world_schema.json").exists():
            return self.world_export_dir
        return None

    def _world_schema_path(self) -> Path | None:
        if self.verified_env is None or not self.accepted:
            return None
        export_dir = self._resolved_world_export_dir()
        if export_dir is None:
            return None
        schema_path = export_dir / "world_schema.json"
        return schema_path if schema_path.exists() else None

    def _slot_world_schema_path(self, slot: VariantSlot) -> Path | None:
        if slot.world_export_dir is not None:
            schema_path = slot.world_export_dir / "world_schema.json"
            if schema_path.exists():
                return schema_path
        if slot.log_dir is not None:
            schema_path = slot.log_dir / "world_export" / "world_schema.json"
            if schema_path.exists():
                return schema_path
        return None

    def _variant_grammar(
        self,
        base: VisualGrammar,
        style: dict[str, object],
        index: int,
    ) -> VisualGrammar:
        palette = style.get("palette") if isinstance(style.get("palette"), dict) else {}

        def color(name: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
            raw = palette.get(name) if isinstance(palette, dict) else None
            if isinstance(raw, list | tuple) and len(raw) >= 3:
                return tuple(max(0, min(255, int(raw[i]))) for i in range(3))
            return fallback

        return VisualGrammar(
            mood=str(style.get("mood") or base.mood),
            background=str(style.get("background") or base.background),
            accent=str(style.get("accent") or base.accent),
            materials=base.materials,
            lighting=base.lighting,
            motion_fx=tuple(style.get("motion_fx") or base.motion_fx),
            surface_fx=tuple(style.get("surface_fx") or base.surface_fx),
            agent_avatar=base.agent_avatar,
            shape_language=base.shape_language,
            presentation=base.presentation,
            background_color=color("background_color", base.background_color),
            primary=color("primary", base.primary),
            secondary=color("secondary", base.secondary),
            hot=color("hot", base.hot),
            seed=base.seed + index * 101,
            source="fast_verified_variant",
            scores=dict(base.scores),
            visual_recipe={
                **dict(base.visual_recipe or {}),
                "fast_variant": True,
                "variant_index": index,
                "variant_label": style.get("label"),
                "variant_note": style.get("note"),
            },
        )

    def _create_fast_variant_export(
        self,
        slot: VariantSlot,
        style: dict[str, object],
    ) -> Path | None:
        base_export = self._resolved_world_export_dir()
        if base_export is None:
            return None
        if self.log_dir is not None:
            root = self.log_dir / "variant_exports"
        else:
            root = Path("exports") / "variants"
        target = root / f"variant_{slot.index:02d}_{str(style.get('label', 'variant')).lower().replace(' ', '_')}"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(base_export, target)
        self._mutate_variant_export_files(target, slot, style)
        return target

    def _mutate_variant_export_files(
        self,
        export_dir: Path,
        slot: VariantSlot,
        style: dict[str, object],
    ) -> None:
        schema_path = export_dir / "world_schema.json"
        visual_path = export_dir / "visual_brief.json"
        manifest_path = export_dir / "manifest.json"
        brief = self._load_json_file(visual_path)
        if not brief and schema_path.exists():
            schema = self._load_json_file(schema_path)
            raw_brief = schema.get("visual_brief") if isinstance(schema, dict) else {}
            brief = raw_brief if isinstance(raw_brief, dict) else {}
        brief = self._variant_visual_brief(brief, slot, style)

        if visual_path.exists():
            self._write_json_file(visual_path, brief)
        schema = self._load_json_file(schema_path)
        if schema:
            schema["visual_brief"] = brief
            source = schema.get("source")
            if isinstance(source, dict):
                source["variant_label"] = style.get("label")
                source["variant_of"] = str(self.verified_env) if self.verified_env else None
            schema["fast_variant"] = {
                "enabled": True,
                "variant_index": slot.index,
                "variant_label": style.get("label"),
                "physics_source_locked": True,
                "note": style.get("note"),
            }
            self._write_json_file(schema_path, schema)

        manifest = self._load_json_file(manifest_path)
        if manifest:
            manifest["fast_variant"] = {
                "enabled": True,
                "variant_index": slot.index,
                "variant_label": style.get("label"),
                "source_export": str(base_export) if (base_export := self._resolved_world_export_dir()) else None,
            }
            self._write_json_file(manifest_path, manifest)

    def _variant_visual_brief(
        self,
        brief: dict[str, object],
        slot: VariantSlot,
        style: dict[str, object],
    ) -> dict[str, object]:
        result = dict(brief or {})
        result["mood"] = style.get("mood", result.get("mood", "training_sim"))
        result["background"] = style.get("background", result.get("background", "neon_grid"))
        result["accent"] = style.get("accent", result.get("accent", "cyan_green"))
        result["motion_fx"] = list(style.get("motion_fx") or result.get("motion_fx") or [])
        result["surface_fx"] = list(style.get("surface_fx") or result.get("surface_fx") or [])
        result["palette"] = style.get("palette", result.get("palette", {}))
        recipe = result.get("recipe") if isinstance(result.get("recipe"), dict) else {}
        result["recipe"] = {
            **recipe,
            "fast_variant": True,
            "variant_index": slot.index,
            "variant_label": style.get("label"),
            "variant_note": style.get("note"),
        }
        program = result.get("visual_program") if isinstance(result.get("visual_program"), dict) else {}
        result["visual_program"] = self._variant_visual_program(program, style, slot.index)
        recipe = result.get("recipe") if isinstance(result.get("recipe"), dict) else {}
        result["recipe"] = {
            **recipe,
            "background_layers": self._variant_recipe_layers(style, slot.index),
            "object_skins": self._variant_object_skins(recipe.get("object_skins"), style),
        }
        result["fast_variant"] = {
            "enabled": True,
            "variant_index": slot.index,
            "variant_label": style.get("label"),
            "physics_source_locked": True,
        }
        return result

    def _variant_recipe_layers(self, style: dict[str, object], index: int) -> list[dict[str, object]]:
        mood = str(style.get("mood", "")).lower()
        background = str(style.get("background", "")).lower()
        layers: list[dict[str, object]] = [{"type": "atmospheric_vignette", "strength": 0.42 + index * 0.04}]
        if "space" in mood or "orbital" in mood or "star" in background:
            layers.extend([
                {"type": "star_dust", "count": 170 + index * 35},
                {"type": "nebula_wash"},
            ])
        elif "volcanic" in mood or "hazard" in mood or "ember" in background:
            layers.extend([
                {"type": "ember_field", "count": 120 + index * 30},
                {"type": "molten_cracks", "count": 16 + index * 5},
                {"type": "heat_shimmer"},
            ])
        elif "organic" in mood or "garden" in mood or "forest" in mood:
            layers.extend([
                {"type": "organic_canopy", "count": 80 + index * 20},
                {"type": "leaf_shadow", "count": 70 + index * 18},
            ])
        elif "water" in mood or "underwater" in mood:
            if index == 1:
                layers.extend([
                    {"type": "cold_mist", "count": 95},
                    {"type": "organic_canopy", "count": 34},
                    {"type": "dust_motes", "count": 65},
                ])
            elif index == 2:
                layers.extend([
                    {"type": "star_dust", "count": 150},
                    {"type": "nebula_wash"},
                    {"type": "oscilloscope_grid", "cell": 54},
                ])
            else:
                layers.extend([
                    {"type": "cold_mist", "count": 170},
                    {"type": "crt_scanlines", "cell": 14},
                    {"type": "heat_shimmer"},
                ])
        else:
            layers.extend([
                {"type": "simulation_grid", "cell": 26 + index * 4},
                {"type": "dust_motes", "count": 85 + index * 25},
            ])
        if index == 1:
            layers.extend([
                {"type": "simulation_grid", "cell": 24},
                {"type": "dust_motes", "count": 55},
                {"type": "set_dressing_props", "theme": "structured", "count": 5},
            ])
        elif index == 2:
            layers.extend([
                {"type": "nebula_wash"},
                {"type": "crt_scanlines", "cell": 18},
                {"type": "set_dressing_props", "theme": "cinematic", "count": 7},
            ])
        else:
            layers.extend([
                {"type": "star_dust", "count": 90},
                {"type": "oscilloscope_grid", "cell": 42},
                {"type": "set_dressing_props", "theme": "alternate", "count": 9},
            ])
        return layers

    def _variant_visual_program(
        self,
        program: dict[str, object],
        style: dict[str, object],
        index: int,
    ) -> dict[str, object]:
        result = dict(program or {})
        result["seed"] = int(result.get("seed") or 0) + index * 137
        palette = style.get("palette") if isinstance(style.get("palette"), dict) else {}
        primary = palette.get("primary", [72, 221, 236]) if isinstance(palette, dict) else [72, 221, 236]
        secondary = palette.get("secondary", [52, 232, 154]) if isinstance(palette, dict) else [52, 232, 154]
        hot = palette.get("hot", [255, 90, 118]) if isinstance(palette, dict) else [255, 90, 118]
        layers = list(result.get("background_layers") or [])
        layers.extend(self._variant_program_background_layers(style, index, primary, secondary, hot))
        if index == 1:
            layers.extend([
                {"primitive": "grid_overlay", "cell": 30, "color": primary},
                {"primitive": "contour_lines", "count": 8, "color": secondary},
            ])
        elif index == 2:
            layers.extend([
                {"primitive": "scanline_roll", "color": secondary},
                {"primitive": "heat_shimmer", "color": hot},
            ])
        else:
            layers.extend([
                {"primitive": "texture_noise", "count": 135, "color": secondary},
                {"primitive": "facet_field", "count": 62, "color": primary},
            ])
        if index == 2:
            layers.append({"primitive": "ribbon_flow", "count": 9, "speed": 0.24, "color": hot})
        elif index == 3:
            layers.append({"primitive": "ribbon_flow", "count": 4, "speed": 0.12, "color": secondary})
        result["background_layers"] = layers
        effects = list(result.get("object_effects") or [])
        effects.extend([
            {"selector": "agent", "primitive": "motion_trail", "color": primary},
            {"selector": "goal|exit|target|beacon", "primitive": "portal_ring", "color": secondary},
            {"selector": "hazard|enemy|fire|lava|bear", "primitive": "danger_pulse", "color": hot},
        ])
        if index == 1:
            effects.extend([
                {"selector": "wall|floor|platform|gate", "primitive": "rim_glow", "color": primary},
                {"selector": "goal|exit|target|beacon", "primitive": "field_ring", "rings": 2, "color": secondary},
            ])
        elif index == 2:
            effects.extend([
                {"selector": "hazard|enemy|fire|lava|bear|ball", "primitive": "spark_burst", "color": hot},
                {"selector": "wall|floor|platform|gate", "primitive": "material_stripes", "color": secondary},
            ])
        else:
            effects.extend([
                {"selector": "wall|floor|platform|gate|rock|stone", "primitive": "cracked_surface", "color": primary},
                {"selector": "agent|ball|crystal|rock", "primitive": "neon_outline", "color": secondary},
            ])
        result["object_effects"] = effects
        return result

    def _variant_program_background_layers(
        self,
        style: dict[str, object],
        index: int,
        primary: object,
        secondary: object,
        hot: object,
    ) -> list[dict[str, object]]:
        mood = str(style.get("mood", "")).lower()
        background = str(style.get("background", "")).lower()
        layers: list[dict[str, object]] = [
            {
                "primitive": "particle_field",
                "count": 80 + index * 36,
                "motion": "drift",
                "speed": 0.16 + index * 0.05,
                "color": primary,
            },
            {"primitive": "radial_glow", "count": 2 + index, "color": secondary},
        ]
        if "space" in mood or "orbital" in mood or "star" in background:
            layers.extend([
                {"primitive": "particle_field", "count": 220, "motion": "orbit", "speed": 0.10 + index * 0.03, "color": secondary},
                {"primitive": "ribbon_flow", "count": 3 + index, "speed": 0.08, "color": primary},
            ])
        elif "volcanic" in mood or "hazard" in mood or "inferno" in background or "magma" in background:
            layers.extend([
                {"primitive": "particle_field", "count": 185, "motion": "updraft", "speed": 0.30 + index * 0.04, "color": hot},
                {"primitive": "heat_shimmer", "color": hot},
            ])
        elif "forest" in mood or "woods" in mood or "grove" in mood:
            layers.extend([
                {"primitive": "texture_noise", "count": 160, "color": secondary},
                {"primitive": "contour_lines", "count": 13, "color": primary},
            ])
        elif "water" in mood or "lagoon" in background or "pool" in background:
            layers.extend([
                {"primitive": "contour_lines", "count": 18, "color": primary},
                {"primitive": "particle_field", "count": 130, "motion": "drift", "speed": 0.08, "color": secondary},
            ])
        return layers

    def _variant_object_skins(
        self,
        existing: object,
        style: dict[str, object],
    ) -> dict[str, object]:
        skins = dict(existing) if isinstance(existing, dict) else {}
        palette = style.get("palette") if isinstance(style.get("palette"), dict) else {}
        primary = palette.get("primary", [72, 221, 236]) if isinstance(palette, dict) else [72, 221, 236]
        secondary = palette.get("secondary", [52, 232, 154]) if isinstance(palette, dict) else [52, 232, 154]
        hot = palette.get("hot", [255, 90, 118]) if isinstance(palette, dict) else [255, 90, 118]
        primary_fill = [max(0, int(c * 0.38)) for c in primary]
        secondary_fill = [max(0, int(c * 0.42)) for c in secondary]
        hot_fill = [max(0, int(c * 0.62)) for c in hot]
        additions = {
            "*goal*": {"fill": secondary_fill, "outline": secondary, "glow": secondary, "fx": ["goal_breathe", "edge_bloom"]},
            "*exit*": {"fill": secondary_fill, "outline": secondary, "glow": secondary, "fx": ["goal_breathe", "edge_bloom"]},
            "*beacon*": {"fill": secondary_fill, "outline": secondary, "glow": secondary, "fx": ["goal_breathe", "edge_bloom"]},
            "*hazard*": {"fill": hot_fill, "outline": hot, "glow": hot, "fx": ["hazard_flash", "edge_bloom"]},
            "*enemy*": {"fill": hot_fill, "outline": hot, "glow": hot, "fx": ["danger_pulse", "edge_bloom"]},
            "*fire*": {"fill": hot, "outline": [255, 235, 140], "glow": hot, "fx": ["hazard_flash", "edge_bloom"]},
            "*wall*": {"fill": primary_fill, "outline": primary, "glow": primary, "fx": ["edge_bloom"]},
            "*floor*": {"fill": primary_fill, "outline": primary, "glow": primary, "fx": ["edge_bloom"]},
            "*platform*": {"fill": primary_fill, "outline": primary, "glow": primary, "fx": ["edge_bloom"]},
            "*ball*": {"fill": secondary, "outline": [245, 250, 255], "glow": secondary, "fx": ["edge_bloom"]},
            "*rock*": {"fill": primary_fill, "outline": primary, "glow": primary, "fx": ["edge_bloom"]},
            "*crystal*": {"fill": secondary, "outline": [245, 250, 255], "glow": secondary, "fx": ["twinkle", "edge_bloom"]},
            "*tree*": {"fill": secondary_fill, "outline": secondary, "trunk": primary_fill, "fx": ["edge_bloom"]},
            "*water*": {"fill": primary_fill, "outline": primary, "foam": secondary, "fx": ["edge_bloom"]},
        }
        for key, value in additions.items():
            skins.setdefault(key, value)
        self._retint_existing_variant_skins(skins, style, primary, secondary, hot)
        return skins

    def _retint_existing_variant_skins(
        self,
        skins: dict[str, object],
        style: dict[str, object],
        primary: object,
        secondary: object,
        hot: object,
    ) -> None:
        label = str(style.get("label", "")).lower()
        mood = str(style.get("mood", "")).lower()

        def update_skin(key: str, **updates: object) -> None:
            current = skins.get(key)
            skin = dict(current) if isinstance(current, dict) else {}
            skin.update(updates)
            skins[key] = skin

        if "water" in mood or label in {"lagoon", "moonpool", "stormwater"}:
            if label == "moonpool":
                water_fill = [20, 48, 125]
                current_fill = [76, 118, 255]
                shore_fill = [44, 48, 76]
                shore_outline = [210, 228, 255]
                beacon_fill = [236, 248, 255]
                agent_fill = [248, 252, 255]
            elif label == "stormwater":
                water_fill = [28, 74, 86]
                current_fill = [84, 154, 172]
                shore_fill = [72, 82, 76]
                shore_outline = [255, 226, 72]
                beacon_fill = [255, 226, 72]
                agent_fill = [255, 245, 190]
            else:
                water_fill = [15, 168, 198]
                current_fill = [64, 232, 235]
                shore_fill = [224, 176, 92]
                shore_outline = [255, 218, 128]
                beacon_fill = [255, 168, 92]
                agent_fill = [255, 250, 215]
            for key in ("water_pool", "*water*", "swim_zone"):
                update_skin(
                    key,
                    fill=water_fill,
                    outline=primary,
                    foam=secondary,
                    glow=primary,
                    alpha=0.64,
                    fx=["surface_wave", "caustics", "bubbles", "edge_bloom"],
                )
            for key in ("buoyancy_current", "water_current", "*current*"):
                update_skin(
                    key,
                    fill=current_fill,
                    outline=secondary,
                    glow=primary,
                    alpha=0.30,
                    fx=["current_arrows", "bubbles", "edge_bloom"],
                )
            for key in ("left_bank", "right_bank", "*bank*", "left_jump_lip", "right_climb_lip"):
                update_skin(
                    key,
                    fill=shore_fill,
                    outline=shore_outline,
                    glow=secondary,
                    fx=["shore_grain", "edge_bloom"],
                )
            update_skin(
                "beacon",
                fill=beacon_fill,
                outline=[245, 250, 255],
                glow=secondary,
                fx=["goal_breathe", "beacon_ray", "edge_bloom"],
            )
            update_skin(
                "agent",
                fill=agent_fill,
                outline=[255, 255, 255],
                glow=primary,
                fx=["swim_stroke", "bubble_trail", "edge_bloom"],
            )
        if label in {"court", "arena", "street"} or mood in {"retro_arcade", "cyber_grid", "industrial_noir"}:
            ball_fill = secondary if label == "arena" else primary
            court_fill = [max(0, int(c * 0.24)) for c in primary]
            if label == "street":
                court_fill = [40, 38, 34]
            update_skin("basketball", fill=ball_fill, outline=[255, 246, 224], glow=hot, fx=["edge_bloom", "impact_ripples"])
            update_skin("*ball*", fill=ball_fill, outline=[255, 246, 224], glow=hot, fx=["edge_bloom", "impact_ripples"])
            update_skin("hoop", fill=hot, outline=[255, 248, 225], glow=hot, fx=["goal_breathe", "edge_bloom"])
            update_skin("*hoop*", fill=hot, outline=[255, 248, 225], glow=hot, fx=["goal_breathe", "edge_bloom"])
            update_skin("backboard", fill=[210, 225, 235], outline=secondary, glow=secondary, fx=["edge_bloom"])
            update_skin("court", fill=court_fill, outline=primary, glow=primary, fx=["edge_bloom", "material_stripes"])
            update_skin("*goal*", fill=[max(0, int(c * 0.32)) for c in secondary], outline=secondary, glow=secondary, fx=["goal_breathe", "edge_bloom"])

        if label in {"workshop", "clockwork", "electric"} or "mechanical" in mood or "clockwork" in mood or "electric" in mood:
            metal_fill = [max(0, int(c * 0.28)) for c in primary]
            brass_fill = [max(0, int(c * 0.42)) for c in secondary]
            weight_fill = [96, 104, 112] if label != "electric" else [64, 92, 128]
            update_skin("seesaw_plank", fill=metal_fill, outline=primary, glow=primary, fx=["edge_bloom", "material_stripes"])
            update_skin("*plank*", fill=metal_fill, outline=primary, glow=primary, fx=["edge_bloom", "material_stripes"])
            update_skin("pivot", fill=brass_fill, outline=secondary, glow=secondary, fx=["edge_bloom"])
            update_skin("*pivot*", fill=brass_fill, outline=secondary, glow=secondary, fx=["edge_bloom"])
            update_skin("heavy_ball", fill=weight_fill, outline=[236, 244, 248], glow=hot, fx=["edge_bloom", "impact_ripples"])
            update_skin("*weight*", fill=weight_fill, outline=[236, 244, 248], glow=hot, fx=["edge_bloom", "impact_ripples"])
            update_skin("launch_pad", fill=[max(0, int(c * 0.34)) for c in hot], outline=hot, glow=hot, fx=["field_pulse", "edge_bloom"])
            update_skin("high_goal", fill=[max(0, int(c * 0.36)) for c in secondary], outline=secondary, glow=secondary, fx=["goal_breathe", "edge_bloom"])

        if label in {"canopy", "twilight", "grove"} or "forest" in mood or "woods" in mood or "grove" in mood:
            tree_fill = [max(0, int(c * 0.36)) for c in primary if isinstance(c, int)]
            if len(tree_fill) < 3:
                tree_fill = [42, 104, 56]
            if label == "twilight":
                tree_fill = [50, 44, 82]
            update_skin("pine_tree", fill=tree_fill, outline=secondary, trunk=[88, 58, 38], glow=primary, fx=["edge_bloom"])
            update_skin("*tree*", fill=tree_fill, outline=secondary, trunk=[88, 58, 38], glow=primary, fx=["edge_bloom"])
            update_skin("bear_chaser", fill=[88, 52, 36] if label != "twilight" else [118, 64, 90], outline=hot, glow=hot, fx=["danger_pulse", "edge_bloom"])
            update_skin("*bear*", fill=[88, 52, 36] if label != "twilight" else [118, 64, 90], outline=hot, glow=hot, fx=["danger_pulse", "edge_bloom"])
            update_skin("cabin_exit", fill=[94, 58, 36], outline=secondary, glow=secondary, fx=["goal_breathe", "edge_bloom"])
            update_skin("mossy_escape_path", fill=[max(0, int(c * 0.28)) for c in secondary], outline=primary, glow=primary, fx=["edge_bloom"])

        if label in {"salvage", "aurora", "asteroid"} or "space" in mood or "asteroid" in mood:
            update_skin("*crystal*", fill=secondary, outline=[245, 250, 255], glow=secondary, fx=["twinkle", "edge_bloom"])
            update_skin("*gem*", fill=secondary, outline=[245, 250, 255], glow=secondary, fx=["twinkle", "edge_bloom"])
            update_skin("*asteroid*", fill=[92, 96, 112] if label == "asteroid" else [74, 78, 98], outline=primary, glow=primary, fx=["cracked_surface", "edge_bloom"])
            update_skin("*rock*", fill=[92, 96, 112], outline=primary, glow=primary, fx=["cracked_surface", "edge_bloom"])
            update_skin("*wall*", fill=[max(0, int(c * 0.22)) for c in primary], outline=primary, glow=primary, fx=["edge_bloom"])

        if label in {"arcade", "hologrid", "crystal"}:
            wall_fill = [max(0, int(c * 0.20)) for c in primary]
            exit_fill = [max(0, int(c * 0.38)) for c in secondary]
            update_skin("*wall*", fill=wall_fill, outline=primary, glow=primary, fx=["edge_bloom"])
            update_skin("*maze*", fill=wall_fill, outline=primary, glow=primary, fx=["edge_bloom"])
            update_skin("*exit*", fill=exit_fill, outline=secondary, glow=secondary, fx=["goal_breathe", "edge_bloom"])
            update_skin("*goal*", fill=exit_fill, outline=secondary, glow=secondary, fx=["goal_breathe", "edge_bloom"])

    def _load_json_file(self, path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_json_file(self, path: Path, payload: dict[str, object]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _update_variant_previews(self) -> None:
        for slot in self.variant_slots:
            if slot.preview_env is None or slot.autoplay is None:
                continue
            try:
                for _ in range(MINI_PREVIEW_STEPS_PER_FRAME):
                    slot.autoplay.step(slot.preview_env)
                agent = slot.preview_env.get_agent_record()
                if agent is not None:
                    slot.trail.append((float(agent.body.position.x), float(agent.body.position.y)))
                    del slot.trail[:-64]
            except Exception as exc:
                slot.autoplay = None
                self._add_console(f"Mini autoplay stopped for V{slot.index}: {type(exc).__name__}: {exc}", RED)

    def _draw(self) -> None:
        self.screen.fill(BACKGROUND)
        self._draw_grid()
        self._draw_header()
        self._draw_prompt_panel()
        self._draw_console_panel()
        self._draw_side_panel()
        if self.variant_slots:
            self._draw_variant_gallery()
        else:
            self._draw_examples()
        if self.prompt_focus_overlay:
            self._draw_prompt_focus_overlay()

    def _draw_grid(self) -> None:
        for x in range(0, self.width, 32):
            pygame.draw.line(self.screen, GRID, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, 32):
            pygame.draw.line(self.screen, GRID, (0, y), (self.width, y), 1)

    def _draw_header(self) -> None:
        title = self.title_font.render("HARNESS ALPHA // WORLD FACTORY", True, TEXT)
        self.screen.blit(title, (34, 26))
        subtitle = self.small_font.render(
            "Prompt-to-Pymunk environments with code-level objectives, reflexion repair, and deterministic validation.",
            True,
            MUTED,
        )
        self.screen.blit(subtitle, (36, 64))
        badge = pygame.Rect(self.width - 294, 28, 250, 42)
        pygame.draw.rect(self.screen, PANEL_DARK, badge, border_radius=7)
        pygame.draw.rect(self.screen, self.status_color, badge, 1, border_radius=7)
        text = self.label_font.render(self.status_text, True, self.status_color)
        self.screen.blit(text, (badge.x + 18, badge.y + 10))

    def _draw_prompt_panel(self) -> None:
        panel = pygame.Rect(28, 100, 860, 128)
        self._panel(panel)
        label = self.label_font.render("TEXT COMMAND", True, CYAN)
        self.screen.blit(label, (44, 116))
        hint = self.small_font.render("Click the box and type/paste your own prompt. Ctrl+V pastes; Ctrl+A replaces; Enter generates.", True, MUTED)
        self.screen.blit(hint, (196, 121))
        clear_rect = self._prompt_clear_rect()
        clear_color = CYAN if not self._busy() else (63, 72, 76)
        pygame.draw.rect(self.screen, (9, 22, 25), clear_rect, border_radius=5)
        pygame.draw.rect(self.screen, clear_color, clear_rect, 1, border_radius=5)
        clear_label = "CLEAR / RESET" if self.generate_locked or self.state != "idle" else "CLEAR PROMPT"
        clear_text = self.small_font.render(clear_label, True, clear_color)
        self.screen.blit(
            clear_text,
            (
                clear_rect.centerx - clear_text.get_width() // 2,
                clear_rect.centery - clear_text.get_height() // 2,
            ),
        )
        input_rect = self._input_rect()
        fill = (11, 24, 28) if self.input_active else (8, 15, 18)
        pygame.draw.rect(self.screen, fill, input_rect, border_radius=7)
        pygame.draw.rect(self.screen, CYAN if self.input_active else BORDER, input_rect, 1, border_radius=7)
        if self.prompt:
            prompt_text = self._fit_text_tail(self.prompt, self.font, input_rect.width - 28) if self.input_active else self._fit_text(self.prompt, self.font, input_rect.width - 28)
            text_color = TEXT
        else:
            prompt_text = "Type a world prompt here..."
            text_color = MUTED
        if self.prompt_selected and self.prompt:
            selection = pygame.Rect(input_rect.x + 9, input_rect.y + 10, input_rect.width - 18, input_rect.height - 20)
            pygame.draw.rect(self.screen, (35, 89, 98), selection, border_radius=4)
        cursor = "_" if self.input_active and pygame.time.get_ticks() % 900 < 450 else ""
        rendered = self.font.render(prompt_text + cursor, True, text_color)
        self.screen.blit(rendered, (input_rect.x + 14, input_rect.y + 15))

    def _draw_console_panel(self) -> None:
        panel = self._console_panel_rect()
        self._panel(panel)
        title = self.label_font.render("LIVE GENERATION TRACE", True, TEXT)
        self.screen.blit(title, (44, 262))
        self._draw_pipeline_status(panel)
        view = self._console_view_rect()
        pygame.draw.rect(self.screen, (14, 18, 21), view, border_radius=4)
        pygame.draw.rect(self.screen, (38, 57, 62), view, 1, border_radius=4)

        max_scroll = self._console_max_scroll()
        self.console_scroll = min(max(self.console_scroll, 0), max_scroll)
        previous_clip = self.screen.get_clip()
        self.screen.set_clip(view)
        y = view.y + 8 - self.console_scroll
        for item in self.console:
            if item.style == "section":
                font = self.label_font
                if view.y - 30 <= y <= view.bottom + 10:
                    pygame.draw.line(self.screen, item.color, (view.x + 8, y - 2), (view.right - 12, y - 2), 1)
                y += 4
            elif item.style == "step":
                font = self.font
            else:
                font = self.small_font
            line_height = self._console_line_height(item)
            if view.y - line_height <= y <= view.bottom:
                rendered = font.render(self._fit_text(item.text, font, view.width - 28), True, item.color)
                self.screen.blit(rendered, (view.x + 12, y))
            y += line_height
        self.screen.set_clip(previous_clip)

        if max_scroll > 0:
            scroll_h = max(26, int(view.height * view.height / max(self._console_content_height(), 1)))
            scroll_y = view.y + int((view.height - scroll_h) * self.console_scroll / max_scroll)
            pygame.draw.rect(self.screen, (31, 43, 47), (view.right - 8, view.y + 4, 4, view.height - 8), border_radius=2)
            pygame.draw.rect(self.screen, CYAN if self.console_follow else MUTED, (view.right - 9, scroll_y, 5, scroll_h), border_radius=3)
            hint = self.small_font.render("scroll", True, MUTED)
            self.screen.blit(hint, (view.right - hint.get_width() - 16, view.y - 18))

    def _draw_prompt_focus_overlay(self) -> None:
        shade = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 82))
        self.screen.blit(shade, (0, 0))

        rect = self._prompt_overlay_rect()
        overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        overlay.fill((2, 4, 6, 220))
        self.screen.blit(overlay, rect.topleft)
        pygame.draw.rect(self.screen, (236, 240, 242), rect, 2, border_radius=14)
        pygame.draw.rect(self.screen, (90, 112, 118), rect.inflate(-10, -10), 1, border_radius=10)

        eyebrow = self.small_font.render("PROMPT COMPOSER // WORLD FACTORY INPUT", True, (184, 196, 200))
        self.screen.blit(eyebrow, (rect.x + 34, rect.y + 28))

        prompt = self.prompt if self.prompt else "Type a world prompt..."
        text_color = (246, 248, 248) if self.prompt else (142, 154, 158)
        cursor = "|" if self.input_active and pygame.time.get_ticks() % 900 < 450 else ""
        lines = self._wrap_text_pixels(prompt + cursor, self.prompt_overlay_font, rect.width - 88, max_lines=7)
        y = rect.y + 90
        for line in lines:
            rendered = self.prompt_overlay_font.render(line, True, text_color)
            self.screen.blit(rendered, (rect.x + 44, y))
            y += 42

        hint = self.small_font.render("Click anywhere to dismiss. Press Enter or use Generate to launch.", True, (166, 176, 180))
        self.screen.blit(hint, (rect.x + 40, rect.bottom - 102))

        button = self._prompt_overlay_generate_rect()
        enabled = self._can_generate()
        fill = (232, 234, 236) if enabled else (108, 112, 116)
        outline = (255, 255, 255) if enabled else (132, 136, 140)
        pygame.draw.rect(self.screen, fill, button, border_radius=8)
        pygame.draw.rect(self.screen, outline, button, 2, border_radius=8)
        label = self.label_font.render("GENERATE", True, (8, 9, 10) if enabled else (42, 44, 46))
        self.screen.blit(label, (button.centerx - label.get_width() // 2, button.centery - label.get_height() // 2))

    def _draw_pipeline_status(self, panel: pygame.Rect) -> None:
        stage_rect = pygame.Rect(panel.x + 16, panel.y + 48, panel.width - 32, 70)
        pygame.draw.rect(self.screen, (7, 17, 20), stage_rect, border_radius=7)
        pygame.draw.rect(self.screen, (35, 93, 100), stage_rect, 1, border_radius=7)

        pulse = 0.55 + 0.45 * math.sin(pygame.time.get_ticks() * 0.006)
        stage_color = self.status_color if self._busy() else (self.status_color if self.state in {"success", "failed"} else CYAN)
        marker = pygame.Rect(stage_rect.x + 14, stage_rect.y + 14, 12, 12)
        pygame.draw.rect(self.screen, stage_color, marker, border_radius=3)
        if self._busy():
            glow = pygame.Surface((36, 36), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*stage_color, int(70 + 60 * pulse)), (18, 18), 14)
            self.screen.blit(glow, (marker.centerx - 18, marker.centery - 18), special_flags=pygame.BLEND_ADD)

        stage = self.label_font.render(self.pipeline_stage.upper(), True, stage_color)
        self.screen.blit(stage, (stage_rect.x + 38, stage_rect.y + 8))
        detail = self.small_font.render(self._fit_text(self.pipeline_detail, self.small_font, stage_rect.width - 52), True, MUTED)
        self.screen.blit(detail, (stage_rect.x + 38, stage_rect.y + 34))

        bar = pygame.Rect(stage_rect.x + 14, stage_rect.y + 54, stage_rect.width - 28, 7)
        pygame.draw.rect(self.screen, (18, 32, 36), bar, border_radius=4)
        progress = self._attempt_progress()
        fill = pygame.Rect(bar.x, bar.y, max(8 if self._busy() else 0, int(bar.width * progress)), bar.height)
        pygame.draw.rect(self.screen, stage_color, fill, border_radius=4)
        if self._busy():
            shine_x = bar.x + int((pygame.time.get_ticks() * 0.16) % max(bar.width, 1))
            pygame.draw.line(self.screen, TEXT, (shine_x, bar.y), (shine_x, bar.bottom), 1)

        attempt = self._attempt_label()
        attempt_text = self.small_font.render(attempt, True, MUTED)
        self.screen.blit(attempt_text, (stage_rect.right - attempt_text.get_width() - 14, stage_rect.y + 10))

    def _attempt_progress(self) -> float:
        if self.pending_demo_load is not None and self.pending_demo_load_at_ms is not None:
            elapsed = pygame.time.get_ticks() - self.pending_demo_load_at_ms
            return min(0.96, max(0.08, elapsed / 3000.0))
        total = max(1, self.max_seeds * self.max_repairs)
        if self.state == "success":
            return 1.0
        if self.state == "failed":
            return 1.0
        if not self._busy():
            return 0.0
        current = self.completed_attempts
        if self.current_seed is not None and self.current_repair is not None:
            current = (self.current_seed - 1) * self.max_repairs + self.current_repair - 0.35
        return max(0.04, min(0.96, current / total))

    def _attempt_label(self) -> str:
        if self.pending_demo_load is not None:
            return "showcase cache"
        if self.current_seed is None or self.current_repair is None:
            return f"0/{self.max_seeds * self.max_repairs} attempts"
        doneish = (self.current_seed - 1) * self.max_repairs + self.current_repair
        return f"seed {self.current_seed} repair {self.current_repair}/{self.max_repairs}  |  {doneish}/{self.max_seeds * self.max_repairs}"

    def _draw_side_panel(self) -> None:
        panel = pygame.Rect(916, 100, 370, 636)
        self._panel(panel)
        title = self.label_font.render("CONTROL ROOM", True, TEXT)
        self.screen.blit(title, (936, 118))
        self._draw_metric("Status", self.status_text, self.status_color, 158)
        self._draw_metric("Tier", str(self.tier if self.tier is not None else "-"), GREEN if self.tier == 5 else MUTED, 210)
        self._draw_metric("Accepted", "yes" if self.accepted else "no", GREEN if self.accepted else MUTED, 262)
        self._draw_metric("Env", self.verified_env.name if self.verified_env else "-", CYAN if self.verified_env else MUTED, 314)
        godot_label, godot_color = self._godot_export_status()
        self._draw_metric("Godot Export", godot_label, godot_color, 366)
        mode_label = "FAST // PARALLEL" if self.execution_mode == "fast" else "NORMAL // SEQUENTIAL"
        mode_color = GREEN if self.execution_mode == "fast" else MUTED
        self._draw_metric("Mode", mode_label, mode_color, 410)
        for button in self._buttons():
            self._draw_button(button)

    def _godot_export_status(self) -> tuple[str, tuple[int, int, int]]:
        if self._busy():
            return ("building..." if self.verified_env else "waiting", YELLOW if self.verified_env else MUTED)
        if self._world_schema_path() is not None:
            return ("ready", GREEN)
        if self.verified_env is not None:
            return ("not exported", RED)
        return ("-", MUTED)

    def _draw_metric(self, label: str, value: str, color: tuple[int, int, int], y: int) -> None:
        label_surf = self.small_font.render(label.upper(), True, MUTED)
        self.screen.blit(label_surf, (938, y))
        value_surf = self.font.render(self._fit_text(value, self.font, 305), True, color)
        self.screen.blit(value_surf, (938, y + 18))

    def _draw_examples(self) -> None:
        panel = self._examples_panel_rect()
        self._panel(panel)
        title = self.example_title_font.render("INSTANT SAVED SHOWCASE", True, TEXT)
        self.screen.blit(title, (44, 626))
        pill = pygame.Rect(300, 622, 218, 25)
        pygame.draw.rect(self.screen, (235, 242, 244), pill, border_radius=12)
        pill_text = self.small_font.render("NO WAIT // PREBUILT", True, (16, 24, 28))
        self.screen.blit(pill_text, (pill.centerx - pill_text.get_width() // 2, pill.centery - pill_text.get_height() // 2))
        hint = self.small_font.render("These example prompts load much faster than fresh generation. Scroll, click one, then Watch AI Solve or Play in Godot.", True, (178, 194, 198))
        self.screen.blit(hint, (44, 653))
        view = self._examples_view_rect()
        old_clip = self.screen.get_clip()
        self.screen.set_clip(view)
        for demo, rect in self._example_card_rects():
            env_path = Path(str(demo.get("env_path") or ""))
            export_dir = Path(str(demo.get("export_dir") or ""))
            ready = env_path.exists()
            exported = (export_dir / "world_schema.json").exists()
            fill = (12, 18, 22) if ready else (22, 14, 18)
            border = (104, 118, 124) if ready and exported else (70, 78, 82)
            pygame.draw.rect(self.screen, fill, rect, border_radius=4)
            pygame.draw.rect(self.screen, border, rect, 1, border_radius=4)
            divider_x = rect.x + 154
            pygame.draw.line(self.screen, (58, 67, 72), (divider_x, rect.y + 8), (divider_x, rect.bottom - 8), 1)
            label = str(demo.get("title", "Saved Demo")).upper()
            self.screen.blit(self.example_font.render(label, True, TEXT if ready else RED), (rect.x + 12, rect.y + 9))
            tier_text = self.small_font.render(f"TIER {demo.get('tier', 5)}", True, (192, 202, 205))
            self.screen.blit(tier_text, (rect.x + 12, rect.y + 31))
            prompt = str(demo.get("prompt") or "")
            rendered = self.example_font.render(self._fit_text(prompt, self.example_font, rect.width - 184), True, TEXT if ready else MUTED)
            self.screen.blit(rendered, (divider_x + 18, rect.y + 18))
        self.screen.set_clip(old_clip)
        content_height = len(SAVED_DEMOS) * EXAMPLE_CARD_HEIGHT
        if content_height > view.height:
            scroll_h = max(24, int(view.height * view.height / content_height))
            max_scroll = max(1, content_height - view.height)
            scroll_y = view.y + int((view.height - scroll_h) * self.example_scroll / max_scroll)
            track = pygame.Rect(view.right - 8, view.y, 5, view.height)
            pygame.draw.rect(self.screen, (34, 45, 50), track, border_radius=3)
            bar = pygame.Rect(view.right - 8, scroll_y, 5, scroll_h)
            pygame.draw.rect(self.screen, (236, 244, 245), bar, border_radius=3)

    def _examples_panel_rect(self) -> pygame.Rect:
        return pygame.Rect(28, 616, 900, 208)

    def _examples_view_rect(self) -> pygame.Rect:
        return pygame.Rect(42, 684, 866, 128)

    def _example_card_rects(self) -> list[tuple[dict[str, object], pygame.Rect]]:
        view = self._examples_view_rect()
        rects: list[tuple[dict[str, object], pygame.Rect]] = []
        for index, demo in enumerate(SAVED_DEMOS):
            y = view.y + index * EXAMPLE_CARD_HEIGHT - self.example_scroll
            rect = pygame.Rect(view.x, y, view.width - 14, 46)
            if rect.bottom >= view.y and rect.y <= view.bottom:
                rects.append((demo, rect))
        return rects

    def _draw_variant_gallery(self) -> None:
        panel = pygame.Rect(28, 676, 1258, 148)
        self._panel(panel)
        title = self.label_font.render("MULTIVERSE VARIANTS", True, TEXT)
        self.screen.blit(title, (44, 690))
        subtitle = self.small_font.render(
            "Fast forks reuse the accepted physics/objective and vary presentation, palette, and runtime export.",
            True,
            MUTED,
        )
        self.screen.blit(subtitle, (270, 695))
        card_width = 394
        card_height = 104
        for index, slot in enumerate(self.variant_slots):
            rect = pygame.Rect(44 + index * (card_width + 17), 714, card_width, card_height)
            self._draw_variant_card(rect, slot)

    def _draw_variant_card(self, rect: pygame.Rect, slot: VariantSlot) -> None:
        color = self._variant_color(slot)
        pygame.draw.rect(self.screen, (8, 16, 19), rect, border_radius=7)
        pygame.draw.rect(self.screen, color, rect, 1, border_radius=7)
        header = f"V{slot.index} // {slot.label}"
        self.screen.blit(self.small_font.render(header, True, color), (rect.x + 10, rect.y + 8))
        status = slot.status.upper()
        if slot.tier is not None:
            status += f" / TIER {slot.tier}"
        self.screen.blit(self.small_font.render(status, True, color), (rect.x + 10, rect.y + 26))
        if slot.preview_env is not None:
            preview = pygame.Rect(rect.x + 196, rect.y + 8, rect.width - 207, rect.height - 16)
            self._draw_preview_world(preview, slot)
            if slot.autoplay is not None:
                pulse = 90 + int(70 * abs((pygame.time.get_ticks() % 1200) / 600.0 - 1.0))
                pygame.draw.rect(self.screen, (*GREEN, pulse), preview, 2, border_radius=5)
                label = f"AI: {slot.autoplay.label}"
                self.screen.blit(self.small_font.render(label, True, GREEN), (rect.x + 10, rect.y + 68))
        else:
            message = {
                "pending": "waiting",
                "running": "generating...",
                "failed": "no verified preview",
            }.get(slot.status, "preview unavailable")
            self.screen.blit(self.small_font.render(message, True, MUTED), (rect.x + 10, rect.y + 49))
        if slot.env_path is not None:
            filename = self._fit_text(slot.env_path.name, self.small_font, 180)
            self.screen.blit(self.small_font.render(filename, True, MUTED), (rect.x + 10, rect.y + 49))

    def _draw_preview_world(self, rect: pygame.Rect, slot: VariantSlot) -> None:
        env = slot.preview_env
        if env is None:
            return
        grammar = slot.grammar or infer_visual_grammar(env)
        self._draw_preview_background(rect, grammar)
        pygame.draw.rect(self.screen, grammar.primary, rect, 1, border_radius=5)
        scale = min(rect.width / max(float(env.width), 1.0), rect.height / max(float(env.height), 1.0))
        left = rect.x + (rect.width - env.width * scale) * 0.5
        top = rect.y + (rect.height - env.height * scale) * 0.5

        def point(world: pymunk.Vec2d | tuple[float, float]) -> tuple[int, int]:
            x = float(world.x) if hasattr(world, "x") else float(world[0])
            y = float(world.y) if hasattr(world, "y") else float(world[1])
            return int(left + x * scale), int(top + (env.height - y) * scale)

        if slot.autoplay is not None:
            self._draw_preview_target_line(rect, slot, point, grammar)

        if len(slot.trail) >= 2:
            trail_points = [point(item) for item in slot.trail]
            for index in range(1, len(trail_points)):
                alpha = max(55, int(225 * index / len(trail_points)))
                trail_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                pygame.draw.line(trail_surface, (*grammar.primary, alpha), trail_points[index - 1], trail_points[index], 3)
                self.screen.blit(trail_surface, (0, 0), special_flags=pygame.BLEND_ADD)

        drawn_agents: set[str] = set()
        for shape in env.space.shapes:
            role = str(getattr(shape, "harness_role", "") or "")
            object_name = getattr(shape, "harness_object_name", None)
            record = env._objects.get(object_name) if object_name else None
            color = (
                color_for_record(record, grammar, sensor=bool(getattr(shape, "sensor", False)))
                if record is not None
                else self._preview_color(role, bool(getattr(shape, "sensor", False)))
            )
            if isinstance(shape, pymunk.Circle):
                center = point(shape.body.local_to_world(shape.offset))
                radius = max(2, int(float(shape.radius) * scale))
                if role.lower() == "agent":
                    if record is not None and record.name not in drawn_agents:
                        self._draw_preview_agent_avatar(rect, env, record, point, scale, grammar)
                        drawn_agents.add(record.name)
                    pygame.draw.circle(self.screen, (45, 88, 102), center, radius, 1)
                    continue
                pygame.draw.circle(self.screen, color, center, radius)
            elif isinstance(shape, pymunk.Segment):
                start = point(shape.body.local_to_world(shape.a))
                end = point(shape.body.local_to_world(shape.b))
                pygame.draw.line(self.screen, color, start, end, max(1, int(float(shape.radius) * scale * 2.0)))
            elif isinstance(shape, pymunk.Poly):
                points = [point(shape.body.local_to_world(vertex)) for vertex in shape.get_vertices()]
                if len(points) >= 3:
                    pygame.draw.polygon(self.screen, color, points, 0)

    def _draw_preview_target_line(self, rect, slot: VariantSlot, point, grammar: VisualGrammar) -> None:
        env = slot.preview_env
        if env is None or slot.autoplay is None or slot.autoplay.index >= len(slot.autoplay.subgoals):
            return
        agent = env.get_agent_record()
        if agent is None:
            return
        subgoal = slot.autoplay.subgoals[slot.autoplay.index]
        target_name = (
            subgoal.get("target")
            or subgoal.get("region")
            or subgoal.get("object")
            or subgoal.get("trigger")
        )
        if not isinstance(target_name, str):
            return
        try:
            target = env.get_object(target_name)
        except Exception:
            target = None
        if target is None:
            return
        start = point(agent.body.position)
        end = point(target.body.position)
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        for segment_index in range(0, 12, 2):
            t0 = segment_index / 12
            t1 = min(1.0, (segment_index + 1) / 12)
            p0 = (int(start[0] + (end[0] - start[0]) * t0), int(start[1] + (end[1] - start[1]) * t0))
            p1 = (int(start[0] + (end[0] - start[0]) * t1), int(start[1] + (end[1] - start[1]) * t1))
            pygame.draw.line(overlay, (*grammar.secondary, 95), p0, p1, 1)
        pygame.draw.circle(overlay, (*grammar.secondary, 135), end, 5, 1)
        self.screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_ADD)

    def _draw_preview_background(self, rect: pygame.Rect, grammar: VisualGrammar) -> None:
        pygame.draw.rect(self.screen, grammar.background_color, rect, border_radius=5)
        clip_before = self.screen.get_clip()
        self.screen.set_clip(rect)
        tick = pygame.time.get_ticks()
        recipe = grammar.visual_recipe
        if recipe:
            layers = recipe.get("background_layers") if isinstance(recipe, dict) else []
            layer_text = json.dumps(layers, default=str).lower() if isinstance(layers, list) else ""
            if any(token in layer_text for token in ("ember", "molten", "basalt", "heat")):
                for x in range(rect.x - rect.height, rect.right, 52):
                    pygame.draw.line(self.screen, self._dim(grammar.hot, 0.75), (x, rect.bottom), (x + rect.height, rect.y), 2)
                for index in range(34):
                    x = rect.x + ((index * 37 + grammar.seed) % max(rect.width, 1))
                    y = rect.y + ((index * 19 - tick // 45) % max(rect.height, 1))
                    pygame.draw.circle(self.screen, grammar.hot, (x, y), 1)
            elif any(token in layer_text for token in ("star", "nebula", "orbit")):
                for index in range(46):
                    x = rect.x + ((index * 43 + grammar.seed + tick // 70) % max(rect.width, 1))
                    y = rect.y + ((index * 23 + grammar.seed // 7) % max(rect.height, 1))
                    pygame.draw.circle(self.screen, grammar.primary if index % 5 else grammar.secondary, (x, y), 1)
            elif any(token in layer_text for token in ("arcade", "crt", "pixel")):
                for y in range(rect.y + (tick // 120) % 8, rect.bottom, 8):
                    pygame.draw.line(self.screen, self._dim(grammar.secondary, 0.55), (rect.x, y), (rect.right, y), 1)
                for x in range(rect.x, rect.right, 28):
                    pygame.draw.line(self.screen, self._dim(grammar.primary, 0.35), (x, rect.y), (x, rect.bottom), 1)
            else:
                for y in range(rect.y + 4 + (tick // 120) % 8, rect.bottom, 8):
                    pygame.draw.line(self.screen, self._dim(grammar.primary, 0.48), (rect.x, y), (rect.right, y), 1)
        elif grammar.background in {"parallax_starfield", "dense_starfield", "orbital_dust"}:
            for index in range(3):
                x = rect.x + ((index * 83 + grammar.seed + tick // 70) % max(rect.width, 1))
                y = rect.y + ((index * 41 + grammar.seed // 11) % max(rect.height, 1))
                pygame.draw.circle(self.screen, self._dim(grammar.secondary if index % 2 else grammar.primary, 0.55), (x, y), 28 + index * 12)
            for index in range(38):
                x = rect.x + ((index * 47 + grammar.seed + tick // 55) % max(rect.width, 1))
                y = rect.y + ((index * 29 + grammar.seed // 7) % max(rect.height, 1))
                color = grammar.secondary if index % 5 == 0 else grammar.primary
                pygame.draw.circle(self.screen, color, (x, y), 1)
        elif grammar.background in {"corn_rows", "organic_noise", "leaf_shadow"}:
            for x in range(rect.x - 40, rect.right + 40, 18):
                sway = int(math.sin(tick * 0.002 + x * 0.05) * 3)
                pygame.draw.line(self.screen, self._dim(grammar.secondary, 0.55), (x + sway, rect.y), (x - 18 + sway, rect.bottom), 1)
        elif grammar.background in {"hazard_warning", "ember_drift", "smoke_haze"}:
            for x in range(rect.x - rect.height, rect.right, 44):
                pygame.draw.line(self.screen, self._dim(grammar.hot, 0.62), (x, rect.bottom), (x + rect.height, rect.y), 2)
        else:
            for y in range(rect.y + 4 + (tick // 120) % 8, rect.bottom, 8):
                pygame.draw.line(self.screen, self._dim(grammar.primary, 0.48), (rect.x, y), (rect.right, y), 1)
        self.screen.set_clip(clip_before)

    def _draw_preview_agent_avatar(
        self,
        rect: pygame.Rect,
        env: BaseEnv,
        record,
        point,
        scale: float,
        grammar: VisualGrammar,
    ) -> None:
        body = record.body
        center = point(body.position)
        radius = max(4, int(_preview_agent_radius(record) * scale))
        unit = max(10, int(radius * 2.0))
        nearest = self._preview_nearest_dynamic(env, record)
        facing = 1
        if nearest is not None:
            facing = 1 if nearest.body.position.x >= body.position.x else -1
        elif abs(float(body.velocity.x)) > 5.0:
            facing = 1 if body.velocity.x >= 0 else -1
        speed = float(body.velocity.length)
        gravity = getattr(env.space, "gravity", pymunk.Vec2d(0, -981)).length
        pushing = nearest is not None and body.position.get_distance(nearest.body.position) < max(48.0, _preview_agent_radius(record) * 3.6)
        kicking = pushing and nearest is not None and self._preview_target_is_kickable(env, nearest)
        throwing = pushing and nearest is not None and self._preview_target_is_throwable(env, nearest)
        zero_g = gravity < 30.0
        jumping = not zero_g and float(body.velocity.y) > 95.0
        falling = not zero_g and float(body.velocity.y) < -95.0 and speed > 110.0
        pose = (
            "throw"
            if throwing
            else "kick"
            if kicking
            else "push"
            if pushing
            else "float"
            if zero_g
            else "jump"
            if jumping
            else "fall"
            if falling
            else "run"
            if speed > 45
            else "idle"
        )
        phase = pygame.time.get_ticks() * 0.01 + body.position.x * 0.02
        avatar = str(getattr(grammar, "agent_avatar", "human") or "human")
        if avatar != "human":
            _draw_preview_nonhuman_agent(self.screen, center, unit, facing, phase, pose, avatar, grammar)
            label = self.small_font.render("AI", True, grammar.primary)
            self.screen.blit(label, (center[0] + unit + 4, center[1] - unit - 7))
            return

        skeleton = _preview_avatar_points(center, unit, facing, phase, pose)
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        pygame.draw.ellipse(
            overlay,
            (*grammar.primary, 35),
            (center[0] - unit, center[1] + int(unit * 0.78), unit * 2, max(3, unit // 3)),
        )
        for a, b in _preview_limb_pairs():
            _draw_preview_round_limb(overlay, (*grammar.primary, 58), skeleton[a], skeleton[b], max(4, int(unit * 0.26)))
        pygame.draw.circle(overlay, (*grammar.primary, 70), skeleton["head"], max(5, int(unit * 0.44)), 3)
        self.screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_ADD)
        stroke = max(2, int(unit * 0.16))
        for a, b in _preview_limb_pairs():
            _draw_preview_round_limb(self.screen, grammar.primary, skeleton[a], skeleton[b], stroke)
        head_radius = max(4, int(unit * 0.38))
        pygame.draw.circle(self.screen, grammar.primary, skeleton["head"], head_radius)
        pygame.draw.circle(self.screen, _bright_preview(grammar.secondary), (skeleton["head"][0] + facing * max(1, head_radius // 4), skeleton["head"][1] - max(1, head_radius // 5)), max(1, head_radius // 5))
        if pose in {"kick", "throw"} and nearest is not None:
            foot = skeleton["foot_front"]
            target = point(nearest.body.position)
            source = skeleton["hand_front"] if pose == "throw" else foot
            pygame.draw.line(overlay, (255, 116, 46, 145), source, target, max(2, stroke))
            pygame.draw.circle(overlay, (255, 116, 46, 125), target, max(4, unit // 4), 2)
            pygame.draw.line(self.screen, (255, 218, 128), source, target, max(2, stroke // 2))
            self.screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_ADD)
        label = self.small_font.render("AI", True, grammar.primary)
        self.screen.blit(label, (center[0] + unit + 4, center[1] - unit - 7))

    def _preview_nearest_dynamic(self, env: BaseEnv, agent_record):
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
        return best if best_distance < max(70.0, _preview_agent_radius(agent_record) * 5.0) else None

    def _preview_target_is_kickable(self, env: BaseEnv, target_record) -> bool:
        try:
            objective = env.get_ground_truth().get("objective", {})
        except Exception:
            objective = {}
        text = " ".join(
            [
                str(getattr(target_record, "name", "") or ""),
                str(getattr(target_record, "role", "") or ""),
                json.dumps(getattr(target_record, "metadata", {}) or {}, default=str),
                json.dumps(objective, default=str),
            ]
        ).lower()
        return any(token in text for token in ("ball", "soccer", "puck", "kick", "strike", "shot", "shoot"))

    def _preview_target_is_throwable(self, env: BaseEnv, target_record) -> bool:
        try:
            objective = env.get_ground_truth().get("objective", {})
        except Exception:
            objective = {}
        text = " ".join(
            [
                str(getattr(target_record, "name", "") or ""),
                str(getattr(target_record, "role", "") or ""),
                json.dumps(getattr(target_record, "metadata", {}) or {}, default=str),
                json.dumps(objective, default=str),
            ]
        ).lower()
        return any(token in text for token in ("throw", "thrown", "toss", "hurl", "lob", "projectile"))

    def _preview_color(self, role: str, sensor: bool) -> tuple[int, int, int]:
        role = role.lower()
        if role == "agent":
            return BLUE
        if role in {"goal", "trigger", "region", "force_zone"} or sensor:
            return GREEN
        if role in {"hazard", "danger"}:
            return RED
        if role == "support":
            return (82, 105, 118)
        return (176, 83, 105)

    def _dim(self, color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
        factor = max(0.0, min(1.0, amount))
        return tuple(max(0, min(255, int(channel * factor))) for channel in color)

    def _variant_color(self, slot: VariantSlot) -> tuple[int, int, int]:
        if slot.status == "success":
            return GREEN
        if slot.status == "failed":
            return RED
        if slot.status == "running":
            return YELLOW
        return MUTED

    def _buttons(self) -> list[Button]:
        x = 936
        w = 330
        mode_label = "Mode: FAST Seed Race" if self.execution_mode == "fast" else "Mode: NORMAL Reflexion"
        generate_enabled = self._can_generate()
        generate_label = "Generate" if not self.generate_locked else "Generate Locked"
        buttons = [
            Button(pygame.Rect(x, 454, w, 34), mode_label, self.toggle_execution_mode, not self._busy(), GREEN if self.execution_mode == "fast" else MUTED),
            Button(pygame.Rect(x, 498, w, 38), generate_label, lambda: self.start_generation(self.prompt), generate_enabled, GREEN),
            Button(pygame.Rect(x, 546, w, 38), "Play in Godot", self.open_godot_runtime, self._world_schema_path() is not None and not self._busy(), GREEN),
            Button(pygame.Rect(x, 594, w, 38), "Watch AI Solve", lambda: self.open_visualizer(autoplay=True), self.verified_env is not None and self.accepted and self.tier == 5 and not self._busy(), PURPLE),
            Button(pygame.Rect(x, 642, w, 34), "Fork 3 Fast Variants", self.start_variants, self.verified_env is not None and self.accepted and not self._busy(), BLUE),
            Button(pygame.Rect(x, 686, 158, 32), "Cancel", self.stop_generation, self._busy(), RED),
            Button(pygame.Rect(x + 172, 686, 158, 32), "Clear / Reset", self.reset_dashboard, not self._busy(), CYAN if self.generate_locked else MUTED),
        ]
        return buttons

    def _draw_button(self, button: Button) -> None:
        color = button.accent if button.enabled else (63, 72, 76)
        fill = (10, 23, 26) if button.enabled else (22, 24, 26)
        pygame.draw.rect(self.screen, fill, button.rect, border_radius=7)
        pygame.draw.rect(self.screen, color, button.rect, 1, border_radius=7)
        text = self.font.render(button.label, True, color if button.enabled else MUTED)
        self.screen.blit(
            text,
            (
                button.rect.centerx - text.get_width() // 2,
                button.rect.centery - text.get_height() // 2,
            ),
        )

    def _panel(self, rect: pygame.Rect) -> None:
        shadow = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        shadow.fill(BLACK_ALPHA)
        self.screen.blit(shadow, (rect.x + 5, rect.y + 6))
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=8)
        pygame.draw.rect(self.screen, BORDER, rect, 1, border_radius=8)

    def _input_rect(self) -> pygame.Rect:
        return pygame.Rect(44, 150, 822, 54)

    def _console_panel_rect(self) -> pygame.Rect:
        return pygame.Rect(28, 246, 860, 362)

    def _console_view_rect(self) -> pygame.Rect:
        panel = self._console_panel_rect()
        return pygame.Rect(panel.x + 16, panel.y + 128, panel.width - 32, panel.height - 142)

    def _prompt_clear_rect(self) -> pygame.Rect:
        return pygame.Rect(720, 114, 146, 26)

    def _prompt_overlay_rect(self) -> pygame.Rect:
        width = min(1080, self.width - 110)
        height = min(500, self.height - 150)
        return pygame.Rect((self.width - width) // 2, 112, width, height)

    def _prompt_overlay_generate_rect(self) -> pygame.Rect:
        overlay = self._prompt_overlay_rect()
        button = pygame.Rect(0, 0, 210, 54)
        button.centerx = overlay.centerx
        button.bottom = overlay.bottom - 30
        return button

    def _busy(self) -> bool:
        return self.pending_demo_load is not None or (self.process is not None and self.process.poll() is None)

    def _can_generate(self) -> bool:
        return bool(self.prompt.strip()) and not self._busy() and not self.generate_locked

    def _add_console(self, text: str, color: tuple[int, int, int], style: str = "normal") -> None:
        for line in self._wrap_text(text, 118):
            self.console.append(ConsoleLine(line, color, style))
        if self.console_follow:
            self.console_scroll = self._console_max_scroll()

    def _console_line_height(self, item: ConsoleLine) -> int:
        if item.style == "section":
            return 27
        if item.style == "step":
            return 20
        if item.style == "result":
            return 18
        return 17

    def _console_content_height(self) -> int:
        if not self.console:
            return 0
        return 16 + sum(self._console_line_height(item) for item in self.console)

    def _console_max_scroll(self) -> int:
        return max(0, self._console_content_height() - self._console_view_rect().height)

    def _line_color(self, line: str) -> tuple[int, int, int]:
        upper = line.upper()
        if "SUCCESS" in upper or "VERIFIED_ENV" in upper or "WORLD_EXPORT" in upper or "ACCEPTED=TRUE" in upper:
            return GREEN
        if "FAILED" in upper or "ERROR" in upper or "POST-MORTEM" in upper:
            return RED
        if "AUTO-REPAIR" in upper or "REPAIR" in upper or "PIVOTING" in upper:
            return YELLOW
        if "REGENERATING" in upper or "VALIDATION" in upper or "VERIFICATION" in upper:
            return CYAN
        return TEXT

    def _line_style(self, line: str) -> str:
        upper = line.upper()
        if line.startswith("VERIFIED_ENV:") or line.startswith("FAILED_ENV:") or line.startswith("LOG_DIR:"):
            return "result"
        if "[SEED" in upper and "REGENERATING" in upper:
            return "step"
        if "RECOMMENDED HEALER ACTION" in upper:
            return "section"
        return "normal"

    def _wrap_text(self, text: str, limit: int) -> list[str]:
        if len(text) <= limit:
            return [text]
        lines = []
        remaining = text
        while len(remaining) > limit:
            split_at = remaining.rfind(" ", 0, limit)
            if split_at < 24:
                split_at = limit
            lines.append(remaining[:split_at].rstrip())
            remaining = "  " + remaining[split_at:].lstrip()
        lines.append(remaining)
        return lines

    def _wrap_text_pixels(
        self,
        text: str,
        font: pygame.font.Font,
        max_width: int,
        *,
        max_lines: int,
    ) -> list[str]:
        words = str(text).split(" ")
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if font.size(candidate)[0] <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = word
            while font.size(current)[0] > max_width and len(current) > 1:
                split = max(1, len(current) - 1)
                while split > 1 and font.size(current[:split])[0] > max_width:
                    split -= 1
                lines.append(current[:split])
                current = current[split:]
            if len(lines) >= max_lines:
                break
        if current and len(lines) < max_lines:
            lines.append(current)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
        if len(lines) == max_lines and words and not lines[-1].endswith("|"):
            while lines[-1] and font.size(lines[-1] + "...")[0] > max_width:
                lines[-1] = lines[-1][:-1]
            lines[-1] += "..."
        return lines or [""]

    def _fit_text(self, text: str, font: pygame.font.Font, max_width: int) -> str:
        if font.size(text)[0] <= max_width:
            return text
        ellipsis = "..."
        while text and font.size(text + ellipsis)[0] > max_width:
            text = text[:-1]
        return text + ellipsis

    def _fit_text_tail(self, text: str, font: pygame.font.Font, max_width: int) -> str:
        if font.size(text)[0] <= max_width:
            return text
        ellipsis = "..."
        while text and font.size(ellipsis + text)[0] > max_width:
            text = text[1:]
        return ellipsis + text


def _preview_agent_radius(record) -> float:
    radii: list[float] = []
    for shape in getattr(record, "shapes", ()) or ():
        if isinstance(shape, pymunk.Circle):
            radii.append(float(shape.radius))
        elif isinstance(shape, pymunk.Poly):
            bb = shape.bb
            radii.append(max(float(bb.right - bb.left), float(bb.top - bb.bottom)) * 0.42)
    return max(radii or [15.0])


def _draw_preview_nonhuman_agent(
    screen: pygame.Surface,
    center: tuple[int, int],
    unit: int,
    facing: int,
    phase: float,
    pose: str,
    style: str,
    grammar: VisualGrammar,
) -> None:
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    cx, cy = center
    if style == "ship":
        nose = (cx + facing * int(unit * 1.1), cy)
        tail_top = (cx - facing * int(unit * 0.78), cy - int(unit * 0.55))
        tail_bottom = (cx - facing * int(unit * 0.78), cy + int(unit * 0.55))
        flame = (cx - facing * int(unit * (1.12 + 0.2 * math.sin(phase * 3))), cy)
        pygame.draw.polygon(overlay, (*grammar.primary, 80), [nose, tail_top, flame, tail_bottom], max(2, unit // 3))
        screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_ADD)
        pygame.draw.polygon(screen, (7, 14, 21), [nose, tail_top, (cx, cy), tail_bottom])
        pygame.draw.polygon(screen, grammar.primary, [nose, tail_top, (cx, cy), tail_bottom], max(1, unit // 5))
        pygame.draw.polygon(overlay, (*grammar.hot, 105), [tail_top, flame, tail_bottom])
        screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_ADD)
    elif style == "robot":
        body = pygame.Rect(0, 0, int(unit * 1.0), int(unit * 1.1))
        body.center = (cx, cy - unit // 3)
        head = pygame.Rect(0, 0, int(unit * 0.82), int(unit * 0.55))
        head.center = (cx + facing * unit // 10, body.top - unit // 4)
        pygame.draw.rect(overlay, (*grammar.primary, 75), body.inflate(unit, unit), border_radius=max(3, unit // 4))
        screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_ADD)
        pygame.draw.rect(screen, (7, 14, 18), body, border_radius=max(2, unit // 6))
        pygame.draw.rect(screen, grammar.primary, body, max(1, unit // 6), border_radius=max(2, unit // 6))
        pygame.draw.rect(screen, grammar.secondary, head, max(1, unit // 7), border_radius=max(2, unit // 6))
        pygame.draw.circle(screen, grammar.secondary, (head.centerx + facing * unit // 8, head.centery), max(1, unit // 8))
    elif style == "arcade_disc":
        radius = int(unit * 0.82)
        mouth = 0.36 + 0.18 * abs(math.sin(phase * 3))
        points = [center]
        for index in range(18):
            angle = -mouth + (2 * math.pi - 2 * mouth) * index / 17.0
            if facing < 0:
                angle = math.pi - angle
            points.append((int(cx + math.cos(angle) * radius), int(cy + math.sin(angle) * radius)))
        pygame.draw.circle(overlay, (*grammar.secondary, 88), center, int(radius * 1.22), 2)
        screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_ADD)
        pygame.draw.polygon(screen, grammar.secondary, points)
        pygame.draw.polygon(screen, grammar.primary, points, max(1, unit // 6))
    elif style in {"marble", "orb"}:
        radius = int(unit * 0.82)
        core = grammar.secondary if style == "orb" else grammar.primary
        pygame.draw.circle(overlay, (*core, 82), center, int(radius * 1.35), 2)
        screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_ADD)
        pygame.draw.circle(screen, tuple(max(0, int(c * 0.55)) for c in core), center, radius)
        pygame.draw.circle(screen, grammar.secondary, center, radius, max(1, unit // 6))
        pygame.draw.circle(screen, (238, 255, 255), (cx - radius // 3, cy - radius // 3), max(1, radius // 5))
    elif style == "drone":
        body = pygame.Rect(0, 0, int(unit * 1.05), int(unit * 0.44))
        body.center = center
        pygame.draw.rect(screen, (8, 16, 20), body, border_radius=max(2, unit // 5))
        pygame.draw.rect(screen, grammar.primary, body, max(1, unit // 7), border_radius=max(2, unit // 5))
        for side in (-1, 1):
            rotor = (cx + side * int(unit * 0.92), cy)
            pygame.draw.line(screen, grammar.primary, body.center, rotor, max(1, unit // 8))
            pygame.draw.circle(overlay, (*grammar.secondary, 70), rotor, max(3, unit // 4), 1)
        screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_ADD)
    else:
        body = pygame.Rect(0, 0, int(unit * 1.45), int(unit * 0.95))
        body.center = center
        pygame.draw.ellipse(overlay, (*grammar.secondary, 72), body.inflate(unit, unit // 2), 2)
        screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_ADD)
        pygame.draw.ellipse(screen, tuple(max(0, int(c * 0.65)) for c in grammar.secondary), body)
        pygame.draw.ellipse(screen, grammar.primary, body, max(1, unit // 6))
        pygame.draw.circle(screen, (6, 12, 14), (cx + facing * unit // 3, cy - unit // 8), max(1, unit // 7))


def _preview_limb_pairs() -> tuple[tuple[str, str], ...]:
    return (
        ("neck", "hip"),
        ("shoulder", "hand_front"),
        ("shoulder", "hand_back"),
        ("hip", "knee_front"),
        ("knee_front", "foot_front"),
        ("hip", "knee_back"),
        ("knee_back", "foot_back"),
    )


def _draw_preview_round_limb(
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


def _bright_preview(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(min(255, channel + 54) for channel in color)


def _preview_avatar_points(
    center: tuple[int, int],
    unit: int,
    facing: int,
    phase: float,
    pose: str,
) -> dict[str, tuple[int, int]]:
    cx, cy = center
    bob = int(math.sin(phase * 1.8) * unit * 0.08)
    swing = math.sin(phase * 2.1)
    hip = (cx, cy - int(unit * 0.08) + bob)
    neck = (cx + int(facing * unit * 0.08), cy - int(unit * 1.12) + bob)
    head = (cx + int(facing * unit * 0.13), cy - int(unit * 1.62) + bob)
    if pose == "throw":
        neck = (neck[0] + int(facing * unit * 0.18), neck[1] - int(unit * 0.04))
        head = (head[0] + int(facing * unit * 0.18), head[1] - int(unit * 0.04))
        hand_front = (neck[0] + int(facing * unit * 1.0), neck[1] - int(unit * 0.38))
        hand_back = (neck[0] - int(facing * unit * 0.58), neck[1] + int(unit * 0.38))
        knee_front = (hip[0] + int(facing * unit * 0.42), hip[1] + int(unit * 0.58))
        foot_front = (hip[0] + int(facing * unit * 0.74), hip[1] + int(unit * 1.02))
        knee_back = (hip[0] - int(facing * unit * 0.4), hip[1] + int(unit * 0.58))
        foot_back = (hip[0] - int(facing * unit * 0.72), hip[1] + int(unit * 1.02))
    elif pose == "kick":
        neck = (neck[0] + int(facing * unit * 0.22), neck[1])
        head = (head[0] + int(facing * unit * 0.22), head[1])
        hand_front = (neck[0] - int(facing * unit * 0.2), neck[1] + int(unit * 0.44))
        hand_back = (neck[0] - int(facing * unit * 0.68), neck[1] + int(unit * 0.34))
        knee_front = (hip[0] + int(facing * unit * 0.66), hip[1] + int(unit * 0.48))
        foot_front = (hip[0] + int(facing * unit * 1.48), hip[1] + int(unit * 0.60))
        knee_back = (hip[0] - int(facing * unit * 0.42), hip[1] + int(unit * 0.66))
        foot_back = (hip[0] - int(facing * unit * 0.88), hip[1] + int(unit * 1.05))
    elif pose == "push":
        neck = (neck[0] + int(facing * unit * 0.34), neck[1] + int(unit * 0.1))
        head = (head[0] + int(facing * unit * 0.34), head[1] + int(unit * 0.1))
        hand_front = (neck[0] + int(facing * unit * 1.12), neck[1] + int(unit * 0.18))
        hand_back = (neck[0] + int(facing * unit * 0.92), neck[1] + int(unit * 0.36))
        foot_front = (hip[0] + int(facing * unit * 0.54), hip[1] + int(unit * 1.05))
        foot_back = (hip[0] - int(facing * unit * 0.9), hip[1] + int(unit * 0.98))
        knee_front = (hip[0] + int(facing * unit * 0.22), hip[1] + int(unit * 0.62))
        knee_back = (hip[0] - int(facing * unit * 0.52), hip[1] + int(unit * 0.6))
    elif pose == "float":
        hand_front = (neck[0] + int(facing * unit * (0.56 + 0.12 * swing)), neck[1] - int(unit * 0.14))
        hand_back = (neck[0] - int(facing * unit * (0.5 - 0.1 * swing)), neck[1] + int(unit * 0.34))
        knee_front = (hip[0] + int(facing * unit * 0.32), hip[1] + int(unit * 0.54))
        knee_back = (hip[0] - int(facing * unit * 0.32), hip[1] + int(unit * 0.52))
        foot_front = (hip[0] + int(facing * unit * 0.72), hip[1] + int(unit * 0.72))
        foot_back = (hip[0] - int(facing * unit * 0.68), hip[1] + int(unit * 0.84))
    elif pose == "jump":
        hand_front = (neck[0] + int(facing * unit * 0.52), neck[1] - int(unit * 0.36))
        hand_back = (neck[0] - int(facing * unit * 0.48), neck[1] - int(unit * 0.28))
        knee_front = (hip[0] + int(facing * unit * 0.42), hip[1] + int(unit * 0.44))
        knee_back = (hip[0] - int(facing * unit * 0.38), hip[1] + int(unit * 0.44))
        foot_front = (hip[0] + int(facing * unit * 0.82), hip[1] + int(unit * 0.76))
        foot_back = (hip[0] - int(facing * unit * 0.78), hip[1] + int(unit * 0.72))
    elif pose == "fall":
        hand_front = (neck[0] + int(facing * unit * 0.58), neck[1] - int(unit * 0.32))
        hand_back = (neck[0] - int(facing * unit * 0.56), neck[1] - int(unit * 0.22))
        knee_front = (hip[0] + int(facing * unit * 0.34), hip[1] + int(unit * 0.48))
        knee_back = (hip[0] - int(facing * unit * 0.38), hip[1] + int(unit * 0.42))
        foot_front = (hip[0] + int(facing * unit * 0.42), hip[1] + int(unit * 0.96))
        foot_back = (hip[0] - int(facing * unit * 0.72), hip[1] + int(unit * 0.82))
    elif pose == "run":
        hand_front = (neck[0] - int(facing * unit * 0.48 * swing), neck[1] + int(unit * 0.42))
        hand_back = (neck[0] + int(facing * unit * 0.48 * swing), neck[1] + int(unit * 0.38))
        knee_front = (hip[0] + int(facing * unit * 0.42 * swing), hip[1] + int(unit * 0.6))
        knee_back = (hip[0] - int(facing * unit * 0.42 * swing), hip[1] + int(unit * 0.6))
        foot_front = (hip[0] + int(facing * unit * (0.84 * swing + 0.08)), hip[1] + int(unit * 1.04))
        foot_back = (hip[0] - int(facing * unit * (0.84 * swing - 0.08)), hip[1] + int(unit * 1.04))
    else:
        hand_front = (neck[0] + int(facing * unit * 0.38), neck[1] + int(unit * 0.54))
        hand_back = (neck[0] - int(facing * unit * 0.34), neck[1] + int(unit * 0.56))
        knee_front = (hip[0] + int(facing * unit * 0.24), hip[1] + int(unit * 0.62))
        knee_back = (hip[0] - int(facing * unit * 0.24), hip[1] + int(unit * 0.62))
        foot_front = (hip[0] + int(facing * unit * 0.35), hip[1] + int(unit * 1.05))
        foot_back = (hip[0] - int(facing * unit * 0.35), hip[1] + int(unit * 1.05))
    return {
        "head": head,
        "neck": neck,
        "shoulder": (neck[0], neck[1] + int(unit * 0.1)),
        "hip": hip,
        "hand_front": hand_front,
        "hand_back": hand_back,
        "knee_front": knee_front,
        "knee_back": knee_back,
        "foot_front": foot_front,
        "foot_back": foot_back,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Harness Alpha Pygame dashboard")
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--max-seeds", type=int, default=2)
    parser.add_argument("--max-repairs", type=int, default=3)
    parser.add_argument("--execution-mode", choices=("normal", "fast"), default="fast")
    parser.add_argument("--smoke-frames", type=int, help="render N frames and exit")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    app = DashboardApp(
        width=args.width,
        height=args.height,
        max_seeds=args.max_seeds,
        max_repairs=args.max_repairs,
        smoke_frames=args.smoke_frames,
    )
    app.execution_mode = args.execution_mode
    app.run()


if __name__ == "__main__":
    main()
