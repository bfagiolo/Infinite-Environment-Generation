"""Renderer-side visual grammar for Harness Alpha worlds.

This module is intentionally presentation-only. It never changes physics,
validation, objective state, or generated code behavior. It maps optional
closed-vocabulary visual tags plus deterministic world metadata into palettes
and drawing hints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import pymunk

from base_env import BaseEnv


Color = tuple[int, int, int]

MOODS = {
    "research_lab",
    "cosmic_research",
    "deep_space",
    "orbital_station",
    "asteroid_belt",
    "organic_maze",
    "cornfield_maze",
    "overgrown_garden",
    "alien_biome",
    "hazard_arena",
    "volcanic_foundry",
    "industrial_noir",
    "mechanical_factory",
    "clockwork_chamber",
    "magnetic_lab",
    "electric_substation",
    "crystal_cavern",
    "ice_facility",
    "desert_ruins",
    "toxic_swamp",
    "retro_arcade",
    "cyber_grid",
    "underwater_lab",
    "tropical_water",
    "moonlit_water",
    "stormwater_channel",
    "organic_forest",
    "twilight_woods",
    "ancient_grove",
    "training_sim",
    "minimalist_sandbox",
}

BACKGROUNDS = {
    "neon_grid",
    "parallax_starfield",
    "dense_starfield",
    "orbital_dust",
    "scanline_lab",
    "circuit_board",
    "oscilloscope",
    "hazard_warning",
    "ember_drift",
    "smoke_haze",
    "organic_noise",
    "leaf_shadow",
    "corn_rows",
    "crystal_speckle",
    "ice_frost",
    "desert_grain",
    "toxic_bubbles",
    "underwater_caustics",
    "retro_vignette",
    "void_gradient",
    "blueprint_grid",
    "radar_sweep",
}

ACCENTS = {
    "cyan_green",
    "cyan_purple",
    "blue_white",
    "electric_blue",
    "emerald_gold",
    "amber_green",
    "red_warning",
    "magenta_orange",
    "violet_lime",
    "toxic_lime",
    "ice_blue",
    "solar_orange",
    "desert_teal",
    "mint_coral",
    "pink_cyan",
    "gold_crimson",
    "acid_purple",
    "white_red",
    "teal_yellow",
    "green_gold",
    "deep_space_neon",
    "monochrome_cyan",
}

LIGHTING = {
    "neon_bloom",
    "soft_glow",
    "hard_shadow",
    "rim_light",
    "warning_pulse",
    "strobe_warning",
    "low_key",
    "high_contrast",
    "holographic",
    "bioluminescent",
    "warm_lantern",
    "cold_facility",
    "volumetric_haze",
    "crt_glow",
}

MOTION_FX = {
    "agent_trail",
    "object_trail",
    "particle_sparks",
    "field_pulse",
    "magnetic_rings",
    "goal_breathe",
    "hazard_flash",
    "gate_unlock_flash",
    "zero_g_drift",
    "dust_motes",
    "falling_embers",
    "bubble_rise",
    "leaf_flutter",
    "scan_sweep",
    "radar_ping",
    "electric_arcs",
    "crystal_shimmer",
    "impact_ripples",
}

AGENT_AVATARS = {
    "human",
    "orb",
    "ship",
    "robot",
    "marble",
    "arcade_disc",
    "drone",
    "creature",
}

SURFACE_FX = {
    "scanlines",
    "crt_vignette",
    "film_grain",
    "procedural_noise",
    "organic_stripes",
    "corn_stripes",
    "metal_panel_lines",
    "circuit_traces",
    "ice_cracks",
    "stone_facets",
    "crystal_facets",
    "hazard_chevrons",
    "warning_stripes",
    "grid_crosshairs",
}

MATERIALS = {
    "dark_metal",
    "brushed_steel",
    "gate_metal",
    "rubber",
    "glass",
    "energy_field",
    "sensor_plate",
    "warning_plate",
    "stone",
    "basalt",
    "hazard_rock",
    "asteroid",
    "crystal",
    "ice",
    "sandstone",
    "wood",
    "corn_wall",
    "vine_wall",
    "leaf_mass",
    "toxic_slime",
    "water_glass",
    "circuit_tile",
    "arcade_plastic",
}

ACCENT_PALETTES: dict[str, dict[str, Color]] = {
    "cyan_green": {"primary": (56, 216, 255), "secondary": (42, 232, 143), "hot": (255, 90, 118)},
    "cyan_purple": {"primary": (76, 221, 255), "secondary": (181, 115, 255), "hot": (255, 91, 165)},
    "blue_white": {"primary": (76, 166, 255), "secondary": (230, 248, 255), "hot": (255, 98, 112)},
    "electric_blue": {"primary": (39, 168, 255), "secondary": (91, 246, 255), "hot": (255, 190, 72)},
    "emerald_gold": {"primary": (48, 231, 155), "secondary": (255, 214, 84), "hot": (255, 84, 92)},
    "amber_green": {"primary": (236, 184, 71), "secondary": (113, 202, 88), "hot": (255, 91, 73)},
    "red_warning": {"primary": (255, 74, 104), "secondary": (255, 180, 62), "hot": (255, 32, 66)},
    "magenta_orange": {"primary": (255, 84, 201), "secondary": (255, 153, 65), "hot": (94, 232, 255)},
    "violet_lime": {"primary": (170, 115, 255), "secondary": (178, 255, 86), "hot": (255, 86, 142)},
    "toxic_lime": {"primary": (184, 255, 60), "secondary": (55, 224, 144), "hot": (255, 86, 62)},
    "ice_blue": {"primary": (134, 225, 255), "secondary": (214, 252, 255), "hot": (255, 96, 144)},
    "solar_orange": {"primary": (255, 169, 66), "secondary": (255, 229, 111), "hot": (78, 215, 255)},
    "desert_teal": {"primary": (229, 178, 101), "secondary": (66, 211, 196), "hot": (255, 92, 80)},
    "mint_coral": {"primary": (114, 246, 193), "secondary": (255, 132, 120), "hot": (255, 225, 98)},
    "pink_cyan": {"primary": (255, 116, 215), "secondary": (87, 232, 255), "hot": (255, 200, 80)},
    "gold_crimson": {"primary": (255, 213, 84), "secondary": (255, 72, 109), "hot": (76, 221, 255)},
    "acid_purple": {"primary": (202, 84, 255), "secondary": (191, 255, 67), "hot": (255, 88, 88)},
    "white_red": {"primary": (236, 246, 248), "secondary": (255, 83, 98), "hot": (255, 188, 70)},
    "teal_yellow": {"primary": (45, 215, 201), "secondary": (255, 231, 94), "hot": (255, 92, 136)},
    "green_gold": {"primary": (83, 214, 104), "secondary": (244, 205, 82), "hot": (255, 86, 86)},
    "deep_space_neon": {"primary": (86, 210, 255), "secondary": (177, 89, 255), "hot": (255, 91, 181)},
    "monochrome_cyan": {"primary": (91, 236, 255), "secondary": (165, 246, 255), "hot": (255, 255, 255)},
}

MOOD_DEFAULTS: dict[str, dict[str, Any]] = {
    "deep_space": {"accent": "deep_space_neon", "background": "parallax_starfield", "background_color": (5, 8, 20)},
    "cosmic_research": {"accent": "cyan_purple", "background": "dense_starfield", "background_color": (7, 10, 24)},
    "orbital_station": {"accent": "blue_white", "background": "blueprint_grid", "background_color": (8, 14, 24)},
    "asteroid_belt": {"accent": "deep_space_neon", "background": "orbital_dust", "background_color": (6, 7, 15)},
    "cornfield_maze": {"accent": "amber_green", "background": "corn_rows", "background_color": (24, 20, 10)},
    "organic_maze": {"accent": "green_gold", "background": "organic_noise", "background_color": (13, 23, 13)},
    "overgrown_garden": {"accent": "mint_coral", "background": "leaf_shadow", "background_color": (9, 25, 18)},
    "alien_biome": {"accent": "violet_lime", "background": "organic_noise", "background_color": (16, 10, 25)},
    "hazard_arena": {"accent": "red_warning", "background": "hazard_warning", "background_color": (24, 12, 13)},
    "volcanic_foundry": {"accent": "solar_orange", "background": "ember_drift", "background_color": (24, 11, 6)},
    "industrial_noir": {"accent": "gold_crimson", "background": "smoke_haze", "background_color": (13, 14, 15)},
    "mechanical_factory": {"accent": "teal_yellow", "background": "circuit_board", "background_color": (13, 18, 19)},
    "clockwork_chamber": {"accent": "emerald_gold", "background": "blueprint_grid", "background_color": (17, 15, 13)},
    "magnetic_lab": {"accent": "cyan_purple", "background": "scanline_lab", "background_color": (8, 12, 19)},
    "electric_substation": {"accent": "electric_blue", "background": "oscilloscope", "background_color": (7, 13, 18)},
    "crystal_cavern": {"accent": "pink_cyan", "background": "crystal_speckle", "background_color": (11, 12, 25)},
    "ice_facility": {"accent": "ice_blue", "background": "ice_frost", "background_color": (9, 18, 24)},
    "desert_ruins": {"accent": "desert_teal", "background": "desert_grain", "background_color": (29, 21, 12)},
    "toxic_swamp": {"accent": "toxic_lime", "background": "toxic_bubbles", "background_color": (9, 21, 13)},
    "retro_arcade": {"accent": "magenta_orange", "background": "retro_vignette", "background_color": (15, 8, 24)},
    "cyber_grid": {"accent": "cyan_purple", "background": "neon_grid", "background_color": (9, 13, 20)},
    "underwater_lab": {"accent": "mint_coral", "background": "underwater_caustics", "background_color": (4, 18, 24)},
    "tropical_water": {"accent": "mint_coral", "background": "underwater_caustics", "background_color": (3, 24, 26)},
    "moonlit_water": {"accent": "blue_white", "background": "underwater_caustics", "background_color": (2, 5, 22)},
    "stormwater_channel": {"accent": "teal_yellow", "background": "scanline_lab", "background_color": (7, 12, 15)},
    "organic_forest": {"accent": "green_gold", "background": "leaf_shadow", "background_color": (7, 18, 12)},
    "twilight_woods": {"accent": "violet_lime", "background": "organic_noise", "background_color": (12, 11, 22)},
    "ancient_grove": {"accent": "cyan_green", "background": "leaf_shadow", "background_color": (6, 20, 18)},
    "research_lab": {"accent": "cyan_green", "background": "scanline_lab", "background_color": (14, 16, 18)},
    "training_sim": {"accent": "monochrome_cyan", "background": "neon_grid", "background_color": (14, 16, 18)},
    "minimalist_sandbox": {"accent": "blue_white", "background": "void_gradient", "background_color": (16, 16, 18)},
}


@dataclass(frozen=True)
class VisualGrammar:
    mood: str = "research_lab"
    background: str = "scanline_lab"
    accent: str = "cyan_green"
    materials: tuple[str, ...] = ("dark_metal", "sensor_plate", "energy_field")
    lighting: tuple[str, ...] = ("neon_bloom", "soft_glow")
    motion_fx: tuple[str, ...] = ("agent_trail", "field_pulse", "goal_breathe")
    surface_fx: tuple[str, ...] = ("scanlines", "grid_crosshairs")
    agent_avatar: str = "human"
    shape_language: str = "technical_rectilinear"
    presentation: str = "clean_demo_mode"
    background_color: Color = (14, 16, 18)
    primary: Color = (56, 216, 255)
    secondary: Color = (42, 232, 143)
    hot: Color = (255, 90, 118)
    seed: int = 0
    source: str = "inferred"
    scores: Mapping[str, int] = field(default_factory=dict)
    visual_recipe: Mapping[str, Any] = field(default_factory=dict)


def infer_visual_grammar(env: BaseEnv) -> VisualGrammar:
    """Infer a renderer grammar from optional tags and deterministic metadata."""

    recipe = load_visual_recipe(env)
    if recipe:
        return _grammar_from_recipe(recipe, env)

    explicit = _valid_visual_tags(getattr(env, "visual_tags", None))
    if explicit:
        return _grammar_from_tags(explicit, env, source="visual_tags")

    text = _world_text(env)
    scores = {mood: 0 for mood in MOODS}
    _score_keywords(scores, text)
    if _gravity_is_zero(env):
        scores["deep_space"] += 4
        scores["cosmic_research"] += 2
    if getattr(env, "_force_zones", None):
        scores["magnetic_lab"] += 5
        scores["electric_substation"] += 2
    objective_type = str(getattr(env, "objective_type", "") or "").lower()
    if "survival" in objective_type:
        scores["hazard_arena"] += 5
    if len([item for item in getattr(env, "_objects", {}).values() if "wall" in item.name.lower()]) >= 5:
        scores["organic_maze"] += 1
        scores["training_sim"] += 1

    mood = max(scores, key=lambda key: scores[key])
    if scores[mood] <= 0:
        mood = "research_lab"
    defaults = MOOD_DEFAULTS[mood]
    tags = {
        "mood": mood,
        "background": defaults["background"],
        "accent": defaults["accent"],
        "materials": _materials_from_text(text, mood),
        "lighting": _lighting_from_mood(mood),
        "motion_fx": _motion_from_env(env, mood),
        "surface_fx": _surface_from_mood(mood),
        "agent_avatar": _agent_avatar_from_text(text, mood),
        "shape_language": _shape_language_from_mood(mood),
        "presentation": "clean_demo_mode",
    }
    return _grammar_from_tags(tags, env, source="inferred", scores=scores)


def load_visual_recipe(env: BaseEnv) -> dict[str, Any] | None:
    """Load renderer-only visual recipe sidecar for an environment."""

    module = sys.modules.get(env.__class__.__module__)
    module_file = getattr(module, "__file__", None) if module is not None else None
    if not module_file:
        return None
    path = Path(module_file).with_suffix(".visual.json")
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def color_for_record(record: Any, grammar: VisualGrammar, *, sensor: bool = False) -> Color:
    """Return material-aware role color for an ObjectRecord-like record."""

    role = str(getattr(record, "role", "") or "").lower()
    name = str(getattr(record, "name", "") or "").lower()
    kind = str(getattr(record, "kind", "") or "").lower()
    metadata = getattr(record, "metadata", None)
    metadata_text = json.dumps(metadata, sort_keys=True, default=str).lower() if isinstance(metadata, Mapping) else ""
    text = " ".join([role, name, kind, metadata_text])
    skin = _skin_for_text(text, grammar.visual_recipe)

    if role == "agent":
        return _skin_color(skin, grammar.primary, grammar)
    if role in {"goal", "target"} or sensor or any(token in text for token in ("goal", "target", "checkpoint")):
        return _skin_color(skin, grammar.secondary, grammar)
    if role in {"hazard", "danger"} or any(token in text for token in ("hazard", "lava", "falling", "danger", "spike")):
        return _skin_color(skin, grammar.hot, grammar)
    if "force_zone" in role or "force_zone" in text or "magnetic" in text or "field" in text:
        return _mix(grammar.primary, grammar.secondary, 0.35)
    if "gate" in text or "door" in text:
        if "mechanism_open" in text or "passable" in text:
            return _mix(grammar.secondary, (230, 255, 248), 0.35)
        return _mix(grammar.hot, grammar.primary, 0.25)
    if any(token in text for token in ("corn", "leaf", "vine", "garden", "organic")):
        return (177, 172, 78) if grammar.mood == "cornfield_maze" else (78, 150, 86)
    if any(token in text for token in ("asteroid", "rock", "stone", "basalt")):
        return (126, 116, 112)
    if any(token in text for token in ("crystal", "gem")):
        return _mix(grammar.primary, (245, 248, 255), 0.45)
    if any(token in text for token in ("ice", "frost")):
        return (160, 220, 236)
    if any(token in text for token in ("support", "floor", "wall", "rail", "metal", "platform")):
        return _mix((90, 108, 118), grammar.primary, 0.18)
    return _mix((138, 86, 108), grammar.primary, 0.2)


def _grammar_from_recipe(recipe: Mapping[str, Any], env: BaseEnv) -> VisualGrammar:
    recipe = dict(recipe)
    identity = str(recipe.get("world_identity") or "")
    original_identity = identity
    recipe_text = json.dumps(recipe, sort_keys=True, default=str).lower()
    allow_identity_inference = not bool(recipe.get("fast_variant"))
    if allow_identity_inference and any(token in recipe_text for token in ("survival", "survive", "falling rocks", "falling balls", "rain down", "avoid hazards")):
        identity = "hazard_survival_arena"
    elif allow_identity_inference and any(token in recipe_text for token in ("lava", "fireball", "molten", "volcano")):
        identity = "volcanic_escape_cavern"
    if identity != original_identity:
        recipe.pop("background_layers", None)
        recipe.pop("particles", None)
        recipe.pop("camera_fx", None)
        recipe["world_identity"] = identity
    mood_by_identity = {
        "volcanic_escape_cavern": "volcanic_foundry",
        "hazard_survival_arena": "hazard_arena",
        "neon_arcade_maze": "retro_arcade",
        "deep_space_salvage": "deep_space",
        "magnetic_research_lab": "magnetic_lab",
        "overgrown_living_maze": "organic_maze",
        "clockwork_mechanism_room": "mechanical_factory",
        "frozen_refraction_chamber": "ice_facility",
        "toxic_industrial_swamp": "toxic_swamp",
        "underwater_lab": "underwater_lab",
    }
    explicit_mood = str(recipe.get("mood") or "")
    mood = explicit_mood if explicit_mood in MOOD_DEFAULTS else mood_by_identity.get(identity, "research_lab")
    defaults = MOOD_DEFAULTS.get(mood, MOOD_DEFAULTS["research_lab"])
    palette = recipe.get("palette") if isinstance(recipe.get("palette"), Mapping) else {}
    primary = _hex_color(palette.get("primary"), ACCENT_PALETTES[defaults["accent"]]["primary"])
    secondary = _hex_color(palette.get("secondary"), ACCENT_PALETTES[defaults["accent"]]["secondary"])
    hot = _hex_color(palette.get("hot") or palette.get("hazard"), ACCENT_PALETTES[defaults["accent"]]["hot"])
    background_color = _hex_color(palette.get("background_color") or palette.get("background"), defaults["background_color"])
    primary = _ensure_contrast(primary, background_color, minimum=0.42, fallback=ACCENT_PALETTES[defaults["accent"]]["primary"])
    secondary = _ensure_contrast(secondary, background_color, minimum=0.36, fallback=ACCENT_PALETTES[defaults["accent"]]["secondary"])
    hot = _ensure_contrast(hot, background_color, minimum=0.40, fallback=ACCENT_PALETTES[defaults["accent"]]["hot"])
    return VisualGrammar(
        mood=mood,
        background=str(recipe.get("background") or defaults["background"]),
        accent=str(recipe.get("accent") or defaults["accent"]),
        materials=_materials_from_recipe(recipe, mood),
        lighting=_lighting_from_mood(mood),
        motion_fx=_motion_from_recipe(recipe, env, mood),
        surface_fx=_surface_from_mood(mood),
        agent_avatar=_agent_avatar_from_recipe(recipe, env, mood),
        shape_language=_shape_language_from_mood(mood),
        presentation="visual_director_recipe",
        background_color=background_color,
        primary=primary,
        secondary=secondary,
        hot=hot,
        seed=int(recipe.get("seed") or _stable_seed(_world_text(env))),
        source="visual_recipe",
        scores={identity: 999} if identity else {},
        visual_recipe=dict(recipe),
    )


def _grammar_from_tags(
    tags: Mapping[str, Any],
    env: BaseEnv,
    *,
    source: str,
    scores: Mapping[str, int] | None = None,
) -> VisualGrammar:
    mood = str(tags.get("mood") or "research_lab")
    if mood not in MOODS:
        mood = "research_lab"
    defaults = MOOD_DEFAULTS.get(mood, MOOD_DEFAULTS["research_lab"])
    background = str(tags.get("background") or defaults["background"])
    if background not in BACKGROUNDS:
        background = defaults["background"]
    accent = str(tags.get("accent") or defaults["accent"])
    if accent not in ACCENTS:
        accent = defaults["accent"]
    palette = ACCENT_PALETTES.get(accent, ACCENT_PALETTES["cyan_green"])
    background_color = tuple(defaults["background_color"])
    primary = _ensure_contrast(palette["primary"], background_color, minimum=0.42, fallback=ACCENT_PALETTES["cyan_green"]["primary"])
    secondary = _ensure_contrast(palette["secondary"], background_color, minimum=0.36, fallback=ACCENT_PALETTES["cyan_green"]["secondary"])
    hot = _ensure_contrast(palette["hot"], background_color, minimum=0.40, fallback=ACCENT_PALETTES["cyan_green"]["hot"])
    return VisualGrammar(
        mood=mood,
        background=background,
        accent=accent,
        materials=_valid_list(tags.get("materials"), MATERIALS, ("dark_metal", "sensor_plate")),
        lighting=_valid_list(tags.get("lighting"), LIGHTING, ("neon_bloom", "soft_glow")),
        motion_fx=_valid_list(tags.get("motion_fx"), MOTION_FX, ("agent_trail", "goal_breathe")),
        surface_fx=_valid_list(tags.get("surface_fx"), SURFACE_FX, ("scanlines",)),
        agent_avatar=_valid_agent_avatar(tags.get("agent_avatar"), env, mood),
        shape_language=str(tags.get("shape_language") or _shape_language_from_mood(mood)),
        presentation=str(tags.get("presentation") or "clean_demo_mode"),
        background_color=background_color,
        primary=primary,
        secondary=secondary,
        hot=hot,
        seed=_stable_seed(_world_text(env)),
        source=source,
        scores=dict(scores or {}),
    )


def _materials_from_recipe(recipe: Mapping[str, Any], mood: str) -> tuple[str, ...]:
    skins = recipe.get("object_skins")
    text = json.dumps(skins, sort_keys=True, default=str).lower() if isinstance(skins, Mapping) else ""
    return _materials_from_text(text, mood)


def _motion_from_recipe(recipe: Mapping[str, Any], env: BaseEnv, mood: str) -> tuple[str, ...]:
    effects = list(_motion_from_env(env, mood))
    cues = recipe.get("animation_cues")
    cue_text = " ".join(str(item).lower() for item in cues) if isinstance(cues, list) else ""
    if any(token in cue_text for token in ("fire", "ember", "lava")):
        effects.extend(["hazard_flash", "falling_embers", "particle_sparks"])
    if any(token in cue_text for token in ("arcade", "chomp", "pixel")):
        effects.extend(["hazard_flash", "agent_trail"])
    if any(token in cue_text for token in ("field", "electric", "magnetic")):
        effects.extend(["field_pulse", "magnetic_rings", "electric_arcs"])
    if any(token in cue_text for token in ("zero", "asteroid", "drift")):
        effects.extend(["zero_g_drift", "object_trail"])
    return tuple(dict.fromkeys(effects))


def _agent_avatar_from_recipe(recipe: Mapping[str, Any], env: BaseEnv, mood: str) -> str:
    avatar = recipe.get("agent_avatar")
    if isinstance(avatar, Mapping):
        value = avatar.get("style") or avatar.get("silhouette") or avatar.get("type")
    else:
        value = avatar
    if isinstance(value, str) and value in AGENT_AVATARS:
        return value
    agent_skin = {}
    skins = recipe.get("object_skins")
    if isinstance(skins, Mapping) and isinstance(skins.get("agent"), Mapping):
        agent_skin = skins["agent"]
    silhouette = str(agent_skin.get("silhouette", "")).lower()
    if silhouette in {"", "hero_orb", "default_agent", "agent"}:
        return _agent_avatar_from_text(_world_text(env), mood)
    if "ship" in silhouette or "spaceship" in silhouette:
        return "ship"
    if "drone" in silhouette:
        return "drone"
    if "robot" in silhouette:
        return "robot"
    if "marble" in silhouette or "ball" in silhouette:
        return "marble"
    if "arcade" in silhouette or "disc" in silhouette or "pac" in silhouette:
        return "arcade_disc"
    if "orb" in silhouette:
        return "orb"
    return _agent_avatar_from_text(_world_text(env), mood)


def _agent_avatar_from_text(text: str, mood: str) -> str:
    normalized = text.lower()
    if any(token in normalized for token in ("pacman", "pac-man", "arcade", "pellet")):
        return "arcade_disc"
    if any(token in normalized for token in ("spaceship", "space ship", "rocket", "fighter", "ship pilot")):
        return "ship"
    if any(token in normalized for token in ("drone", "quadrotor", "hovercraft", "ufo")):
        return "drone"
    if any(token in normalized for token in ("robot", "android", "machine", "mech")):
        return "robot"
    if _agent_is_ball_like(normalized):
        return "marble"
    if any(token in normalized for token in ("creature", "bug", "alien", "monster", "slime")):
        return "creature"
    if any(_contains_keyword(normalized, token) for token in ("particle", "orb", "charge", "energy", "magnetic")):
        return "orb"
    if any(token in normalized for token in ("survival", "survive", "escape", "dodge", "avoid falling", "falling rocks", "falling balls", "rain down")):
        return "human"
    if mood in {"deep_space", "asteroid_belt", "cosmic_research", "orbital_station"}:
        return "ship"
    if mood in {"retro_arcade"}:
        return "arcade_disc"
    if mood in {"magnetic_lab", "electric_substation"}:
        return "orb"
    return "human"


def _agent_is_ball_like(text: str) -> bool:
    agent_patterns = (
        r"\b(agent|player|hero|avatar)\s+(is|as|becomes|controls?)\s+(a\s+)?(ball|marble|sphere)\b",
        r"\b(ball|marble|sphere)\s+(agent|player|hero|avatar)\b",
        r"\b(agent|player|hero|avatar)\s+rolls?\b",
    )
    if "marble" in text and "falling marble" not in text:
        return True
    return any(re.search(pattern, text) for pattern in agent_patterns)


def _valid_agent_avatar(value: Any, env: BaseEnv, mood: str) -> str:
    if isinstance(value, str) and value in AGENT_AVATARS:
        return value
    return _agent_avatar_from_text(_world_text(env), mood)


def _skin_for_text(text: str, recipe: Mapping[str, Any]) -> Mapping[str, Any]:
    skins = recipe.get("object_skins")
    if not isinstance(skins, Mapping):
        return {}
    for pattern, skin in skins.items():
        if not isinstance(skin, Mapping):
            continue
        pattern_text = str(pattern).lower()
        if pattern_text == "agent" and "agent" in text:
            return skin
        if pattern_text == "goal" and any(token in text for token in ("goal", "target", "exit")):
            return skin
        if pattern_text.startswith("*") and pattern_text.endswith("*"):
            token = pattern_text.strip("*")
            if token and token in text:
                return skin
        elif pattern_text in text:
            return skin
    return {}


def _skin_color(skin: Mapping[str, Any], fallback: Color, grammar: VisualGrammar) -> Color:
    if not skin:
        return fallback
    text = json.dumps(skin, sort_keys=True, default=str).lower()
    if any(token in text for token in ("fire", "flame", "molten", "lava", "ember")):
        return _mix(grammar.hot, (255, 225, 118), 0.28)
    if any(token in text for token in ("ice", "frost", "cold")):
        return (170, 234, 255)
    if any(token in text for token in ("neon", "arcade", "pixel")):
        return _mix(fallback, grammar.secondary, 0.35)
    if any(token in text for token in ("basalt", "rock", "asteroid")):
        return _mix((93, 78, 72), grammar.hot, 0.12)
    if any(token in text for token in ("portal", "exit")):
        return _mix(grammar.secondary, (240, 255, 245), 0.24)
    return fallback


def _hex_color(value: Any, fallback: Color) -> Color:
    if not isinstance(value, str):
        return fallback
    text = value.strip().lstrip("#")
    if len(text) != 6:
        return fallback
    try:
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    except ValueError:
        return fallback


def _ensure_contrast(color: Color, background: Color, *, minimum: float, fallback: Color) -> Color:
    if _luminance_delta(color, background) >= minimum:
        return color
    if _luminance_delta(fallback, background) >= minimum:
        return fallback
    brightened = _mix(color, (255, 255, 255), 0.70)
    if _luminance_delta(brightened, background) >= minimum:
        return brightened
    return _mix(color, (0, 0, 0), 0.70)


def _luminance_delta(a: Color, b: Color) -> float:
    return abs(_relative_luminance(a) - _relative_luminance(b))


def _relative_luminance(color: Color) -> float:
    r, g, b = (channel / 255.0 for channel in color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _valid_visual_tags(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    mood = value.get("mood")
    if mood is not None and str(mood) not in MOODS:
        return None
    return dict(value)


def _valid_list(value: Any, allowed: set[str], fallback: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = []
    clean = tuple(str(item) for item in values if str(item) in allowed)
    return clean or fallback


def _world_text(env: BaseEnv) -> str:
    parts: list[str] = [env.__class__.__name__, str(getattr(env, "objective_type", ""))]
    module = sys.modules.get(env.__class__.__module__)
    if module is not None:
        parts.append(str(getattr(module, "SOURCE_PROMPT", "")))
    try:
        objective = env.get_ground_truth().get("objective", {})
        parts.append(json.dumps(objective, sort_keys=True, default=str))
    except Exception:
        pass
    for record in getattr(env, "_objects", {}).values():
        parts.extend([record.name, str(record.role), record.kind, json.dumps(record.metadata, sort_keys=True, default=str)])
    return " ".join(parts).lower()


def _score_keywords(scores: dict[str, int], text: str) -> None:
    keyword_scores = {
        "deep_space": ("outer space", "spaceship", "zero gravity", "zero_g", "orbit", "salvage", "cosmic"),
        "asteroid_belt": ("asteroid", "meteor", "drifting rock", "salvage field"),
        "magnetic_lab": ("magnetic", "charged", "force zone", "field_force", "gravity well"),
        "electric_substation": ("electric", "current", "conveyor", "energy"),
        "cornfield_maze": ("corn", "cornfield", "farm"),
        "organic_maze": ("maze", "garden", "overgrown", "vine", "leaf"),
        "hazard_arena": ("survival", "hazard", "falling rocks", "danger", "arena", "spike"),
        "volcanic_foundry": ("lava", "volcano", "ember", "foundry", "molten"),
        "mechanical_factory": ("factory", "gear", "mechanical", "conveyor", "machine", "sliding gate", "pressure plate", "gate"),
        "clockwork_chamber": ("clockwork", "gear", "seesaw", "lever", "pivot"),
        "crystal_cavern": ("crystal", "gem", "cavern"),
        "ice_facility": ("ice", "frost", "frozen"),
        "desert_ruins": ("desert", "sand", "ruin"),
        "toxic_swamp": ("toxic", "slime", "swamp", "acid"),
        "retro_arcade": ("pacman", "arcade", "pinball"),
        "underwater_lab": ("underwater", "water", "bubble"),
        "research_lab": ("lab", "research", "simulation"),
    }
    for mood, keywords in keyword_scores.items():
        for keyword in keywords:
            if _contains_keyword(text, keyword):
                scores[mood] += 2 if " " in keyword else 1


def _contains_keyword(text: str, keyword: str) -> bool:
    if " " in keyword:
        return keyword in text
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def _materials_from_text(text: str, mood: str) -> tuple[str, ...]:
    materials = []
    for token in MATERIALS:
        if token.replace("_", " ") in text or token in text:
            materials.append(token)
    mood_materials = {
        "deep_space": ("asteroid", "energy_field", "dark_metal"),
        "asteroid_belt": ("asteroid", "crystal", "energy_field"),
        "cornfield_maze": ("corn_wall", "wood", "leaf_mass"),
        "organic_maze": ("vine_wall", "leaf_mass", "wood"),
        "hazard_arena": ("hazard_rock", "warning_plate", "dark_metal"),
        "magnetic_lab": ("energy_field", "brushed_steel", "sensor_plate"),
        "mechanical_factory": ("brushed_steel", "gate_metal", "rubber"),
        "crystal_cavern": ("crystal", "stone", "energy_field"),
        "ice_facility": ("ice", "glass", "brushed_steel"),
        "desert_ruins": ("sandstone", "stone", "wood"),
    }
    return tuple(dict.fromkeys([*materials, *mood_materials.get(mood, ("dark_metal", "sensor_plate"))]))[:5]


def _lighting_from_mood(mood: str) -> tuple[str, ...]:
    if mood in {"hazard_arena", "volcanic_foundry"}:
        return ("warning_pulse", "high_contrast", "volumetric_haze")
    if mood in {"deep_space", "cosmic_research", "asteroid_belt"}:
        return ("neon_bloom", "holographic", "low_key")
    if mood in {"cornfield_maze", "organic_maze", "overgrown_garden"}:
        return ("warm_lantern", "soft_glow", "bioluminescent")
    if mood in {"magnetic_lab", "electric_substation", "cyber_grid"}:
        return ("neon_bloom", "crt_glow", "holographic")
    return ("neon_bloom", "soft_glow")


def _motion_from_env(env: BaseEnv, mood: str) -> tuple[str, ...]:
    effects = ["agent_trail", "goal_breathe"]
    if getattr(env, "_force_zones", None) or mood in {"magnetic_lab", "electric_substation"}:
        effects.extend(["field_pulse", "magnetic_rings", "electric_arcs"])
    if _gravity_is_zero(env) or mood in {"deep_space", "asteroid_belt"}:
        effects.extend(["zero_g_drift", "particle_sparks"])
    if mood in {"hazard_arena", "volcanic_foundry"}:
        effects.extend(["hazard_flash", "falling_embers"])
    if mood in {"cornfield_maze", "organic_maze", "overgrown_garden"}:
        effects.extend(["leaf_flutter", "dust_motes"])
    return tuple(dict.fromkeys(effects))


def _surface_from_mood(mood: str) -> tuple[str, ...]:
    mapping = {
        "cornfield_maze": ("corn_stripes", "organic_noise"),
        "organic_maze": ("organic_stripes", "procedural_noise"),
        "magnetic_lab": ("scanlines", "circuit_traces", "grid_crosshairs"),
        "electric_substation": ("scanlines", "circuit_traces"),
        "hazard_arena": ("warning_stripes", "hazard_chevrons"),
        "volcanic_foundry": ("film_grain", "stone_facets"),
        "deep_space": ("film_grain", "grid_crosshairs"),
        "asteroid_belt": ("film_grain", "stone_facets"),
        "crystal_cavern": ("crystal_facets", "procedural_noise"),
        "ice_facility": ("ice_cracks", "scanlines"),
    }
    return mapping.get(mood, ("scanlines", "grid_crosshairs"))


def _shape_language_from_mood(mood: str) -> str:
    if mood in {"asteroid_belt", "volcanic_foundry"}:
        return "jagged_asteroid"
    if mood in {"cornfield_maze", "organic_maze", "overgrown_garden", "alien_biome"}:
        return "organic_irregular"
    if mood in {"crystal_cavern", "ice_facility"}:
        return "crystalline"
    if mood in {"mechanical_factory", "industrial_noir"}:
        return "heavy_industrial"
    return "technical_rectilinear"


def _gravity_is_zero(env: BaseEnv) -> bool:
    try:
        gravity = env.config.gravity
        vector = pymunk.Vec2d(float(gravity[0]), float(gravity[1]))
        return vector.length < 1.0
    except Exception:
        return False


def _stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16)


def _mix(a: Color, b: Color, amount: float) -> Color:
    t = max(0.0, min(1.0, amount))
    return tuple(int(a[i] * (1.0 - t) + b[i] * t) for i in range(3))
