"""Visual Director agent for renderer-only world identity recipes.

The Visual Director is deliberately sandboxed: it writes JSON recipes that the
renderer may interpret, but it never changes generated environment code,
physics, objectives, or validation. This gives Harness Alpha a creative layer
without weakening the deterministic simulation contract.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv

from prompt_cache import load_cached_json, save_cached_json


Color = str
DEFAULT_VISUAL_MODEL = os.getenv("OPENAI_VISUAL_MODEL", "gpt-5.2")
DEFAULT_VISUAL_REASONING = os.getenv("OPENAI_VISUAL_REASONING_EFFORT", "low")
DEFAULT_VISUAL_MAX_OUTPUT = int(os.getenv("OPENAI_VISUAL_MAX_OUTPUT_TOKENS", "3200"))
VISUAL_RECIPE_CACHE_VERSION = "visual_recipe.v2"

load_dotenv()

VISUAL_PROGRAM_PRIMITIVES = {
    "particle_field",
    "ribbon_flow",
    "contour_lines",
    "parallax_dots",
    "grid_overlay",
    "facet_field",
    "texture_noise",
    "silhouette_motes",
    "heat_shimmer",
    "scanline_roll",
    "radial_glow",
}

OBJECT_EFFECT_PRIMITIVES = {
    "flame_orb",
    "portal_ring",
    "neon_outline",
    "danger_pulse",
    "motion_trail",
    "cracked_surface",
    "rim_glow",
    "spark_burst",
    "field_ring",
    "material_stripes",
    "impact_shadow",
}


def recipe_path_for_env(env_path: str | Path) -> Path:
    """Return the sidecar path for a generated environment's visual recipe."""

    path = Path(env_path)
    return path.with_suffix(".visual.json")


def _cache_key(prompt: str, env_path: str | Path, validation: Any | None) -> dict[str, Any]:
    path = Path(env_path)
    return {
        "version": VISUAL_RECIPE_CACHE_VERSION,
        "prompt": prompt,
        "env_path": str(path),
        "env_fingerprint": _file_fingerprint(path),
        "validation": _validation_text(validation),
        "model": DEFAULT_VISUAL_MODEL,
        "reasoning_effort": DEFAULT_VISUAL_REASONING,
        "max_output_tokens": DEFAULT_VISUAL_MAX_OUTPUT,
        "llm_disabled": os.getenv("HARNESS_DISABLE_LLM_VISUAL_DIRECTOR", "").lower()
        in {"1", "true", "yes"},
    }


def _file_fingerprint(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "missing"


def create_visual_recipe(
    *,
    prompt: str,
    env_path: str | Path,
    validation: Any | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Create and persist a procedural visual recipe for a generated world."""

    env_path = Path(env_path)
    recipe = VisualDirector().compose(prompt=prompt, env_path=env_path, validation=validation)
    path = Path(output_path) if output_path else recipe_path_for_env(env_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recipe, indent=2, sort_keys=True), encoding="utf-8")
    return path


class VisualDirector:
    """Prompt-to-procedural-art-brief agent.

    This local implementation behaves like a specialist agent: it reads the
    user prompt and environment object names, chooses a world identity, then
    composes renderer primitives. It is intentionally flexible: object skins
    and background layers are descriptive recipes, not a tiny theme enum.
    """

    def compose(
        self,
        *,
        prompt: str,
        env_path: Path,
        validation: Any | None = None,
    ) -> dict[str, Any]:
        cache_key = _cache_key(prompt, env_path, validation)
        cached = load_cached_json("visual_recipe", cache_key)
        if cached is not None:
            return cached

        if os.getenv("OPENAI_API_KEY") and os.getenv("HARNESS_DISABLE_LLM_VISUAL_DIRECTOR", "").lower() not in {"1", "true", "yes"}:
            try:
                recipe = self._compose_with_openai(prompt=prompt, env_path=env_path, validation=validation)
                save_cached_json("visual_recipe", cache_key, recipe)
                return recipe
            except Exception:
                pass
        recipe = self._compose_fallback(prompt=prompt, env_path=env_path, validation=validation)
        save_cached_json("visual_recipe", cache_key, recipe)
        return recipe

    def _compose_with_openai(
        self,
        *,
        prompt: str,
        env_path: Path,
        validation: Any | None = None,
    ) -> dict[str, Any]:
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model=DEFAULT_VISUAL_MODEL,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are the Visual Director for Harness Alpha, a deterministic 2D physics world factory. "
                        "Output only JSON. Never write Godot code. Create a renderer-only visual_program that "
                        "uses the allowed primitive names. You may be imaginative in combinations, parameters, "
                        "labels, rhythm, and art direction, but the program must be safe for a procedural renderer."
                    ),
                },
                {
                    "role": "user",
                    "content": _visual_program_prompt(prompt, env_path, validation),
                },
            ],
            reasoning={"effort": DEFAULT_VISUAL_REASONING},
            max_output_tokens=DEFAULT_VISUAL_MAX_OUTPUT,
        )
        text = str(getattr(response, "output_text", "") or "").strip()
        if not text:
            raise RuntimeError("empty visual director response")
        raw = _extract_json(text)
        return self._normalize_recipe(raw, prompt=prompt, env_path=env_path, validation=validation)

    def _compose_fallback(
        self,
        *,
        prompt: str,
        env_path: Path,
        validation: Any | None = None,
    ) -> dict[str, Any]:
        text = " ".join([prompt, env_path.stem, _validation_text(validation)]).lower()
        seed = _stable_seed(text)
        identity = self._identity(text)
        palette = self._palette(identity, seed)
        recipe = {
            "schema_version": 1,
            "agent": "visual_director",
            "source_prompt": prompt,
            "source_env_path": str(env_path),
            "seed": seed,
            "world_identity": identity,
            "palette": palette,
            "agent_avatar": self._agent_avatar(identity, text),
            "background_layers": self._background_layers(identity, seed),
            "object_skins": self._object_skins(identity, text),
            "semantic_props": self._semantic_props(identity, text),
            "particles": self._particles(identity, seed),
            "camera_fx": self._camera_fx(identity),
            "animation_cues": self._animation_cues(identity),
            "visual_program": self._visual_program(identity, text, seed, palette),
            "renderer_contract": {
                "physics_mutation_allowed": False,
                "objective_mutation_allowed": False,
                "fallback": "ignore_unknown_recipe_fields",
            },
        }
        return self._normalize_recipe(recipe, prompt=prompt, env_path=env_path, validation=validation)

    def _normalize_recipe(
        self,
        recipe: dict[str, Any],
        *,
        prompt: str,
        env_path: Path,
        validation: Any | None,
    ) -> dict[str, Any]:
        text = " ".join([prompt, env_path.stem, json.dumps(recipe, default=str), _validation_text(validation)]).lower()
        seed = int(recipe.get("seed") or _stable_seed(text))
        identity = str(recipe.get("world_identity") or self._identity(text))
        palette = recipe.get("palette") if isinstance(recipe.get("palette"), dict) else self._palette(identity, seed)
        normalized = {
            "schema_version": 2,
            "agent": "visual_director",
            "source": str(recipe.get("source") or ("llm_visual_director" if os.getenv("OPENAI_API_KEY") else "deterministic_visual_director")),
            "source_prompt": prompt,
            "source_env_path": str(env_path),
            "seed": seed,
            "world_identity": identity,
            "palette": _normalize_palette(palette, self._palette(identity, seed)),
            "agent_avatar": self._normalize_agent_avatar(recipe.get("agent_avatar"), identity, text),
            "background_layers": recipe.get("background_layers") if isinstance(recipe.get("background_layers"), list) else self._background_layers(identity, seed),
            "object_skins": recipe.get("object_skins") if isinstance(recipe.get("object_skins"), dict) else self._object_skins(identity, text),
            "semantic_props": self._normalize_semantic_props(recipe.get("semantic_props"), identity, text),
            "particles": recipe.get("particles") if isinstance(recipe.get("particles"), list) else self._particles(identity, seed),
            "camera_fx": _string_list(recipe.get("camera_fx")) or self._camera_fx(identity),
            "animation_cues": _string_list(recipe.get("animation_cues")) or self._animation_cues(identity),
            "visual_program": self._normalize_visual_program(recipe.get("visual_program"), identity, text, seed),
            "renderer_contract": {
                "physics_mutation_allowed": False,
                "objective_mutation_allowed": False,
                "fallback": "ignore_unknown_recipe_fields",
                "allowed_background_primitives": sorted(VISUAL_PROGRAM_PRIMITIVES),
                "allowed_object_effect_primitives": sorted(OBJECT_EFFECT_PRIMITIVES),
            },
        }
        return normalized

    def _normalize_semantic_props(self, value: Any, identity: str, text: str) -> list[dict[str, Any]]:
        fallback = self._semantic_props(identity, text)
        if not isinstance(value, list):
            return fallback
        normalized: list[dict[str, Any]] = []
        for item in value[:14]:
            if not isinstance(item, dict):
                continue
            tags = _string_list(item.get("semantic_tags")) or _string_list(item.get("tags"))
            kind = str(item.get("kind") or item.get("label") or (tags[0] if tags else "world_prop"))
            if not tags:
                tags = _string_list(kind.replace("_", " "))
            normalized.append(
                {
                    "kind": _safe_slug(kind) or "world_prop",
                    "semantic_tags": [_safe_slug(tag) for tag in tags if _safe_slug(tag)][:6],
                    "role": str(item.get("role") or item.get("importance") or "background_prop"),
                    "placement": str(item.get("placement") or "background"),
                    "renderer_only": True,
                }
            )
        return normalized or fallback

    def _normalize_agent_avatar(self, value: Any, identity: str, text: str) -> dict[str, Any]:
        if isinstance(value, dict):
            style = value.get("style") or value.get("silhouette") or value.get("type")
            description = value.get("description")
        else:
            style = value
            description = None
        fallback = self._agent_avatar(identity, text)
        if not isinstance(style, str) or style not in {"human", "orb", "ship", "robot", "marble", "arcade_disc", "drone", "creature"}:
            style = str(fallback.get("style") or "human")
        return {
            "style": style,
            "description": _string_list(description) or _string_list(fallback.get("description")),
            "renderer_only": True,
        }

    def _normalize_visual_program(self, value: Any, identity: str, text: str, seed: int) -> dict[str, Any]:
        fallback = self._visual_program(identity, text, seed, self._palette(identity, seed))
        program = value if isinstance(value, dict) else fallback
        layers = _normalize_program_layers(program.get("background_layers"), fallback["background_layers"])
        object_effects = _normalize_object_effects(program.get("object_effects"), fallback["object_effects"])
        return {
            "schema_version": "harness-alpha-visual-program-v1",
            "identity": str(program.get("identity") or identity),
            "art_direction": str(program.get("art_direction") or fallback["art_direction"]),
            "background_layers": layers,
            "object_effects": object_effects,
            "agent_style": program.get("agent_style") if isinstance(program.get("agent_style"), dict) else fallback["agent_style"],
            "motion_rules": _string_list(program.get("motion_rules")) or fallback["motion_rules"],
        }

    def _visual_program(self, identity: str, text: str, seed: int, palette: dict[str, Color]) -> dict[str, Any]:
        style_note = identity.replace("_", " ")
        layers: list[dict[str, Any]]
        effects: list[dict[str, Any]] = [
            {"selector": "goal|exit|target", "primitive": "portal_ring", "pulse": 0.55, "color_role": "secondary"},
            {"selector": "agent", "primitive": "motion_trail", "opacity": 0.35, "color_role": "primary"},
        ]
        if identity == "volcanic_escape_cavern":
            layers = [
                {"primitive": "texture_noise", "style": "charred_basalt", "density": 0.62, "color_role": "shadow"},
                {"primitive": "ribbon_flow", "style": "lava_falls", "count": 8 + seed % 5, "speed": 0.28, "color_role": "hazard"},
                {"primitive": "particle_field", "style": "embers", "count": 150 + seed % 60, "motion": "updraft", "color_role": "hazard"},
                {"primitive": "heat_shimmer", "style": "air_distortion", "strength": 0.62, "color_role": "primary"},
            ]
            effects.extend(
                [
                    {"selector": "fire|fireball|lava|ember", "primitive": "flame_orb", "trail": True, "color_role": "hazard"},
                    {"selector": "floor|wall|platform", "primitive": "cracked_surface", "glow": "molten", "color_role": "hazard"},
                ]
            )
        elif identity == "deep_space_salvage":
            layers = [
                {"primitive": "parallax_dots", "style": "distant_stars", "count": 190 + seed % 90, "speed": 0.08, "color_role": "primary"},
                {"primitive": "ribbon_flow", "style": "nebula_threads", "count": 5 + seed % 4, "speed": 0.10, "color_role": "secondary"},
                {"primitive": "particle_field", "style": "orbital_dust", "count": 110, "motion": "drift", "color_role": "secondary"},
            ]
            effects.extend(
                [
                    {"selector": "asteroid|rock|crystal", "primitive": "rim_glow", "opacity": 0.22, "color_role": "primary"},
                    {"selector": "field|zone|gravity", "primitive": "field_ring", "rings": 4, "color_role": "secondary"},
                ]
            )
        elif identity == "neon_arcade_maze":
            layers = [
                {"primitive": "scanline_roll", "style": "crt", "opacity": 0.38, "color_role": "secondary"},
                {"primitive": "grid_overlay", "style": "arcade_tiles", "cell": 28 + seed % 10, "color_role": "primary"},
                {"primitive": "particle_field", "style": "pixel_sparks", "count": 75, "motion": "sparkle", "color_role": "secondary"},
            ]
            effects.extend(
                [
                    {"selector": "wall|maze", "primitive": "neon_outline", "color_role": "primary"},
                    {"selector": "pellet|dot|goal", "primitive": "spark_burst", "color_role": "secondary"},
                ]
            )
        elif identity == "overgrown_living_maze":
            layers = [
                {"primitive": "silhouette_motes", "style": "leaf_canopy", "count": 95, "motion": "sway", "color_role": "secondary"},
                {"primitive": "ribbon_flow", "style": "vine_strands", "count": 18, "speed": 0.12, "color_role": "primary"},
                {"primitive": "texture_noise", "style": "organic_floor", "density": 0.45, "color_role": "shadow"},
            ]
            effects.append({"selector": "wall|vine|corn|leaf", "primitive": "material_stripes", "color_role": "secondary"})
        elif identity == "magnetic_research_lab":
            layers = [
                {"primitive": "grid_overlay", "style": "oscilloscope", "cell": 42, "color_role": "primary"},
                {"primitive": "contour_lines", "style": "magnetic_flux", "count": 12, "speed": 0.35, "color_role": "secondary"},
                {"primitive": "particle_field", "style": "ion_sparks", "count": 70, "motion": "orbit", "color_role": "secondary"},
            ]
            effects.extend(
                [
                    {"selector": "field|force|magnetic|charged", "primitive": "field_ring", "rings": 5, "color_role": "secondary"},
                    {"selector": "ball|orb|charge", "primitive": "neon_outline", "color_role": "primary"},
                ]
            )
        else:
            layers = [
                {"primitive": "grid_overlay", "style": "research_floor", "cell": 32, "color_role": "primary"},
                {"primitive": "particle_field", "style": "holographic_noise", "count": 70 + seed % 40, "motion": "drift", "color_role": "secondary"},
                {"primitive": "radial_glow", "style": "lab_light_pool", "count": 3, "color_role": "primary"},
            ]
        return {
            "schema_version": "harness-alpha-visual-program-v1",
            "identity": identity,
            "art_direction": f"Procedural {style_note} treatment generated from prompt semantics.",
            "background_layers": layers,
            "object_effects": effects,
            "agent_style": {"motion_language": "pose_from_intent", "contact_actions": ["push", "kick", "throw"]},
            "motion_rules": [
                "Visual effects must not mutate physics or objective state.",
                "Use object names and roles as selectors; ignore unknown selectors safely.",
                "Prefer visible readable motion over photorealism.",
            ],
        }

    def _agent_avatar(self, identity: str, text: str) -> dict[str, Any]:
        if any(token in text for token in ("pacman", "pac-man", "arcade", "pellet")):
            style = "arcade_disc"
        elif _agent_is_ball_like(text):
            style = "marble"
        elif any(token in text for token in ("spaceship", "space ship", "rocket", "fighter", "ship")):
            style = "ship"
        elif any(token in text for token in ("drone", "quadrotor", "hovercraft", "ufo")):
            style = "drone"
        elif any(token in text for token in ("robot", "android", "mech")):
            style = "robot"
        elif any(token in text for token in ("creature", "alien", "bug", "monster", "slime")):
            style = "creature"
        elif any(token in text for token in ("particle", "orb", "charge", "energy")):
            style = "orb"
        elif any(token in text for token in ("human", "person", "runner", "player", "escape", "worker", "explorer", "survival", "survive", "dodge", "avoid falling", "rain down")):
            style = "human"
        elif identity == "deep_space_salvage":
            style = "ship"
        elif identity == "neon_arcade_maze":
            style = "arcade_disc"
        elif identity == "magnetic_research_lab":
            style = "orb"
        else:
            style = "human"
        descriptors = {
            "human": ["procedural stick-runner", "limb poses from velocity", "push/float/fall animation"],
            "ship": ["triangular spacecraft", "engine plume", "banking nose"],
            "drone": ["hover chassis", "rotor glow", "stabilizer bob"],
            "robot": ["blocky research bot", "servo limbs", "lit faceplate"],
            "arcade_disc": ["chomping arcade disc", "neon rim", "directional mouth"],
            "marble": ["rolling glass sphere", "specular highlight", "spin rings"],
            "creature": ["small alien critter", "antenna bounce", "squash-and-stretch"],
            "orb": ["energy core", "orbiting particles", "ion trail"],
        }
        return {
            "style": style,
            "description": descriptors.get(style, descriptors["human"]),
            "renderer_only": True,
        }

    def _identity(self, text: str) -> str:
        scores = {
            "volcanic_escape_cavern": _score(text, "lava", "fire", "fireball", "volcano", "molten", "ember", "ash"),
            "hazard_survival_arena": _score(text, "survival", "survive", "falling rocks", "falling balls", "rain down", "dodge", "avoid hazards", "hazard arena"),
            "neon_arcade_maze": _score(text, "pacman", "arcade", "pellet", "ghost", "maze", "chase"),
            "deep_space_salvage": _score(text, "space", "asteroid", "zero gravity", "zero-g", "orbit", "spaceship"),
            "magnetic_research_lab": _score(text, "magnetic", "field", "charged", "force", "lab", "electric"),
            "overgrown_living_maze": _score(text, "forest", "garden", "corn", "vine", "overgrown", "organic"),
            "clockwork_mechanism_room": _score(text, "gear", "seesaw", "lever", "gate", "pressure plate", "factory"),
            "frozen_refraction_chamber": _score(text, "ice", "frozen", "crystal", "glass", "slippery"),
            "toxic_industrial_swamp": _score(text, "toxic", "acid", "slime", "swamp", "pipe"),
        }
        identity = max(scores, key=lambda key: scores[key])
        return identity if scores[identity] > 0 else "research_simulation_stage"

    def _palette(self, identity: str, seed: int) -> dict[str, Color]:
        palettes: dict[str, tuple[Color, Color, Color, Color, Color]] = {
            "volcanic_escape_cavern": ("#090605", "#ff5a1f", "#ffb13d", "#4a1711", "#78f7ff"),
            "hazard_survival_arena": ("#10090b", "#ff496d", "#ffc247", "#ff234f", "#2b1118"),
            "neon_arcade_maze": ("#070615", "#ffe84a", "#ff4be8", "#2bf7ff", "#351064"),
            "deep_space_salvage": ("#030713", "#62d8ff", "#b66cff", "#f087ff", "#2b315f"),
            "magnetic_research_lab": ("#061019", "#51ddff", "#b77cff", "#9cff57", "#17233b"),
            "overgrown_living_maze": ("#07140b", "#c4b65a", "#4fce65", "#ff9d5c", "#16351f"),
            "clockwork_mechanism_room": ("#11100c", "#ffcf63", "#35d7c9", "#c64c42", "#383021"),
            "frozen_refraction_chamber": ("#06131d", "#a4edff", "#ffffff", "#75a8ff", "#173247"),
            "toxic_industrial_swamp": ("#061108", "#baff35", "#45df91", "#a66cff", "#172f16"),
            "research_simulation_stage": ("#101214", "#38d8ff", "#2ae88f", "#ff5a76", "#1e2a2d"),
        }
        bg, primary, secondary, hot, shadow = palettes.get(identity, palettes["research_simulation_stage"])
        return {
            "background": bg,
            "primary": primary,
            "secondary": secondary,
            "hazard": hot,
            "shadow": shadow,
            "accent_variation": str(seed % 7),
        }

    def _background_layers(self, identity: str, seed: int) -> list[dict[str, Any]]:
        common = [{"type": "atmospheric_vignette", "strength": 0.25 + (seed % 5) * 0.04}]
        layers = {
            "volcanic_escape_cavern": [
                {"type": "basalt_noise", "density": 0.82},
                {"type": "molten_cracks", "count": 16 + seed % 9},
                {"type": "ember_field", "count": 120 + seed % 60},
                {"type": "heat_shimmer", "strength": 0.55},
            ],
            "hazard_survival_arena": [
                {"type": "danger_grid", "cell": 36},
                {"type": "warning_chevrons", "count": 22 + seed % 9},
                {"type": "dust_and_impact_sparks", "count": 95},
                {"type": "fall_shadow_bands", "count": 9},
            ],
            "neon_arcade_maze": [
                {"type": "crt_scanlines", "opacity": 0.35},
                {"type": "arcade_grid", "cell": 28 + seed % 10},
                {"type": "neon_pixel_sparks", "count": 70},
            ],
            "deep_space_salvage": [
                {"type": "parallax_starfield", "count": 150 + seed % 90},
                {"type": "nebula_wash", "bands": 4 + seed % 4},
                {"type": "orbital_dust", "count": 90},
            ],
            "magnetic_research_lab": [
                {"type": "oscilloscope_grid", "cell": 44},
                {"type": "field_contours", "rings": 8 + seed % 5},
                {"type": "circuit_traces", "count": 42},
            ],
            "overgrown_living_maze": [
                {"type": "organic_canopy", "density": 0.7},
                {"type": "leaf_shadow", "count": 90},
                {"type": "dust_motes", "count": 80},
            ],
            "clockwork_mechanism_room": [
                {"type": "brass_plate_noise", "density": 0.5},
                {"type": "gear_silhouettes", "count": 7 + seed % 5},
                {"type": "workshop_dust", "count": 60},
            ],
            "frozen_refraction_chamber": [
                {"type": "ice_facets", "count": 38},
                {"type": "cold_mist", "count": 75},
                {"type": "refraction_lines", "count": 24},
            ],
            "toxic_industrial_swamp": [
                {"type": "toxic_bubbles", "count": 72},
                {"type": "pipe_shadows", "count": 16},
                {"type": "acid_vapor", "count": 90},
            ],
        }
        return [*layers.get(identity, [{"type": "simulation_grid", "cell": 32}]), *common]

    def _object_skins(self, identity: str, text: str) -> dict[str, dict[str, Any]]:
        base = {
            "agent": {"silhouette": "hero_orb", "fx": ["rim_glow", "motion_trail"]},
            "goal": {"silhouette": "portal_or_exit", "fx": ["breathing_ring", "arrival_glow"]},
            "*gate*": {"material": "machined_door", "fx": ["edge_light"]},
        }
        avatar = self._agent_avatar(identity, text)["style"]
        agent_silhouettes = {
            "human": "stick_runner",
            "ship": "small_spacecraft",
            "drone": "hover_drone",
            "robot": "blocky_robot",
            "arcade_disc": "arcade_player_disc",
            "marble": "glass_marble",
            "creature": "alien_critter",
            "orb": "energy_orb",
        }
        base["agent"] = {
            "silhouette": agent_silhouettes.get(avatar, "stick_runner"),
            "fx": ["rim_glow", "motion_trail", f"{avatar}_motion"],
        }
        if identity == "volcanic_escape_cavern":
            base.update(
                {
                    "*fire*": {"silhouette": "flame_core_orb", "fx": ["flicker_aura", "ember_trail", "heat_halo"]},
                    "*lava*": {"material": "molten_surface", "fx": ["lava_wave", "crack_glow"]},
                    "*floor*": {"material": "basalt_shelf", "fx": ["molten_edge"]},
                    "*wall*": {"material": "charred_basalt", "fx": ["ash_speckle"]},
                }
            )
        elif identity == "hazard_survival_arena":
            base.update(
                {
                    "*falling*": {"silhouette": "warning_projectile", "fx": ["impact_shadow", "danger_flash"]},
                    "*rock*": {"material": "hazard_rock", "fx": ["impact_shadow", "dust_rim"]},
                    "*fire*": {"silhouette": "flame_core_orb", "fx": ["flicker_aura", "ember_trail"]},
                    "*wall*": {"material": "dark_metal", "fx": ["warning_stripes"]},
                    "*floor*": {"material": "warning_plate", "fx": ["hazard_chevrons"]},
                }
            )
        elif identity == "neon_arcade_maze":
            base.update(
                {
                    "agent": {"silhouette": "arcade_player_disc", "fx": ["mouth_chomp", "neon_rim"]},
                    "*ghost*": {"silhouette": "enemy_blob", "fx": ["afterimage"]},
                    "*pellet*": {"silhouette": "tiny_neon_dot", "fx": ["twinkle"]},
                    "*wall*": {"material": "arcade_neon_wall", "fx": ["edge_bloom"]},
                }
            )
        elif identity == "deep_space_salvage":
            base.update(
                {
                    "*asteroid*": {"material": "pitted_rock", "fx": ["slow_spin_shadow"]},
                    "*rock*": {"material": "pitted_rock", "fx": ["dust_rim"]},
                    "*field*": {"material": "gravity_lens", "fx": ["orbital_rings"]},
                }
            )
        return base

    def _particles(self, identity: str, seed: int) -> list[dict[str, Any]]:
        mapping = {
            "volcanic_escape_cavern": [{"type": "embers", "count": 140}, {"type": "smoke_wisps", "count": 38}],
            "hazard_survival_arena": [{"type": "impact_sparks", "count": 95}, {"type": "dust_motes", "count": 60}],
            "neon_arcade_maze": [{"type": "pixel_sparks", "count": 70}],
            "deep_space_salvage": [{"type": "star_dust", "count": 160}],
            "overgrown_living_maze": [{"type": "leaf_bits", "count": 90}],
            "frozen_refraction_chamber": [{"type": "ice_motes", "count": 80}],
            "toxic_industrial_swamp": [{"type": "acid_bubbles", "count": 80}],
        }
        return mapping.get(identity, [{"type": "holographic_noise", "count": 55 + seed % 40}])

    def _camera_fx(self, identity: str) -> list[str]:
        mapping = {
            "volcanic_escape_cavern": ["heat_shimmer", "warm_bloom"],
            "hazard_survival_arena": ["warning_pulse", "impact_vignette"],
            "neon_arcade_maze": ["crt_bloom", "scanline_roll"],
            "deep_space_salvage": ["subtle_star_parallax"],
            "magnetic_research_lab": ["field_distortion"],
            "frozen_refraction_chamber": ["cold_vignette"],
        }
        return mapping.get(identity, ["subtle_bloom"])

    def _animation_cues(self, identity: str) -> list[str]:
        mapping = {
            "volcanic_escape_cavern": ["fireball_flicker", "lava_pulse", "ember_rise", "exit_shimmer"],
            "hazard_survival_arena": ["hazard_flash", "impact_shadow", "falling_dust", "agent_trail"],
            "neon_arcade_maze": ["maze_edge_pulse", "player_chomp", "pellet_twinkle"],
            "deep_space_salvage": ["zero_g_drift", "asteroid_shadow_roll"],
            "magnetic_research_lab": ["field_rings", "electric_arc_jitter"],
        }
        return mapping.get(identity, ["goal_breathe", "agent_trail"])

    def _semantic_props(self, identity: str, text: str) -> list[dict[str, Any]]:
        props: list[dict[str, Any]] = []

        def add(kind: str, tags: list[str], role: str = "background_prop", placement: str = "background") -> None:
            props.append(
                {
                    "kind": kind,
                    "semantic_tags": tags,
                    "role": role,
                    "placement": placement,
                    "renderer_only": True,
                }
            )

        if identity == "volcanic_escape_cavern" or any(token in text for token in ("lava", "fire", "volcano", "fireball", "ember")):
            add("lava_fire_particles", ["fire", "flame", "spark"], "effect_prop", "hazard")
            add("smoke_heat_wisps", ["smoke", "particle"], "background_prop", "sky")
            add("molten_sun_or_moon", ["sun", "moon"], "background_prop", "horizon")
        if identity == "deep_space_salvage" or any(token in text for token in ("space", "asteroid", "zero-g", "spaceship", "rocket")):
            add("spacecraft_silhouette", ["ship", "spaceship", "satellite"], "background_prop", "sky")
            add("asteroid_field_props", ["asteroid", "meteor"], "background_prop", "sky")
            add("stellar_specks", ["star", "space"], "background_prop", "sky")
        if identity == "overgrown_living_maze" or any(token in text for token in ("forest", "garden", "corn", "tree", "woods")):
            add("tree_line", ["tree", "forest"], "background_prop", "edges")
            add("grass_and_leaf_edges", ["grass", "leaf"], "set_dressing", "floor")
            add("soft_clouds", ["cloud"], "background_prop", "sky")
        if any(token in text for token in ("basketball", "hoop", "court")):
            add("basketball_equipment", ["basketball", "ball"], "foreground_prop", "objective")
            add("score_marker", ["flag", "goal"], "foreground_prop", "objective")
        if any(token in text for token in ("soccer", "football", "kick", "goal line")):
            add("soccer_equipment", ["soccer", "ball"], "foreground_prop", "objective")
            add("goal_marker", ["goal", "flag"], "foreground_prop", "objective")
        if any(token in text for token in ("platform", "jump", "climb", "tower")):
            add("platformer_set_dressing", ["platform", "crate", "ladder"], "set_dressing", "platforms")
        if not props:
            add("research_stage_props", ["platform", "particle"], "set_dressing", "background")
        return props


def _score(text: str, *terms: str) -> int:
    score = 0
    for term in terms:
        if " " in term:
            score += 3 if term in text else 0
        elif re.search(rf"\b{re.escape(term)}\b", text):
            score += 2
    return score


def _safe_slug(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:48]


def _agent_is_ball_like(text: str) -> bool:
    agent_patterns = (
        r"\b(agent|player|hero|avatar)\s+(is|as|becomes|controls?)\s+(a\s+)?(ball|marble|sphere)\b",
        r"\b(ball|marble|sphere)\s+(agent|player|hero|avatar)\b",
        r"\b(agent|player|hero|avatar)\s+rolls?\b",
    )
    return "marble" in text or any(re.search(pattern, text) for pattern in agent_patterns)


def _stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16)


def _validation_text(validation: Any | None) -> str:
    if validation is None:
        return ""
    details = getattr(validation, "details", None)
    if isinstance(details, dict):
        try:
            return json.dumps(details, sort_keys=True, default=str)
        except Exception:
            return str(details)
    return str(validation)


def _visual_program_prompt(prompt: str, env_path: Path, validation: Any | None) -> str:
    return f"""
Create a renderer-only visual recipe JSON for this prompt:

{prompt}

Generated environment path/stem:
{env_path}

Validation context:
{_validation_text(validation)[:1800]}

Output JSON only. Required keys:
- world_identity: short snake_case world identity
- palette: object with background, primary, secondary, hazard, shadow hex colors
- agent_avatar: object with style and description
- visual_program: object with:
  - identity
  - art_direction
  - background_layers: list of primitive objects
  - object_effects: list of primitive objects
  - agent_style: object
  - motion_rules: list of strings
- object_skins: object keyed by selectors like agent, goal, *fire*, *wall*
- semantic_props: list of renderer-only local asset hints, each with kind,
  semantic_tags, role, and placement
- animation_cues: list of strings
- camera_fx: list of strings

Allowed background primitive names:
{", ".join(sorted(VISUAL_PROGRAM_PRIMITIVES))}

Allowed object effect primitive names:
{", ".join(sorted(OBJECT_EFFECT_PRIMITIVES))}

Primitive objects may include safe parameters such as style, count, density,
speed, opacity, strength, color_role, selector, rings, trail, glow, cell.

Rules:
- Do not include code.
- Do not modify physics/objectives.
- Be imaginative through composition and parameters, not through new APIs.
- For semantic_props, request recognizable props from the local renderer, not
  internet assets. Examples: tree, grass, cloud, ship, asteroid, meteor,
  basketball, soccer, goal, fire, flame, smoke, spark, platform, crate.
- If the prompt says lava/fire, make hazards visibly fiery.
- If the prompt says falling/raining hazards, include falling/telegraph/trail visuals.
- If the prompt implies a human/player, prefer agent_avatar.style="human".
- Unknown fields may be ignored by the renderer, so keep the core fields clear.
""".strip()


def _extract_json(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _normalize_palette(value: Any, fallback: dict[str, Color]) -> dict[str, Color]:
    palette = value if isinstance(value, dict) else {}
    normalized = dict(fallback)
    for key in ("background", "primary", "secondary", "hazard", "shadow"):
        candidate = palette.get(key)
        if isinstance(candidate, str) and re.fullmatch(r"#?[0-9a-fA-F]{6}", candidate.strip()):
            text = candidate.strip()
            normalized[key] = text if text.startswith("#") else f"#{text}"
    normalized["accent_variation"] = str(palette.get("accent_variation", normalized.get("accent_variation", "0")))
    return normalized


def _normalize_program_layers(value: Any, fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return fallback
    normalized: list[dict[str, Any]] = []
    for item in value[:10]:
        if not isinstance(item, dict):
            continue
        primitive = str(item.get("primitive") or item.get("type") or "")
        if primitive not in VISUAL_PROGRAM_PRIMITIVES:
            continue
        normalized.append(_safe_program_item(item, primitive=primitive))
    return normalized or fallback


def _normalize_object_effects(value: Any, fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return fallback
    normalized: list[dict[str, Any]] = []
    for item in value[:16]:
        if not isinstance(item, dict):
            continue
        primitive = str(item.get("primitive") or item.get("type") or "")
        if primitive not in OBJECT_EFFECT_PRIMITIVES:
            continue
        safe = _safe_program_item(item, primitive=primitive)
        safe["selector"] = str(item.get("selector") or "*")
        normalized.append(safe)
    return normalized or fallback


def _safe_program_item(item: dict[str, Any], *, primitive: str) -> dict[str, Any]:
    safe: dict[str, Any] = {"primitive": primitive}
    allowed = {
        "style",
        "count",
        "density",
        "speed",
        "opacity",
        "strength",
        "color_role",
        "motion",
        "rings",
        "trail",
        "glow",
        "cell",
        "selector",
        "pulse",
    }
    for key in allowed:
        value = item.get(key)
        if isinstance(value, str | int | float | bool):
            safe[key] = value
    return safe


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str | int | float | bool)][:24]
