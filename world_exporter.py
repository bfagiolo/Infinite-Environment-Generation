"""Export verified Harness Alpha worlds to a renderer/runtime schema.

Phase 1 of the Godot bridge: Python remains the source of truth, and this
module serializes deterministic Pymunk state plus visual and verification
metadata into JSON files a richer runtime can consume.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
from typing import Any

from base_env import BaseEnv
from validator import ValidationResult, ValidatorConfig, load_env_class, validate_generated_env
from visual_grammar import VisualGrammar, infer_visual_grammar, load_visual_recipe

try:
    from asset_resolver import resolve_semantic_assets
except Exception:  # pragma: no cover - renderer props are optional.
    resolve_semantic_assets = None  # type: ignore[assignment]


EXPORT_ROOT = Path("exports")
LATEST_EXPORT_DIR = EXPORT_ROOT / "latest_world"
SCHEMA_VERSION = "harness-alpha-world-v1"


def export_verified_world(
    env_path: str | Path,
    *,
    output_dir: str | Path = LATEST_EXPORT_DIR,
    prompt: str | None = None,
    validation: ValidationResult | None = None,
    visual_recipe_path: str | Path | None = None,
    class_name: str | None = None,
    require_accepted: bool = False,
) -> Path:
    """Export a generated environment to world/visual/verification JSON files.

    Returns the output directory.
    """

    source_env_path = Path(env_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    env_class = load_env_class(source_env_path, class_name=class_name)
    env = env_class()
    ground_truth = env.get_ground_truth()

    active_validation = validation
    if active_validation is None:
        active_validation = validate_generated_env(
            source_env_path,
            class_name=class_name,
            config=ValidatorConfig(simulation_steps=4),
        )
    if require_accepted and not active_validation.accepted:
        raise ValueError(
            "Refusing to export unaccepted world: "
            f"tier {active_validation.achieved_tier}/{active_validation.minimum_acceptance_tier} "
            f"({active_validation.reason})"
        )

    grammar = infer_visual_grammar(env)
    recipe = _load_recipe_for_export(env, visual_recipe_path)

    visual_brief = _visual_brief(grammar, recipe, prompt=prompt, ground_truth=ground_truth)
    world_schema = _build_world_schema(
        env=env,
        ground_truth=ground_truth,
        env_path=source_env_path,
        prompt=prompt,
        visual_brief=visual_brief,
        validation=active_validation,
    )
    verification = _verification_payload(active_validation, source_env_path, prompt)

    _write_json(output_path / "world_schema.json", world_schema)
    _write_json(output_path / "visual_brief.json", visual_brief)
    _write_json(output_path / "verification.json", verification)
    visual_program = visual_brief.get("visual_program")
    if isinstance(visual_program, dict):
        _write_json(output_path / "visual_program.json", visual_program)
    semantic_assets = visual_brief.get("semantic_assets")
    if isinstance(semantic_assets, list):
        _write_json(output_path / "semantic_assets.json", {"assets": semantic_assets})
    shutil.copyfile(source_env_path, output_path / "source_env.py")
    if visual_recipe_path:
        recipe_source = Path(visual_recipe_path)
        if recipe_source.exists():
            shutil.copyfile(recipe_source, output_path / "source_env.visual.json")
    else:
        inferred_recipe = source_env_path.with_suffix(".visual.json")
        if inferred_recipe.exists():
            shutil.copyfile(inferred_recipe, output_path / "source_env.visual.json")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": _utc_now(),
        "files": {
            "world_schema": "world_schema.json",
            "visual_brief": "visual_brief.json",
            "visual_program": "visual_program.json"
            if isinstance(visual_brief.get("visual_program"), dict)
            else None,
            "semantic_assets": "semantic_assets.json"
            if isinstance(visual_brief.get("semantic_assets"), list)
            else None,
            "verification": "verification.json",
            "source_env": "source_env.py",
        },
        "source_env_path": str(source_env_path),
        "accepted": bool(active_validation.accepted),
        "achieved_tier": active_validation.achieved_tier,
        "tier_name": active_validation.tier_name,
    }
    _write_json(output_path / "manifest.json", manifest)
    return output_path


def _build_world_schema(
    *,
    env: BaseEnv,
    ground_truth: dict[str, Any],
    env_path: Path,
    prompt: str | None,
    visual_brief: dict[str, Any],
    validation: ValidationResult,
) -> dict[str, Any]:
    config = ground_truth.get("config", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "exported_at": _utc_now(),
        "source": {
            "env_path": str(env_path),
            "env_class": env.__class__.__name__,
            "prompt": prompt,
            "seed": ground_truth.get("seed"),
        },
        "semantic_context": _semantic_context(prompt, ground_truth),
        "world": {
            "width": config.get("width", getattr(env, "width", None)),
            "height": config.get("height", getattr(env, "height", None)),
            "gravity": config.get("gravity"),
            "time_step": config.get("time_step"),
            "damping": config.get("damping"),
            "iterations": config.get("iterations"),
            "agent_strength": config.get("agent_strength", getattr(env, "agent_strength", None)),
        },
        "objective": ground_truth.get("objective", {}),
        "solvability_check": ground_truth.get("solvability_check", {}),
        "objects": _export_objects(ground_truth.get("objects", {})),
        "constraints": _export_named_records(ground_truth.get("constraints", {})),
        "force_zones": _export_named_records(ground_truth.get("force_zones", {})),
        "mechanisms": _export_named_records(ground_truth.get("mechanisms", {})),
        "visual_brief": visual_brief,
        "verification": {
            "accepted": validation.accepted,
            "achieved_tier": validation.achieved_tier,
            "tier_name": validation.tier_name,
            "minimum_acceptance_tier": validation.minimum_acceptance_tier,
            "objective_tier": validation.objective_tier,
            "reason": validation.reason,
        },
    }


def _export_objects(objects: Any) -> list[dict[str, Any]]:
    if not isinstance(objects, dict):
        return []
    exported: list[dict[str, Any]] = []
    for name, record in objects.items():
        if not isinstance(record, dict):
            continue
        body = record.get("body") if isinstance(record.get("body"), dict) else {}
        shapes = record.get("shapes") if isinstance(record.get("shapes"), list) else []
        exported.append(
            {
                "name": name,
                "kind": record.get("kind"),
                "role": record.get("role"),
                "metadata": record.get("metadata", {}),
                "body": {
                    "type": body.get("type"),
                    "position": body.get("position"),
                    "velocity": body.get("velocity"),
                    "angle": body.get("angle"),
                    "angular_velocity": body.get("angular_velocity"),
                    "mass": body.get("mass"),
                    "moment": body.get("moment"),
                },
                "shapes": [_export_shape(shape) for shape in shapes if isinstance(shape, dict)],
            }
        )
    return exported


def _export_shape(shape: dict[str, Any]) -> dict[str, Any]:
    shape_type = str(shape.get("type") or "")
    base = {
        "type": _normalized_shape_type(shape_type),
        "source_type": shape_type,
        "sensor": bool(shape.get("sensor")),
        "collision_type": shape.get("collision_type"),
        "friction": shape.get("friction"),
        "elasticity": shape.get("elasticity"),
        "bb": shape.get("bb"),
    }
    if shape_type == "Circle":
        base.update(
            {
                "radius": shape.get("radius"),
                "center": shape.get("world_center"),
                "offset": shape.get("offset"),
            }
        )
    elif shape_type == "Segment":
        base.update(
            {
                "a": shape.get("world_a"),
                "b": shape.get("world_b"),
                "local_a": shape.get("a"),
                "local_b": shape.get("b"),
                "radius": shape.get("radius"),
            }
        )
    elif shape_type == "Poly":
        base.update(
            {
                "vertices": shape.get("world_vertices"),
                "local_vertices": shape.get("vertices"),
            }
        )
    return base


def _normalized_shape_type(shape_type: str) -> str:
    if shape_type == "Circle":
        return "circle"
    if shape_type == "Segment":
        return "segment"
    if shape_type == "Poly":
        return "polygon"
    return shape_type.lower() or "unknown"


def _export_named_records(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, dict):
        return []
    exported: list[dict[str, Any]] = []
    for name, payload in records.items():
        if isinstance(payload, dict):
            exported.append({"name": name, **payload})
    return exported


def _visual_brief(
    grammar: VisualGrammar,
    recipe: dict[str, Any] | None,
    *,
    prompt: str | None = None,
    ground_truth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recipe_payload = recipe or dict(grammar.visual_recipe or {})
    semantic_assets: list[dict[str, Any]] = []
    if resolve_semantic_assets is not None:
        try:
            semantic_assets = resolve_semantic_assets(
                prompt=prompt,
                recipe=recipe_payload,
                ground_truth=ground_truth or {},
                max_assets=14,
            )
        except Exception:
            semantic_assets = []
    return {
        "schema_version": "harness-alpha-visual-brief-v1",
        "prompt": prompt,
        "semantic_context": _semantic_context(prompt, ground_truth or {}),
        "mood": grammar.mood,
        "background": grammar.background,
        "accent": grammar.accent,
        "materials": list(grammar.materials),
        "lighting": grammar.lighting,
        "motion_fx": grammar.motion_fx,
        "surface_fx": grammar.surface_fx,
        "agent_avatar": grammar.agent_avatar,
        "shape_language": grammar.shape_language,
        "presentation": grammar.presentation,
        "palette": {
            "background_color": list(grammar.background_color),
            "primary": list(grammar.primary),
            "secondary": list(grammar.secondary),
            "hot": list(grammar.hot),
        },
        "source": grammar.source,
        "scores": dict(grammar.scores),
        "recipe": recipe_payload,
        "semantic_props": recipe_payload.get("semantic_props", [])
        if isinstance(recipe_payload.get("semantic_props"), list)
        else [],
        "semantic_assets": semantic_assets,
        "visual_program": _visual_program_from_recipe(recipe_payload),
    }


def _visual_program_from_recipe(recipe: dict[str, Any]) -> dict[str, Any] | None:
    program = recipe.get("visual_program") if isinstance(recipe, dict) else None
    return program if isinstance(program, dict) else None


def _semantic_context(prompt: str | None, ground_truth: dict[str, Any]) -> dict[str, Any]:
    objects = ground_truth.get("objects", {})
    names: list[str] = []
    roles: list[str] = []
    metadata_terms: list[str] = []
    if isinstance(objects, dict):
        for name, record in objects.items():
            names.append(str(name))
            if isinstance(record, dict):
                role = record.get("role")
                kind = record.get("kind")
                if role:
                    roles.append(str(role))
                if kind:
                    metadata_terms.append(str(kind))
                metadata = record.get("metadata")
                if isinstance(metadata, dict):
                    metadata_terms.extend(str(key) for key in metadata.keys())
                    metadata_terms.extend(str(value) for value in metadata.values() if isinstance(value, str | int | float | bool))
    objective = ground_truth.get("objective", {})
    prompt_text = str(prompt or "")
    combined = " ".join(
        [
            prompt_text,
            " ".join(names),
            " ".join(roles),
            " ".join(metadata_terms),
            json.dumps(objective, sort_keys=True, default=str) if isinstance(objective, dict) else str(objective),
        ]
    ).lower()
    keywords = sorted(
        {
            token
            for token in (
                "soccer",
                "ball",
                "goal",
                "lava",
                "fire",
                "space",
                "asteroid",
                "maze",
                "gate",
                "plate",
                "magnetic",
                "field",
                "cave",
                "survival",
                "falling",
                "throw",
                "kick",
                "push",
                "jump",
                "escape",
            )
            if token in combined
        }
    )
    return {
        "prompt": prompt_text,
        "object_names": names,
        "roles": sorted(set(roles)),
        "keywords": keywords,
        "objective_type": objective.get("objective_type") if isinstance(objective, dict) else None,
    }


def _verification_payload(
    validation: ValidationResult,
    env_path: Path,
    prompt: str | None,
) -> dict[str, Any]:
    payload = validation.to_dict()
    payload.update(
        {
            "schema_version": "harness-alpha-verification-v1",
            "exported_at": _utc_now(),
            "env_path": str(env_path),
            "prompt": prompt,
        }
    )
    return payload


def _load_recipe_for_export(env: BaseEnv, visual_recipe_path: str | Path | None) -> dict[str, Any] | None:
    if visual_recipe_path:
        path = Path(visual_recipe_path)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            return data if isinstance(data, dict) else None
    return load_visual_recipe(env)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    safe_payload = _json_safe(payload)
    path.write_text(
        json.dumps(safe_payload, indent=2, sort_keys=True, default=str, allow_nan=False),
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_safe(item) for item in value), key=str)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a Harness Alpha world schema for external runtimes.")
    parser.add_argument("env_path", type=Path, help="Generated environment .py file")
    parser.add_argument("--output-dir", type=Path, default=LATEST_EXPORT_DIR)
    parser.add_argument("--prompt")
    parser.add_argument("--visual-recipe", type=Path)
    parser.add_argument("--class-name")
    parser.add_argument("--require-accepted", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = export_verified_world(
        args.env_path,
        output_dir=args.output_dir,
        prompt=args.prompt,
        visual_recipe_path=args.visual_recipe,
        class_name=args.class_name,
        require_accepted=args.require_accepted,
    )
    print(f"WORLD_EXPORT: {output_dir}")
    print(f"WORLD_SCHEMA: {output_dir / 'world_schema.json'}")
    print(f"VISUAL_BRIEF: {output_dir / 'visual_brief.json'}")
    print(f"VERIFICATION: {output_dir / 'verification.json'}")


if __name__ == "__main__":
    main()
