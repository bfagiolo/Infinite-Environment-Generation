"""Semantic asset index and resolver for renderer-only world props.

The resolver is intentionally outside the physics contract. It maps prompt and
Visual Director semantics to local CC0 PNGs so richer runtimes can decorate a
world without changing collision shapes, objectives, or validation truth.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


ASSET_ROOT = Path("assets/library/kenney")
INDEX_PATH = Path("assets/asset_index.json")
SCHEMA_VERSION = "harness-alpha-asset-index-v1"
LICENSE_NOTE = "Kenney CC0 local asset packs; see each pack's License.txt."

SKIP_NAMES = {"preview", "sample", "sample1", "sample2", "preview_kenneynl"}

TAG_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("space", ("simple-space", "space", "ship", "meteor", "asteroid", "satellite", "enemy")),
    ("spaceship", ("ship_", "spaceship", "rocket")),
    ("ship", ("ship_", "satellite", "rocket")),
    ("asteroid", ("meteor", "asteroid")),
    ("meteor", ("meteor",)),
    ("star", ("star",)),
    ("forest", ("forest", "tree", "foliage", "leaf")),
    ("tree", ("tree", "foliagepack")),
    ("grass", ("grass", "foliage", "leaf")),
    ("leaf", ("leaf", "foliage")),
    ("cloud", ("cloud",)),
    ("sun", ("sun",)),
    ("moon", ("moon",)),
    ("sports", ("sports-pack", "ball_", "goal", "flag", "racket", "bat_")),
    ("ball", ("ball_", "ball")),
    ("soccer", ("soccer", "football")),
    ("basketball", ("basket", "basketball")),
    ("goal", ("goal", "flag_checkered", "flag_green")),
    ("flag", ("flag",)),
    ("platform", ("platform", "tile", "terrain_", "ground")),
    ("crate", ("crate", "box")),
    ("box", ("crate", "box")),
    ("coin", ("coin",)),
    ("crystal", ("crystal", "gem", "diamond")),
    ("gem", ("gem", "diamond")),
    ("spike", ("spike",)),
    ("ladder", ("ladder",)),
    ("fire", ("fire", "flame", "flare")),
    ("lava", ("fire", "flame", "orange")),
    ("smoke", ("smoke", "dirt")),
    ("spark", ("spark", "muzzle", "magic", "light", "flare")),
    ("particle", ("particle-pack", "circle", "magic", "light", "spark", "flame")),
    ("castle", ("castle", "tower", "temple")),
    ("building", ("house", "castle", "tower", "temple")),
)

PROMPT_PROP_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...], str, str], ...] = (
    (("space", "asteroid", "zero gravity", "zero-g", "spaceship", "rocket"), ("ship", "asteroid", "star", "meteor"), "background_prop", "sky"),
    (("crystal", "gem", "collect"), ("crystal", "gem", "star"), "foreground_prop", "objective"),
    (("forest", "garden", "corn", "jungle", "tree", "woods"), ("tree", "grass", "leaf", "cloud"), "background_prop", "edges"),
    (("lava", "fire", "volcano", "fireball", "ember"), ("fire", "spark", "smoke", "lava"), "effect_prop", "hazard"),
    (("basketball", "hoop", "basketball court"), ("basketball", "ball", "flag"), "foreground_prop", "objective"),
    (("soccer", "football", "goal line", "kick"), ("soccer", "ball", "goal", "flag"), "foreground_prop", "objective"),
    (("platform", "jump", "climb", "tower"), ("platform", "crate", "ladder", "spike"), "set_dressing", "platforms"),
    (("castle", "temple", "village", "house"), ("castle", "building", "tree"), "background_prop", "horizon"),
)


@dataclass(frozen=True)
class AssetRecord:
    path: str
    pack: str
    name: str
    role: str
    semantic_tags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "pack": self.pack,
            "name": self.name,
            "role": self.role,
            "semantic_tags": list(self.semantic_tags),
        }


def build_asset_index(
    root: Path = ASSET_ROOT,
    output_path: Path = INDEX_PATH,
) -> dict[str, Any]:
    """Scan local asset packs and write a compact semantic index."""

    root = Path(root)
    records: list[AssetRecord] = []
    if not root.exists():
        raise FileNotFoundError(f"asset root not found: {root}")

    for path in sorted(root.rglob("*.png")):
        if "_downloads" in path.parts:
            continue
        if any(part.lower() in {"spritesheet", "tilesheet", "vector"} for part in path.parts):
            continue
        stem = _normalize_token_text(path.stem)
        if stem in SKIP_NAMES or stem.startswith("preview") or stem.startswith("sample") or "sheet" in stem:
            continue
        pack = _pack_name(root, path)
        tags = _tags_for_path(root, path)
        records.append(
            AssetRecord(
                path=_repo_relative(path),
                pack=pack,
                name=path.name,
                role=_role_for_path(path, tags),
                semantic_tags=tuple(sorted(tags)),
            )
        )

    by_tag: dict[str, list[str]] = defaultdict(list)
    for record in records:
        for tag in record.semantic_tags:
            by_tag[tag].append(record.path)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "root": _repo_relative(root),
        "license": LICENSE_NOTE,
        "asset_count": len(records),
        "assets": [record.to_dict() for record in records],
        "by_tag": {tag: paths[:80] for tag, paths in sorted(by_tag.items())},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_asset_index(index_path: Path = INDEX_PATH) -> dict[str, Any]:
    """Load the semantic asset index, building it when local packs exist."""

    index_path = Path(index_path)
    if not index_path.exists() and ASSET_ROOT.exists():
        return build_asset_index(output_path=index_path)
    if not index_path.exists():
        return {}
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def resolve_semantic_assets(
    *,
    prompt: str | None = None,
    recipe: dict[str, Any] | None = None,
    ground_truth: dict[str, Any] | None = None,
    max_assets: int = 12,
    index_path: Path = INDEX_PATH,
) -> list[dict[str, Any]]:
    """Choose a small, diverse set of local decorative assets for a world."""

    index = load_asset_index(index_path)
    assets = index.get("assets", [])
    if not isinstance(assets, list) or not assets:
        return []

    requested = _requested_semantics(prompt=prompt, recipe=recipe, ground_truth=ground_truth)
    if not requested:
        return []

    chosen: list[dict[str, Any]] = []
    used_paths: set[str] = set()
    seed = _stable_seed(" ".join([str(prompt or ""), json.dumps(requested, sort_keys=True, default=str)]))
    for semantic in requested:
        tags = [str(tag).lower() for tag in semantic.get("tags", []) if str(tag).strip()]
        generic_tags = {"goal", "target", "collectible", "glowing", "touch_contact", "feedback"}
        specific_tags = [tag for tag in tags if tag not in generic_tags]
        search_tags = specific_tags or tags
        matches = _rank_assets(assets, search_tags, seed)
        for record in matches:
            path = str(record.get("path", ""))
            if not path or path in used_paths:
                continue
            used_paths.add(path)
            chosen.append(
                {
                    "semantic": semantic.get("kind", tags[0] if tags else "prop"),
                    "requested_tags": tags,
                    "path": path,
                    "pack": record.get("pack"),
                    "name": record.get("name"),
                    "asset_tags": record.get("semantic_tags", []),
                    "role": semantic.get("role") or record.get("role") or "background_prop",
                    "placement": semantic.get("placement", "background"),
                    "renderer_only": True,
                }
            )
            break
        if len(chosen) >= max_assets:
            break
    return chosen


def _requested_semantics(
    *,
    prompt: str | None,
    recipe: dict[str, Any] | None,
    ground_truth: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    text = _semantic_query_text(prompt=prompt, recipe=recipe, ground_truth=ground_truth)
    requested: list[dict[str, Any]] = []

    if isinstance(recipe, dict):
        for prop in recipe.get("semantic_props", []) or []:
            if not isinstance(prop, dict):
                continue
            tags = _semantic_tags_from_prop(prop)
            if tags:
                requested.append(
                    {
                        "kind": str(prop.get("kind") or tags[0]),
                        "tags": tags,
                        "role": str(prop.get("role") or prop.get("importance") or "background_prop"),
                        "placement": str(prop.get("placement") or "background"),
                    }
                )

    for triggers, tags, role, placement in PROMPT_PROP_RULES:
        if _contains_any_semantic_term(text, triggers):
            for tag in tags:
                requested.append({"kind": tag, "tags": [tag], "role": role, "placement": placement})

    object_names = []
    objects = (ground_truth or {}).get("objects", {})
    if isinstance(objects, dict):
        object_names = [str(name).lower() for name in objects.keys()]
    for name in object_names:
        for tag in ("fire", "ball", "rock", "asteroid", "gate", "goal", "box", "tree", "ship"):
            if tag in name:
                requested.append({"kind": tag, "tags": [tag], "role": "object_accent", "placement": "object"})

    return _dedupe_requested(requested)


def _semantic_query_text(
    *,
    prompt: str | None,
    recipe: dict[str, Any] | None,
    ground_truth: dict[str, Any] | None,
) -> str:
    parts: list[str] = [str(prompt or "")]
    if isinstance(recipe, dict):
        for key in ("world_identity", "agent_avatar"):
            value = recipe.get(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, dict):
                parts.extend(str(item) for item in value.values() if isinstance(item, str))
        for prop in recipe.get("semantic_props", []) or []:
            if isinstance(prop, dict):
                parts.extend(_semantic_tags_from_prop(prop))
    objective = (ground_truth or {}).get("objective", {})
    if isinstance(objective, dict):
        for key in ("objective_type", "description", "objective_description"):
            value = objective.get(key)
            if isinstance(value, str):
                parts.append(value)
    return " ".join(parts).lower()


def _contains_any_semantic_term(text: str, terms: Iterable[str]) -> bool:
    for term in terms:
        normalized = str(term).lower().strip()
        if not normalized:
            continue
        if " " in normalized:
            if normalized in text:
                return True
        elif re.search(rf"\b{re.escape(normalized)}\b", text):
            return True
    return False


def _semantic_tags_from_prop(prop: dict[str, Any]) -> list[str]:
    raw: list[Any] = []
    raw.extend(prop.get("semantic_tags", []) if isinstance(prop.get("semantic_tags"), list) else [])
    raw.extend(prop.get("tags", []) if isinstance(prop.get("tags"), list) else [])
    for key in ("kind", "label", "asset_hint", "description"):
        value = prop.get(key)
        if isinstance(value, str):
            raw.extend(re.split(r"[^a-zA-Z0-9]+", value))
    tags = [tag.lower() for tag in raw if isinstance(tag, str) and tag.strip()]
    return [tag for tag in tags if len(tag) > 1]


def _dedupe_requested(requested: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in requested:
        tags = [str(tag).lower() for tag in item.get("tags", []) if str(tag).strip()]
        if not tags:
            continue
        key = "|".join([str(item.get("kind", "")), ",".join(tags), str(item.get("placement", ""))])
        if key in seen:
            continue
        seen.add(key)
        deduped.append({**item, "tags": tags})
    return deduped


def _rank_assets(assets: list[Any], tags: list[str], seed: int) -> list[dict[str, Any]]:
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        asset_tags = {str(tag).lower() for tag in asset.get("semantic_tags", [])}
        name = str(asset.get("name", "")).lower()
        pack = str(asset.get("pack", "")).lower()
        score = 0
        for tag in tags:
            if tag in asset_tags:
                score += 8
            if tag in name:
                score += 5
            if tag in pack:
                score += 3
        if score <= 0:
            continue
        path_text = str(asset.get("path", "")).lower()
        if "transparent" in path_text:
            score += 3
        if "black background" in path_text:
            score -= 6
        jitter = _stable_seed(str(asset.get("path", "")) + str(seed)) % 997
        ranked.append((-score, jitter, asset))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [asset for _, _, asset in ranked]


def _tags_for_path(root: Path, path: Path) -> set[str]:
    relative = path.relative_to(root).as_posix().lower()
    normalized = _normalize_token_text(relative)
    tags: set[str] = set()
    for tag, needles in TAG_RULES:
        if any(_needle_matches(normalized, needle) for needle in needles):
            tags.add(tag)
    pack = _pack_name(root, path)
    tags.add(pack)
    if "background" in normalized:
        tags.add("background")
    if "equipment" in normalized:
        tags.add("equipment")
    if "characters" in normalized or "character" in normalized:
        tags.add("character")
    return tags


def _needle_matches(text: str, needle: str) -> bool:
    normalized = needle.lower().strip()
    if not normalized:
        return False
    if normalized.endswith("_") or normalized.endswith("-"):
        return normalized in text
    if "_" in normalized or "-" in normalized:
        return normalized in text
    return re.search(rf"\b{re.escape(normalized)}(?:\b|[_\-\d])", text) is not None


def _role_for_path(path: Path, tags: set[str]) -> str:
    text = path.as_posix().lower()
    if {"fire", "spark", "particle", "smoke"} & tags:
        return "effect_prop"
    if {"tree", "grass", "cloud", "building", "castle", "space", "star"} & tags or "background" in text:
        return "background_prop"
    if {"ball", "soccer", "basketball", "goal", "flag"} & tags:
        return "foreground_prop"
    if "character" in tags:
        return "avatar_prop"
    return "set_dressing"


def _pack_name(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).parts[0]
    except (ValueError, IndexError):
        return "unknown"


def _normalize_token_text(value: str) -> str:
    return re.sub(r"[^a-z0-9_ -]+", " ", value.lower())


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or query the Harness Alpha semantic asset index.")
    subparsers = parser.add_subparsers(dest="command")
    build = subparsers.add_parser("build", help="scan assets/library/kenney and write assets/asset_index.json")
    build.add_argument("--root", type=Path, default=ASSET_ROOT)
    build.add_argument("--output", type=Path, default=INDEX_PATH)
    resolve = subparsers.add_parser("resolve", help="print resolved assets for a prompt")
    resolve.add_argument("prompt")
    resolve.add_argument("--max-assets", type=int, default=12)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "build":
        payload = build_asset_index(root=args.root, output_path=args.output)
        print(f"ASSET_INDEX: {args.output}")
        print(f"ASSET_COUNT: {payload.get('asset_count', 0)}")
    elif args.command == "resolve":
        assets = resolve_semantic_assets(prompt=args.prompt, max_assets=args.max_assets)
        print(json.dumps(assets, indent=2, sort_keys=True))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
