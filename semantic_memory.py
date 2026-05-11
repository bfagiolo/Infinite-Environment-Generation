"""Conceptual retrieval over learned Harness Alpha generation memory.

This module is deliberately not a prompt cache. Exact prompt caching lives in
``prompt_cache.py``. Here we retrieve small, reusable lessons from prior
successes and failures so a new prompt can borrow physical constraints without
being forced into an old template.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parent
LEARNED_SKILLS_DIR = ROOT / "skills" / "learned"
BASE_SKILLS_DIR = ROOT / "skills"
BASE_AFFORDANCE_BLOCKS_DIR = ROOT / "affordance_blocks"
LEARNED_AFFORDANCE_BLOCKS_DIR = BASE_AFFORDANCE_BLOCKS_DIR / "learned"
CAPABILITY_GAPS_DIR = ROOT / "capability_gaps"

MAX_RECORD_CHARS = 9000
STOPWORDS = {
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
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "must",
    "of",
    "on",
    "onto",
    "or",
    "the",
    "to",
    "while",
    "where",
    "with",
}

CONCEPT_ALIASES: dict[str, set[str]] = {
    "move_object_to_region": {
        "box",
        "barrel",
        "crate",
        "nudge",
        "push",
        "rock",
        "roll",
        "shove",
        "slide",
    },
    "pressure_trigger": {"button", "plate", "pressure", "switch", "trigger"},
    "mechanism_gate": {"door", "gate", "lock", "open", "sliding"},
    "agent_reach_region": {"escape", "exit", "goal", "maze", "navigate", "reach"},
    "avoid_contact_until_goal": {
        "avoid",
        "bear",
        "chase",
        "chaser",
        "enemy",
        "pursue",
        "pursuer",
        "touched",
    },
    "survival_timer": {"seconds", "survive", "timer", "until"},
    "ballistic_object_to_region": {
        "arc",
        "basket",
        "basketball",
        "clear",
        "hoop",
        "lob",
        "over",
        "throw",
    },
    "strike_impulse": {"ball", "kick", "shot", "soccer", "strike"},
    "projectile_hazard": {
        "bullet",
        "laser",
        "projectile",
        "shoot",
        "shot",
        "spaceship",
        "zap",
    },
    "falling_hazard": {"drop", "falling", "fireball", "hazard", "lava", "rain"},
    "zero_g_motion": {"asteroid", "drift", "floating", "salvage", "space", "zero", "zero-g"},
    "water_buoyancy": {"buoyancy", "float", "swim", "water"},
    "pivot_mechanism": {"balance", "lever", "pivot", "seesaw", "teeter"},
    "field_force": {"charge", "electric", "field", "force", "magnetic"},
    "support_exit": {"cliff", "edge", "fall", "off", "platform", "support"},
}

CONCEPT_EXCLUSIONS: dict[str, set[str]] = {
    "strike_impulse": {"laser", "projectile", "spaceship"},
    "projectile_hazard": {"basketball", "hoop", "soccer"},
}


@dataclass(frozen=True)
class MemoryRecord:
    """Loaded learned/base memory record."""

    kind: str
    path: Path
    payload: dict[str, Any]
    tokens: set[str]
    concepts: set[str]
    surface_concepts: set[str]


def retrieve_semantic_memory(
    prompt: str,
    *,
    simulation_brief: dict[str, Any] | None = None,
    gameplay_profile: dict[str, Any] | None = None,
    physics_relations: dict[str, Any] | None = None,
    layout_plan: dict[str, Any] | None = None,
    max_matches: int = 6,
) -> dict[str, Any]:
    """Return compact conceptual memories relevant to a new prompt.

    The returned JSON is meant to be injected into the generation prompt. It is
    intentionally advisory: it describes reusable physics lessons and failure
    modes, not code snippets or fixed layouts.
    """

    query_tokens = _extract_tokens(
        _query_text(
            prompt,
            simulation_brief=simulation_brief,
            gameplay_profile=gameplay_profile,
            physics_relations=physics_relations,
            layout_plan=layout_plan,
        )
    )
    query_concepts = _infer_concepts(query_tokens)
    relation_concepts = _relation_concepts(physics_relations)
    query_concepts |= relation_concepts

    records = _load_memory_records()
    scored: list[tuple[float, MemoryRecord]] = []
    for record in records:
        score = _score_record(
            query_tokens=query_tokens,
            query_concepts=query_concepts,
            record=record,
        )
        if score >= 0.35:
            scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    matches = [
        _compact_match(record, score)
        for score, record in _diverse_top_matches(scored, max_matches=max_matches)
    ]

    return {
        "version": "semantic-memory-1.0",
        "mode": "conceptual_retrieval_not_template_cache",
        "query_concepts": sorted(query_concepts),
        "query_keywords": sorted(_important_keywords(query_tokens))[:28],
        "memory_count": len(records),
        "matches": matches,
        "prompt_guidance": _memory_guidance(matches),
    }


def _load_memory_records() -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    specs = [
        ("learned_skill", LEARNED_SKILLS_DIR),
        ("base_skill", BASE_SKILLS_DIR),
        ("base_affordance", BASE_AFFORDANCE_BLOCKS_DIR),
        ("learned_affordance", LEARNED_AFFORDANCE_BLOCKS_DIR),
        ("capability_gap", CAPABILITY_GAPS_DIR),
    ]
    for kind, directory in specs:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.json")):
            if "learned" in path.parts and kind in {"base_skill", "base_affordance"}:
                continue
            payload = _read_json(path)
            if not payload:
                continue
            text = _record_text(payload)
            tokens = _extract_tokens(text)
            surface_concepts = _infer_concepts(_extract_tokens(_surface_record_text(payload)))
            concepts = _infer_concepts(tokens)
            concepts |= _relation_concepts(payload.get("physics_relations"))
            validator_route = payload.get("validator_route")
            if isinstance(validator_route, dict):
                concepts |= _relation_concepts(validator_route.get("physics_relations"))
            abstract_relation = payload.get("abstract_relation")
            if isinstance(abstract_relation, str):
                concepts.add(_normalize_concept(abstract_relation))
            records.append(
                MemoryRecord(
                    kind=kind,
                    path=path,
                    payload=payload,
                    tokens=tokens,
                    concepts=concepts,
                    surface_concepts=surface_concepts,
                )
            )
    return records


def _score_record(
    *,
    query_tokens: set[str],
    query_concepts: set[str],
    record: MemoryRecord,
) -> float:
    concept_overlap = query_concepts & record.concepts
    surface_overlap = query_concepts & record.surface_concepts
    token_overlap = _important_keywords(query_tokens) & _important_keywords(record.tokens)
    score = len(surface_overlap) * 0.42 + len(concept_overlap) * 0.14 + len(token_overlap) * 0.035

    if concept_overlap and not surface_overlap:
        score -= 0.28
    if {"move_object_to_region", "pressure_trigger"} <= query_concepts:
        if "move_object_to_region" not in record.surface_concepts:
            score -= 0.22
        if "pressure_trigger" not in record.surface_concepts:
            score -= 0.16
    if {"projectile_hazard", "survival_timer"} <= query_concepts:
        if "projectile_hazard" not in record.surface_concepts:
            score -= 0.2
        if "survival_timer" not in record.surface_concepts:
            score -= 0.12

    if record.kind in {"learned_affordance", "learned_skill"}:
        score += 0.08
    if record.kind == "capability_gap":
        score += 0.04
    if _contains_any(record.tokens, {"failed", "failure", "gap"}) and concept_overlap:
        score += 0.03

    for concept, exclusions in CONCEPT_EXCLUSIONS.items():
        if concept in record.concepts and concept in query_concepts:
            if exclusions & query_tokens:
                score -= 0.45
    return max(score, 0.0)


def _diverse_top_matches(
    scored: list[tuple[float, MemoryRecord]],
    *,
    max_matches: int,
) -> list[tuple[float, MemoryRecord]]:
    selected: list[tuple[float, MemoryRecord]] = []
    seen_fingerprints: set[str] = set()
    for score, record in scored:
        fingerprint = _match_fingerprint(record)
        if fingerprint in seen_fingerprints:
            continue
        selected.append((round(score, 3), record))
        seen_fingerprints.add(fingerprint)
        if len(selected) >= max_matches:
            break
    return selected


def _compact_match(record: MemoryRecord, score: float) -> dict[str, Any]:
    payload = record.payload
    source_prompt = _string(payload.get("source_prompt"))
    verification = payload.get("verification")
    if not isinstance(verification, dict):
        verification = payload.get("verification_summary")
    if not isinstance(verification, dict):
        verification = {}

    return {
        "kind": record.kind,
        "score": score,
        "id": _string(payload.get("block_id") or payload.get("skill_id") or payload.get("gap_id") or record.path.stem),
        "source_prompt": source_prompt,
        "summary": _string(payload.get("summary") or payload.get("capability_gap")),
        "abstract_relation": _string(payload.get("abstract_relation") or payload.get("objective_type") or payload.get("task_family")),
        "matched_concepts": sorted(record.concepts)[:16],
        "reusable_constraints": _limited_list(payload.get("constraints"), 5),
        "repair_lessons": _repair_lessons(payload),
        "validator_checks": _limited_list(payload.get("validator_checks") or payload.get("validator_focus"), 6),
        "verification": {
            "accepted": verification.get("accepted"),
            "achieved_tier": verification.get("achieved_tier"),
            "tier_name": verification.get("tier_name"),
            "objective_tier": verification.get("objective_tier"),
        },
        "source_path": str(record.path.relative_to(ROOT)),
        "reuse_policy": (
            "Conceptual prior only. Reuse relation constraints, telemetry lessons, "
            "and repair knobs; do not copy object names, theme, or layout unless the "
            "current prompt explicitly asks for the same thing."
        ),
    }


def _repair_lessons(payload: dict[str, Any]) -> list[str]:
    lessons = []
    for key in ("repair_guidance", "generation_guidance", "common_failures", "risk_flags"):
        lessons.extend(_limited_list(payload.get(key), 4))
    missing = _string(payload.get("missing_capability"))
    if missing:
        lessons.append(f"Known missing capability to avoid: {missing}")
    return lessons[:8]


def _memory_guidance(matches: list[dict[str, Any]]) -> list[str]:
    guidance = [
        "Use semantic memory as a physics prior, not as a replacement prompt.",
        "Preserve the current prompt's nouns, style, gravity context, objective, and constraints even when a prior match is close.",
        "Borrow only the reusable physical relation: alignment, mass/friction ranges, support geometry, sensor placement, route clearance, and validator probes.",
        "If a prior failure match appears, proactively avoid its named failure mode before generation.",
    ]
    if not matches:
        guidance.append("No strong prior memory matched; compose from the physics relation graph and BaseEnv primitives.")
    return guidance


def _extract_tokens(value: Any) -> set[str]:
    text = _record_text(value)
    raw_tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())
    tokens = {token for token in raw_tokens if len(token) > 1 and token not in STOPWORDS}
    normalized = set(tokens)
    if "zero" in tokens and "gravity" in tokens:
        normalized.add("zero-g")
    if "top" in tokens and "right" in tokens:
        normalized.add("top-right")
    return normalized


def _infer_concepts(tokens: set[str]) -> set[str]:
    concepts: set[str] = set()
    for concept, aliases in CONCEPT_ALIASES.items():
        if aliases & tokens:
            concepts.add(concept)
    if {"agent_reach_region", "avoid_contact_until_goal"} <= concepts:
        concepts.add("finite_avoidance_navigation")
    if {"move_object_to_region", "pressure_trigger"} <= concepts:
        concepts.add("mechanism_activation")
    if {"ballistic_object_to_region", "strike_impulse"} <= concepts:
        concepts.add("ballistic_strike_goal")
    if {"falling_hazard", "agent_reach_region"} <= concepts:
        concepts.add("hazard_navigation")
    if {"projectile_hazard", "survival_timer"} <= concepts:
        concepts.add("projectile_survival")
    return concepts


def _relation_concepts(value: Any) -> set[str]:
    concepts: set[str] = set()
    if not isinstance(value, dict):
        return concepts
    for relation in value.get("relations", []) if isinstance(value.get("relations"), list) else []:
        if not isinstance(relation, dict):
            continue
        relation_type = _normalize_concept(_string(relation.get("type")))
        if relation_type:
            concepts.add(relation_type)
        for probe in relation.get("probes", []) if isinstance(relation.get("probes"), list) else []:
            concepts.add(_normalize_concept(str(probe)))
    for subgoal in value.get("suggested_subgoals", []) if isinstance(value.get("suggested_subgoals"), list) else []:
        if isinstance(subgoal, dict):
            concepts.add(_normalize_concept(_string(subgoal.get("kind"))))
    return {concept for concept in concepts if concept}


def _important_keywords(tokens: set[str]) -> set[str]:
    return {
        token
        for token in tokens
        if token not in STOPWORDS
        and not token.isdigit()
        and len(token) > 2
        and token not in {"agent", "object", "region", "target", "goal", "world"}
    }


def _contains_any(tokens: set[str], options: set[str]) -> bool:
    return bool(tokens & options)


def _match_fingerprint(record: MemoryRecord) -> str:
    payload = record.payload
    relation = _string(payload.get("abstract_relation") or payload.get("objective_type") or payload.get("task_family"))
    source_prompt = _string(payload.get("source_prompt"))
    if relation and source_prompt:
        return f"{record.kind}:{relation}:{source_prompt[:80]}"
    return f"{record.kind}:{record.path.stem}"


def _record_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:MAX_RECORD_CHARS]
    try:
        return json.dumps(value, sort_keys=True, default=str)[:MAX_RECORD_CHARS]
    except TypeError:
        return str(value)[:MAX_RECORD_CHARS]


def _surface_record_text(payload: dict[str, Any]) -> str:
    """Return fields that describe the memory, excluding deep validator noise."""

    fields = {
        key: payload.get(key)
        for key in (
            "source_prompt",
            "summary",
            "abstract_relation",
            "objective_type",
            "task_family",
            "trigger_keywords",
            "creates",
            "constraints",
            "repair_guidance",
            "generation_guidance",
            "common_failures",
            "risk_flags",
        )
        if key in payload
    }
    return _record_text(fields)


def _query_text(
    prompt: str,
    *,
    simulation_brief: dict[str, Any] | None,
    gameplay_profile: dict[str, Any] | None,
    physics_relations: dict[str, Any] | None,
    layout_plan: dict[str, Any] | None,
) -> str:
    """Build query text from semantic facts, excluding generic instruction prose."""

    query: dict[str, Any] = {"prompt": prompt}
    if isinstance(simulation_brief, dict):
        query["simulation_brief"] = {
            "intent_summary": simulation_brief.get("intent_summary"),
            "semantic_must_happen": simulation_brief.get("semantic_must_happen"),
            "objective": simulation_brief.get("objective"),
            "world_context": simulation_brief.get("world_context"),
            "entities": _entity_briefs(simulation_brief.get("entities")),
        }
    if isinstance(gameplay_profile, dict):
        query["gameplay_profile"] = {
            "gameplay_loop": gameplay_profile.get("gameplay_loop"),
            "difficulty": gameplay_profile.get("difficulty"),
            "world_context": gameplay_profile.get("world_context"),
            "dynamics": _entity_briefs(gameplay_profile.get("dynamics")),
            "simulation_brief_alignment": gameplay_profile.get("simulation_brief_alignment"),
        }
    if isinstance(physics_relations, dict):
        query["physics_relations"] = {
            "intent_summary": physics_relations.get("intent_summary"),
            "world_context": physics_relations.get("world_context"),
            "relations": physics_relations.get("relations"),
            "suggested_subgoals": physics_relations.get("suggested_subgoals"),
            "physical_parameters": physics_relations.get("physical_parameters"),
        }
    if isinstance(layout_plan, dict):
        query["layout_plan"] = {
            "layout_type": layout_plan.get("layout_type"),
            "objective_critical_structure": layout_plan.get("objective_critical_structure"),
            "route_model": layout_plan.get("route_model"),
        }
    return _record_text(query)


def _entity_briefs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    briefs = []
    for item in value:
        if not isinstance(item, dict):
            continue
        briefs.append(
            {
                "name": item.get("name"),
                "role": item.get("role"),
                "type": item.get("type"),
                "expected_motion": item.get("expected_motion"),
            }
        )
    return briefs


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _limited_list(value: Any, limit: int) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [_string(item) for item in value[:limit] if _string(item)]


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _normalize_concept(value: str) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", value.lower()))
