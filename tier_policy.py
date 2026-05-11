"""Independent tier policy critic for Harness Alpha.

The Architect is allowed to be creative, but it should not grade its own work.
This module reviews the user's prompt plus generated objective/capability
profiles and returns a strict JSON policy that the validator can enforce.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from validator import load_env_class


POLICY_MEMORY_DIR = Path("policy_memory")
load_dotenv()

DEFAULT_POLICY_MODEL = os.getenv("OPENAI_POLICY_MODEL", os.getenv("OPENAI_ARCHITECT_MODEL", "gpt-5.2"))
DEFAULT_POLICY_REASONING_EFFORT = os.getenv("OPENAI_POLICY_REASONING_EFFORT", "low")
DEFAULT_POLICY_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_POLICY_MAX_OUTPUT_TOKENS", "1600"))


@dataclass(frozen=True)
class PolicyCriticConfig:
    """Runtime settings for the LLM policy critic."""

    model: str = DEFAULT_POLICY_MODEL
    reasoning_effort: str = DEFAULT_POLICY_REASONING_EFFORT
    max_output_tokens: int = DEFAULT_POLICY_MAX_OUTPUT_TOKENS
    enabled: bool = os.getenv("HARNESS_POLICY_CRITIC", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


class PolicyCritic:
    """LLM-backed reviewer for minimum validation tier and validator focus."""

    def __init__(self, config: PolicyCriticConfig | None = None) -> None:
        self.config = config or PolicyCriticConfig()
        self._client = None

    async def evaluate_env(
        self,
        *,
        prompt: str,
        env_path: str | Path,
        class_name: str | None = None,
    ) -> dict[str, Any]:
        """Return a strict tier policy for a generated environment."""

        profile_bundle = _load_profile_bundle(env_path, class_name)
        memory = _load_policy_memory()
        fallback = _fallback_policy(prompt, profile_bundle, memory)
        if not self.config.enabled or not os.getenv("OPENAI_API_KEY"):
            return fallback

        try:
            raw = await self._call_llm(prompt, profile_bundle, memory, fallback)
            policy = _parse_policy_json(raw)
        except Exception as exc:
            policy = {
                **fallback,
                "critic_backend": "fallback",
                "critic_error": f"{type(exc).__name__}: {exc}",
            }
        return _sanitize_policy(policy, fallback)

    async def _call_llm(
        self,
        prompt: str,
        profile_bundle: dict[str, Any],
        memory: dict[str, Any],
        fallback: dict[str, Any],
    ) -> str:
        client = self._get_client()
        response = await client.responses.create(
            model=self.config.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are the independent Harness Alpha Policy Critic. "
                        "You choose the required verification tier for generated "
                        "2D physics environments. Compare prompts conceptually by "
                        "physical intent, object roles, and success predicates. "
                        "Return only valid JSON with no markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": "Choose the minimum validation tier and validator focus.",
                            "user_prompt": prompt,
                            "generated_profiles": profile_bundle,
                            "policy_memory": memory,
                            "fallback_policy": fallback,
                            "required_schema": {
                                "objective_tier": "integer 0-5: ideal proof standard the task deserves in principle",
                                "operational_acceptance_tier": "integer 0-5: proof standard this harness should enforce today",
                                "required_tier": "integer 0-5 alias for operational_acceptance_tier",
                                "task_family": "short conceptual label",
                                "reason": "one or two sentences",
                                "capability_gap": "short explanation when operational tier is below objective tier",
                                "risk_flags": ["strings"],
                                "suggested_validator_focus": ["strings"],
                                "memory_matches": ["strings"],
                                "confidence": "float 0-1"
                            },
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                },
            ],
            reasoning={"effort": self.config.reasoning_effort},
            max_output_tokens=self.config.max_output_tokens,
        )
        output_text = getattr(response, "output_text", None)
        if not output_text:
            raise RuntimeError("OpenAI policy response did not include output_text")
        return str(output_text)

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError("The openai package is not installed.") from exc
            self._client = AsyncOpenAI()
        return self._client


def _load_profile_bundle(env_path: str | Path, class_name: str | None) -> dict[str, Any]:
    env_class = load_env_class(env_path, class_name=class_name)
    env = env_class()
    truth = env.get_ground_truth()
    objective = truth.get("objective") if isinstance(truth, dict) else {}
    objective = objective if isinstance(objective, dict) else {}
    return {
        "env_class": env_class.__name__,
        "objective_type": objective.get("objective_type"),
        "objective_targets": objective.get("objective_targets"),
        "objective_profile": objective.get("objective_profile"),
        "capability_profile": objective.get("capability_profile"),
        "objective_satisfied_initially": objective.get("objective_satisfied"),
    }


def _load_policy_memory(memory_dir: Path = POLICY_MEMORY_DIR) -> dict[str, Any]:
    memory: dict[str, Any] = {"rulebook": {}, "examples": []}
    rulebook_path = memory_dir / "tier_policy_rulebook.json"
    if rulebook_path.exists():
        memory["rulebook"] = _read_json(rulebook_path)
    examples_dir = memory_dir / "examples"
    if examples_dir.exists():
        for path in sorted(examples_dir.glob("*.json")):
            memory["examples"].append(_read_json(path))
    return memory


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_policy_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.strip().startswith("```"))
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Policy critic returned non-object JSON")
    return parsed


def _fallback_policy(
    prompt: str,
    profile_bundle: dict[str, Any],
    memory: dict[str, Any],
) -> dict[str, Any]:
    profile = profile_bundle.get("objective_profile")
    profile = profile if isinstance(profile, dict) else {}
    objective_type = str(profile.get("objective_type") or profile_bundle.get("objective_type") or "")
    subgoals = profile.get("subgoals")
    if not isinstance(subgoals, list):
        subgoals = []
    prompt_lower = prompt.lower()
    kinds = {str(item.get("kind") or "") for item in subgoals if isinstance(item, dict)}
    direct_kinds = {"agent_reach_region", "agent_touch_object", "move_object_to_region"}
    open_ended_markers = {
        "pacman",
        "game",
        "score",
        "enemy",
        "opponent",
        "survive",
        "avoid",
        "balance",
        "strategy",
    }
    if kinds & direct_kinds:
        objective_tier = 5
        required_tier = 5
        task_family = "finite_direct_subgoal_plan"
        reason = "The profile exposes executable finite subgoals, so solved verification is required."
    elif objective_type in {"navigation_goal", "single_target_touch", "multi_target_touch", "push_object"}:
        objective_tier = 5
        required_tier = 5
        task_family = f"finite_{objective_type}"
        reason = "This objective type has a direct deterministic predicate that should be solved."
    elif any(marker in prompt_lower for marker in open_ended_markers):
        objective_tier = 5
        required_tier = 4
        task_family = "open_ended_or_long_horizon"
        reason = "The prompt appears strategy-heavy or open-ended, so deterministic progress is acceptable."
    else:
        objective_tier = int(profile.get("objective_tier") or 5)
        required_tier = int(profile.get("minimum_acceptance_tier") or 4)
        task_family = objective_type or "custom_physics"
        reason = "Fallback policy used the generated profile with guardrails."
    return _sanitize_policy(
        {
            "objective_tier": objective_tier,
            "operational_acceptance_tier": required_tier,
            "required_tier": required_tier,
            "task_family": task_family,
            "reason": reason,
            "capability_gap": (
                "Current harness accepts progress evidence for this family."
                if required_tier < objective_tier
                else ""
            ),
            "risk_flags": [],
            "suggested_validator_focus": sorted(kinds) or list(profile.get("validator_skills") or []),
            "memory_matches": _conceptual_memory_matches(prompt_lower, memory),
            "confidence": 0.55,
            "critic_backend": "fallback",
        },
        None,
    )


def _conceptual_memory_matches(prompt_lower: str, memory: dict[str, Any]) -> list[str]:
    matches: list[str] = []
    for example in memory.get("examples") or []:
        if not isinstance(example, dict):
            continue
        aliases = example.get("conceptual_aliases") or {}
        alias_values: list[str] = []
        if isinstance(aliases, dict):
            for value in aliases.values():
                if isinstance(value, list):
                    alias_values.extend(str(item).lower() for item in value)
        if any(alias and alias in prompt_lower for alias in alias_values):
            matches.append(str(example.get("abstract_task") or example.get("task_family") or "memory_example"))
    return sorted(set(matches))


def _sanitize_policy(policy: dict[str, Any], fallback: dict[str, Any] | None) -> dict[str, Any]:
    source = fallback or {}
    sanitized = dict(policy)

    try:
        objective_tier = int(sanitized.get("objective_tier"))
    except (TypeError, ValueError):
        try:
            objective_tier = int(source.get("objective_tier", source.get("required_tier", 5)))
        except (TypeError, ValueError):
            objective_tier = 5
    objective_tier = max(0, min(5, objective_tier))

    try:
        operational_tier = int(
            sanitized.get(
                "operational_acceptance_tier",
                sanitized.get("required_tier"),
            )
        )
    except (TypeError, ValueError):
        operational_tier = int(
            source.get(
                "operational_acceptance_tier",
                source.get("required_tier", 4),
            )
        )
    operational_tier = max(0, min(5, operational_tier))
    fallback_tier = source.get("operational_acceptance_tier", source.get("required_tier"))
    if isinstance(fallback_tier, int):
        operational_tier = max(operational_tier, fallback_tier)
    if operational_tier > objective_tier:
        objective_tier = operational_tier

    sanitized["objective_tier"] = objective_tier
    sanitized["operational_acceptance_tier"] = operational_tier
    sanitized["required_tier"] = operational_tier
    sanitized["task_family"] = str(sanitized.get("task_family") or source.get("task_family") or "unknown")
    sanitized["reason"] = str(sanitized.get("reason") or source.get("reason") or "No policy reason supplied.")
    sanitized["capability_gap"] = str(
        sanitized.get("capability_gap")
        or source.get("capability_gap")
        or (
            "Operational tier is lower than the ideal objective tier because the current validator only proves progress for this family."
            if operational_tier < objective_tier
            else ""
        )
    )
    sanitized["risk_flags"] = _string_list(sanitized.get("risk_flags"))
    sanitized["suggested_validator_focus"] = _string_list(sanitized.get("suggested_validator_focus"))
    sanitized["memory_matches"] = _string_list(sanitized.get("memory_matches"))
    try:
        sanitized["confidence"] = max(0.0, min(1.0, float(sanitized.get("confidence"))))
    except (TypeError, ValueError):
        sanitized["confidence"] = float(source.get("confidence") or 0.0)
    sanitized.setdefault("critic_backend", "openai")
    return sanitized


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]
