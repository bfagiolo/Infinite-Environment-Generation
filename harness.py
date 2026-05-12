"""Recursive Harness Alpha brain: architect, validate, repair, and pivot."""

from __future__ import annotations

import argparse
import asyncio
import ast
from dataclasses import dataclass, field, replace
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap
from typing import Any, Protocol

from architect import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_REASONING_EFFORT,
    OpenAIArchitect,
    OpenAIArchitectConfig,
    diagnose_spec_error,
    extract_python_code,
    render_prompt,
    save_generated_code,
)
from auto_repair import auto_repair_generated_env
from gameplay_architect import GameplayArchitect
from layout_planner import infer_layout_plan
from physics_relations import infer_physics_relation_graph
from semantic_memory import retrieve_semantic_memory
from simulation_brief import SimulationBriefArchitect
from tier_policy import PolicyCritic
from validator import ValidationResult, ValidatorConfig, validate_generated_env
from visual_director import create_visual_recipe
from world_exporter import export_verified_world


LOGS_DIR = Path("logs")
CANDIDATE_ENVS_DIR = Path("candidate_envs")
GENERATED_ENVS_DIR = Path("generated_envs")
LEARNED_SKILLS_DIR = Path("skills") / "learned"
LEARNED_AFFORDANCE_BLOCKS_DIR = Path("affordance_blocks") / "learned"
CAPABILITY_GAPS_DIR = Path("capability_gaps")
MAX_SEEDS = 5
MAX_REPAIRS = 3
MAX_AUTO_REPAIRS = 4
MAX_CONTRACT_FAST_LANE_RETRIES = 2
TIER4_LOCAL_PIVOTS = 4
TIER3_LOCAL_PIVOTS = 2

DIVERSITY_DIRECTIVES = {
    1: "Baseline layout: open space with central obstacles and a clear primary route.",
    2: "Inverse layout: peripheral obstacles with a central goal region and a reversed route.",
    3: "High-entropy layout: complex maze with dynamic or moving elements, but still solvable.",
    4: "Mechanics-first layout: compact interaction space with short, readable cause-and-effect chains.",
    5: "Wide safety-margin layout: simple geometry, generous clearances, and validator-friendly object placement.",
}


@dataclass(frozen=True)
class HarnessConfig:
    """Control parameters for the recursive Reflexion loop."""

    max_seeds: int = MAX_SEEDS
    max_repairs: int = MAX_REPAIRS
    validator: ValidatorConfig = field(default_factory=lambda: ValidatorConfig(simulation_steps=4))
    execution_mode: str = "normal"


@dataclass(frozen=True)
class GenerationContext:
    """Context sent to an architect backend for a single attempt."""

    original_prompt: str
    enhanced_request: str
    class_name: str
    seed_index: int
    repair_index: int
    diversity_directive: str
    correction_prompt: str
    previous_result: ValidationResult | None = None
    previous_error: str | None = None


@dataclass(frozen=True)
class AttemptRecord:
    """Durable record for one seed/repair attempt."""

    seed_index: int
    repair_index: int
    attempt_dir: Path
    class_name: str
    generated_code_path: Path | None
    validation_path: Path
    result: ValidationResult


@dataclass(frozen=True)
class AttemptExecution:
    """Result of executing one seed/repair attempt."""

    record: AttemptRecord
    candidate_env_path: Path | None
    generated_code_path: Path | None
    validation: ValidationResult
    error: str | None = None


@dataclass(frozen=True)
class HarnessResult:
    """Final harness outcome."""

    success: bool
    prompt: str
    run_dir: Path
    generated_env_path: Path | None
    validation: ValidationResult | None
    attempts: tuple[AttemptRecord, ...]
    post_mortem: str | None = None
    capability_gap_path: Path | None = None
    visual_recipe_path: Path | None = None
    world_export_dir: Path | None = None
    simulation_brief: dict[str, Any] | None = None
    gameplay_profile: dict[str, Any] | None = None
    physics_relations: dict[str, Any] | None = None
    layout_plan: dict[str, Any] | None = None
    semantic_memory: dict[str, Any] | None = None


class ArchitectBackend(Protocol):
    """Async boundary for future LLM providers."""

    async def generate(self, context: GenerationContext) -> str:
        """Return an LLM response containing a generated Python code block."""


class LocalTemplateArchitect:
    """Deterministic local stand-in for Codex/LLM smoke tests.

    The harness is designed to swap this backend for a real LLM provider. This
    local backend exists so the complete Reflexion pipeline can be tested in CI
    and offline development without credentials.
    """

    async def generate(self, context: GenerationContext) -> str:
        await asyncio.sleep(0)
        code = _build_local_env_code(context)
        return f"```python\n{code}\n```"


def launch_visualizer(env_path: str | Path) -> subprocess.Popen:
    """Launch the Pygame telemetry dashboard for a verified environment."""

    return subprocess.Popen([sys.executable, "visualizer.py", str(env_path)])


async def _design_gameplay_profile(
    gameplay_architect: GameplayArchitect,
    prompt: str,
    run_dir: Path,
    simulation_brief: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the run-level gameplay contract without blocking on failure."""

    try:
        profile = await gameplay_architect.design(prompt, simulation_brief=simulation_brief)
    except Exception as exc:
        profile = {
            "source": "harness_error_fallback",
            "source_prompt": prompt,
            "gameplay_loop": "physics_objective",
            "difficulty": "medium",
            "dynamics": [],
            "feel_targets": {
                "readability": "high",
                "responsiveness": "medium",
                "forgiveness": "medium",
                "feedback": "visible",
            },
            "fairness_rules": ["do not let gameplay-profile failure block world generation"],
            "validator_expectations": ["objective_can_be_checked_from_pymunk_state"],
            "implementation_notes": [f"Gameplay Architect failed: {type(exc).__name__}: {exc}"],
        }
    _write_json(run_dir / "gameplay_profile.json", profile)
    return profile


async def _design_simulation_brief(
    brief_architect: SimulationBriefArchitect,
    prompt: str,
    run_dir: Path,
) -> dict[str, Any]:
    """Create the run-level prompt-to-physics interpretation contract."""

    try:
        brief = await brief_architect.design(prompt)
    except Exception as exc:
        brief = {
            "source": "harness_error_fallback",
            "intent_summary": prompt,
            "world_context": {
                "theme": "research_simulation",
                "perspective": "ground_lane_physics",
                "gravity": "normal",
                "movement_model": "ground_force",
                "support_model": "stable_floor_support",
                "rationale": f"Simulation Brief failed: {type(exc).__name__}: {exc}",
            },
            "agent": {
                "form": "simple_agent",
                "role": "agent",
                "controls": ["apply_force_x", "apply_force_y", "brake"],
                "capability_assumptions": ["ground_force", "normal"],
            },
            "objective": {
                "type": "navigation_goal",
                "success_condition": "agent reaches objective region",
                "failure_condition": "agent cannot reach objective region",
                "target_names_hint": ["goal"],
            },
            "entities": [],
            "semantic_must_happen": [],
            "validation": {
                "required_tier": 4,
                "semantic_checks": [],
                "objective_checks": ["objective state becomes true or progress is verified"],
                "capability_checks": ["agent can move"],
            },
            "visuals": {
                "style_intent": "research_simulation",
                "agent_avatar": "simple_agent",
                "important_props": [],
                "effect_notes": ["neon_grid"],
            },
            "ambiguity_notes": [],
            "warnings": [f"Simulation Brief failed: {type(exc).__name__}: {exc}"],
        }
    _write_json(run_dir / "simulation_brief.json", brief)
    return brief


async def _generate_candidate_artifacts(
    *,
    backend: ArchitectBackend,
    policy_critic: PolicyCritic,
    context: GenerationContext,
    attempt_dir: Path,
    clean_prompt: str,
    enhanced_request: str,
    class_name: str,
    artifact_suffix: str = "",
) -> tuple[Path, Path, dict[str, object]]:
    """Generate, contract-check, save, and tier-policy a candidate env."""

    suffix = f"_{artifact_suffix}" if artifact_suffix else ""
    llm_response = await backend.generate(context)
    _write_text(attempt_dir / f"llm_response{suffix}.txt", llm_response)
    generated_code = extract_python_code(llm_response)
    if not _code_has_method(generated_code, "check_objective"):
        raise ValueError(
            "Missing Code-Level Objective: Every environment must define its own win condition."
        )
    attempt_code_path = attempt_dir / f"generated_code{suffix}.py"
    _write_text(attempt_code_path, generated_code)
    candidate_env_path = await asyncio.to_thread(
        save_generated_code,
        enhanced_request,
        llm_response,
        class_name=class_name,
        output_dir=CANDIDATE_ENVS_DIR,
    )
    tier_policy = await policy_critic.evaluate_env(
        prompt=clean_prompt,
        env_path=candidate_env_path,
        class_name=class_name,
    )
    _write_json(attempt_dir / f"tier_policy{suffix}.json", tier_policy)
    return attempt_code_path, candidate_env_path, tier_policy


async def run_harness(
    prompt: str,
    *,
    backend: ArchitectBackend | None = None,
    config: HarnessConfig | None = None,
    run_dir: Path | None = None,
) -> HarnessResult:
    """Run the 3x3 Global Pivot + Local Repair Reflexion loop."""

    active_backend = backend or OpenAIArchitect()
    active_config = config or HarnessConfig()
    policy_critic = PolicyCritic()
    simulation_brief_architect = SimulationBriefArchitect()
    gameplay_architect = GameplayArchitect()
    clean_prompt = _normalize_prompt(prompt)
    active_run_dir = run_dir or _create_run_dir(clean_prompt)
    active_run_dir.mkdir(parents=True, exist_ok=True)
    simulation_brief = await _design_simulation_brief(
        simulation_brief_architect,
        clean_prompt,
        active_run_dir,
    )
    gameplay_profile = await _design_gameplay_profile(
        gameplay_architect,
        clean_prompt,
        active_run_dir,
        simulation_brief=simulation_brief,
    )
    physics_relations = infer_physics_relation_graph(
        clean_prompt,
        simulation_brief=simulation_brief,
        gameplay_profile=gameplay_profile,
    )
    _write_json(active_run_dir / "physics_relations.json", physics_relations)
    layout_plan = infer_layout_plan(
        clean_prompt,
        simulation_brief=simulation_brief,
        gameplay_profile=gameplay_profile,
        physics_relations=physics_relations,
    )
    _write_json(active_run_dir / "layout_plan.json", layout_plan)
    semantic_memory = retrieve_semantic_memory(
        clean_prompt,
        simulation_brief=simulation_brief,
        gameplay_profile=gameplay_profile,
        physics_relations=physics_relations,
        layout_plan=layout_plan,
    )
    _write_json(active_run_dir / "semantic_memory.json", semantic_memory)
    validator_with_relation_graph = replace(
        active_config.validator,
        expected_physics_relations=physics_relations,
        expected_layout_plan=layout_plan,
    )

    attempts: list[AttemptRecord] = []
    last_code_path: Path | None = None
    last_validation: ValidationResult | None = None
    last_error: str | None = None
    completed_attempt_keys: set[tuple[int, int]] = set()

    if active_config.execution_mode == "fast" and active_config.max_seeds > 1:
        _print_fast_status(
            f"SPECULATIVE SEED RACE: launching {active_config.max_seeds} first-attempt worlds in parallel..."
        )
        fast_tasks = [
            asyncio.create_task(
                _execute_attempt(
                    backend=active_backend,
                    policy_critic=policy_critic,
                    clean_prompt=clean_prompt,
                    active_run_dir=active_run_dir,
                    validator_config=validator_with_relation_graph,
                    max_repairs=active_config.max_repairs,
                    seed_index=seed_index,
                    repair_index=1,
                    simulation_brief=simulation_brief,
                    gameplay_profile=gameplay_profile,
                    physics_relations=physics_relations,
                    layout_plan=layout_plan,
                    semantic_memory=semantic_memory,
                    previous_result=None,
                    previous_error=None,
                    attempt_history=[],
                )
            )
            for seed_index in range(1, active_config.max_seeds + 1)
        ]
        pending = set(fast_tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                execution = task.result()
                attempts.append(execution.record)
                completed_attempt_keys.add(
                    (execution.record.seed_index, execution.record.repair_index)
                )
                last_code_path = execution.generated_code_path or last_code_path
                last_validation = execution.validation
                last_error = execution.error or execution.validation.reason
                if execution.validation.accepted and execution.candidate_env_path is not None:
                    _print_fast_status(
                        f"SEED {execution.record.seed_index} won the race; cancelling slower speculative candidates."
                    )
                    for pending_task in pending:
                        pending_task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    return _finalize_success(
                        active_run_dir=active_run_dir,
                        clean_prompt=clean_prompt,
                        candidate_env_path=execution.candidate_env_path,
                        validation=execution.validation,
                        attempts=attempts,
                        simulation_brief=simulation_brief,
                        gameplay_profile=gameplay_profile,
                        physics_relations=physics_relations,
                        layout_plan=layout_plan,
                        semantic_memory=semantic_memory,
                    )
        if attempts:
            best_attempt = max(attempts, key=lambda item: _attempt_rank(item.result))
            last_validation = best_attempt.result
            last_error = best_attempt.result.reason
            last_code_path = best_attempt.generated_code_path
            _print_fast_status(
                "No first-attempt seed solved it; continuing targeted repairs from the strongest completed candidate."
            )

    for seed_index in range(1, active_config.max_seeds + 1):
        directive = DIVERSITY_DIRECTIVES.get(seed_index, DIVERSITY_DIRECTIVES[MAX_SEEDS])
        repair_index = 1
        seed_repair_limit = active_config.max_repairs
        while repair_index <= seed_repair_limit:
            if (seed_index, repair_index) in completed_attempt_keys:
                repair_index += 1
                continue
            class_name = _class_name(clean_prompt, seed_index, repair_index)
            seed_history = [attempt for attempt in attempts if attempt.seed_index == seed_index]
            seed_previous_result = seed_history[-1].result if seed_history else last_validation
            seed_previous_error = (
                seed_previous_result.reason if seed_previous_result else last_error
            )
            enhanced_request = _enhanced_request(
                clean_prompt,
                directive,
                seed_index,
                repair_index,
                simulation_brief=simulation_brief,
                gameplay_profile=gameplay_profile,
                physics_relations=physics_relations,
                layout_plan=layout_plan,
                semantic_memory=semantic_memory,
            )
            architect_prompt = render_prompt(enhanced_request, class_name=class_name)
            correction_prompt = _build_correction_prompt(
                original_prompt=clean_prompt,
                seed_index=seed_index,
                repair_index=repair_index,
                diversity_directive=directive,
                architect_prompt=architect_prompt,
                previous_result=seed_previous_result,
                previous_error=seed_previous_error,
                attempt_history=seed_history,
                max_repairs=seed_repair_limit,
            )
            context = GenerationContext(
                original_prompt=clean_prompt,
                enhanced_request=enhanced_request,
                class_name=class_name,
                seed_index=seed_index,
                repair_index=repair_index,
                diversity_directive=directive,
                correction_prompt=correction_prompt,
                previous_result=seed_previous_result,
                previous_error=seed_previous_error,
            )

            attempt_dir = active_run_dir / f"seed_{seed_index:02d}_repair_{repair_index:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            _write_text(attempt_dir / "architect_prompt.txt", architect_prompt)
            _write_text(attempt_dir / "correction_prompt.txt", correction_prompt)
            attempt_code_path: Path | None = None
            candidate_env_path: Path | None = None
            tier_policy: dict[str, object] | None = None

            _print_status(seed_index, repair_index, seed_repair_limit, "REGENERATING...")
            validation: ValidationResult | None = None
            generation_failed = False
            contract_retry_index = 0
            while True:
                retry_suffix = (
                    f"contract_retry_{contract_retry_index}"
                    if contract_retry_index
                    else ""
                )
                retry_context = context
                if contract_retry_index:
                    retry_context = replace(
                        context,
                        correction_prompt=(
                            context.correction_prompt
                            + "\n\n"
                            + _contract_fast_lane_prompt(last_error or "")
                        ),
                        previous_error=last_error,
                        previous_result=last_validation,
                    )
                    _print_status(
                        seed_index,
                        repair_index,
                        seed_repair_limit,
                        "CONTRACT FAST-LANE RETRY "
                        f"{contract_retry_index}/{MAX_CONTRACT_FAST_LANE_RETRIES}...",
                    )
                try:
                    attempt_code_path, candidate_env_path, tier_policy = (
                        await _generate_candidate_artifacts(
                            backend=active_backend,
                            policy_critic=policy_critic,
                            context=retry_context,
                            attempt_dir=attempt_dir,
                            clean_prompt=clean_prompt,
                            enhanced_request=enhanced_request,
                            class_name=class_name,
                            artifact_suffix=retry_suffix,
                        )
                    )
                    last_code_path = attempt_code_path
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"
                    diagnostic = diagnose_spec_error(error_text)
                    last_error = diagnostic or f"Generation or contract verification failed: {exc}"
                    failed_result = _with_failure_category(_failure_result(last_error, class_name))
                    last_validation = failed_result
                    _write_json(
                        attempt_dir
                        / f"validation_contract_fast_lane_{contract_retry_index}.json",
                        failed_result.to_dict(),
                    )
                    if _contract_fast_lane_retry_allowed(
                        failed_result,
                        contract_retry_index,
                    ):
                        contract_retry_index += 1
                        continue
                    validation_path = _write_validation(attempt_dir, failed_result)
                    attempts.append(
                        AttemptRecord(
                            seed_index=seed_index,
                            repair_index=repair_index,
                            attempt_dir=attempt_dir,
                            class_name=class_name,
                            generated_code_path=attempt_code_path,
                            validation_path=validation_path,
                            result=failed_result,
                        )
                    )
                    _print_status(
                        seed_index,
                        repair_index,
                        seed_repair_limit,
                        f"CODE ERROR - {last_error}",
                    )
                    generation_failed = True
                    break

                try:
                    validation = await asyncio.to_thread(
                        validate_generated_env,
                        candidate_env_path,
                        class_name=class_name,
                        config=validator_with_relation_graph,
                        tier_policy=tier_policy,
                    )
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"
                    diagnostic = diagnose_spec_error(error_text)
                    validation = _failure_result(
                        diagnostic
                        or f"Validation/import/runtime failed: {error_text}",
                        class_name,
                    )
                validation = _with_failure_category(validation)
                if _contract_fast_lane_retry_allowed(validation, contract_retry_index):
                    last_validation = validation
                    last_error = validation.reason
                    _write_json(
                        attempt_dir
                        / f"validation_contract_fast_lane_{contract_retry_index}.json",
                        validation.to_dict(),
                    )
                    contract_retry_index += 1
                    continue
                break

            if generation_failed:
                repair_index += 1
                continue
            assert validation is not None
            assert candidate_env_path is not None
            attempt_record_code_path = attempt_code_path
            for auto_repair_index in range(1, MAX_AUTO_REPAIRS + 1):
                if validation.accepted:
                    break
                _write_json(
                    attempt_dir / f"validation_before_auto_repair_{auto_repair_index}.json",
                    validation.to_dict(),
                )
                repaired_env_path = (
                    attempt_dir
                    / f"{candidate_env_path.stem}_auto_repaired_{auto_repair_index}.py"
                )
                auto_repair = await asyncio.to_thread(
                    auto_repair_generated_env,
                    candidate_env_path,
                    validation,
                    repaired_env_path,
                )
                _write_json(
                    attempt_dir / f"auto_repair_result_{auto_repair_index}.json",
                    auto_repair.to_dict(),
                )
                if auto_repair.applied and auto_repair.output_path:
                    _print_status(
                        seed_index,
                        repair_index,
                        seed_repair_limit,
                        "AUTO-REPAIRING MEASURED PHYSICS CONSTRAINT "
                        f"({auto_repair_index}/{MAX_AUTO_REPAIRS})...",
                    )
                    try:
                        repaired_validation = await asyncio.to_thread(
                            validate_generated_env,
                            Path(auto_repair.output_path),
                            class_name=class_name,
                            config=validator_with_relation_graph,
                            tier_policy=tier_policy,
                        )
                    except Exception as exc:
                        error_text = f"{type(exc).__name__}: {exc}"
                        diagnostic = diagnose_spec_error(error_text)
                        repaired_validation = _failure_result(
                            diagnostic
                            or f"Auto-repaired validation/import/runtime failed: {error_text}",
                            class_name,
                        )
                    repaired_validation = _with_failure_category(repaired_validation)
                    _write_json(
                        attempt_dir / f"validation_after_auto_repair_{auto_repair_index}.json",
                        repaired_validation.to_dict(),
                    )
                    if _auto_repair_improved(validation, repaired_validation):
                        validation = repaired_validation
                        candidate_env_path = Path(auto_repair.output_path)
                        last_code_path = candidate_env_path
                        attempt_record_code_path = candidate_env_path
                        continue
                break
            last_validation = validation
            last_error = validation.reason
            validation_path = _write_validation(attempt_dir, validation)
            attempts.append(
                AttemptRecord(
                    seed_index=seed_index,
                    repair_index=repair_index,
                    attempt_dir=attempt_dir,
                    class_name=class_name,
                    generated_code_path=attempt_record_code_path,
                    validation_path=validation_path,
                    result=validation,
                )
            )

            if validation.accepted:
                verified_env_path = _promote_verified_env(candidate_env_path)
                visual_recipe_path = _record_visual_recipe(
                    prompt=clean_prompt,
                    env_path=verified_env_path,
                    validation=validation,
                    run_dir=active_run_dir,
                )
                world_export_dir = _record_world_export(
                    prompt=clean_prompt,
                    env_path=verified_env_path,
                    validation=validation,
                    visual_recipe_path=visual_recipe_path,
                    run_dir=active_run_dir,
                )
                learned_skill_path = _record_learned_skill(
                    prompt=clean_prompt,
                    env_path=verified_env_path,
                    validation=validation,
                )
                learned_affordance_path = _record_learned_affordance_block(
                    prompt=clean_prompt,
                    env_path=verified_env_path,
                    validation=validation,
                )
                capability_gap_path = _record_capability_gap(
                    prompt=clean_prompt,
                    validation=validation,
                    attempts=attempts,
                    run_dir=active_run_dir,
                    generated_env_path=verified_env_path,
                    post_mortem=None,
                )
                _print_status(
                    seed_index,
                    repair_index,
                    seed_repair_limit,
                    _success_status(validation),
                )
                _write_summary(
                    active_run_dir,
                    success=True,
                    prompt=clean_prompt,
                    generated_env_path=verified_env_path,
                    validation=validation,
                    post_mortem=None,
                    learned_skill_path=learned_skill_path,
                    learned_affordance_path=learned_affordance_path,
                    capability_gap_path=capability_gap_path,
                    visual_recipe_path=visual_recipe_path,
                    world_export_dir=world_export_dir,
                    simulation_brief=simulation_brief,
                    gameplay_profile=gameplay_profile,
                    physics_relations=physics_relations,
                    layout_plan=layout_plan,
                    semantic_memory=semantic_memory,
                )
                return HarnessResult(
                    success=True,
                    prompt=clean_prompt,
                    run_dir=active_run_dir,
                    generated_env_path=verified_env_path,
                    validation=validation,
                    attempts=tuple(attempts),
                    capability_gap_path=capability_gap_path,
                    visual_recipe_path=visual_recipe_path,
                    world_export_dir=world_export_dir,
                    simulation_brief=simulation_brief,
                    gameplay_profile=gameplay_profile,
                    physics_relations=physics_relations,
                    layout_plan=layout_plan,
                    semantic_memory=semantic_memory,
                )

            status = _blocked_status(validation)
            if repair_index < seed_repair_limit:
                _print_status(seed_index, repair_index, seed_repair_limit, status)
            elif seed_index < active_config.max_seeds:
                extra_budget = _near_miss_local_pivot_budget(validation)
                extra_used = max(0, repair_index - active_config.max_repairs)
                if extra_used < extra_budget:
                    seed_repair_limit += 1
                    _print_status(
                        seed_index,
                        repair_index,
                        seed_repair_limit,
                        f"{status}; LOCAL NEAR-MISS PIVOT {extra_used + 1}/{extra_budget} "
                        "WITHIN SAME SEED...",
                    )
                else:
                    _print_status(
                        seed_index,
                        repair_index,
                        seed_repair_limit,
                        f"{status}; PIVOTING TO SEED {seed_index + 1}...",
                    )
            else:
                extra_budget = _near_miss_local_pivot_budget(validation)
                extra_used = max(0, repair_index - active_config.max_repairs)
                if extra_used < extra_budget:
                    seed_repair_limit += 1
                    _print_status(
                        seed_index,
                        repair_index,
                        seed_repair_limit,
                        f"{status}; FINAL-SEED NEAR-MISS PIVOT {extra_used + 1}/{extra_budget}...",
                    )
                else:
                    _print_status(seed_index, repair_index, seed_repair_limit, status)
            repair_index += 1

    post_mortem = _build_post_mortem(clean_prompt, attempts, last_validation, last_error)
    failed_path = active_run_dir / "env_failed_final.py.fail"
    if last_code_path and last_code_path.exists():
        shutil.copyfile(last_code_path, failed_path)
    else:
        _write_text(failed_path, "# No valid generated code was produced.\n")
    _write_text(active_run_dir / "post_mortem.txt", post_mortem)
    capability_gap_path = _record_capability_gap(
        prompt=clean_prompt,
        validation=last_validation,
        attempts=attempts,
        run_dir=active_run_dir,
        generated_env_path=failed_path,
        post_mortem=post_mortem,
    )
    _write_summary(
        active_run_dir,
        success=False,
        prompt=clean_prompt,
        generated_env_path=failed_path,
        validation=last_validation,
        post_mortem=post_mortem,
        learned_skill_path=None,
        learned_affordance_path=None,
        capability_gap_path=capability_gap_path,
        visual_recipe_path=None,
        world_export_dir=None,
        simulation_brief=simulation_brief,
        gameplay_profile=gameplay_profile,
        physics_relations=physics_relations,
        layout_plan=layout_plan,
        semantic_memory=semantic_memory,
    )
    return HarnessResult(
        success=False,
        prompt=clean_prompt,
        run_dir=active_run_dir,
        generated_env_path=failed_path,
        validation=last_validation,
        attempts=tuple(attempts),
        post_mortem=post_mortem,
        capability_gap_path=capability_gap_path,
        visual_recipe_path=None,
        world_export_dir=None,
        simulation_brief=simulation_brief,
        gameplay_profile=gameplay_profile,
        physics_relations=physics_relations,
        layout_plan=layout_plan,
        semantic_memory=semantic_memory,
    )


async def _execute_attempt(
    *,
    backend: ArchitectBackend,
    policy_critic: PolicyCritic,
    clean_prompt: str,
    active_run_dir: Path,
    validator_config: ValidatorConfig,
    max_repairs: int,
    seed_index: int,
    repair_index: int,
    simulation_brief: dict[str, Any] | None,
    gameplay_profile: dict[str, Any] | None,
    physics_relations: dict[str, Any] | None,
    layout_plan: dict[str, Any] | None,
    semantic_memory: dict[str, Any] | None,
    previous_result: ValidationResult | None,
    previous_error: str | None,
    attempt_history: list[AttemptRecord],
) -> AttemptExecution:
    """Execute one complete seed/repair attempt for speculative fast mode."""

    directive = DIVERSITY_DIRECTIVES.get(seed_index, DIVERSITY_DIRECTIVES[MAX_SEEDS])
    class_name = _class_name(clean_prompt, seed_index, repair_index)
    enhanced_request = _enhanced_request(
        clean_prompt,
        directive,
        seed_index,
        repair_index,
        simulation_brief=simulation_brief,
        gameplay_profile=gameplay_profile,
        physics_relations=physics_relations,
        layout_plan=layout_plan,
        semantic_memory=semantic_memory,
    )
    architect_prompt = render_prompt(enhanced_request, class_name=class_name)
    correction_prompt = _build_correction_prompt(
        original_prompt=clean_prompt,
        seed_index=seed_index,
        repair_index=repair_index,
        diversity_directive=directive,
        architect_prompt=architect_prompt,
        previous_result=previous_result,
        previous_error=previous_error,
        attempt_history=attempt_history,
        max_repairs=max_repairs,
    )
    context = GenerationContext(
        original_prompt=clean_prompt,
        enhanced_request=enhanced_request,
        class_name=class_name,
        seed_index=seed_index,
        repair_index=repair_index,
        diversity_directive=directive,
        correction_prompt=correction_prompt,
        previous_result=previous_result,
        previous_error=previous_error,
    )
    attempt_dir = active_run_dir / f"seed_{seed_index:02d}_repair_{repair_index:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    _write_text(attempt_dir / "architect_prompt.txt", architect_prompt)
    _write_text(attempt_dir / "correction_prompt.txt", correction_prompt)

    _print_status(seed_index, repair_index, max_repairs, "FAST SPECULATIVE GENERATION...")
    attempt_code_path: Path | None = None
    candidate_env_path: Path | None = None
    tier_policy: dict[str, object] | None = None
    validation: ValidationResult | None = None
    error: str | None = None
    contract_retry_index = 0
    while True:
        retry_suffix = f"contract_retry_{contract_retry_index}" if contract_retry_index else ""
        retry_context = context
        if contract_retry_index:
            retry_context = replace(
                context,
                correction_prompt=(
                    context.correction_prompt + "\n\n" + _contract_fast_lane_prompt(error or "")
                ),
                previous_error=error,
                previous_result=validation,
            )
            _print_status(
                seed_index,
                repair_index,
                max_repairs,
                "CONTRACT FAST-LANE RETRY "
                f"{contract_retry_index}/{MAX_CONTRACT_FAST_LANE_RETRIES}...",
            )
        try:
            attempt_code_path, candidate_env_path, tier_policy = await _generate_candidate_artifacts(
                backend=backend,
                policy_critic=policy_critic,
                context=retry_context,
                attempt_dir=attempt_dir,
                clean_prompt=clean_prompt,
                enhanced_request=enhanced_request,
                class_name=class_name,
                artifact_suffix=retry_suffix,
            )
        except Exception as exc:
            diagnostic = diagnose_spec_error(f"{type(exc).__name__}: {exc}")
            error = diagnostic or f"Generation or contract verification failed: {exc}"
            validation = _with_failure_category(_failure_result(error, class_name))
            _write_json(
                attempt_dir / f"validation_contract_fast_lane_{contract_retry_index}.json",
                validation.to_dict(),
            )
            if _contract_fast_lane_retry_allowed(validation, contract_retry_index):
                contract_retry_index += 1
                continue
            validation_path = _write_validation(attempt_dir, validation)
            record = AttemptRecord(
                seed_index=seed_index,
                repair_index=repair_index,
                attempt_dir=attempt_dir,
                class_name=class_name,
                generated_code_path=attempt_code_path,
                validation_path=validation_path,
                result=validation,
            )
            _print_status(seed_index, repair_index, max_repairs, f"CODE ERROR - {error}")
            return AttemptExecution(
                record=record,
                candidate_env_path=None,
                generated_code_path=attempt_code_path,
                validation=validation,
                error=error,
            )

        assert candidate_env_path is not None
        try:
            validation = await asyncio.to_thread(
                validate_generated_env,
                candidate_env_path,
                class_name=class_name,
                config=validator_config,
                tier_policy=tier_policy,
            )
        except Exception as exc:
            diagnostic = diagnose_spec_error(f"{type(exc).__name__}: {exc}")
            validation = _failure_result(
                diagnostic or f"Validation/import/runtime failed: {type(exc).__name__}: {exc}",
                class_name,
            )
        validation = _with_failure_category(validation)
        if _contract_fast_lane_retry_allowed(validation, contract_retry_index):
            error = validation.reason
            _write_json(
                attempt_dir / f"validation_contract_fast_lane_{contract_retry_index}.json",
                validation.to_dict(),
            )
            contract_retry_index += 1
            continue
        break

    assert validation is not None
    attempt_record_code_path = attempt_code_path
    for auto_repair_index in range(1, MAX_AUTO_REPAIRS + 1):
        if validation.accepted:
            break
        _write_json(
            attempt_dir / f"validation_before_auto_repair_{auto_repair_index}.json",
            validation.to_dict(),
        )
        repaired_env_path = (
            attempt_dir / f"{candidate_env_path.stem}_auto_repaired_{auto_repair_index}.py"
        )
        auto_repair = await asyncio.to_thread(
            auto_repair_generated_env,
            candidate_env_path,
            validation,
            repaired_env_path,
        )
        _write_json(
            attempt_dir / f"auto_repair_result_{auto_repair_index}.json",
            auto_repair.to_dict(),
        )
        if auto_repair.applied and auto_repair.output_path:
            _print_status(
                seed_index,
                repair_index,
                max_repairs,
                "AUTO-REPAIRING MEASURED PHYSICS CONSTRAINT "
                f"({auto_repair_index}/{MAX_AUTO_REPAIRS})...",
            )
            try:
                repaired_validation = await asyncio.to_thread(
                    validate_generated_env,
                    Path(auto_repair.output_path),
                    class_name=class_name,
                    config=validator_config,
                    tier_policy=tier_policy,
                )
            except Exception as exc:
                diagnostic = diagnose_spec_error(f"{type(exc).__name__}: {exc}")
                repaired_validation = _failure_result(
                    diagnostic
                    or f"Auto-repaired validation/import/runtime failed: {type(exc).__name__}: {exc}",
                    class_name,
                )
            repaired_validation = _with_failure_category(repaired_validation)
            _write_json(
                attempt_dir / f"validation_after_auto_repair_{auto_repair_index}.json",
                repaired_validation.to_dict(),
            )
            if _auto_repair_improved(validation, repaired_validation):
                validation = repaired_validation
                candidate_env_path = Path(auto_repair.output_path)
                attempt_record_code_path = candidate_env_path
                continue
        break

    validation_path = _write_validation(attempt_dir, validation)
    record = AttemptRecord(
        seed_index=seed_index,
        repair_index=repair_index,
        attempt_dir=attempt_dir,
        class_name=class_name,
        generated_code_path=attempt_record_code_path,
        validation_path=validation_path,
        result=validation,
    )
    status = _success_status(validation) if validation.accepted else _blocked_status(validation)
    _print_status(seed_index, repair_index, max_repairs, status)
    return AttemptExecution(
        record=record,
        candidate_env_path=candidate_env_path,
        generated_code_path=attempt_record_code_path,
        validation=validation,
        error=validation.reason,
    )


def _finalize_success(
    *,
    active_run_dir: Path,
    clean_prompt: str,
    candidate_env_path: Path,
    validation: ValidationResult,
    attempts: list[AttemptRecord],
    simulation_brief: dict[str, Any] | None,
    gameplay_profile: dict[str, Any] | None,
    physics_relations: dict[str, Any] | None,
    layout_plan: dict[str, Any] | None,
    semantic_memory: dict[str, Any] | None,
) -> HarnessResult:
    verified_env_path = _promote_verified_env(candidate_env_path)
    visual_recipe_path = _record_visual_recipe(
        prompt=clean_prompt,
        env_path=verified_env_path,
        validation=validation,
        run_dir=active_run_dir,
    )
    world_export_dir = _record_world_export(
        prompt=clean_prompt,
        env_path=verified_env_path,
        validation=validation,
        visual_recipe_path=visual_recipe_path,
        run_dir=active_run_dir,
    )
    learned_skill_path = _record_learned_skill(
        prompt=clean_prompt,
        env_path=verified_env_path,
        validation=validation,
    )
    learned_affordance_path = _record_learned_affordance_block(
        prompt=clean_prompt,
        env_path=verified_env_path,
        validation=validation,
    )
    capability_gap_path = _record_capability_gap(
        prompt=clean_prompt,
        validation=validation,
        attempts=attempts,
        run_dir=active_run_dir,
        generated_env_path=verified_env_path,
        post_mortem=None,
    )
    _write_summary(
        active_run_dir,
        success=True,
        prompt=clean_prompt,
        generated_env_path=verified_env_path,
        validation=validation,
        post_mortem=None,
        learned_skill_path=learned_skill_path,
        learned_affordance_path=learned_affordance_path,
        capability_gap_path=capability_gap_path,
        visual_recipe_path=visual_recipe_path,
        world_export_dir=world_export_dir,
        simulation_brief=simulation_brief,
        gameplay_profile=gameplay_profile,
        physics_relations=physics_relations,
        layout_plan=layout_plan,
        semantic_memory=semantic_memory,
    )
    return HarnessResult(
        success=True,
        prompt=clean_prompt,
        run_dir=active_run_dir,
        generated_env_path=verified_env_path,
        validation=validation,
        attempts=tuple(attempts),
        capability_gap_path=capability_gap_path,
        visual_recipe_path=visual_recipe_path,
        world_export_dir=world_export_dir,
        simulation_brief=simulation_brief,
        gameplay_profile=gameplay_profile,
        physics_relations=physics_relations,
        layout_plan=layout_plan,
        semantic_memory=semantic_memory,
    )


def _attempt_rank(result: ValidationResult) -> tuple[int, int, int, int]:
    return (
        1 if result.accepted else 0,
        int(result.achieved_tier or 0),
        int(result.objective_tier or 0),
        -int(result.verification_gap or 0),
    )


def _print_fast_status(status: str) -> None:
    print(f"[FAST MODE] -> Status: {status}", flush=True)


def _build_correction_prompt(
    *,
    original_prompt: str,
    seed_index: int,
    repair_index: int,
    diversity_directive: str,
    architect_prompt: str,
    previous_result: ValidationResult | None,
    previous_error: str | None,
    attempt_history: list[AttemptRecord] | None = None,
    max_repairs: int = MAX_REPAIRS,
) -> str:
    if previous_result is None and previous_error is None:
        feedback = "No previous failure. Generate the initial candidate for this seed."
        failure_category = "initial_generation"
        diagnostic_axis = "initial_generation"
        verification_line = "Verification tier: not yet evaluated"
    elif previous_result is not None:
        feedback = json.dumps(previous_result.to_dict(), indent=2, sort_keys=True)
        probe_feedback = _probe_feedback_summary(previous_result)
        failure_category = _failure_category(previous_result)
        diagnostic_axis = _diagnostic_axis_for_category(failure_category)
        verification_line = (
            f"Verification tier: {previous_result.achieved_tier}/"
            f"{previous_result.operational_acceptance_tier} "
            f"({previous_result.tier_name}); accepted={previous_result.accepted}; "
            f"objective_tier={previous_result.objective_tier}"
        )
    else:
        feedback = previous_error or "Unknown failure."
        probe_feedback = "No structured probe evidence was available."
        failure_category = _failure_category_from_text(feedback)
        diagnostic_axis = _diagnostic_axis_for_category(failure_category)
        verification_line = "Verification tier: unavailable"
    if previous_result is None and previous_error is None:
        probe_feedback = "No structured probe evidence was available yet."
    history_feedback = _probe_history_feedback_summary(attempt_history or [], previous_result)
    repeat_feedback = _repeated_failure_feedback(attempt_history or [], previous_result, previous_error)

    repair_rule = _repair_rule_for_category(failure_category, feedback)

    return textwrap.dedent(
        f"""
        Reflexion correction request for Harness Alpha.

        Original user prompt:
        {original_prompt}

        Current strategy:
        - Seed: {seed_index}
        - Repair: {repair_index}/{max_repairs}
        - Diversity directive: {diversity_directive}
        - Failure category: {failure_category}
        - Diagnostic axis: {diagnostic_axis}
        - {verification_line}

        Probe evidence summary:
        {probe_feedback}

        Cumulative seed repair memory:
        {history_feedback}

        Repeated-failure guard:
        {repeat_feedback}

        Previous failure telemetry:
        {feedback}

        Repair rule:
        {repair_rule}

        Full architect contract prompt:
        {architect_prompt}
        """
    ).strip()


def _auto_repair_improved(
    before: ValidationResult,
    after: ValidationResult,
) -> bool:
    """Return whether deterministic repair should replace the current attempt."""

    if after.accepted:
        return True
    if after.achieved_tier > before.achieved_tier:
        return True
    if after.achieved_tier == before.achieved_tier and after.verification_gap < before.verification_gap:
        return True
    if (
        after.achieved_tier == before.achieved_tier
        and after.kinetic_progress
        and not before.kinetic_progress
    ):
        return True
    if after.achieved_tier == before.achieved_tier and _post_subgoal_repair_progress(before, after):
        return True
    if _completed_subgoal_count(after) > _completed_subgoal_count(before):
        return True
    return False


def _post_subgoal_repair_progress(before: ValidationResult, after: ValidationResult) -> bool:
    before_summary = _subgoal_summary(before)
    after_summary = _subgoal_summary(after)
    before_blocker = str(before_summary.get("final_blocking_object") or before_summary.get("initial_blocking_object") or "")
    after_blocker = str(after_summary.get("final_blocking_object") or after_summary.get("initial_blocking_object") or "")
    before_threshold = _float_or_none(before_summary.get("threshold"))
    after_threshold = _float_or_none(after_summary.get("threshold"))
    before_distance = _float_or_none(before_summary.get("final_distance"))
    after_distance = _float_or_none(after_summary.get("final_distance"))
    if before_blocker and after_blocker and before_blocker != after_blocker:
        return True
    if before_threshold is not None and after_threshold is not None and after_threshold > before_threshold:
        return True
    if before_distance is not None and after_distance is not None and after_distance < before_distance:
        return True
    return False


def _subgoal_summary(result: ValidationResult) -> dict[str, Any]:
    diagnostics = result.details.get("subgoal_diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    summary = diagnostics.get("summary")
    return summary if isinstance(summary, dict) else {}


def _completed_subgoal_count(result: ValidationResult) -> int:
    completed = result.details.get("completed_subgoals")
    return len(completed) if isinstance(completed, list) else 0


def _near_miss_local_pivot_budget(result: ValidationResult) -> int:
    """Return extra same-seed repair attempts for promising near-misses only."""

    if result.accepted:
        return 0
    category = _failure_category(result)
    hard_reset_categories = {
        "api_keyword_error",
        "capability_mismatch",
        "code_contract_error",
        "object_model_error",
        "objective_logic_error",
        "semantic_contract_error",
        "semantic_profile_error",
        "structural_invalidity",
        "subgoal_plan_error",
    }
    if category in hard_reset_categories:
        return 0
    near_miss_categories = {
        "contact_control_failure",
        "gameplay_dynamics_failure",
        "kinetic_partial_progress",
        "mechanical_leverage",
        "physical_blockage",
        "physics_instability",
        "post_subgoal_navigation_failure",
        "prompt_anticheat_failure",
        "semantic_dynamics_failure",
        "subgoal_affordance_failure",
        "subgoal_execution_failure",
        "tier_below_acceptance",
        "validator_solver_weakness",
    }
    if category not in near_miss_categories and not result.kinetic_progress:
        return 0
    if result.achieved_tier >= 4:
        return TIER4_LOCAL_PIVOTS
    if result.achieved_tier >= 3:
        return TIER3_LOCAL_PIVOTS
    return 0


def _probe_history_feedback_summary(
    attempt_history: list[AttemptRecord],
    current_result: ValidationResult | None,
) -> str:
    """Summarize probe memory across repairs within the active seed."""

    if not attempt_history:
        return "No previous attempts in this seed yet."

    prior_probes: list[tuple[AttemptRecord, dict[str, Any]]] = []
    for attempt in attempt_history:
        for probe in _collect_probe_dicts(attempt.result.details):
            prior_probes.append((attempt, probe))
    if not prior_probes:
        return "Previous attempts did not emit structured probe evidence."

    current_failures = {
        _probe_key(probe): probe
        for probe in _collect_probe_dicts(current_result.details if current_result else {})
        if not bool(probe.get("passed"))
    }
    passed_by_key: dict[tuple[str, tuple[str, ...]], tuple[AttemptRecord, dict[str, Any]]] = {}
    best_by_key: dict[tuple[str, tuple[str, ...]], tuple[AttemptRecord, dict[str, Any]]] = {}
    latest_by_key: dict[tuple[str, tuple[str, ...]], tuple[AttemptRecord, dict[str, Any]]] = {}

    for attempt, probe in prior_probes:
        key = _probe_key(probe)
        latest_by_key[key] = (attempt, probe)
        if bool(probe.get("passed")):
            passed_by_key[key] = (attempt, probe)
        if _probe_is_better(probe, best_by_key.get(key, (None, None))[1]):
            best_by_key[key] = (attempt, probe)

    lines: list[str] = []
    regressions = [
        (key, passed_by_key[key], current_failures[key])
        for key in current_failures
        if key in passed_by_key
    ]
    if regressions:
        lines.append("Regression warnings:")
        for _, (attempt, prior_probe), current_probe in regressions[:4]:
            lines.append(
                "- "
                + _format_probe_history_line(
                    "Previously passed; restore this invariant",
                    prior_probe,
                    attempt,
                    current_probe=current_probe,
                )
            )

    invariants = [
        (key, item)
        for key, item in passed_by_key.items()
        if key not in current_failures
    ]
    if invariants:
        lines.append("Probe invariants to preserve:")
        for _, (attempt, probe) in invariants[:5]:
            lines.append(
                "- "
                + _format_probe_history_line(
                    "Preserve",
                    probe,
                    attempt,
                )
            )

    improving_failures = [
        (key, item)
        for key, item in best_by_key.items()
        if key in current_failures and key not in passed_by_key
    ]
    if improving_failures:
        lines.append("Best partial progress so far:")
        for _, (attempt, probe) in improving_failures[:4]:
            lines.append(
                "- "
                + _format_probe_history_line(
                    "Best observed",
                    probe,
                    attempt,
                )
            )

    if not lines:
        lines.append("No reusable probe invariants yet; prioritize the current failed probes.")
    lines.append(
        "Repair instruction: fix the current failed probes while preserving every listed invariant; do not rewrite working subgoals unless a probe says they regressed."
    )
    return "\n".join(lines)


def _probe_key(probe: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    objects = probe.get("objects")
    if not isinstance(objects, list):
        objects = []
    return (str(probe.get("name") or "unknown_probe"), tuple(str(item) for item in objects))


def _repeated_failure_feedback(
    attempt_history: list[AttemptRecord],
    current_result: ValidationResult | None,
    previous_error: str | None,
) -> str:
    """Warn the Architect when the same failure repeats inside a seed."""

    current_reason = _normalized_failure_signature(
        current_result.reason if current_result is not None else previous_error
    )
    if not current_reason:
        return "No repeated failure signature detected yet."

    matching_attempts = [
        attempt
        for attempt in attempt_history
        if _normalized_failure_signature(attempt.result.reason) == current_reason
    ]
    if len(matching_attempts) < 1:
        return "No repeated failure signature detected yet."

    count = len(matching_attempts) + 1
    categories = sorted({_failure_category(attempt.result) for attempt in matching_attempts})
    category_text = ", ".join(categories) or "unknown"
    return (
        f"The same failure signature has appeared {count} times in this seed "
        f"(categories: {category_text}). Do not make another cosmetic/layout-only "
        "variation. Change the mistaken interpretation or API pattern that caused it. "
        "If this is a code-contract error, remove the offending construct entirely "
        "and choose a different valid objective/subgoal model before regenerating."
    )


def _normalized_failure_signature(reason: str | None) -> str:
    if not reason:
        return ""
    text = str(reason).lower()
    text = re.sub(r"seed[_\\/\s-]*\d+", "seed_n", text)
    text = re.sub(r"repair[_\\/\s-]*\d+", "repair_n", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "#", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:700]


def _probe_is_better(probe: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    if bool(probe.get("passed")) and not bool(incumbent.get("passed")):
        return True
    if not bool(probe.get("passed")) and bool(incumbent.get("passed")):
        return False
    probe_score = _probe_score(probe)
    incumbent_score = _probe_score(incumbent)
    if probe_score is None:
        return False
    if incumbent_score is None:
        return True
    return probe_score > incumbent_score


def _probe_score(probe: dict[str, Any]) -> float | None:
    metrics = probe.get("metrics") if isinstance(probe.get("metrics"), dict) else {}
    if bool(probe.get("passed")):
        try:
            return 1000.0 + float(probe.get("tier_evidence") or 0)
        except (TypeError, ValueError):
            return 1000.0
    lower_is_better = (
        "object_displacement",
        "distance",
        "final_distance",
        "best_distance",
        "object_to_region_distance",
        "agent_to_object_distance",
        "alignment_angle_degrees",
    )
    for key in lower_is_better:
        value = _float_or_none(metrics.get(key))
        if value is not None:
            return -value
    higher_is_better = (
        "distance_reduced",
        "max_object_velocity_toward_region",
        "max_agent_velocity_toward_target",
        "progress_delta",
        "displacement",
    )
    for key in higher_is_better:
        value = _float_or_none(metrics.get(key))
        if value is not None:
            return value
    try:
        return float(probe.get("tier_evidence") or 0)
    except (TypeError, ValueError):
        return None


def _format_probe_history_line(
    prefix: str,
    probe: dict[str, Any],
    attempt: AttemptRecord,
    *,
    current_probe: dict[str, Any] | None = None,
) -> str:
    name = str(probe.get("name") or "unknown_probe")
    objects = probe.get("objects") if isinstance(probe.get("objects"), list) else []
    metrics = probe.get("metrics") if isinstance(probe.get("metrics"), dict) else {}
    base = (
        f"{prefix}: {name} objects={objects} at seed {attempt.seed_index} "
        f"repair {attempt.repair_index}; diagnosis={probe.get('diagnosis')}. "
        f"{_compact_probe_metrics(metrics)}"
    ).strip()
    if current_probe is None:
        return base
    current_metrics = (
        current_probe.get("metrics") if isinstance(current_probe.get("metrics"), dict) else {}
    )
    return (
        f"{base} Current regression: diagnosis={current_probe.get('diagnosis')}; "
        f"{_compact_probe_metrics(current_metrics)}"
    ).strip()


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _probe_feedback_summary(result: ValidationResult) -> str:
    """Summarize probe evidence into repair-oriented feedback for the Architect."""

    probes = _collect_probe_dicts(result.details)
    if not probes:
        return "No structured probe evidence was available."

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for probe in probes:
        key = json.dumps(probe, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(probe)
    probes = deduped

    failed = [probe for probe in probes if not bool(probe.get("passed"))]
    passed = [probe for probe in probes if bool(probe.get("passed"))]
    lines: list[str] = []
    if failed:
        lines.append("Failed probes to repair:")
        for probe in failed[:6]:
            lines.append(f"- {_format_probe_for_repair(probe)}")
    if passed:
        lines.append("Passed probes to preserve:")
        for probe in passed[:4]:
            lines.append(f"- {_format_probe_for_repair(probe, preserve=True)}")
    return "\n".join(lines)


def _format_probe_for_repair(probe: dict[str, Any], *, preserve: bool = False) -> str:
    name = str(probe.get("name") or "unknown_probe")
    objects = probe.get("objects") if isinstance(probe.get("objects"), list) else []
    diagnosis = str(probe.get("diagnosis") or "")
    repair = str(probe.get("repair") or "")
    metrics = probe.get("metrics") if isinstance(probe.get("metrics"), dict) else {}
    metric_text = _compact_probe_metrics(metrics)
    object_text = f" objects={objects}" if objects else ""
    if preserve:
        return f"{name}{object_text}: {diagnosis}. Preserve this relation. {metric_text}".strip()
    return f"{name}{object_text}: {diagnosis}. {repair} {metric_text}".strip()


def _compact_probe_metrics(metrics: dict[str, Any]) -> str:
    preferred_keys = (
        "distance",
        "threshold",
        "object_displacement",
        "recommended_max_displacement",
        "object_to_region_distance",
        "recommended_object_to_region_max",
        "agent_to_object_distance",
        "recommended_agent_to_object_max",
        "alignment_angle_degrees",
        "recommended_alignment_angle_max",
        "distance_reduced",
        "final_distance",
        "best_distance",
        "failure_modes",
    )
    parts: list[str] = []
    for key in preferred_keys:
        if key in metrics and metrics.get(key) is not None:
            parts.append(f"{key}={metrics.get(key)}")
    if not parts:
        return ""
    return "Metrics: " + "; ".join(parts[:8]) + "."


def _collect_probe_dicts(value: Any, *, _depth: int = 0) -> list[dict[str, Any]]:
    if _depth > 8:
        return []
    if isinstance(value, dict):
        if isinstance(value.get("name"), str) and "passed" in value and "tier_evidence" in value:
            return [value]
        probes: list[dict[str, Any]] = []
        for child in value.values():
            probes.extend(_collect_probe_dicts(child, _depth=_depth + 1))
        return probes
    if isinstance(value, list | tuple):
        probes: list[dict[str, Any]] = []
        for child in value:
            probes.extend(_collect_probe_dicts(child, _depth=_depth + 1))
        return probes
    return []


def _build_local_env_code(context: GenerationContext) -> str:
    seed = context.seed_index
    repair = context.repair_index
    class_name = context.class_name
    source_prompt = context.enhanced_request

    if seed == 1:
        first_wall_height = 220 - (repair - 1) * 20
        gate_height = 120 - (repair - 1) * 20
        return _maze_env_code(
            class_name,
            source_prompt,
            layout_note="baseline maze with two narrow passages and an offset sliding gate",
            first_wall_center=(300, 170),
            first_wall_size=(34, first_wall_height),
            second_wall_center=(520, 390),
            second_wall_size=(34, 220),
            gate_center=(690, 260),
            gate_size=(28, gate_height),
            hazard_positions=((385, 430),),
        )

    if seed == 2:
        return _maze_env_code(
            class_name,
            source_prompt,
            layout_note="inverse layout with peripheral blockers and central target lane",
            first_wall_center=(250, 380),
            first_wall_size=(36, 230),
            second_wall_center=(540, 170),
            second_wall_size=(36, 220),
            gate_center=(690, 350),
            gate_size=(28, 110),
            hazard_positions=((430, 130), (740, 430)),
            start=(90, 430),
            goal=(820, 130),
        )

    return _maze_env_code(
        class_name,
        source_prompt,
        layout_note="high entropy maze with dynamic falling balls and a sliding gate",
        first_wall_center=(285, 165),
        first_wall_size=(34, 210),
        second_wall_center=(505, 385),
        second_wall_size=(34, 210),
        gate_center=(665, 250),
        gate_size=(28, 95),
        hazard_positions=((230, 455), (435, 125), (760, 445)),
    )


def _is_attribute_error_feedback(feedback: str) -> bool:
    return "AttributeError" in feedback or "does not exist" in feedback


def _repair_rule_for_category(category: str, feedback: str) -> str:
    lowered_feedback = feedback.lower()
    if category == "semantic_contract_error":
        return (
            "Repair the declared semantic contract before changing geometry. Keep the "
            "same prompt semantics, but make objective_profile, capability_profile, "
            "registered objects, and registered interaction resources agree exactly. "
            "For field_force_interaction, the subgoal field name must match a real "
            "self.register_force_zone(name, ...) entry; do not point it at a plain "
            "sensor or decorative region. For generated math, do not call "
            "pymunk.Vec2d with one tuple/list argument; use plain (x, y) tuples or "
            "pymunk.Vec2d(x, y). Treat these as contract repairs, not layout repairs."
        )
    if category == "semantic_profile_error":
        return (
            "Repair the semantic/profile contract. Ensure objective_profile and "
            "capability_profile are present, internally consistent, and aligned with "
            "the prompt. objective_profile['targets'] must match objective_targets, "
            "minimum_acceptance_tier must reflect the task, and validator_skills must "
            "describe how the oracle should test the declared objective."
        )
    if category == "prompt_anticheat_failure":
        return (
            "Repair the prompt-fidelity anti-cheat failure without lowering the objective tier. "
            "The code-level objective may be passable, but the main challenge from the prompt was "
            "neutralized. Use anti_cheat_probes telemetry literally: if hazards are static or "
            "blocked, open their ingress/egress lanes and give them velocity/force; if they never "
            "enter the agent lane, move spawn points, route openings, or hazard lanes so they cross "
            "the playable challenge; if chasers are sealed away, connect them to the route; if "
            "projectiles are parked, activate them within the first seconds. Preserve object names, "
            "check_objective semantics, and required duration."
        )
    if category == "semantic_dynamics_failure":
        return (
            "Repair prompt-fidelity dynamics, not just objective solvability. The world "
            "contains the right nouns but failed to implement the requested physical verb. "
            "Use the semantic_probes telemetry literally. If the prompt says falling/raining/"
            "dropping hazards, create dynamic role='hazard' bodies above the play route, "
            "remove any supporting shelf/platform under them, enable gravity or downward "
            "initial velocity, and ensure at least one hazard moves downward by the required "
            "pixels during passive simulation. If a hazard stops near the top boundary, "
            "spawn it below the ceiling or cut ceiling gaps above every drop lane. If the "
            "prompt says chasing/pursuing, use zero gravity or stable support and apply "
            "bounded force toward self.agent.body.position in after_step(self) so the chaser "
            "distance to the agent decreases. If the prompt says drifting/floating, give "
            "dynamic objects visible velocity or force-zone motion. If the prompt says "
            "spaceship/laser/bullet/projectile shots, make at least one role='hazard' "
            "projectile active immediately or within 30 steps and give it enough velocity "
            "or bounded force to travel 120+ px; do not use parked inactive projectile "
            "pool bodies that match semantic_requirements. Preserve check_objective and "
            "do not lower the verification tier."
        )
    if category == "gameplay_dynamics_failure":
        return (
            "Repair the Gameplay Architect contract without rewriting the whole world. "
            "Preserve the objective_profile, capability_profile, object names, and "
            "check_objective semantics. Implement self.gameplay_profile exactly as the "
            "provided design contract and export it in get_ground_truth objective metadata. "
            "For recurring/staggered falling hazards, add registry state for phase offsets, "
            "spawn lanes, reset_y, bottom_y, cadence timers, and active hazard names; implement "
            "after_step(self) so hazards begin at different times, fall visibly through the "
            "play area, reset to the top after exiting below the world, and continue until "
            "check_objective() is true. Do not place shelves, ceilings, platforms, or walls "
            "that catch hazards before they fall. Preserve at least one readable safe lane "
            "or dodge window. For heavy-but-movable push gameplay, tune mass, friction, "
            "agent_strength, and lane alignment so pushing produces visibly significant "
            "motion while remaining stable. "
            "For endless/sequential cars, trains, traffic, rolling logs, or side-view crossing hazards, "
            "use BaseEnv create_recurring_lateral_hazards(...). Hazards must spawn offscreen, stay "
            "lane-locked with near-zero vertical drift, move horizontally through the agent lane, and "
            "reset after exiting. Do not let them fall off-screen or get blocked by world bounds."
        )
    if category == "subgoal_plan_error":
        return (
            "Repair the objective subgoal plan. Decompose the task into ordered generic "
            "subgoals such as agent_reach_region, agent_touch_object, "
            "move_object_to_region, and activate_mechanism. Ensure each subgoal uses "
            "registered object names and that check_objective matches the final subgoal "
            "sequence exactly."
        )
    if category == "subgoal_execution_failure":
        if "ballistic_object_to_region" in lowered_feedback or "ballistic" in lowered_feedback:
            return (
                "Repair the ballistic relation, not the whole theme. Preserve the "
                "object, wall/barrier, goal sensor, and Tier 5 objective. Align "
                "agent -> ball/object -> barrier -> goal on one clear axis, keep "
                "the goal as a generous sensor=True target, keep posts/frames "
                "sensor-only, stage the object within 260-300 px of the goal, "
                "lower/cap barrier height if apex telemetry is below the required "
                "clearance, reduce object friction/mass, and increase agent_strength "
                "through declared registry/layout knobs. If check_objective requires "
                "crossing or apex state, objective_profile/semantic_requirements must "
                "declare the same relation so the validator can verify and repair it."
            )
        if "field_force_interaction" in lowered_feedback:
            return (
                "Repair the field-force subgoal with measured physics changes, not a "
                "theme rewrite. Ensure the object starts inside or immediately before "
                "the registered force zone, remove solid blockers between object and "
                "zone, enlarge the non-blocking force-zone/sensor footprint, increase "
                "bounded field strength, reduce falloff, and lower affected object "
                "mass/friction if the field effect is too weak. The subgoal's field "
                "name must match self.register_force_zone(...), and the force direction "
                "or attract/repel mode must improve the declared progress metric."
            )
        if "survive_duration" in lowered_feedback or '"objective_type": "survival"' in lowered_feedback:
            return (
                "Repair the survival-duration objective specifically. Do not treat it as "
                "navigation or object pushing. Ensure hazards are present and registered, "
                "spawn outside the agent's initial overlap, leave at least one dodge/safe "
                "corridor, track elapsed survival time in deterministic state, and make "
                "check_objective true only when the agent remains unhit for the required "
                "duration. Do not shorten the requested duration to pass; instead make "
                "projectile cadence, warmup, speed, radius, and safe-lane geometry fair "
                "enough for the deterministic survival oracle."
            )
        return (
            "Repair the failed physical subgoal, not the whole theme. For "
            "move_object_to_region, place the movable object between the agent and "
            "region, keep the object dynamic and pushable, reduce friction/mass if "
            "needed, and make the region threshold achievable. Pressure plates, goals, "
            "and trigger regions should be created with sensor=True so they do not "
            "physically block the object that must enter them. Keep the first push "
            "short and validator-friendly: agent -> movable object -> target region "
            "should be roughly collinear, with the object-to-region start distance "
            "under 140 px unless guides or rails assist the motion. For "
            "agent_reach_region, ensure the target region is reachable after prior "
            "subgoals complete."
        )
    if category == "contact_control_failure":
        if "ballistic_object_to_region" in lowered_feedback or "ballistic" in lowered_feedback:
            return (
                "Repair the ballistic contact-control failure with measured relation "
                "knobs. Do not remove the required wall/barrier. Put the agent close "
                "behind the ball/object on the object-to-goal axis, keep the ball "
                "light/round/low-friction, enlarge the non-blocking goal sensor, "
                "make all goal posts/frames/decor sensor-only, cap barrier height "
                "to a solvable visible arc, and increase agent_strength enough for "
                "a crisp impulse. If sideways drift appears, remove angled contacts "
                "and keep the approach lane straight; if object_blocked_or_pinned "
                "appears, only sensorize decorations, not the required barrier."
            )
        return (
            "Repair the concrete contact-control failure using the validator's push "
            "diagnostics. Preserve the objective and Tier 5 requirement. If "
            "object_moved_away_from_region appears, place the agent directly behind "
            "the object on the object-to-region axis and add short guide rails or "
            "corridor walls. If agent_never_reached_push_side appears, move the "
            "agent closer to the staging point with clear free space. If "
            "agent_or_object_drifted_sideways appears, narrow and straighten the "
            "push lane. If object_never_gained_forward_velocity appears, lower "
            "object mass/friction or increase agent_strength through the registry. "
            "Keep target regions sensor=True and do not replace the task with simple navigation."
        )
    if category in {"post_subgoal_navigation_failure", "post_mechanism_navigation_failure"}:
        return (
            "NARROW LLM REPAIR ONLY: repair the final post-subgoal reach step using "
            "the validator's reach/path diagnostics. Preserve completed earlier subgoals "
            "exactly: do not change the movable object, trigger/plate, mechanism, object "
            "names, objective_type, objective_targets, or success predicate. Only simplify "
            "the route after the earlier subgoal has completed. Valid narrow edits: move "
            "the final goal onto the open path, enlarge the final goal sensor when the "
            "agent got close, remove/relocate route blockers, straighten the final corridor, "
            "or make decorative/support geometry truly non-blocking/passable. "
            "If post_subgoal_path_blocked appears, clear/sensorize the named blocker or "
            "move the agent/goal onto a reachable open lane. If target_threshold_or_sensor_too_tight "
            "appears, enlarge the goal sensor or align check_objective with the visible "
            "goal radius. Do not lower Tier 5 and do not replace the task with simple navigation."
        )
    if category == "subgoal_affordance_failure":
        return (
            "Repair the exact physical affordance failure reported by the validator. "
            "Use the numeric telemetry literally: move named objects closer when "
            "object_to_region_distance exceeds the recommendation, align "
            "agent -> object -> region when alignment_angle_degrees is too high, "
            "make target regions sensor=True when region_not_sensor appears, and "
            "remove or relocate any named solid_blocker_between_object_and_region. "
            "If object_not_passively_stable appears, add a stable floor, shelf, "
            "rail, or flat guide under the object so it does not fall or drift "
            "before the agent touches it. For seesaw/lever launch tasks, if the "
            "prompt does not explicitly require pushing the weight, remove the "
            "fragile push-to-impact subgoal and use a lever_launch subgoal with "
            "the weight staged on/above the load side. If a push lane is kept, "
            "place rails outside the object's initial bounding circle; do not let "
            "horizontal rails or stop blocks overlap the ball or cross the push lane. "
            "Do not change the task semantics or lower the required tier."
        )
    if category in {"code_contract_error", "api_keyword_error"}:
        if "soccer-style" in lowered_feedback or "create_strike_shot_lane" in lowered_feedback:
            return (
                "Repair the contextual task interpretation, not just syntax. Use "
                "create_strike_shot_lane only for sports/ball strike tasks such as "
                "soccer, hockey, basketball, billiards, pinball, kick-ball-into-goal, "
                "or throw-ball-into-hoop. For spaceships, lasers, bullets, missiles, "
                "turrets, enemy shots, or dodge/avoid shooting prompts, model shots as "
                "dynamic role='hazard' projectiles or weapon-fire dynamics with "
                "survival/avoidance/navigation subgoals. Do not declare "
                "strike_object_to_region for projectile combat."
            )
        if "projectile-combat prompts require" in lowered_feedback:
            return (
                "Repair projectile-combat semantics directly. Create/register actual "
                "dynamic shot/laser/projectile/bolt bodies with role='hazard'; enemy "
                "ships or turrets alone are shooters, not the projectile hazards. Give "
                "at least one projectile immediate velocity or bounded force so it "
                "travels 120+ px during passive validation, and keep every projectile "
                "timer/counter/flag declared in __init__."
            )
        if "set_solvability_hint" in lowered_feedback and (
            "survival" in lowered_feedback or "survive_duration" in lowered_feedback
        ):
            return (
                "Repair the survival/projectile contract, not route geometry. Survival "
                "or pure dodge worlds do not need to become navigation tasks. Keep "
                "objective_type='survival', declare survive_duration, register moving "
                "hazards/projectiles, and only add set_solvability_hint if the objective "
                "also contains an explicit agent_reach_region route."
            )
        return (
            "Repair the generated code contract before changing geometry. Cross-reference "
            "environment_spec.json, remove forbidden or unsupported helper arguments, "
            "preserve the BaseEnv skeleton, and keep every self.* state variable in the "
            "mandatory __init__ registry. If you implement after_step, its signature must "
            "be exactly after_step(self); BaseEnv does not pass dt or any other argument."
        )
    if category == "object_model_error":
        return (
            "Repair object access, not geometry. Store object names in registry lists, "
            "retrieve records with self.get_object(name), use record.body.position for "
            "coordinates, and prefer self.distance_between('agent', target_name) for "
            "touch/proximity objectives."
        )
    if category == "objective_logic_error":
        return (
            "Repair check_objective and objective metadata before changing geometry. "
            "Ensure objective_type and objective_targets match the task, use persistent "
            "state such as self.targets_touched for collection tasks, and return True "
            "exactly when the code-level objective is satisfied."
        )
    if category == "structural_invalidity":
        return (
            "Repair the world structure. Ensure a dynamic role='agent' object exists, "
            "all objective_targets are registered object names, required objects are "
            "created before validation, and solvability_check matches the registered "
            "start/goal or target region."
        )
    if category == "mechanical_leverage":
        return (
            "Repair interaction mechanics. Increase usable agent leverage, reduce moved "
            "object masses, lower friction/damping where appropriate, or add a clear "
            "mechanical advantage such as a ramp, lever, or pivot."
        )
    if category == "kinetic_partial_progress":
        return (
            "The oracle made progress but did not finish. Preserve the working objective "
            "logic, then adjust target placement, spacing, thresholds, or interaction "
            "layout so the baseline kinetic oracle can complete all required targets."
        )
    if category == "kinetic_no_progress":
        return (
            "The code contract is valid but the kinetic oracle made no progress. Bring "
            "the first objective target closer, improve free space, increase agent "
            "strength through registry tuning, or simplify the first interaction."
        )
    if category == "tier_below_acceptance":
        return (
            "The validator observed progress, but the world did not reach the objective "
            "profile's required verification tier. Either make the task fully solvable "
            "by the declared validator skill, or only lower minimum_acceptance_tier when "
            "the objective is genuinely a novel or mechanism-style progress task."
        )
    if category == "capability_mismatch":
        return (
            "Repair the profile/code mismatch. The objective requires an interaction "
            "that the declared capability_profile does not permit. Add the necessary "
            "physical control capability, such as apply_force_x/apply_force_y or "
            "push_contact, or redesign the objective so it matches the declared agent."
        )
    if category == "validator_solver_weakness":
        return (
            "The world may be physically plausible, but the current validator skill did "
            "not produce enough evidence. Add clearer progress metrics, choose a more "
            "specific validator_skill, simplify the first objective interaction, or mark "
            "the objective as custom_physics/mechanism with Tier 4 only when progress "
            "verification is the honest target."
        )
    if category == "physics_instability":
        return (
            "Repair physics stability before changing objective logic. Remove initial "
            "overlaps, separate jointed bodies, reduce extreme elasticity, keep masses "
            "within interaction-possible ranges, and avoid jitter-prone tiny gaps."
        )
    if _is_attribute_error_feedback(feedback):
        return (
            "The previous failure is a code-definition error, not a layout or "
            "pathfinding failure. Do not widen paths, move obstacles, or change "
            "geometry as the primary fix. Define the missing self.<name> variable "
            "in __init__ before any method reads it."
        )
    return (
        "If the previous attempt was blocked, preserve the user's requested theme "
        "while changing geometry so the agent has a mathematically valid route to "
        "the goal. Widen narrow passages, move or shrink the named blocker, and "
        "update solvability_check so it matches the new route."
    )


def _contract_fast_lane_retry_allowed(
    validation: ValidationResult,
    retry_index: int,
) -> bool:
    """Return True when a failure should be repaired before spending an attempt."""

    if retry_index >= MAX_CONTRACT_FAST_LANE_RETRIES:
        return False
    category = _failure_category(validation)
    if category in {
        "code_contract_error",
        "api_keyword_error",
        "object_model_error",
        "semantic_contract_error",
    }:
        return True
    lowered = str(validation.reason or "").lower()
    return any(
        marker in lowered
        for marker in (
            "missing code-level objective",
            "mandatory state registry",
            "failed verification",
            "unexpected keyword argument",
            "attributeerror",
            "typeerror",
            "after_step",
            "pymunk.vec2d",
        )
    )


def _contract_fast_lane_prompt(previous_error: str) -> str:
    return textwrap.dedent(
        f"""
        CONTRACT FAST-LANE RETRY:
        The previous candidate failed the Python/BaseEnv contract before the
        harness could judge the world design. Do not redesign the task, layout,
        theme, objective, entities, or visual intent. Only fix the code contract.

        Previous contract error:
        {previous_error}

        Required narrow fixes:
        - Preserve class name, objective_profile, capability_profile, layout_plan,
          physics_relations, semantic_requirements, and check_objective semantics.
        - Declare every self.* variable used in build_world/add_objects,
          after_step, reset_objective_state, check_objective, or get_ground_truth
          in the mandatory __init__ registry first.
        - Keep after_step signature exactly after_step(self); BaseEnv does not
          pass dt or action.
        - Use only BaseEnv helper APIs and allowed arguments from environment_spec.json.
        - Do not lower the verification tier or convert the task into a simpler one.
        """
    ).strip()


def _maze_env_code(
    class_name: str,
    source_prompt: str,
    *,
    layout_note: str,
    first_wall_center: tuple[int, int],
    first_wall_size: tuple[int, int],
    second_wall_center: tuple[int, int],
    second_wall_size: tuple[int, int],
    gate_center: tuple[int, int],
    gate_size: tuple[int, int],
    hazard_positions: tuple[tuple[int, int], ...],
    start: tuple[int, int] = (90, 120),
    goal: tuple[int, int] = (820, 430),
) -> str:
    hazard_lines = "\n".join(
        f"""        self.create_dynamic_circle(
            "falling_ball_{index}",
            pos={position},
            radius=14,
            role="hazard",
            metadata={{"purpose": "dynamic pressure element", "layout": "{layout_note}"}},
        )"""
        for index, position in enumerate(hazard_positions, start=1)
    )
    if not hazard_lines:
        hazard_lines = "        # No dynamic hazards for this seed."

    return textwrap.dedent(
        f'''
        from base_env import BaseEnv, EnvConfig


        class {class_name}(BaseEnv):
            def __init__(self, config: EnvConfig | None = None):
                super().__init__(config=config or EnvConfig(width=920, height=560), auto_reset=False)
                self.objective_type = "navigation_goal"
                self.objective_targets = ["goal"]
                self.semantic_requirements = []
                self.layout_plan = {{
                    "layout_type": "local_smoke_navigation",
                    "route_model": "two_passage_smoke_route",
                    "start": {{"name": "agent", "position": {start}}},
                    "goal": {{"name": "goal", "position": {goal}, "size": (34, 34), "sensor": False}},
                    "critical_path_points": [{start}, {goal}],
                    "protected_zones": [],
                    "construction_rules": ["local deterministic smoke template"],
                }}
                self.layout = {{
                    "agent_start": {start},
                    "goal_center": {goal},
                    "goal_size": (34, 34),
                    "first_wall_center": {first_wall_center},
                    "first_wall_size": {first_wall_size},
                    "second_wall_center": {second_wall_center},
                    "second_wall_size": {second_wall_size},
                    "gate_center": {gate_center},
                    "gate_size": {gate_size},
                    "hazard_positions": {list(hazard_positions)!r},
                    "success_x_threshold": 130.0,
                }}
                self.objective_profile = {{
                    "objective_type": self.objective_type,
                    "objective_description": "Navigate through two narrow passages and reach the goal.",
                    "success_predicate": "agent crosses the verified progress threshold toward the goal",
                    "targets": self.objective_targets,
                    "required_capabilities": ["ground_force", "touch_contact"],
                    "progress_metrics": ["agent_x_position", "agent_goal_distance"],
                    "subgoals": [
                        {{"kind": "agent_reach_region", "target": "goal"}},
                    ],
                    "validator_skills": ["navigation_probe"],
                    "failure_modes": ["blocked_path", "agent_underpowered"],
                    "minimum_acceptance_tier": 5,
                }}
                self.capability_profile = {{
                    "movement": "ground_force",
                    "interaction": ["touch_contact"],
                    "gravity": "normal",
                    "allowed_controls": ["apply_force_x", "apply_force_y", "brake"],
                    "forbidden_controls": [
                        "teleport",
                        "direct_object_move",
                        "direct_object_rotation",
                        "direct_goal_state_write",
                    ],
                    "notes": "Local smoke template uses physical force only.",
                }}
                self.physics_relations = {{}}
                self.agent_radius = 15
                self.agent_strength = 2500
                self.touch_threshold = 20
                self.reset()

            def build_world(self) -> None:
                self.add_objects()

            def add_objects(self) -> None:
                self.create_static_segment(
                    "floor",
                    a=(40, 60),
                    b=(880, 60),
                    radius=3,
                    role="terrain",
                    metadata={{"purpose": "lower world boundary", "layout": "{layout_note}"}},
                )
                self.create_static_segment(
                    "ceiling",
                    a=(40, 520),
                    b=(880, 520),
                    radius=3,
                    role="terrain",
                    metadata={{"purpose": "upper world boundary", "layout": "{layout_note}"}},
                )
                self.create_static_segment(
                    "left_wall",
                    a=(40, 60),
                    b=(40, 520),
                    radius=3,
                    role="terrain",
                    metadata={{"purpose": "left world boundary", "layout": "{layout_note}"}},
                )
                self.create_static_segment(
                    "right_wall",
                    a=(880, 60),
                    b=(880, 520),
                    radius=3,
                    role="terrain",
                    metadata={{"purpose": "right world boundary", "layout": "{layout_note}"}},
                )
                self.create_static_box(
                    "maze_wall_top_passage",
                    center={first_wall_center},
                    size={first_wall_size},
                    role="obstacle",
                    metadata={{"purpose": "forces first narrow passage", "passage": "upper"}},
                )
                self.create_static_box(
                    "maze_wall_bottom_passage",
                    center={second_wall_center},
                    size={second_wall_size},
                    role="obstacle",
                    metadata={{"purpose": "forces second narrow passage", "passage": "lower"}},
                )
                self.create_static_box(
                    "sliding_gate_open",
                    center={gate_center},
                    size={gate_size},
                    role="obstacle",
                    metadata={{
                        "purpose": "sliding gate represented at a verified open offset",
                        "motion_axis": "vertical",
                    }},
                )
                self.create_dynamic_circle(
                    "agent",
                    pos={start},
                    radius=14,
                    mass=1.0,
                    role="agent",
                    metadata={{"purpose": "validator start body", "route": "two passage maze"}},
                )
                self.create_static_box(
                    "goal",
                    center={goal},
                    size=(34, 34),
                    role="goal",
                    metadata={{"purpose": "target region after the sliding gate"}},
                )
        {hazard_lines}
                self.set_solvability_hint(
                    start={start},
                    goal={goal},
                    grid_size=22,
                    agent_radius=14,
                    notes="Route uses the two narrow passages and the open side of the sliding gate.",
                    metadata={{"layout": "{layout_note}"}},
                )

            def check_objective(self) -> bool:
                return self.get_object("agent").body.position.x > 130.0

            def get_ground_truth(self):
                truth = super().get_ground_truth()
                truth["objective"].update({{
                    "objective_type": self.objective_type,
                    "objective_targets": list(self.objective_targets),
                    "objective_profile": self.objective_profile,
                    "capability_profile": self.capability_profile,
                    "physics_relations": self.physics_relations,
                    "layout_plan": self.layout_plan,
                    "objective_satisfied": self.check_objective(),
                    "agent_goal_distance": self.distance_between("agent", "goal"),
                    "agent_x_threshold": 130.0,
                }})
                return truth


        GENERATED_ENV_CLASS = "{class_name}"
        SOURCE_PROMPT = {source_prompt!r}
        '''
    ).strip()


def _failure_result(reason: str, class_name: str) -> ValidationResult:
    return ValidationResult(
        solvable=False,
        reason=reason,
        env_class=class_name,
        details={"stage": "architect_or_contract"},
    )


def _with_failure_category(result: ValidationResult) -> ValidationResult:
    """Attach a diagnostic failure category for healer/post-mortem routing."""

    if result.accepted:
        category = "success"
    else:
        category = _classify_failure(result)
    details = dict(result.details)
    details["failure_category"] = category
    details["diagnostic_axis"] = _diagnostic_axis_for_category(category)
    details["healer_focus"] = _healer_focus_for_category(category)
    return replace(result, details=details)


def _failure_category(result: ValidationResult) -> str:
    category = result.details.get("failure_category")
    if isinstance(category, str) and category:
        return category
    if result.accepted:
        return "success"
    return _classify_failure(result)


def _classify_failure(result: ValidationResult) -> str:
    text = _result_text(result)
    lowered = text.lower()
    observed_modes = _observed_failure_modes(result)
    reason_lowered = str(result.reason or "").lower()

    if result.details.get("contract_errors") or "contract validation failed" in lowered:
        return "semantic_contract_error"
    if "prompt anti-cheat validation failed" in lowered or result.details.get("anti_cheat_failures"):
        return "prompt_anticheat_failure"
    if "semantic dynamics validation failed" in lowered or result.details.get("semantic_failures"):
        return "semantic_dynamics_failure"
    if "gameplay dynamics validation failed" in lowered or result.details.get("gameplay_failures"):
        return "gameplay_dynamics_failure"
    if _is_post_subgoal_reach_failure(result):
        return "post_subgoal_navigation_failure"
    if "affordance check failed" in lowered or result.details.get("affordance_failures"):
        return "subgoal_affordance_failure"
    if "nan" in lowered or "jitter" in lowered or "unstable" in lowered:
        return "physics_instability"
    if observed_modes & {
        "object_moved_away_from_region",
        "object_made_insufficient_forward_progress",
        "object_never_gained_forward_velocity",
        "agent_object_collision_filtered",
        "agent_force_not_applied",
        "agent_force_not_aligned_with_push_axis",
        "agent_pinned_or_immobile",
        "agent_never_contacted_object",
        "agent_never_reached_push_side",
        "agent_or_object_drifted_sideways",
        "object_stationary",
        "object_blocked_or_pinned",
        "ballistic_apex_below_barrier_clearance",
        "ballistic_object_never_crossed_barrier",
        "generic_push_controller_failed",
    }:
        return "contact_control_failure"
    if (
        observed_modes
        & {
            "post_mechanism_path_blocked",
            "post_subgoal_path_blocked",
            "target_offset_from_reachable_lane",
            "target_threshold_or_sensor_too_tight",
        }
    ):
        return "post_subgoal_navigation_failure"
    if "subgoal" in reason_lowered and (
        "missing kind" in reason_lowered or "registered object" in reason_lowered
    ):
        return "subgoal_plan_error"
    if "subgoal" in reason_lowered:
        return "subgoal_execution_failure"
    if result.kinetic_progress and result.achieved_tier < result.operational_acceptance_tier:
        return "tier_below_acceptance"
    if "capability_mismatch" in lowered:
        return "capability_mismatch"
    if "objective_profile" in lowered or "capability_profile" in lowered:
        return "semantic_profile_error"
    if "objectrecord" in lowered or "direct position" in lowered:
        return "object_model_error"
    if "unsupported keyword" in lowered or "forbidden" in lowered:
        return "api_keyword_error"
    if (
        result.details.get("stage") == "architect_or_contract"
        or not result.contract_valid
        or "generated environment failed verification" in lowered
        or "missing code-level objective" in lowered
        or "mandatory state registry" in lowered
        or "references self." in lowered
        or "assigns self." in lowered
        or "must define mandatory variable" in lowered
    ):
        return "code_contract_error"
    if not result.structurally_valid:
        return "structural_invalidity"
    if not result.objective_valid or "objective check failed" in lowered:
        return "objective_logic_error"
    if "inadequate mechanical leverage" in lowered:
        return "mechanical_leverage"
    if result.kinetic_progress and not result.kinetic_solved:
        return "kinetic_partial_progress"
    if result.objective_valid and result.structurally_valid and not result.kinetic_progress:
        return "kinetic_no_progress"
    if (
        result.objective_valid
        and result.structurally_valid
        and result.achieved_tier < result.operational_acceptance_tier
    ):
        return "validator_solver_weakness"
    if result.blocking_object:
        return "physical_blockage"
    return "unknown_failure"


def _is_post_subgoal_reach_failure(result: ValidationResult) -> bool:
    failed = result.details.get("failed_subgoal")
    if not isinstance(failed, dict) or str(failed.get("kind") or "") != "agent_reach_region":
        return False
    completed = result.details.get("completed_subgoals")
    if not isinstance(completed, list):
        return False
    completed_kinds = {
        str(subgoal.get("kind") or "")
        for subgoal in completed
        if isinstance(subgoal, dict)
    }
    return bool(
        completed_kinds
        & {
            "activate_mechanism",
            "move_object_to_region",
            "low_friction_slide_to_region",
            "lever_launch",
            "field_force_interaction",
            "survive_duration",
            "maintain_balance",
        }
    )


def _observed_failure_modes(result: ValidationResult) -> set[str]:
    modes: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "failure_modes":
                    modes.update(str(item) for item in _list_field(child))
                else:
                    visit(child)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for key in ("subgoal_diagnostics", "progress_events", "mechanism_diagnostics"):
        visit(result.details.get(key))
    return modes


def _list_field(value: Any) -> list[Any]:
    if isinstance(value, list | tuple | set):
        return list(value)
    if value is None:
        return []
    return [value]


def _diagnostic_axis_for_category(category: str) -> str:
    axis_by_category = {
        "success": "accepted",
        "initial_generation": "initial_generation",
        "semantic_contract_error": "code_contract",
        "code_contract_error": "code_contract",
        "api_keyword_error": "code_contract",
        "object_model_error": "code_contract",
        "semantic_profile_error": "semantic_profile",
        "prompt_anticheat_failure": "prompt_fidelity_anticheat",
        "semantic_dynamics_failure": "semantic_dynamics",
        "gameplay_dynamics_failure": "gameplay_dynamics",
        "subgoal_plan_error": "objective_decomposition",
        "subgoal_execution_failure": "objective_decomposition",
        "subgoal_affordance_failure": "objective_affordance",
        "contact_control_failure": "contact_control",
        "post_subgoal_navigation_failure": "post_subgoal_navigation",
        "post_mechanism_navigation_failure": "post_subgoal_navigation",
        "structural_invalidity": "semantic_structure",
        "objective_logic_error": "objective_definition",
        "capability_mismatch": "capability_profile",
        "mechanical_leverage": "physics_interaction",
        "physical_blockage": "physics_geometry",
        "physics_instability": "physics_stability",
        "kinetic_no_progress": "validator_or_actionability",
        "kinetic_partial_progress": "validator_solver_weakness",
        "tier_below_acceptance": "validator_solver_weakness",
        "validator_solver_weakness": "validator_solver_weakness",
        "unknown_failure": "unknown",
    }
    return axis_by_category.get(category, "unknown")


def _healer_focus_for_category(category: str) -> str:
    if category == "semantic_contract_error":
        return "Fix semantic/API agreement between profiles, subgoals, and registered resources."
    focus_by_axis = {
        "code_contract": "Fix Python/API contract errors before touching layout.",
        "semantic_profile": "Align objective_profile, capability_profile, and prompt semantics.",
        "prompt_fidelity_anticheat": "Prevent degenerate wins that neutralize the prompt's intended challenge.",
        "semantic_dynamics": "Implement the prompt's physical verbs as observable simulation behavior.",
        "gameplay_dynamics": "Implement the gameplay loop, cadence, fairness, and feel targets.",
        "objective_decomposition": "Fix or physically support the ordered objective subgoals.",
        "objective_affordance": "Fix concrete pre-rollout geometry/physics affordances for the declared subgoals.",
        "contact_control": "Fix the physical contact trajectory using rollout diagnostics.",
        "post_subgoal_navigation": "Fix the final path after completed subgoals.",
        "semantic_structure": "Create the declared agent, targets, and named regions.",
        "objective_definition": "Fix check_objective and persistent objective state.",
        "capability_profile": "Align allowed controls with the task's required interaction.",
        "physics_interaction": "Tune masses, force, friction, and mechanical advantage.",
        "physics_geometry": "Move blockers, widen valid passages, or relocate start/target.",
        "physics_stability": "Remove overlap/jitter sources and stabilize constraints.",
        "validator_or_actionability": "Make the first interaction actionable by the declared agent.",
        "validator_solver_weakness": "Improve validator skill selection or reduce the required tier only for honest progress tasks.",
        "accepted": "No repair needed.",
        "initial_generation": "Generate the first candidate.",
        "unknown": "Inspect telemetry and repair the highest-confidence concrete failure.",
    }
    return focus_by_axis.get(_diagnostic_axis_for_category(category), focus_by_axis["unknown"])


def _failure_category_from_text(text: str) -> str:
    return _classify_failure(
        ValidationResult(
            solvable=False,
            reason=text,
            env_class="UnknownEnv",
            details={"stage": "text_only"},
        )
    )


def _result_text(result: ValidationResult) -> str:
    return "\n".join(
        [
            result.reason,
            result.blocking_object or "",
            json.dumps(result.details, sort_keys=True, default=str),
        ]
    )


def _promote_verified_env(candidate_env_path: Path) -> Path:
    """Copy a validation-passing candidate into generated_envs/."""

    GENERATED_ENVS_DIR.mkdir(parents=True, exist_ok=True)
    verified_env_path = GENERATED_ENVS_DIR / candidate_env_path.name
    shutil.copyfile(candidate_env_path, verified_env_path)
    return verified_env_path


def _record_visual_recipe(
    *,
    prompt: str,
    env_path: Path,
    validation: ValidationResult,
    run_dir: Path,
) -> Path | None:
    """Ask the Visual Director for a renderer-only sidecar recipe."""

    try:
        recipe_path = create_visual_recipe(
            prompt=prompt,
            env_path=env_path,
            validation=validation,
        )
        run_recipe_path = run_dir / recipe_path.name
        shutil.copyfile(recipe_path, run_recipe_path)
        return recipe_path
    except Exception as exc:
        _write_text(
            run_dir / "visual_director_error.txt",
            f"Visual Director failed without affecting validation: {type(exc).__name__}: {exc}\n",
        )
        return None


def _record_world_export(
    *,
    prompt: str,
    env_path: Path,
    validation: ValidationResult,
    visual_recipe_path: Path | None,
    run_dir: Path,
) -> Path | None:
    """Export the verified world schema for external runtimes such as Godot."""

    try:
        latest_export_dir = export_verified_world(
            env_path,
            prompt=prompt,
            validation=validation,
            visual_recipe_path=visual_recipe_path,
        )
        run_export_dir = run_dir / "world_export"
        if run_export_dir.exists():
            shutil.rmtree(run_export_dir)
        shutil.copytree(latest_export_dir, run_export_dir)
        return latest_export_dir
    except Exception as exc:
        _write_text(
            run_dir / "world_export_error.txt",
            f"World export failed without affecting validation: {type(exc).__name__}: {exc}\n",
        )
        return None


def _record_capability_gap(
    *,
    prompt: str,
    validation: ValidationResult | None,
    attempts: list[AttemptRecord],
    run_dir: Path,
    generated_env_path: Path | None,
    post_mortem: str | None,
) -> Path | None:
    """Persist a reusable memory record for failed or only-progress-verified task families."""

    if validation is None and _is_external_provider_failure(post_mortem or ""):
        return None
    if validation is not None and validation.accepted and validation.verification_gap <= 0:
        return None

    objective_profile = _dict_field(validation.details.get("objective_profile")) if validation else {}
    capability_profile = _dict_field(validation.details.get("capability_profile")) if validation else {}
    tier_policy = _dict_field(validation.details.get("tier_policy")) if validation else {}
    validator_route = _dict_field(validation.details.get("validator_route")) if validation else {}
    failure_category = _failure_category(validation) if validation else "generation_or_runtime_failure"
    diagnostic_axis = (
        validation.details.get("diagnostic_axis")
        if validation
        else None
    ) or _diagnostic_axis_for_category(failure_category)
    task_family = str(
        tier_policy.get("task_family")
        or objective_profile.get("objective_type")
        or "unknown_task_family"
    )
    failed_subgoals = _capability_gap_failed_subgoals(validation, attempts)
    failed_probes = _capability_gap_failed_probes(validation, attempts)
    missing_capability = _capability_gap_missing_capability(
        failure_category,
        failed_subgoals,
        failed_probes,
        validation,
    )
    gap_id = _capability_gap_id(prompt, task_family, missing_capability)
    gap = {
        "gap_id": gap_id,
        "version": "capability-gap-1.0",
        "source": "harness_failure" if not (validation and validation.accepted) else "harness_partial_acceptance",
        "source_prompt": prompt,
        "source_run_dir": str(run_dir),
        "source_env_path": str(generated_env_path) if generated_env_path else None,
        "task_family": task_family,
        "objective_type": objective_profile.get("objective_type") or (validation.details.get("objective_type") if validation else None),
        "diagnostic_axis": str(diagnostic_axis or "unknown"),
        "failure_category": failure_category,
        "missing_capability": missing_capability,
        "capability_gap": _capability_gap_text(validation, missing_capability),
        "reason": validation.reason if validation else (post_mortem or "No validation result was produced."),
        "objective_tier": validation.objective_tier if validation else 5,
        "operational_acceptance_tier": validation.operational_acceptance_tier if validation else 0,
        "achieved_tier": validation.achieved_tier if validation else 0,
        "verification_gap": validation.verification_gap if validation else 5,
        "accepted": validation.accepted if validation else False,
        "blocking_object": validation.blocking_object if validation else None,
        "failed_subgoals": failed_subgoals,
        "failed_probes": failed_probes,
        "suggested_new_blocks": _capability_gap_suggested_blocks(failed_subgoals, failure_category),
        "suggested_new_probes": _capability_gap_suggested_probes(failed_subgoals, failure_category),
        "repair_guidance": _capability_gap_repair_guidance(validation, failure_category),
        "risk_flags": _string_list(tier_policy.get("risk_flags")),
        "validator_focus": _string_list(tier_policy.get("suggested_validator_focus"))
        or _string_list(validator_route.get("validator_skills")),
        "trigger_keywords": _learned_trigger_keywords(prompt, objective_profile, capability_profile),
        "post_mortem_excerpt": _post_mortem_excerpt(post_mortem),
    }
    CAPABILITY_GAPS_DIR.mkdir(parents=True, exist_ok=True)
    family_slug = "_".join(re.findall(r"[a-z0-9]+", task_family.lower())) or "unknown"
    directory = CAPABILITY_GAPS_DIR / family_slug
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{gap_id}.json"
    path.write_text(json.dumps(gap, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _is_external_provider_failure(text: str) -> bool:
    lowered = text.lower()
    provider_markers = (
        "insufficient_quota",
        "rate limit",
        "rate_limit",
        "openai api",
        "api key",
        "authentication",
        "connection error",
        "service unavailable",
    )
    return any(marker in lowered for marker in provider_markers)


def _capability_gap_id(prompt: str, task_family: str, missing_capability: str) -> str:
    family_slug = "_".join(re.findall(r"[a-z0-9]+", task_family.lower())) or "unknown"
    capability_slug = "_".join(re.findall(r"[a-z0-9]+", missing_capability.lower())[:8]) or "gap"
    prompt_slug = "_".join(re.findall(r"[a-z0-9]+", prompt.lower())[:8]) or "world"
    return f"gap_{family_slug}_{capability_slug}_{prompt_slug}"


def _capability_gap_failed_subgoals(
    validation: ValidationResult | None,
    attempts: list[AttemptRecord],
) -> list[str]:
    names: list[str] = []
    sources: list[dict[str, Any]] = []
    if validation is not None:
        sources.append(validation.details)
    sources.extend(attempt.result.details for attempt in attempts[-3:])
    for details in sources:
        for key in ("failed_subgoal", "subgoal"):
            value = details.get(key)
            if isinstance(value, dict) and value.get("kind"):
                names.append(str(value.get("kind")))
        objective_profile = _dict_field(details.get("objective_profile"))
        if details.get("objective_type") in {"survival", "custom_physics"}:
            subgoals = objective_profile.get("subgoals")
            if isinstance(subgoals, list):
                for subgoal in subgoals:
                    if isinstance(subgoal, dict) and subgoal.get("kind"):
                        names.append(str(subgoal.get("kind")))
        failures = details.get("affordance_failures")
        if isinstance(failures, list):
            for failure in failures:
                subgoal = failure.get("subgoal") if isinstance(failure, dict) else None
                if isinstance(subgoal, dict) and subgoal.get("kind"):
                    names.append(str(subgoal.get("kind")))
        progress_events = details.get("progress_events")
        if isinstance(progress_events, list):
            for event in progress_events:
                subgoal = event.get("subgoal") if isinstance(event, dict) else None
                if isinstance(subgoal, dict) and subgoal.get("kind"):
                    names.append(str(subgoal.get("kind")))
    return sorted(set(names))


def _capability_gap_failed_probes(
    validation: ValidationResult | None,
    attempts: list[AttemptRecord],
) -> list[str]:
    probes: list[str] = []
    details_list = []
    if validation is not None:
        details_list.append(validation.details)
    details_list.extend(attempt.result.details for attempt in attempts[-3:])
    for details in details_list:
        for probe in _collect_probe_dicts(details):
            if not bool(probe.get("passed")):
                name = str(probe.get("name") or "")
                diagnosis = str(probe.get("diagnosis") or "")
                if name:
                    probes.append(f"{name}:{diagnosis}" if diagnosis else name)
    return sorted(set(probes))


def _capability_gap_missing_capability(
    failure_category: str,
    failed_subgoals: list[str],
    failed_probes: list[str],
    validation: ValidationResult | None,
) -> str:
    if failed_subgoals:
        if "bounce_to_target" in failed_subgoals:
            return "robust bounce_to_target trajectory solver and repair probes"
        if "activate_mechanism" in failed_subgoals:
            return "direct agent-triggered mechanism activation solver"
        if "maintain_balance" in failed_subgoals:
            return "balance-duration controller and stability validator"
        if "classify_objects_to_regions" in failed_subgoals:
            return "multi-object classification and placement validator"
        if "field_force_interaction" in failed_subgoals:
            return "field-force trajectory/progress validator"
        if "move_object_to_region" in failed_subgoals:
            return "more robust object manipulation affordance or push controller"
        if "agent_reach_region" in failed_subgoals and failure_category in {"post_subgoal_navigation_failure", "post_mechanism_navigation_failure"}:
            return "post-subgoal route simplification and final reach repair"
        if "survive_duration" in failed_subgoals:
            return "survival-duration hazard policy and objective progress validator"
    if any("passive_stability" in probe for probe in failed_probes):
        return "automatic stable staging/support design for dynamic objects"
    if validation:
        objective_type = str(validation.details.get("objective_type") or "").lower()
        if objective_type == "survival":
            return "survival-duration hazard policy and objective progress validator"
    if failure_category == "code_contract_error":
        return "generation contract repair memory"
    if failure_category == "semantic_contract_error":
        return "semantic contract repair memory"
    if failure_category == "physics_instability":
        return "physics stability and jitter repair memory"
    if validation and validation.verification_gap > 0:
        return "validator progress-to-solution upgrade for this task family"
    return f"{failure_category} repair capability"


def _capability_gap_suggested_blocks(
    failed_subgoals: list[str],
    failure_category: str,
) -> list[str]:
    suggestions: list[str] = []
    mapping = {
        "bounce_to_target": "elastic_bounce_to_target",
        "activate_mechanism": "agent_trigger_button_gate",
        "maintain_balance": "balance_duration_platform",
        "classify_objects_to_regions": "multi_object_sorting_room",
        "move_object_to_region": "guided_push_lane",
        "field_force_interaction": "field_force_interaction",
        "survive_duration": "survival_hazard_arena",
        "agent_reach_region": "reachable_post_mechanism_goal_corridor",
    }
    for subgoal in failed_subgoals:
        if subgoal in mapping:
            suggestions.append(mapping[subgoal])
    if failure_category == "physics_instability":
        suggestions.append("stable_floor_support")
    return sorted(set(suggestions))


def _capability_gap_suggested_probes(
    failed_subgoals: list[str],
    failure_category: str,
) -> list[str]:
    suggestions: list[str] = []
    mapping = {
        "bounce_to_target": ["trajectory_intersection", "elastic_collision_angle", "target_cup_capture"],
        "activate_mechanism": ["agent_trigger_contact", "mechanism_state_change", "post_activation_reachability"],
        "maintain_balance": ["angle_duration_stability", "platform_level_window", "agent_balance_response"],
        "classify_objects_to_regions": ["class_region_membership", "multi_object_assignment_progress"],
        "move_object_to_region": ["object_region_affordance", "push_contact", "object_inside_region"],
        "field_force_interaction": ["field_effect", "trajectory_change", "progress_metric_change"],
        "survive_duration": ["hazard_presence", "safe_spawn_margin", "survival_timer_progress", "agent_hazard_clearance"],
        "agent_reach_region": ["post_activation_reachability", "target_containment", "agent_moves_toward_target"],
    }
    for subgoal in failed_subgoals:
        suggestions.extend(mapping.get(subgoal, []))
    if failure_category == "physics_instability":
        suggestions.append("passive_stability")
    return sorted(set(suggestions))


def _capability_gap_repair_guidance(
    validation: ValidationResult | None,
    failure_category: str,
) -> list[str]:
    objective_type = str(validation.details.get("objective_type") or "").lower() if validation else ""
    failed_subgoals = _capability_gap_failed_subgoals(validation, []) if validation else []
    is_survival_gap = objective_type == "survival" or "survive_duration" in failed_subgoals
    guidance = []
    if not is_survival_gap:
        guidance.append(_repair_rule_for_category(failure_category, validation.reason if validation else ""))
    if validation is not None:
        if is_survival_gap:
            guidance.append(
                "For survive_duration, do not repair it like a navigation or push task. "
                "Ensure hazards exist, spawn outside the agent's starting overlap, leave at least one dodge corridor, "
                "track elapsed survival time in deterministic state, and make check_objective true only after the required duration."
            )
        for probe in _collect_probe_dicts(validation.details):
            if bool(probe.get("passed")):
                continue
            repair = str(probe.get("repair") or "").strip()
            if repair:
                guidance.append(repair)
    return sorted(set(item for item in guidance if item))


def _capability_gap_text(
    validation: ValidationResult | None,
    missing_capability: str,
) -> str:
    if validation is None:
        return f"No validation result; missing capability likely: {missing_capability}."
    if validation.verification_gap > 0:
        return (
            f"Achieved Tier {validation.achieved_tier}, objective Tier {validation.objective_tier}; "
            f"missing capability: {missing_capability}."
        )
    return f"Missing capability: {missing_capability}."


def _post_mortem_excerpt(post_mortem: str | None, *, max_chars: int = 900) -> str:
    if not post_mortem:
        return ""
    compact = " ".join(post_mortem.split())
    return compact[:max_chars]


def _record_learned_skill(
    *,
    prompt: str,
    env_path: Path,
    validation: ValidationResult,
) -> Path | None:
    """Persist an accepted profile/validator pattern for future retrieval."""

    if not validation.accepted or not validation.kinetic_solved:
        return None

    objective_profile = _dict_field(validation.details.get("objective_profile"))
    capability_profile = _dict_field(validation.details.get("capability_profile"))
    validator_route = _dict_field(validation.details.get("validator_route"))
    objective_type = str(
        objective_profile.get("objective_type")
        or validation.details.get("objective_type")
        or "custom_physics"
    )
    skill_id = _learned_skill_id(prompt, objective_type, validation.tier_name)
    learned_skill = {
        "skill_id": skill_id,
        "version": "learned-1.0",
        "source": "harness_success",
        "source_prompt": prompt,
        "source_env_path": str(env_path),
        "objective_type": objective_type,
        "summary": _learned_skill_summary(
            objective_profile,
            capability_profile,
            validation,
        ),
        "trigger_keywords": _learned_trigger_keywords(prompt, objective_profile, capability_profile),
        "objective_profile": objective_profile,
        "capability_profile": capability_profile,
        "validator_route": validator_route,
        "verification": {
            "accepted": validation.accepted,
            "achieved_tier": validation.achieved_tier,
            "tier_name": validation.tier_name,
            "objective_tier": validation.objective_tier,
            "operational_acceptance_tier": validation.operational_acceptance_tier,
            "minimum_acceptance_tier": validation.minimum_acceptance_tier,
            "reason": validation.reason,
        },
        "generation_guidance": _learned_generation_guidance(
            objective_profile,
            capability_profile,
            validator_route,
        ),
        "common_failures": _learned_common_failures(validation),
    }

    LEARNED_SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    path = LEARNED_SKILLS_DIR / f"{skill_id}.json"
    path.write_text(json.dumps(learned_skill, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _record_learned_affordance_block(
    *,
    prompt: str,
    env_path: Path,
    validation: ValidationResult,
) -> Path | None:
    """Persist a solved physical motif as future construction-kit memory."""

    if not validation.accepted or not validation.kinetic_solved:
        return None

    objective_profile = _dict_field(validation.details.get("objective_profile"))
    capability_profile = _dict_field(validation.details.get("capability_profile"))
    subgoals = objective_profile.get("subgoals")
    subgoals = subgoals if isinstance(subgoals, list) else []
    block_id = _learned_affordance_block_id(prompt, objective_profile, validation)
    block = {
        "block_id": block_id,
        "version": "learned-affordance-1.0",
        "source": "harness_tier5_success",
        "source_prompt": prompt,
        "source_env_path": str(env_path),
        "summary": _learned_affordance_summary(objective_profile, capability_profile, validation),
        "abstract_relation": _learned_abstract_relation(objective_profile, subgoals),
        "trigger_keywords": _learned_trigger_keywords(prompt, objective_profile, capability_profile),
        "creates": _learned_affordance_creates(objective_profile, subgoals),
        "constraints": _learned_affordance_constraints(validation),
        "composes_with": _learned_affordance_compositions(subgoals),
        "validator_checks": _learned_affordance_validator_checks(validation, subgoals),
        "repair_guidance": _learned_generation_guidance(
            objective_profile,
            capability_profile,
            _dict_field(validation.details.get("validator_route")),
        ),
        "verification": {
            "achieved_tier": validation.achieved_tier,
            "tier_name": validation.tier_name,
            "objective_tier": validation.objective_tier,
            "operational_acceptance_tier": validation.operational_acceptance_tier,
            "minimum_acceptance_tier": validation.minimum_acceptance_tier,
            "reason": validation.reason,
        },
    }
    LEARNED_AFFORDANCE_BLOCKS_DIR.mkdir(parents=True, exist_ok=True)
    path = LEARNED_AFFORDANCE_BLOCKS_DIR / f"{block_id}.json"
    path.write_text(json.dumps(block, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _learned_affordance_block_id(
    prompt: str,
    objective_profile: dict[str, object],
    validation: ValidationResult,
) -> str:
    relation = _learned_abstract_relation(
        objective_profile,
        objective_profile.get("subgoals") if isinstance(objective_profile.get("subgoals"), list) else [],
    )
    relation_slug = "_".join(re.findall(r"[a-z0-9]+", relation.lower())) or "custom"
    prompt_slug = "_".join(re.findall(r"[a-z0-9]+", prompt.lower())[:7]) or "world"
    tier_slug = "_".join(re.findall(r"[a-z0-9]+", validation.tier_name.lower())) or "verified"
    return f"learned_block_{relation_slug}_{tier_slug}_{prompt_slug}"


def _learned_affordance_summary(
    objective_profile: dict[str, object],
    capability_profile: dict[str, object],
    validation: ValidationResult,
) -> str:
    description = str(objective_profile.get("objective_description") or "Solved affordance motif")
    movement = capability_profile.get("movement", "unknown_movement")
    return f"{description} Solved with movement={movement} at Tier {validation.achieved_tier}."


def _learned_abstract_relation(
    objective_profile: dict[str, object],
    subgoals: list[object],
) -> str:
    kinds = [
        str(item.get("kind"))
        for item in subgoals
        if isinstance(item, dict) and item.get("kind")
    ]
    if kinds:
        return " + ".join(kinds)
    return str(objective_profile.get("objective_type") or "custom_physics")


def _learned_affordance_creates(
    objective_profile: dict[str, object],
    subgoals: list[object],
) -> list[str]:
    creates = ["deterministic check_objective predicate", "validator-readable objective_profile"]
    for item in subgoals:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "")
        if kind == "move_object_to_region":
            creates.extend(["dynamic movable object", "target sensor region", "pushable lane"])
        elif kind == "agent_reach_region":
            creates.append("reachable target/goal region")
        elif kind == "agent_touch_object":
            creates.append("touchable named object")
        elif kind == "activate_mechanism":
            creates.append("deterministic trigger/mechanism state")
    targets = objective_profile.get("targets")
    if isinstance(targets, list) and targets:
        creates.append("registered targets: " + ", ".join(map(str, targets)))
    return sorted(set(creates))


def _learned_affordance_constraints(validation: ValidationResult) -> list[str]:
    constraints = [
        "Preserve the same objective subgoal structure unless the prompt asks for a different task.",
        "Keep generated geometry validator-friendly and physically stable.",
        "Do not lower the required verification tier for finite direct objectives.",
    ]
    for progress in validation.details.get("subgoal_progress", []) or []:
        if isinstance(progress, dict):
            kind = progress.get("kind")
            start = progress.get("start_distance")
            if kind and start is not None:
                constraints.append(f"For {kind}, solved example start_distance was {start} px.")
    return constraints


def _learned_affordance_compositions(subgoals: list[object]) -> list[str]:
    kinds = {
        str(item.get("kind"))
        for item in subgoals
        if isinstance(item, dict) and item.get("kind")
    }
    compositions = []
    if "move_object_to_region" in kinds:
        compositions.extend(["stable_floor_support", "sensor_region", "push_lane"])
    if "activate_mechanism" in kinds:
        compositions.extend(["gated_path", "sensor_region"])
    if "agent_touch_object" in kinds:
        compositions.extend(["target_cluster", "sensor_region"])
    if "agent_reach_region" in kinds:
        compositions.append("sensor_region")
    return sorted(set(compositions))


def _learned_affordance_validator_checks(
    validation: ValidationResult,
    subgoals: list[object],
) -> list[str]:
    checks = ["check_objective", str(validation.details.get("oracle") or "subgoal_plan")]
    for item in subgoals:
        if isinstance(item, dict) and item.get("kind"):
            checks.append(str(item["kind"]))
    return sorted(set(checks))


def _dict_field(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple | set):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _learned_skill_id(prompt: str, objective_type: str, tier_name: str) -> str:
    slug = "_".join(re.findall(r"[a-z0-9]+", prompt.lower())[:8]) or "world"
    objective_slug = "_".join(re.findall(r"[a-z0-9]+", objective_type.lower())) or "custom"
    tier_slug = "_".join(re.findall(r"[a-z0-9]+", tier_name.lower())) or "verified"
    return f"learned_{objective_slug}_{tier_slug}_{slug}"


def _learned_skill_summary(
    objective_profile: dict[str, object],
    capability_profile: dict[str, object],
    validation: ValidationResult,
) -> str:
    description = str(objective_profile.get("objective_description") or "Accepted world pattern")
    movement = capability_profile.get("movement", "unknown_movement")
    gravity = capability_profile.get("gravity", "unknown_gravity")
    return (
        f"{description} Validated as {validation.tier_name} "
        f"with movement={movement}, gravity={gravity}."
    )


def _learned_trigger_keywords(
    prompt: str,
    objective_profile: dict[str, object],
    capability_profile: dict[str, object],
) -> list[str]:
    tokens = set(re.findall(r"[a-z0-9]+", prompt.lower()))
    for value in (
        objective_profile.get("objective_type"),
        capability_profile.get("movement"),
        capability_profile.get("gravity"),
    ):
        if value:
            tokens.update(re.findall(r"[a-z0-9]+", str(value).lower()))
    for field_name in ("validator_skills", "required_capabilities", "progress_metrics"):
        values = objective_profile.get(field_name, [])
        if isinstance(values, list):
            for value in values:
                tokens.update(re.findall(r"[a-z0-9]+", str(value).lower()))
    return sorted(tokens)[:40]


def _learned_generation_guidance(
    objective_profile: dict[str, object],
    capability_profile: dict[str, object],
    validator_route: dict[str, object],
) -> list[str]:
    guidance = [
        "Reuse this objective/capability profile shape when the prompt semantics match.",
        "Keep check_objective aligned with objective_profile['success_predicate'].",
        "Do not use validator controls outside capability_profile['allowed_controls'].",
    ]
    route = validator_route.get("oracle")
    if route:
        guidance.append(f"Preferred validator oracle: {route}.")
    metrics = objective_profile.get("progress_metrics")
    if isinstance(metrics, list) and metrics:
        guidance.append(f"Expose progress metrics: {', '.join(map(str, metrics))}.")
    return guidance


def _learned_common_failures(validation: ValidationResult) -> list[str]:
    failures = [
        "Do not lower minimum_acceptance_tier for simple known tasks just to pass validation.",
        "Keep objective_profile['targets'] synchronized with self.objective_targets.",
        "Keep capability_profile controls honest; no teleporting or direct object mutation.",
    ]
    route = _dict_field(validation.details.get("validator_route"))
    if route.get("oracle") == "multi_target_touch":
        failures.append("Persist touched target state so check_objective does not require simultaneous contact.")
    if validation.objective_tier >= 5:
        failures.append("Tier 5 patterns must be physically solved by check_objective(), not just progress-verified.")
    return failures


def _code_has_method(code: str, method_name: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return True
    return False


def _write_validation(attempt_dir: Path, validation: ValidationResult) -> Path:
    path = attempt_dir / "validation_result.json"
    path.write_text(json.dumps(validation.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_summary(
    run_dir: Path,
    *,
    success: bool,
    prompt: str,
    generated_env_path: Path | None,
    validation: ValidationResult | None,
    post_mortem: str | None,
    learned_skill_path: Path | None,
    learned_affordance_path: Path | None,
    capability_gap_path: Path | None,
    visual_recipe_path: Path | None,
    world_export_dir: Path | None,
    simulation_brief: dict[str, Any] | None,
    gameplay_profile: dict[str, Any] | None,
    physics_relations: dict[str, Any] | None,
    layout_plan: dict[str, Any] | None,
    semantic_memory: dict[str, Any] | None,
) -> None:
    summary = {
        "success": success,
        "prompt": prompt,
        "generated_env_path": str(generated_env_path) if generated_env_path else None,
        "validation": validation.to_dict() if validation else None,
        "post_mortem": post_mortem,
        "learned_skill_path": str(learned_skill_path) if learned_skill_path else None,
        "learned_affordance_path": str(learned_affordance_path)
        if learned_affordance_path
        else None,
        "capability_gap_path": str(capability_gap_path) if capability_gap_path else None,
        "visual_recipe_path": str(visual_recipe_path) if visual_recipe_path else None,
        "world_export_dir": str(world_export_dir) if world_export_dir else None,
        "simulation_brief": simulation_brief,
        "gameplay_profile": gameplay_profile,
        "physics_relations": physics_relations,
        "layout_plan": layout_plan,
        "semantic_memory": semantic_memory,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _build_post_mortem(
    prompt: str,
    attempts: list[AttemptRecord],
    last_validation: ValidationResult | None,
    last_error: str | None,
) -> str:
    blocker_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    axis_counts: dict[str, int] = {}
    reasons: list[str] = []
    for attempt in attempts:
        category = _failure_category(attempt.result)
        axis = _diagnostic_axis_for_category(category)
        category_counts[category] = category_counts.get(category, 0) + 1
        axis_counts[axis] = axis_counts.get(axis, 0) + 1
        reasons.append(
            f"seed {attempt.seed_index} repair {attempt.repair_index} "
            f"[{axis}/{category}] "
            f"tier {attempt.result.achieved_tier}/{attempt.result.operational_acceptance_tier} "
            f"(objective {attempt.result.objective_tier}): "
            f"{attempt.result.reason}"
        )
        if attempt.result.blocking_object:
            blocker_counts[attempt.result.blocking_object] = (
                blocker_counts.get(attempt.result.blocking_object, 0) + 1
            )

    dominant = "none"
    if blocker_counts:
        dominant = max(blocker_counts, key=blocker_counts.get)

    final_reason = last_validation.reason if last_validation else (last_error or "unknown failure")
    dominant_category = "unknown_failure"
    if category_counts:
        dominant_category = max(category_counts, key=category_counts.get)
    dominant_axis = _diagnostic_axis_for_category(dominant_category)
    if axis_counts:
        dominant_axis = max(axis_counts, key=axis_counts.get)

    if dominant_category in {
        "semantic_contract_error",
        "code_contract_error",
        "object_model_error",
        "objective_logic_error",
        "api_keyword_error",
        "semantic_profile_error",
        "capability_mismatch",
    }:
        bottleneck_label = "Dominant non-physics bottleneck"
        dominant = dominant_category
    else:
        bottleneck_label = "Dominant physical bottleneck"
    repair_category = _failure_category(last_validation) if last_validation is not None else dominant_category
    repair_feedback = (
        json.dumps(last_validation.to_dict(), sort_keys=True)
        if last_validation is not None
        else final_reason
    )
    recommended_action = _repair_rule_for_category(repair_category, repair_feedback)
    final_tier = "unavailable"
    if last_validation is not None:
        final_tier = (
            f"{last_validation.achieved_tier}/{last_validation.operational_acceptance_tier} "
            f"({last_validation.tier_name}, accepted={last_validation.accepted}, "
            f"objective_tier={last_validation.objective_tier})"
        )
    category_breakdown = _indent_block(_format_count_breakdown(category_counts), "        ")
    axis_breakdown = _indent_block(_format_count_breakdown(axis_counts), "        ")
    attempt_trace = _indent_block("\n".join(reasons), "        ")

    return textwrap.dedent(
        f"""
        Post-Mortem Diagnostic
        Prompt: {prompt}
        Attempts: {len(attempts)}
        Final reason: {final_reason}
        Final verification tier: {final_tier}
        Dominant diagnostic axis: {dominant_axis}
        Dominant failure category: {dominant_category}
        {bottleneck_label}: {dominant}

        Diagnostic axis breakdown:
{axis_breakdown}

        Failure category breakdown:
{category_breakdown}

        Attempt trace:
{attempt_trace}

        Recommended healer action:
        {recommended_action}
        """
    ).strip()


def _format_count_breakdown(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    parts = [
        f"- {name}: {count}"
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return "\n".join(parts)


def _indent_block(text: str, prefix: str) -> str:
    return "\n".join(f"{prefix}{line}" if line else line for line in text.splitlines())


def _blocked_status(validation: ValidationResult) -> str:
    axis = validation.details.get("diagnostic_axis") or _diagnostic_axis_for_category(
        _failure_category(validation)
    )
    if validation.blocking_object:
        return f"{axis.upper()} - BLOCKED BY {validation.blocking_object}"
    tier = (
        f"TIER {validation.achieved_tier}/{validation.operational_acceptance_tier} "
        f"{validation.tier_name.upper()}"
    )
    return f"{axis.upper()} - {tier} - {validation.reason.upper()}"


def _success_status(validation: ValidationResult) -> str:
    if validation.kinetic_solved:
        return (
            "SUCCESS - VERIFIED SOLVABLE "
            f"(TIER {validation.achieved_tier}: {validation.tier_name.upper()})"
        )
    return (
        "SUCCESS - ACCEPTED "
        f"(TIER {validation.achieved_tier}: {validation.tier_name.upper()}, "
        f"OPERATIONAL {validation.operational_acceptance_tier}, "
        f"OBJECTIVE {validation.objective_tier})"
    )


def _print_status(seed_index: int, repair_index: int, max_repairs: int, status: str) -> None:
    print(
        f"[SEED {seed_index}] [REPAIR {repair_index}/{max_repairs}] -> Status: {status}",
        flush=True,
    )


def _enhanced_request(
    prompt: str,
    directive: str,
    seed_index: int,
    repair_index: int,
    *,
    simulation_brief: dict[str, Any] | None = None,
    gameplay_profile: dict[str, Any] | None = None,
    physics_relations: dict[str, Any] | None = None,
    layout_plan: dict[str, Any] | None = None,
    semantic_memory: dict[str, Any] | None = None,
) -> str:
    brief_block = ""
    if simulation_brief:
        brief_block = (
            "\nSIMULATION BRIEF (physical interpretation contract; implement this meaning):\n"
            f"{json.dumps(simulation_brief, indent=2, sort_keys=True)}\n"
            "Simulation brief rules: this is what the prompt means physically. "
            "Do not contradict its gravity, agent form, objective type, hazard interpretation, "
            "semantic_must_happen, or validation expectations. If the brief says shots are "
            "projectiles, do not use sports/strike-shot logic. If it says zero gravity, do "
            "not add normal gravity just to make hazards move.\n"
        )
    gameplay_block = ""
    if gameplay_profile:
        gameplay_block = (
            "\nGAMEPLAY ARCHITECT PROFILE (implement this as self.gameplay_profile and export it):\n"
            f"{json.dumps(gameplay_profile, indent=2, sort_keys=True)}\n"
            "Gameplay profile rules: preserve the user's task, but account for game-feel expectations. "
            "Obey gameplay_profile.world_context when choosing gravity, perspective, movement model, support geometry, and route placement. "
            "If it asks for recurring/staggered hazards, implement timers/phase offsets/reset behavior. "
            "If it asks for heavy-but-movable pushing, tune mass/friction/agent_strength so motion is visibly significant.\n"
        )
    relation_block = ""
    if physics_relations:
        relation_block = (
            "\nPHYSICS RELATION GRAPH (mechanic source of truth; implement/export this as self.physics_relations):\n"
            f"{json.dumps(physics_relations, indent=2, sort_keys=True)}\n"
            "Relation graph rules: this graph decomposes the prompt into low-level physical relations, not old task templates. "
            "Use it to build objective_profile['subgoals'], check_objective telemetry, and repairable layout state. "
            "If a helper primitive conflicts with a relation, compose from lower-level BaseEnv primitives instead of forcing the helper. "
            "Preserve relation names and registered object/region names as much as possible so the validator can inspect failed relations.\n"
        )
    layout_block = ""
    if layout_plan:
        layout_block = (
            "\nROUTE-AWARE LAYOUT PLAN (spatial skeleton contract; implement/export this as self.layout_plan):\n"
            f"{json.dumps(layout_plan, indent=2, sort_keys=True)}\n"
            "Layout plan rules: build the guaranteed route/mechanism skeleton first, then add obstacles, hazards, enemies, and props around it. "
            "Start, goal, triggers, boundaries, critical_path_points/cells, hazard lanes, and protected_zones are spatial contracts, not decoration. "
            "No solid static blocker may overlap protected zones, goal sensors, trigger sensors, or the critical path corridor. "
            "For maze/escape prompts, carve the path before rasterizing walls. For side-view/top-right prompts, create continuous support under the planned route unless world_context explicitly says zero_g. "
            "For push, ballistic, and support-exit prompts, keep the lane/arc corridor short, aligned, and free of decorative blockers.\n"
        )
    semantic_memory_block = ""
    if semantic_memory:
        semantic_memory_block = (
            "\nSEMANTIC MEMORY RETRIEVAL (conceptual priors from prior runs; do not template-copy):\n"
            f"{json.dumps(semantic_memory, indent=2, sort_keys=True)}\n"
            "Semantic memory rules: use these memories only as reusable physics lessons. "
            "Do not replace the current prompt with a prior prompt. Do not copy object names, "
            "theme, layout, gravity, or objective unless the current prompt explicitly asks for the same thing. "
            "Borrow the matched relation constraints, stable parameter ranges, validator checks, and known failure repairs. "
            "If a capability_gap memory matches, proactively avoid that failure before returning code.\n"
        )
    return (
        f"{prompt}\n"
        f"Seed {seed_index} diversity directive: {directive}\n"
        f"Repair pass {repair_index}: preserve requested theme while maximizing solvability."
        f"{brief_block}"
        f"{gameplay_block}"
        f"{relation_block}"
        f"{layout_block}"
        f"{semantic_memory_block}"
    )


def _class_name(prompt: str, seed_index: int, repair_index: int) -> str:
    words = re.findall(r"[A-Za-z0-9]+", prompt)
    stem = "".join(word.capitalize() for word in words[:5]) or "GeneratedWorld"
    if stem[0].isdigit():
        stem = f"World{stem}"
    return f"{stem}Seed{seed_index}Repair{repair_index}Env"


def _create_run_dir(prompt: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = "_".join(re.findall(r"[a-z0-9]+", prompt.lower())[:8]) or "world"
    return LOGS_DIR / f"{timestamp}_{slug}"


def _normalize_prompt(prompt: str) -> str:
    clean = " ".join(prompt.strip().split())
    if not clean:
        raise ValueError("prompt cannot be empty")
    return clean


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Harness Alpha recursive healer")
    parser.add_argument("prompt", help="natural language environment prompt")
    parser.add_argument(
        "--backend",
        choices=("openai", "local"),
        default="openai",
        help="architect backend to use; openai generates unique prompt-specific code",
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_ARCHITECT_MODEL", DEFAULT_OPENAI_MODEL))
    parser.add_argument(
        "--reasoning-effort",
        default=os.getenv("OPENAI_ARCHITECT_REASONING_EFFORT", DEFAULT_REASONING_EFFORT),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=int(os.getenv("OPENAI_ARCHITECT_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)),
    )
    parser.add_argument("--max-seeds", type=int, default=MAX_SEEDS)
    parser.add_argument("--max-repairs", type=int, default=MAX_REPAIRS)
    parser.add_argument(
        "--execution-mode",
        choices=("normal", "fast"),
        default=os.getenv("HARNESS_EXECUTION_MODE", "normal"),
        help="normal runs the classic sequential Reflexion loop; fast races first-attempt seeds in parallel",
    )
    parser.add_argument("--grid-size", type=float, default=None)
    parser.add_argument("--agent-radius", type=float, default=None)
    parser.add_argument("--json", action="store_true", help="print final result as JSON")
    return parser


async def _async_main() -> int:
    args = _build_parser().parse_args()
    if args.backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Set it to use the real OpenAI architect, "
            "or pass --backend local for the deterministic smoke backend.",
            file=sys.stderr,
        )
        return 2

    backend: ArchitectBackend
    if args.backend == "local":
        backend = LocalTemplateArchitect()
    else:
        backend = OpenAIArchitect(
            OpenAIArchitectConfig(
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
            )
        )

    validator_defaults = ValidatorConfig(simulation_steps=4)
    validator_config = ValidatorConfig(
        grid_size=args.grid_size or validator_defaults.grid_size,
        agent_radius=args.agent_radius or validator_defaults.agent_radius,
        simulation_steps=validator_defaults.simulation_steps,
        substeps=validator_defaults.substeps,
        include_dynamic_blockers=validator_defaults.include_dynamic_blockers,
        max_cells=validator_defaults.max_cells,
        kinetic_validation=validator_defaults.kinetic_validation,
        kinetic_steps=validator_defaults.kinetic_steps,
        kinetic_displacement_threshold=validator_defaults.kinetic_displacement_threshold,
        kinetic_substeps=validator_defaults.kinetic_substeps,
    )
    result = await run_harness(
        args.prompt,
        backend=backend,
        config=HarnessConfig(
            max_seeds=args.max_seeds,
            max_repairs=args.max_repairs,
            validator=validator_config,
            execution_mode=args.execution_mode,
        ),
    )

    if args.json:
        print(
            json.dumps(
                {
                    "success": result.success,
                    "run_dir": str(result.run_dir),
                    "generated_env_path": str(result.generated_env_path)
                    if result.generated_env_path
                    else None,
                    "visual_recipe_path": str(result.visual_recipe_path)
                    if result.visual_recipe_path
                    else None,
                    "world_export_dir": str(result.world_export_dir)
                    if result.world_export_dir
                    else None,
                    "simulation_brief": result.simulation_brief,
                    "gameplay_profile": result.gameplay_profile,
                    "layout_plan": result.layout_plan,
                    "validation": result.validation.to_dict() if result.validation else None,
                    "post_mortem": result.post_mortem,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif result.success:
        print(f"VERIFIED_ENV: {result.generated_env_path}")
        if result.visual_recipe_path:
            print(f"VISUAL_RECIPE: {result.visual_recipe_path}")
        if result.world_export_dir:
            print(f"WORLD_EXPORT: {result.world_export_dir}")
        if result.validation:
            print(
                "VERIFICATION: "
                f"TIER {result.validation.achieved_tier} "
                f"({result.validation.tier_name}), "
                f"OPERATIONAL {result.validation.operational_acceptance_tier}, "
                f"OBJECTIVE {result.validation.objective_tier}, "
                f"ACCEPTED={result.validation.accepted}"
            )
        print(f"LOG_DIR: {result.run_dir}")
    else:
        print(result.post_mortem)
        print(f"FAILED_ENV: {result.generated_env_path}")
        print(f"LOG_DIR: {result.run_dir}")

    return 0 if result.success else 1


def main() -> None:
    raise SystemExit(asyncio.run(_async_main()))


if __name__ == "__main__":
    main()
