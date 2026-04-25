"""Recursive Harness Alpha brain: architect, validate, repair, and pivot."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import textwrap
from typing import Protocol

from architect import extract_python_code, render_prompt, save_generated_code
from validator import ValidationResult, ValidatorConfig, validate_generated_env


LOGS_DIR = Path("logs")
GENERATED_ENVS_DIR = Path("generated_envs")
MAX_SEEDS = 3
MAX_REPAIRS = 3

DIVERSITY_DIRECTIVES = {
    1: "Baseline layout: open space with central obstacles and a clear primary route.",
    2: "Inverse layout: peripheral obstacles with a central goal region and a reversed route.",
    3: "High-entropy layout: complex maze with dynamic or moving elements, but still solvable.",
}


@dataclass(frozen=True)
class HarnessConfig:
    """Control parameters for the recursive Reflexion loop."""

    max_seeds: int = MAX_SEEDS
    max_repairs: int = MAX_REPAIRS
    validator: ValidatorConfig = field(default_factory=lambda: ValidatorConfig(simulation_steps=4))


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
class HarnessResult:
    """Final harness outcome."""

    success: bool
    prompt: str
    run_dir: Path
    generated_env_path: Path | None
    validation: ValidationResult | None
    attempts: tuple[AttemptRecord, ...]
    post_mortem: str | None = None


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


async def run_harness(
    prompt: str,
    *,
    backend: ArchitectBackend | None = None,
    config: HarnessConfig | None = None,
    run_dir: Path | None = None,
) -> HarnessResult:
    """Run the 3x3 Global Pivot + Local Repair Reflexion loop."""

    active_backend = backend or LocalTemplateArchitect()
    active_config = config or HarnessConfig()
    clean_prompt = _normalize_prompt(prompt)
    active_run_dir = run_dir or _create_run_dir(clean_prompt)
    active_run_dir.mkdir(parents=True, exist_ok=True)

    attempts: list[AttemptRecord] = []
    last_code_path: Path | None = None
    last_validation: ValidationResult | None = None
    last_error: str | None = None

    for seed_index in range(1, active_config.max_seeds + 1):
        directive = DIVERSITY_DIRECTIVES.get(seed_index, DIVERSITY_DIRECTIVES[MAX_SEEDS])
        for repair_index in range(1, active_config.max_repairs + 1):
            class_name = _class_name(clean_prompt, seed_index, repair_index)
            enhanced_request = _enhanced_request(clean_prompt, directive, seed_index, repair_index)
            architect_prompt = render_prompt(enhanced_request, class_name=class_name)
            correction_prompt = _build_correction_prompt(
                original_prompt=clean_prompt,
                seed_index=seed_index,
                repair_index=repair_index,
                diversity_directive=directive,
                architect_prompt=architect_prompt,
                previous_result=last_validation,
                previous_error=last_error,
                max_repairs=active_config.max_repairs,
            )
            context = GenerationContext(
                original_prompt=clean_prompt,
                enhanced_request=enhanced_request,
                class_name=class_name,
                seed_index=seed_index,
                repair_index=repair_index,
                diversity_directive=directive,
                correction_prompt=correction_prompt,
                previous_result=last_validation,
                previous_error=last_error,
            )

            attempt_dir = active_run_dir / f"seed_{seed_index:02d}_repair_{repair_index:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            _write_text(attempt_dir / "architect_prompt.txt", architect_prompt)
            _write_text(attempt_dir / "correction_prompt.txt", correction_prompt)
            attempt_code_path: Path | None = None

            _print_status(seed_index, repair_index, active_config.max_repairs, "REGENERATING...")
            try:
                llm_response = await active_backend.generate(context)
                _write_text(attempt_dir / "llm_response.txt", llm_response)
                generated_code = extract_python_code(llm_response)
                attempt_code_path = attempt_dir / "generated_code.py"
                _write_text(attempt_code_path, generated_code)
                last_code_path = attempt_code_path
                generated_env_path = await asyncio.to_thread(
                    save_generated_code,
                    enhanced_request,
                    llm_response,
                    class_name=class_name,
                    output_dir=GENERATED_ENVS_DIR,
                )
            except Exception as exc:
                last_error = f"Generation or contract verification failed: {exc}"
                failed_result = _failure_result(last_error, class_name)
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
                    active_config.max_repairs,
                    f"CODE ERROR - {exc}",
                )
                continue

            validation = await asyncio.to_thread(
                validate_generated_env,
                generated_env_path,
                class_name=class_name,
                config=active_config.validator,
            )
            last_validation = validation
            last_error = validation.reason
            validation_path = _write_validation(attempt_dir, validation)
            attempts.append(
                AttemptRecord(
                    seed_index=seed_index,
                    repair_index=repair_index,
                    attempt_dir=attempt_dir,
                    class_name=class_name,
                    generated_code_path=attempt_code_path,
                    validation_path=validation_path,
                    result=validation,
                )
            )

            if validation.solvable:
                _print_status(
                    seed_index,
                    repair_index,
                    active_config.max_repairs,
                    "SUCCESS - VERIFIED SOLVABLE",
                )
                _write_summary(
                    active_run_dir,
                    success=True,
                    prompt=clean_prompt,
                    generated_env_path=generated_env_path,
                    validation=validation,
                    post_mortem=None,
                )
                return HarnessResult(
                    success=True,
                    prompt=clean_prompt,
                    run_dir=active_run_dir,
                    generated_env_path=generated_env_path,
                    validation=validation,
                    attempts=tuple(attempts),
                )

            status = _blocked_status(validation)
            if repair_index < active_config.max_repairs:
                _print_status(seed_index, repair_index, active_config.max_repairs, status)
            elif seed_index < active_config.max_seeds:
                _print_status(
                    seed_index,
                    repair_index,
                    active_config.max_repairs,
                    f"{status}; PIVOTING TO SEED {seed_index + 1}...",
                )
            else:
                _print_status(seed_index, repair_index, active_config.max_repairs, status)

    post_mortem = _build_post_mortem(clean_prompt, attempts, last_validation, last_error)
    failed_path = active_run_dir / "env_failed_final.py.fail"
    if last_code_path and last_code_path.exists():
        shutil.copyfile(last_code_path, failed_path)
    else:
        _write_text(failed_path, "# No valid generated code was produced.\n")
    _write_text(active_run_dir / "post_mortem.txt", post_mortem)
    _write_summary(
        active_run_dir,
        success=False,
        prompt=clean_prompt,
        generated_env_path=failed_path,
        validation=last_validation,
        post_mortem=post_mortem,
    )
    return HarnessResult(
        success=False,
        prompt=clean_prompt,
        run_dir=active_run_dir,
        generated_env_path=failed_path,
        validation=last_validation,
        attempts=tuple(attempts),
        post_mortem=post_mortem,
    )


def _build_correction_prompt(
    *,
    original_prompt: str,
    seed_index: int,
    repair_index: int,
    diversity_directive: str,
    architect_prompt: str,
    previous_result: ValidationResult | None,
    previous_error: str | None,
    max_repairs: int = MAX_REPAIRS,
) -> str:
    if previous_result is None and previous_error is None:
        feedback = "No previous failure. Generate the initial candidate for this seed."
    elif previous_result is not None:
        feedback = json.dumps(previous_result.to_dict(), indent=2, sort_keys=True)
    else:
        feedback = previous_error or "Unknown failure."

    return textwrap.dedent(
        f"""
        Reflexion correction request for Harness Alpha.

        Original user prompt:
        {original_prompt}

        Current strategy:
        - Seed: {seed_index}
        - Repair: {repair_index}/{max_repairs}
        - Diversity directive: {diversity_directive}

        Previous failure telemetry:
        {feedback}

        Repair rule:
        If the previous attempt was blocked, preserve the user's requested theme
        while changing geometry so the agent has a mathematically valid route to
        the goal. Widen narrow passages, move or shrink the named blocker, and
        update solvability_check so it matches the new route.

        Full architect contract prompt:
        {architect_prompt}
        """
    ).strip()


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
            position={position},
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
            def __init__(self, *, seed: int = 0):
                super().__init__(config=EnvConfig(width=920, height=560), seed=seed)

            def build_world(self) -> None:
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
                    position={start},
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
) -> None:
    summary = {
        "success": success,
        "prompt": prompt,
        "generated_env_path": str(generated_env_path) if generated_env_path else None,
        "validation": validation.to_dict() if validation else None,
        "post_mortem": post_mortem,
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
    reasons: list[str] = []
    for attempt in attempts:
        reasons.append(
            f"seed {attempt.seed_index} repair {attempt.repair_index}: {attempt.result.reason}"
        )
        if attempt.result.blocking_object:
            blocker_counts[attempt.result.blocking_object] = (
                blocker_counts.get(attempt.result.blocking_object, 0) + 1
            )

    dominant = "none"
    if blocker_counts:
        dominant = max(blocker_counts, key=blocker_counts.get)

    final_reason = last_validation.reason if last_validation else (last_error or "unknown failure")
    return textwrap.dedent(
        f"""
        Post-Mortem Diagnostic
        Prompt: {prompt}
        Attempts: {len(attempts)}
        Final reason: {final_reason}
        Dominant physical bottleneck: {dominant}

        Attempt trace:
        {chr(10).join(reasons)}

        Recommended healer action:
        Widen or relocate the dominant blocker, verify start/goal are not inside
        inflated obstacle geometry, and regenerate with a larger grid passage
        than the agent radius.
        """
    ).strip()


def _blocked_status(validation: ValidationResult) -> str:
    if validation.blocking_object:
        return f"BLOCKED BY {validation.blocking_object}"
    return validation.reason.upper()


def _print_status(seed_index: int, repair_index: int, max_repairs: int, status: str) -> None:
    print(
        f"[SEED {seed_index}] [REPAIR {repair_index}/{max_repairs}] -> Status: {status}",
        flush=True,
    )


def _enhanced_request(prompt: str, directive: str, seed_index: int, repair_index: int) -> str:
    return (
        f"{prompt}\n"
        f"Seed {seed_index} diversity directive: {directive}\n"
        f"Repair pass {repair_index}: preserve requested theme while maximizing solvability."
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
    path.write_text(content, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Harness Alpha recursive healer")
    parser.add_argument("prompt", help="natural language environment prompt")
    parser.add_argument("--max-seeds", type=int, default=MAX_SEEDS)
    parser.add_argument("--max-repairs", type=int, default=MAX_REPAIRS)
    parser.add_argument("--grid-size", type=float, default=None)
    parser.add_argument("--agent-radius", type=float, default=None)
    parser.add_argument("--json", action="store_true", help="print final result as JSON")
    return parser


async def _async_main() -> int:
    args = _build_parser().parse_args()
    validator_defaults = ValidatorConfig(simulation_steps=4)
    validator_config = ValidatorConfig(
        grid_size=args.grid_size or validator_defaults.grid_size,
        agent_radius=args.agent_radius or validator_defaults.agent_radius,
        simulation_steps=validator_defaults.simulation_steps,
        substeps=validator_defaults.substeps,
        include_dynamic_blockers=validator_defaults.include_dynamic_blockers,
        max_cells=validator_defaults.max_cells,
    )
    result = await run_harness(
        args.prompt,
        config=HarnessConfig(
            max_seeds=args.max_seeds,
            max_repairs=args.max_repairs,
            validator=validator_config,
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
                    "validation": result.validation.to_dict() if result.validation else None,
                    "post_mortem": result.post_mortem,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif result.success:
        print(f"VERIFIED_ENV: {result.generated_env_path}")
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
