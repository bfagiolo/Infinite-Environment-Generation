"""Benchmark runner for Harness Alpha prompt suites.

This script is intentionally lightweight: it runs the real harness on a prompt
suite, records the verification tiers, and writes a submission-friendly JSON
and Markdown report.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from architect import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_REASONING_EFFORT,
    OpenAIArchitect,
    OpenAIArchitectConfig,
)
from harness import HarnessConfig, LocalTemplateArchitect, run_harness
from validator import ValidatorConfig


BENCHMARKS_DIR = Path("benchmarks")


SCOUT_SUITE = [
    {
        "id": "maze_navigation",
        "category": "navigation",
        "prompt": "A tiny maze where the agent must reach a glowing exit.",
    },
    {
        "id": "zero_g_collection",
        "category": "zero_g_collection",
        "prompt": "A zero-gravity salvage field where the agent must touch four drifting crystals.",
    },
    {
        "id": "push_gate_mechanism",
        "category": "mechanism_push",
        "prompt": "A room with a sliding gate where the agent must push a blue box onto a pressure plate to open the path.",
    },
    {
        "id": "survival_hazards",
        "category": "survival",
        "prompt": "A survival arena where falling rocks rain down and the agent must survive for ten seconds.",
    },
    {
        "id": "seesaw_launch",
        "category": "lever_launch",
        "prompt": "A weighted seesaw puzzle where a heavy ball must launch the agent to a high goal.",
    },
    {
        "id": "magnetic_force_zone",
        "category": "field_force",
        "prompt": "A magnetic field puzzle where the agent must push a charged ball through a force zone into a target.",
    },
]


@dataclass(frozen=True)
class BenchmarkRow:
    """One prompt result in a benchmark run."""

    id: str
    category: str
    prompt: str
    success: bool
    accepted: bool
    achieved_tier: int | None
    tier_name: str | None
    operational_acceptance_tier: int | None
    objective_tier: int | None
    verification_gap: int | None
    attempts: int
    duration_seconds: float
    failure_category: str | None
    diagnostic_axis: str | None
    reason: str | None
    generated_env_path: str | None
    run_dir: str


def _load_prompt_suite(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return [dict(item) for item in SCOUT_SUITE]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("prompt suite JSON must be a list of strings or objects")
    suite: list[dict[str, str]] = []
    for index, item in enumerate(payload, start=1):
        if isinstance(item, str):
            prompt = item
            suite.append(
                {
                    "id": f"prompt_{index:02d}",
                    "category": "custom",
                    "prompt": prompt,
                }
            )
            continue
        if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
            raise ValueError("each prompt suite item must be a string or object with a prompt")
        prompt = str(item["prompt"])
        suite.append(
            {
                "id": str(item.get("id") or _slug(prompt, fallback=f"prompt_{index:02d}")),
                "category": str(item.get("category") or "custom"),
                "prompt": prompt,
            }
        )
    return suite


async def _run_benchmark(args: argparse.Namespace) -> int:
    if args.backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Set it to use the real OpenAI architect, "
            "or pass --backend local for the deterministic smoke backend."
        )
        return 2

    suite = _load_prompt_suite(Path(args.prompt_file) if args.prompt_file else None)
    if args.max_prompts is not None:
        suite = suite[: max(0, int(args.max_prompts))]
    if not suite:
        print("No prompts to benchmark.")
        return 2

    backend = (
        LocalTemplateArchitect()
        if args.backend == "local"
        else OpenAIArchitect(
            OpenAIArchitectConfig(
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
            )
        )
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.name or f"{timestamp}_{args.suite}"
    output_dir = Path(args.output_dir) / _slug(run_name, fallback="benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

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
    config = HarnessConfig(
        max_seeds=args.max_seeds,
        max_repairs=args.max_repairs,
        validator=validator_config,
    )

    rows: list[BenchmarkRow] = []
    print(
        f"Benchmark: {run_name} | prompts={len(suite)} | "
        f"budget={args.max_seeds}x{args.max_repairs} | backend={args.backend}"
    )
    print(f"Output: {output_dir}")

    for index, item in enumerate(suite, start=1):
        prompt = item["prompt"]
        prompt_id = item["id"]
        prompt_dir = output_dir / f"{index:02d}_{_slug(prompt_id, fallback='prompt')}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        print("")
        print(f"[{index}/{len(suite)}] {prompt_id}: {prompt}")
        started = time.perf_counter()
        try:
            result = await run_harness(
                prompt,
                backend=backend,
                config=config,
                run_dir=prompt_dir / "harness",
            )
            duration = time.perf_counter() - started
            validation = result.validation
            details = validation.details if validation else {}
            row = BenchmarkRow(
                id=prompt_id,
                category=item["category"],
                prompt=prompt,
                success=result.success,
                accepted=bool(validation.accepted) if validation else False,
                achieved_tier=validation.achieved_tier if validation else None,
                tier_name=validation.tier_name if validation else None,
                operational_acceptance_tier=validation.operational_acceptance_tier
                if validation
                else None,
                objective_tier=validation.objective_tier if validation else None,
                verification_gap=validation.verification_gap if validation else None,
                attempts=len(result.attempts),
                duration_seconds=round(duration, 3),
                failure_category=_string_or_none(details.get("failure_category")),
                diagnostic_axis=_string_or_none(details.get("diagnostic_axis")),
                reason=validation.reason if validation else result.post_mortem,
                generated_env_path=str(result.generated_env_path)
                if result.generated_env_path
                else None,
                run_dir=str(result.run_dir),
            )
        except Exception as exc:  # noqa: BLE001 - benchmark should record failures.
            duration = time.perf_counter() - started
            row = BenchmarkRow(
                id=prompt_id,
                category=item["category"],
                prompt=prompt,
                success=False,
                accepted=False,
                achieved_tier=None,
                tier_name=None,
                operational_acceptance_tier=None,
                objective_tier=None,
                verification_gap=None,
                attempts=0,
                duration_seconds=round(duration, 3),
                failure_category="benchmark_exception",
                diagnostic_axis="runner",
                reason=f"{type(exc).__name__}: {exc}",
                generated_env_path=None,
                run_dir=str(prompt_dir / "harness"),
            )
        rows.append(row)
        _write_json(prompt_dir / "result.json", asdict(row))
        print(_format_console_row(row))

    report = _build_report(
        rows,
        suite_name=args.suite,
        backend=args.backend,
        model=args.model if args.backend == "openai" else "local",
        max_seeds=args.max_seeds,
        max_repairs=args.max_repairs,
    )
    payload = {
        "suite": args.suite,
        "backend": args.backend,
        "model": args.model if args.backend == "openai" else "local",
        "max_seeds": args.max_seeds,
        "max_repairs": args.max_repairs,
        "output_dir": str(output_dir),
        "summary": _summary(rows),
        "results": [asdict(row) for row in rows],
    }
    _write_json(output_dir / "results.json", payload)
    (output_dir / "results.md").write_text(report, encoding="utf-8")

    print("")
    print("===== BENCHMARK SUMMARY =====")
    print(_summary_line(rows))
    print(f"JSON: {output_dir / 'results.json'}")
    print(f"Markdown: {output_dir / 'results.md'}")
    return 0 if all(row.accepted for row in rows) else 1


def _summary(rows: list[BenchmarkRow]) -> dict[str, Any]:
    total = len(rows)
    accepted = sum(1 for row in rows if row.accepted)
    tier5 = sum(1 for row in rows if row.achieved_tier == 5)
    tier4_or_better = sum(1 for row in rows if (row.achieved_tier or 0) >= 4)
    crashes = sum(1 for row in rows if row.failure_category == "benchmark_exception")
    return {
        "total": total,
        "accepted": accepted,
        "tier5": tier5,
        "tier4_or_better": tier4_or_better,
        "crashes": crashes,
        "acceptance_rate": round(accepted / total, 3) if total else 0.0,
        "tier5_rate": round(tier5 / total, 3) if total else 0.0,
    }


def _summary_line(rows: list[BenchmarkRow]) -> str:
    summary = _summary(rows)
    return (
        f"accepted={summary['accepted']}/{summary['total']} "
        f"tier5={summary['tier5']}/{summary['total']} "
        f"tier4+={summary['tier4_or_better']}/{summary['total']} "
        f"crashes={summary['crashes']}"
    )


def _build_report(
    rows: list[BenchmarkRow],
    *,
    suite_name: str,
    backend: str,
    model: str,
    max_seeds: int,
    max_repairs: int,
) -> str:
    lines = [
        f"# Harness Alpha Benchmark: {suite_name}",
        "",
        f"- Backend: `{backend}`",
        f"- Model: `{model}`",
        f"- Budget: `{max_seeds} seeds x {max_repairs} repairs`",
        f"- Summary: {_summary_line(rows)}",
        "",
        "| ID | Category | Accepted | Tier | Attempts | Failure Category | Reason |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        tier = "n/a" if row.achieved_tier is None else f"{row.achieved_tier} ({row.tier_name})"
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row.id),
                    _md(row.category),
                    "yes" if row.accepted else "no",
                    _md(tier),
                    str(row.attempts),
                    _md(row.failure_category or ""),
                    _md(_shorten(row.reason or "", 140)),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Prompts", ""])
    for row in rows:
        lines.append(f"- `{row.id}`: {row.prompt}")
    lines.append("")
    return "\n".join(lines)


def _format_console_row(row: BenchmarkRow) -> str:
    tier = "n/a" if row.achieved_tier is None else f"T{row.achieved_tier}"
    accepted = "ACCEPTED" if row.accepted else "FAILED"
    reason = _shorten(row.reason or "", 110)
    return (
        f"  -> {accepted} {tier} attempts={row.attempts} "
        f"axis={row.diagnostic_axis or 'n/a'} reason={reason}"
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _shorten(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _md(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _slug(value: str, *, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug[:80] or fallback


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Harness Alpha prompt benchmarks")
    parser.add_argument("--suite", default="scout", help="suite label for reporting")
    parser.add_argument("--prompt-file", default=None, help="JSON list of prompts or prompt objects")
    parser.add_argument("--max-prompts", type=int, default=None, help="limit prompts for quick smoke tests")
    parser.add_argument("--output-dir", default=str(BENCHMARKS_DIR))
    parser.add_argument("--name", default=None, help="optional benchmark run folder name")
    parser.add_argument("--backend", choices=("openai", "local"), default="openai")
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
    parser.add_argument("--max-seeds", type=int, default=2)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--grid-size", type=float, default=None)
    parser.add_argument("--agent-radius", type=float, default=None)
    return parser


def main() -> None:
    raise SystemExit(asyncio.run(_run_benchmark(_build_parser().parse_args())))


if __name__ == "__main__":
    main()
