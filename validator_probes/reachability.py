"""Reachability evidence probes built from path diagnostics."""

from __future__ import annotations

from typing import Any

from .contract import ProbeResult


def path_reachability_probe(
    *,
    start_name: str,
    target_name: str,
    path_diagnostics: dict[str, Any] | None,
) -> ProbeResult:
    """Convert deterministic path diagnostics into probe evidence."""

    diagnostics = path_diagnostics or {}
    passed = bool(diagnostics.get("path_found"))
    blocker = diagnostics.get("blocking_object")
    return ProbeResult(
        name="path_reachability",
        passed=passed,
        tier_evidence=3 if passed else 2,
        objects=(start_name, target_name),
        metrics=dict(diagnostics),
        diagnosis="path_found" if passed else str(diagnostics.get("reason") or "path_not_found"),
        repair=(
            "Path is reachable."
            if passed
            else f"Clear or move blocker {blocker!r}, or relocate the target onto the reachable lane."
        ),
        severity="info" if passed else "error",
    )
