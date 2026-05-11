"""Field and force-effect probes for abstract physical interactions."""

from __future__ import annotations

from typing import Any

from .contract import ProbeResult


def field_effect_probe(
    *,
    object_name: str,
    field_name: str | None = None,
    summary: dict[str, Any],
    min_displacement: float = 5.0,
    min_progress_delta: float = 5.0,
) -> ProbeResult:
    """Check whether a declared field or force produced measurable state change."""

    displacement = _float_or_none(summary.get("displacement"))
    progress_delta = _float_or_none(summary.get("progress_delta"))
    velocity_delta = _float_or_none(summary.get("velocity_delta"))
    displacement_passed = displacement is not None and displacement >= min_displacement
    progress_passed = progress_delta is not None and progress_delta >= min_progress_delta
    velocity_passed = velocity_delta is not None and abs(velocity_delta) > 1.0
    passed = displacement_passed or progress_passed or velocity_passed
    objects = (object_name,) if not field_name else (object_name, field_name)
    return ProbeResult(
        name="field_effect",
        passed=passed,
        tier_evidence=4 if passed else 2,
        objects=objects,
        metrics={
            "displacement": displacement,
            "progress_delta": progress_delta,
            "velocity_delta": velocity_delta,
            "recommended_min_displacement": float(min_displacement),
            "recommended_min_progress_delta": float(min_progress_delta),
        },
        diagnosis="field_effect_observed" if passed else "field_effect_not_observed",
        repair=(
            "Field/force changed object motion."
            if passed
            else "Increase field strength, place the affected object inside the field region, or expose a progress metric tied to the field effect."
        ),
        severity="info" if passed else "warning",
    )


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
