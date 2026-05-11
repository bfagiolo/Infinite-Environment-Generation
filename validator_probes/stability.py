"""Passive stability probes."""

from __future__ import annotations

from typing import Any

from .common import copy_vec2d, record_by_name
from .contract import ProbeResult


def passive_stability_probe(
    env: Any,
    object_name: str,
    *,
    steps: int = 60,
    substeps: int = 5,
    max_displacement: float = 8.0,
    max_vertical_settle: float = 24.0,
) -> ProbeResult:
    """Check whether an object stays controllable without agent action."""

    record = record_by_name(env, object_name)
    if record is None:
        return ProbeResult(
            name="passive_stability",
            passed=False,
            tier_evidence=0,
            objects=(object_name,),
            diagnosis="missing_object",
            repair=f"Create a registered object named {object_name!r}.",
            severity="error",
        )

    start = copy_vec2d(record.body.position)
    try:
        for _ in range(steps):
            env.step(substeps=substeps)
        refreshed = record_by_name(env, object_name)
        end = copy_vec2d(refreshed.body.position) if refreshed is not None else start
        displacement = float(start.get_distance(end))
        horizontal_displacement = abs(float(end.x - start.x))
        vertical_displacement = float(end.y - start.y)
    finally:
        env.reset()

    settled_on_support = (
        horizontal_displacement <= max_displacement
        and abs(vertical_displacement) <= max_vertical_settle
    )
    passed = displacement <= max_displacement or settled_on_support
    metrics = {
        "object_displacement": round(displacement, 3),
        "horizontal_displacement": round(horizontal_displacement, 3),
        "vertical_displacement": round(vertical_displacement, 3),
        "start_position": [round(float(start.x), 3), round(float(start.y), 3)],
        "end_position": [round(float(end.x), 3), round(float(end.y), 3)],
        "passive_steps": int(steps),
        "recommended_max_displacement": float(max_displacement),
        "recommended_max_vertical_settle": float(max_vertical_settle),
        "settled_on_support": settled_on_support,
    }
    return ProbeResult(
        name="passive_stability",
        passed=passed,
        tier_evidence=2 if passed else 1,
        objects=(object_name,),
        metrics=metrics,
        diagnosis="object_passively_stable" if passed else "object_not_passively_stable",
        repair=(
            "Object is passively stable."
            if passed
            else "Add a stable floor, shelf, rail, or guide so the object does not fall or drift before contact."
        ),
        severity="info" if passed else "error",
    )
