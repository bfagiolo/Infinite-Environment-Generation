"""Contact and proximity probes for touch-style objectives."""

from __future__ import annotations

from typing import Any

from .common import record_by_name, record_radius
from .contract import ProbeResult


def contact_or_proximity_probe(
    env: Any,
    *,
    first_name: str,
    second_name: str,
    threshold: float | None = None,
) -> ProbeResult:
    """Check deterministic contact possibility using body distance and radii."""

    first = record_by_name(env, first_name)
    second = record_by_name(env, second_name)
    missing = [
        name
        for name, record in ((first_name, first), (second_name, second))
        if record is None
    ]
    if missing:
        return ProbeResult(
            name="contact_or_proximity",
            passed=False,
            tier_evidence=0,
            objects=(first_name, second_name),
            metrics={"missing": missing},
            diagnosis="missing_contact_objects",
            repair="Register both named bodies/regions before declaring a touch objective.",
            severity="error",
        )

    first_radius = record_radius(first, 12.0)
    second_radius = record_radius(second, 12.0)
    required_distance = (
        float(threshold)
        if threshold is not None
        else first_radius + second_radius + 8.0
    )
    distance = float(first.body.position.get_distance(second.body.position))
    passed = distance <= required_distance
    return ProbeResult(
        name="contact_or_proximity",
        passed=passed,
        tier_evidence=5 if passed else 3,
        objects=(first_name, second_name),
        metrics={
            "distance": round(distance, 3),
            "threshold": round(required_distance, 3),
            "first_radius": round(first_radius, 3),
            "second_radius": round(second_radius, 3),
        },
        diagnosis="contact_or_proximity_satisfied" if passed else "contact_not_reached",
        repair=(
            "Bodies are touching or within the declared objective threshold."
            if passed
            else "Move the target onto the reachable path, enlarge the sensor/threshold, or clear blockers preventing contact."
        ),
        severity="info" if passed else "warning",
    )
