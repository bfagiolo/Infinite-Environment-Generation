"""Containment probes for object-in-region objective evidence."""

from __future__ import annotations

from typing import Any

from .common import record_by_name, record_radius
from .contract import ProbeResult


def object_inside_region_probe(
    env: Any,
    *,
    object_name: str,
    region_name: str,
    threshold: float | None = None,
) -> ProbeResult:
    """Check whether an object is inside or close enough to a named region."""

    object_record = record_by_name(env, object_name)
    region_record = record_by_name(env, region_name)
    missing = [
        name
        for name, record in ((object_name, object_record), (region_name, region_record))
        if record is None
    ]
    if missing:
        return ProbeResult(
            name="object_inside_region",
            passed=False,
            tier_evidence=0,
            objects=(object_name, region_name),
            metrics={"missing": missing},
            diagnosis="missing_containment_objects",
            repair="Register both the moved object and target region before declaring containment.",
            severity="error",
        )

    object_radius = record_radius(object_record, 12.0)
    region_radius = record_radius(region_record, 24.0)
    required_distance = (
        float(threshold)
        if threshold is not None
        else max(28.0, object_radius + region_radius * 0.65)
    )
    distance = float(object_record.body.position.get_distance(region_record.body.position))
    passed = distance <= required_distance
    return ProbeResult(
        name="object_inside_region",
        passed=passed,
        tier_evidence=5 if passed else 3,
        objects=(object_name, region_name),
        metrics={
            "distance": round(distance, 3),
            "threshold": round(required_distance, 3),
            "object_radius": round(object_radius, 3),
            "region_radius": round(region_radius, 3),
        },
        diagnosis="object_inside_region" if passed else "object_outside_region",
        repair=(
            "Object is inside the target region."
            if passed
            else "Move the region closer, enlarge the target sensor, or improve the push/slide path so the object crosses the threshold."
        ),
        severity="info" if passed else "warning",
    )
