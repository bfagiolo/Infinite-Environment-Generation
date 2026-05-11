"""Class-membership probes for sorting and grouping objectives."""

from __future__ import annotations

from typing import Any

from .common import record_by_name, record_radius, string_list
from .contract import ProbeResult


def class_membership_probe(
    env: Any,
    *,
    object_names: list[str] | tuple[str, ...],
    region_name: str,
    threshold: float | None = None,
) -> ProbeResult:
    """Check whether a group of objects occupies a target class region."""

    region = record_by_name(env, region_name)
    names = string_list(object_names)
    missing = [region_name] if region is None else []
    object_records = []
    for name in names:
        record = record_by_name(env, name)
        if record is None:
            missing.append(name)
        else:
            object_records.append((name, record))
    if missing:
        return ProbeResult(
            name="class_membership",
            passed=False,
            tier_evidence=0,
            objects=tuple([region_name, *names]),
            metrics={"missing": missing},
            diagnosis="missing_classification_objects",
            repair="Register every sorted object and its target class region with exact names.",
            severity="error",
        )

    region_radius = record_radius(region, 24.0)
    per_object = []
    all_inside = True
    for name, record in object_records:
        object_radius = record_radius(record, 12.0)
        required_distance = (
            float(threshold)
            if threshold is not None
            else max(28.0, object_radius + region_radius * 0.65)
        )
        distance = float(record.body.position.get_distance(region.body.position))
        inside = distance <= required_distance
        all_inside = all_inside and inside
        per_object.append(
            {
                "object": name,
                "distance": round(distance, 3),
                "threshold": round(required_distance, 3),
                "inside": inside,
            }
        )

    return ProbeResult(
        name="class_membership",
        passed=all_inside,
        tier_evidence=5 if all_inside else (4 if any(item["inside"] for item in per_object) else 3),
        objects=tuple([region_name, *names]),
        metrics={
            "region": region_name,
            "objects": per_object,
            "inside_count": sum(1 for item in per_object if item["inside"]),
            "total_count": len(per_object),
        },
        diagnosis="all_objects_in_class_region" if all_inside else "class_membership_incomplete",
        repair=(
            "All class members occupy their target region."
            if all_inside
            else "Shorten or guide the path for the remaining class objects and ensure target regions are non-blocking sensors."
        ),
        severity="info" if all_inside else "warning",
    )
