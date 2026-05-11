"""Small shared helpers for validator probes."""

from __future__ import annotations

from typing import Any

import pymunk


def record_by_name(env: Any, name: str | None):
    if not name:
        return None
    objects = getattr(env, "_objects", {})
    return objects.get(str(name))


def record_radius(record: Any, fallback: float = 12.0) -> float:
    if record is None:
        return float(fallback)
    for shape in getattr(record, "shapes", ()):
        if isinstance(shape, pymunk.Circle):
            return float(shape.radius)
    for shape in getattr(record, "shapes", ()):
        bb = shape.bb
        width = float(bb.right - bb.left)
        height = float(bb.top - bb.bottom)
        if width > 0.0 and height > 0.0:
            return max(width, height) / 2.0
    return float(fallback)


def record_is_sensor(record: Any) -> bool:
    shapes = getattr(record, "shapes", ())
    return bool(shapes) and all(bool(shape.sensor) for shape in shapes)


def body_type_name(body: pymunk.Body) -> str:
    if body.body_type == pymunk.Body.STATIC:
        return "static"
    if body.body_type == pymunk.Body.KINEMATIC:
        return "kinematic"
    return "dynamic"


def copy_vec2d(vector: Any) -> pymunk.Vec2d:
    return pymunk.Vec2d(float(vector.x), float(vector.y))


def string_list(value: Any) -> list[str]:
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]
