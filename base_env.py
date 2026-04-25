"""Physics contract for Harness Alpha generated environments.

Generated worlds should subclass ``BaseEnv`` and implement ``build_world``.
The harness owns stepping, reset, deterministic state export, and metadata
registration so validators can reason from Pymunk state instead of pixels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import math
import random
from typing import Any, Mapping, Sequence

import pymunk


VectorLike = tuple[float, float] | list[float] | pymunk.Vec2d
Action = Mapping[str, Any] | Sequence[float] | None
GroundTruth = dict[str, Any]


@dataclass(frozen=True)
class EnvConfig:
    """Deterministic physics settings shared by generated environments."""

    width: int = 1024
    height: int = 768
    gravity: tuple[float, float] = (0.0, -981.0)
    time_step: float = 1.0 / 60.0
    damping: float = 1.0
    iterations: int = 10


@dataclass
class ObjectRecord:
    """Internal registry entry for telemetry and ground-truth export."""

    name: str
    body: pymunk.Body
    shapes: tuple[pymunk.Shape, ...]
    kind: str
    role: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseEnv(ABC):
    """Base contract for deterministic Pymunk environments.

    Subclasses must implement ``build_world`` and register every relevant
    object with ``register_object`` or the convenience creation helpers.
    """

    def __init__(
        self,
        config: EnvConfig | None = None,
        *,
        seed: int = 0,
        auto_reset: bool = True,
    ) -> None:
        self.config = config or EnvConfig()
        self.seed = seed
        self.rng = random.Random(seed)
        self.space = self._new_space()
        self.solvability_check: dict[str, Any] = {}
        self._objects: dict[str, ObjectRecord] = {}
        self._time = 0.0
        self._step_count = 0

        if auto_reset:
            self.reset(seed=seed)

    @abstractmethod
    def build_world(self) -> None:
        """Populate ``self.space`` with bodies, shapes, and metadata."""

    def apply_action(self, action: Action) -> None:
        """Apply an agent/control action before physics advances.

        Generated environments can override this hook. The base implementation
        is intentionally a no-op for passive validation rollouts.
        """

    def after_step(self) -> None:
        """Hook called after each completed simulation step."""

    def reset(self, *, seed: int | None = None) -> GroundTruth:
        """Rebuild the world and return its initial code-level ground truth."""

        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
        self.space = self._new_space()
        self._objects = {}
        self.solvability_check = {}
        self._time = 0.0
        self._step_count = 0

        self.build_world()
        ground_truth = self.get_ground_truth()
        self._assert_json_serializable(ground_truth)
        return ground_truth

    def step(
        self,
        action: Action = None,
        *,
        dt: float | None = None,
        substeps: int = 1,
    ) -> GroundTruth:
        """Advance deterministic physics and return code-level ground truth."""

        if substeps < 1:
            raise ValueError("substeps must be >= 1")

        frame_dt = self.config.time_step if dt is None else dt
        if frame_dt <= 0.0:
            raise ValueError("dt must be positive")

        self.apply_action(action)
        physics_dt = frame_dt / substeps
        for _ in range(substeps):
            self.space.step(physics_dt)

        self._time += frame_dt
        self._step_count += 1
        self.after_step()

        ground_truth = self.get_ground_truth()
        self._assert_json_serializable(ground_truth)
        return ground_truth

    def get_ground_truth(self) -> GroundTruth:
        """Return a JSON-serializable snapshot of deterministic physics state."""

        return {
            "env": self.__class__.__name__,
            "seed": self.seed,
            "time": self._finite_or_none(self._time),
            "step_count": self._step_count,
            "config": {
                "width": self.config.width,
                "height": self.config.height,
                "gravity": self._vec_to_list(self.config.gravity),
                "time_step": self.config.time_step,
                "damping": self.config.damping,
                "iterations": self.config.iterations,
            },
            "objects": {
                name: self._object_to_ground_truth(record)
                for name, record in self._objects.items()
            },
            "solvability_check": self.solvability_check,
        }

    def get_ground_truth_json(self, *, indent: int | None = None) -> str:
        """Return the ground-truth snapshot as a JSON string."""

        return json.dumps(self.get_ground_truth(), indent=indent, sort_keys=True)

    def register_object(
        self,
        name: str,
        body: pymunk.Body,
        shapes: Sequence[pymunk.Shape] | pymunk.Shape,
        *,
        kind: str,
        role: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ObjectRecord:
        """Register a Pymunk object for physics, validation, and telemetry."""

        if not name:
            raise ValueError("registered object name cannot be empty")
        if name in self._objects:
            raise ValueError(f"duplicate registered object name: {name}")

        shape_tuple = (shapes,) if isinstance(shapes, pymunk.Shape) else tuple(shapes)
        if not shape_tuple:
            raise ValueError(f"{name!r} must include at least one shape")

        for shape in shape_tuple:
            if shape.body is not body:
                raise ValueError(f"shape for {name!r} is attached to a different body")
            shape.harness_object_name = name
            shape.harness_role = role

        self._add_to_space(body, shape_tuple)
        record = ObjectRecord(
            name=name,
            body=body,
            shapes=shape_tuple,
            kind=kind,
            role=role,
            metadata=dict(metadata or {}),
        )
        self._objects[name] = record
        return record

    def create_dynamic_circle(
        self,
        name: str,
        *,
        position: VectorLike,
        radius: float,
        mass: float = 1.0,
        kind: str = "dynamic_circle",
        role: str | None = None,
        elasticity: float = 0.0,
        friction: float = 0.8,
        metadata: Mapping[str, Any] | None = None,
    ) -> ObjectRecord:
        """Create and register a dynamic circular body."""

        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if mass <= 0.0:
            raise ValueError("mass must be positive")

        moment = pymunk.moment_for_circle(mass, 0.0, radius)
        body = pymunk.Body(mass, moment)
        body.position = self._to_vec2d(position)
        shape = pymunk.Circle(body, radius)
        shape.elasticity = elasticity
        shape.friction = friction
        return self.register_object(
            name,
            body,
            shape,
            kind=kind,
            role=role,
            metadata=metadata,
        )

    def create_static_segment(
        self,
        name: str,
        *,
        a: VectorLike,
        b: VectorLike,
        radius: float = 1.0,
        kind: str = "static_segment",
        role: str | None = None,
        elasticity: float = 0.0,
        friction: float = 0.9,
        metadata: Mapping[str, Any] | None = None,
    ) -> ObjectRecord:
        """Create and register a static line segment."""

        if radius < 0.0:
            raise ValueError("radius cannot be negative")

        body = self.space.static_body
        shape = pymunk.Segment(body, self._to_vec2d(a), self._to_vec2d(b), radius)
        shape.elasticity = elasticity
        shape.friction = friction
        return self.register_object(
            name,
            body,
            shape,
            kind=kind,
            role=role,
            metadata=metadata,
        )

    def create_static_box(
        self,
        name: str,
        *,
        center: VectorLike,
        size: VectorLike,
        kind: str = "static_box",
        role: str | None = None,
        elasticity: float = 0.0,
        friction: float = 0.9,
        metadata: Mapping[str, Any] | None = None,
    ) -> ObjectRecord:
        """Create and register an axis-aligned static box."""

        width, height = self._to_pair(size)
        if width <= 0.0 or height <= 0.0:
            raise ValueError("box size values must be positive")

        cx, cy = self._to_pair(center)
        half_w = width / 2.0
        half_h = height / 2.0
        vertices = [
            (-half_w, -half_h),
            (-half_w, half_h),
            (half_w, half_h),
            (half_w, -half_h),
        ]
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (cx, cy)
        shape = pymunk.Poly(body, vertices)
        shape.elasticity = elasticity
        shape.friction = friction
        return self.register_object(
            name,
            body,
            shape,
            kind=kind,
            role=role,
            metadata=metadata,
        )

    def set_solvability_hint(
        self,
        *,
        start: VectorLike,
        goal: VectorLike,
        grid_size: float = 24.0,
        agent_radius: float | None = None,
        notes: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Attach the required hint consumed by the headless validator."""

        if grid_size <= 0.0:
            raise ValueError("grid_size must be positive")

        self.solvability_check = {
            "start": self._vec_to_list(start),
            "goal": self._vec_to_list(goal),
            "grid_size": grid_size,
            "agent_radius": agent_radius,
            "notes": notes,
            "metadata": dict(metadata or {}),
        }

    def get_object(self, name: str) -> ObjectRecord:
        """Return a registered object record by name."""

        try:
            return self._objects[name]
        except KeyError as exc:
            raise KeyError(f"unknown object: {name}") from exc

    def distance_between(self, first: str, second: str) -> float:
        """Return center-to-center distance between two registered objects."""

        first_pos = self.get_object(first).body.position
        second_pos = self.get_object(second).body.position
        return float(first_pos.get_distance(second_pos))

    def _new_space(self) -> pymunk.Space:
        space = pymunk.Space(threaded=False)
        space.gravity = self.config.gravity
        space.damping = self.config.damping
        space.iterations = self.config.iterations
        return space

    def _add_to_space(
        self,
        body: pymunk.Body,
        shapes: tuple[pymunk.Shape, ...],
    ) -> None:
        additions: list[Any] = []
        if body.body_type != pymunk.Body.STATIC or body is not self.space.static_body:
            if body not in self.space.bodies:
                additions.append(body)
        for shape in shapes:
            if shape not in self.space.shapes:
                additions.append(shape)
        if additions:
            self.space.add(*additions)

    def _object_to_ground_truth(self, record: ObjectRecord) -> dict[str, Any]:
        body = record.body
        return {
            "kind": record.kind,
            "role": record.role,
            "metadata": self._json_safe(record.metadata),
            "body": {
                "type": self._body_type_name(body),
                "position": self._vec_to_list(body.position),
                "velocity": self._vec_to_list(body.velocity),
                "angle": self._finite_or_none(body.angle),
                "angular_velocity": self._finite_or_none(body.angular_velocity),
                "force": self._vec_to_list(body.force),
                "torque": self._finite_or_none(body.torque),
                "mass": self._finite_or_none(body.mass),
                "moment": self._finite_or_none(body.moment),
            },
            "shapes": [self._shape_to_ground_truth(shape) for shape in record.shapes],
        }

    def _shape_to_ground_truth(self, shape: pymunk.Shape) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": shape.__class__.__name__,
            "sensor": bool(shape.sensor),
            "collision_type": int(shape.collision_type),
            "elasticity": self._finite_or_none(shape.elasticity),
            "friction": self._finite_or_none(shape.friction),
            "bb": self._bb_to_ground_truth(shape),
        }

        body = shape.body
        if isinstance(shape, pymunk.Circle):
            data.update(
                {
                    "radius": self._finite_or_none(shape.radius),
                    "offset": self._vec_to_list(shape.offset),
                    "world_center": self._vec_to_list(body.local_to_world(shape.offset)),
                }
            )
        elif isinstance(shape, pymunk.Segment):
            data.update(
                {
                    "a": self._vec_to_list(shape.a),
                    "b": self._vec_to_list(shape.b),
                    "world_a": self._vec_to_list(body.local_to_world(shape.a)),
                    "world_b": self._vec_to_list(body.local_to_world(shape.b)),
                    "radius": self._finite_or_none(shape.radius),
                }
            )
        elif isinstance(shape, pymunk.Poly):
            local_vertices = list(shape.get_vertices())
            data.update(
                {
                    "vertices": [self._vec_to_list(vertex) for vertex in local_vertices],
                    "world_vertices": [
                        self._vec_to_list(body.local_to_world(vertex))
                        for vertex in local_vertices
                    ],
                }
            )

        return data

    def _bb_to_ground_truth(self, shape: pymunk.Shape) -> dict[str, float | None]:
        bb = shape.cache_bb()
        return {
            "left": self._finite_or_none(bb.left),
            "right": self._finite_or_none(bb.right),
            "bottom": self._finite_or_none(bb.bottom),
            "top": self._finite_or_none(bb.top),
        }

    @staticmethod
    def _body_type_name(body: pymunk.Body) -> str:
        if body.body_type == pymunk.Body.DYNAMIC:
            return "dynamic"
        if body.body_type == pymunk.Body.KINEMATIC:
            return "kinematic"
        if body.body_type == pymunk.Body.STATIC:
            return "static"
        return "unknown"

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, pymunk.Vec2d):
            return cls._vec_to_list(value)
        if isinstance(value, float):
            return cls._finite_or_none(value)
        return value

    @staticmethod
    def _assert_json_serializable(payload: GroundTruth) -> None:
        json.dumps(payload, allow_nan=False)

    @staticmethod
    def _finite_or_none(value: float) -> float | None:
        value = float(value)
        if math.isfinite(value):
            return value
        return None

    @staticmethod
    def _to_vec2d(value: VectorLike) -> pymunk.Vec2d:
        if isinstance(value, pymunk.Vec2d):
            return value
        x, y = value
        return pymunk.Vec2d(float(x), float(y))

    @classmethod
    def _to_pair(cls, value: VectorLike) -> tuple[float, float]:
        vector = cls._to_vec2d(value)
        return float(vector.x), float(vector.y)

    @classmethod
    def _vec_to_list(cls, value: VectorLike) -> list[float | None]:
        vector = cls._to_vec2d(value)
        return [cls._finite_or_none(vector.x), cls._finite_or_none(vector.y)]
