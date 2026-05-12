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
    damping: float = 0.99
    iterations: int = 30
    angular_damping: float = 0.5


@dataclass
class ObjectRecord:
    """Internal registry entry for telemetry and ground-truth export."""

    name: str
    body: pymunk.Body
    shapes: tuple[pymunk.Shape, ...]
    kind: str
    role: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstraintRecord:
    """Internal registry entry for joints and mechanical telemetry."""

    name: str
    constraint: pymunk.Constraint
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForceZoneRecord:
    """Internal registry entry for deterministic environmental force fields."""

    name: str
    object_name: str
    mode: str
    force: pymunk.Vec2d
    strength: float
    affected_names: tuple[str, ...] = ()
    affected_roles: tuple[str, ...] = ()
    falloff: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MechanismRecord:
    """Internal registry entry for deterministic trigger/gate mechanisms."""

    name: str
    trigger_name: str
    gate_name: str
    activator_names: tuple[str, ...]
    activation_distance: float | None = None
    open_mode: str = "sensorize"
    activated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecurringHazardRecord:
    """Internal registry entry for reusable falling-hazard emitters."""

    name: str
    object_names: tuple[str, ...]
    spawn_lanes: tuple[tuple[float, float], ...]
    bottom_y: float
    speed_y: float
    phase_gap_steps: int
    active_names: set[str] = field(default_factory=set)
    next_release_steps: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecurringLateralHazardRecord:
    """Internal registry entry for reusable side-scroller crossing hazards."""

    name: str
    object_names: tuple[str, ...]
    spawn_lanes: tuple[tuple[float, float], ...]
    exit_x: float
    speed_x: float
    angular_speed: float
    phase_gap_steps: int
    active_names: set[str] = field(default_factory=set)
    next_release_steps: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReadableChaserRecord:
    """Internal registry entry for deterministic pursuit hazards/enemies."""

    name: str
    chaser_name: str
    target_name: str
    force_strength: float
    max_speed: float
    stop_radius: float
    axis: str = "x"
    anchor_y: float | None = None
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
        self.width = self.config.width
        self.height = self.config.height
        self.seed = seed
        self.rng = random.Random(seed)
        self.space = self._new_space()
        self.solvability_check: dict[str, Any] = {}
        self._objects: dict[str, ObjectRecord] = {}
        self._constraints: dict[str, ConstraintRecord] = {}
        self._force_zones: dict[str, ForceZoneRecord] = {}
        self._mechanisms: dict[str, MechanismRecord] = {}
        self._recurring_hazards: dict[str, RecurringHazardRecord] = {}
        self._recurring_lateral_hazards: dict[str, RecurringLateralHazardRecord] = {}
        self._readable_chasers: dict[str, ReadableChaserRecord] = {}
        self.agent: ObjectRecord | None = None
        self.agent_strength = 1.0
        self._time = 0.0
        self._step_count = 0
        self._last_agent_jump_step = -10_000

        if auto_reset:
            self.reset(seed=seed)

    @abstractmethod
    def build_world(self) -> None:
        """Populate ``self.space`` with bodies, shapes, and metadata."""

    def apply_action(self, action: Action) -> None:
        """Apply an agent/control action before physics advances.

        Generated environments can override this hook. The base implementation
        accepts a direction vector as a sequence or as {"move": [x, y]}.
        """

        if action is None:
            return
        if isinstance(action, Mapping):
            direction = action.get("move") or action.get("direction")
        else:
            direction = action
        if direction is not None:
            self.apply_agent_force(direction)

    def after_step(self) -> None:
        """Hook called after each completed simulation step."""
        self._update_readable_chasers()
        self._update_recurring_falling_hazards()
        self._update_recurring_lateral_hazards()

    def reset_objective_state(self) -> None:
        """Reset subclass objective flags before each world rebuild.

        Generated environments that track success over time should override
        this hook and reset booleans, counters, timers, and per-run sets here.
        Keeping this separate from ``__init__`` makes visual replays and
        repeated validation runs deterministic.
        """

    @property
    def time(self) -> float:
        """Current deterministic simulation time in seconds."""

        return self._time

    @property
    def step_count(self) -> int:
        """Number of completed high-level simulation steps."""

        return self._step_count

    @property
    def dt(self) -> float:
        """Default high-level simulation time step in seconds."""

        return self.config.time_step

    def reset(self, *, seed: int | None = None) -> GroundTruth:
        """Rebuild the world and return its initial code-level ground truth."""

        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
        self.space = self._new_space()
        self._objects = {}
        self._constraints = {}
        self._force_zones = {}
        self._mechanisms = {}
        self._recurring_hazards = {}
        self._recurring_lateral_hazards = {}
        self._readable_chasers = {}
        self.agent = None
        self.agent_strength = 1.0
        self.solvability_check = {}
        self._time = 0.0
        self._step_count = 0
        self._last_agent_jump_step = -10_000

        self.reset_objective_state()

        self.build_world()
        self._sanitize_body_callbacks()
        self._update_agent_strength()
        self._update_mechanisms()
        ground_truth = self._json_safe(self.get_ground_truth())
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
        self._sanitize_body_callbacks()
        physics_dt = frame_dt / substeps
        for _ in range(substeps):
            self._update_mechanisms()
            self._apply_force_zones()
            self._sanitize_body_callbacks()
            self.space.step(physics_dt)
            self._apply_dynamic_angular_damping(physics_dt)

        self._time += frame_dt
        self._step_count += 1
        self._update_mechanisms()
        self.after_step()
        self._update_mechanisms()

        ground_truth = self._json_safe(self.get_ground_truth())
        self._assert_json_serializable(ground_truth)
        return ground_truth

    def _sanitize_body_callbacks(self) -> None:
        """Restore Pymunk's required integration callbacks if generated code broke them."""

        for body in list(getattr(self.space, "bodies", ())):
            # Pymunk's public getter returns the default callable even after
            # `body.velocity_func = None`, but the C callback still points at a
            # Python None. Assigning the default unconditionally restores the
            # native Chipmunk callback and prevents CFFI popup errors.
            body.velocity_func = pymunk.Body.update_velocity
            body.position_func = pymunk.Body.update_position

    def get_ground_truth(self) -> GroundTruth:
        """Return a JSON-serializable snapshot of deterministic physics state."""

        ground_truth = {
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
                "angular_damping": self.config.angular_damping,
                "agent_strength": self.agent_strength,
            },
            "objects": {
                name: self._object_to_ground_truth(record)
                for name, record in self._objects.items()
            },
            "constraints": {
                name: self._constraint_to_ground_truth(record)
                for name, record in self._constraints.items()
            },
            "force_zones": {
                name: self._force_zone_to_ground_truth(record)
                for name, record in self._force_zones.items()
            },
            "mechanisms": {
                name: self._mechanism_to_ground_truth(record)
                for name, record in self._mechanisms.items()
            },
            "recurring_hazards": {
                name: self._recurring_hazard_to_ground_truth(record)
                for name, record in self._recurring_hazards.items()
            },
            "recurring_lateral_hazards": {
                name: self._recurring_lateral_hazard_to_ground_truth(record)
                for name, record in self._recurring_lateral_hazards.items()
            },
            "readable_chasers": {
                name: self._readable_chaser_to_ground_truth(record)
                for name, record in self._readable_chasers.items()
            },
            "solvability_check": self.solvability_check,
        }
        ground_truth["objective"] = self._objective_to_ground_truth()
        return ground_truth

    def get_ground_truth_json(self, *, indent: int | None = None) -> str:
        """Return the ground-truth snapshot as a JSON string."""

        return json.dumps(self._json_safe(self.get_ground_truth()), indent=indent, sort_keys=True)

    def _objective_to_ground_truth(self) -> dict[str, Any]:
        """Return default objective metadata for validator/oracle routing."""

        objective_active = callable(getattr(self, "check_objective", None))
        objective_satisfied = False
        objective_error = None
        if objective_active:
            try:
                objective_satisfied = bool(self.check_objective())
            except Exception as exc:
                objective_error = f"{type(exc).__name__}: {exc}"

        metadata: dict[str, Any] = {
            "objective_active": objective_active,
            "objective_type": getattr(self, "objective_type", None),
            "objective_targets": self._json_safe(
                list(getattr(self, "objective_targets", []) or [])
            ),
            "objective_profile": self._objective_profile_to_ground_truth(),
            "capability_profile": self._capability_profile_to_ground_truth(),
            "gameplay_profile": self._gameplay_profile_to_ground_truth(),
            "physics_relations": self._physics_relations_to_ground_truth(),
            "layout_plan": self._layout_plan_to_ground_truth(),
            "semantic_requirements": self._semantic_requirements_to_ground_truth(),
            "anti_cheat_profile": self._anti_cheat_profile_to_ground_truth(),
            "objective_satisfied": objective_satisfied,
        }
        if objective_error is not None:
            metadata["objective_error"] = objective_error
        if hasattr(self, "targets_touched"):
            targets_touched = getattr(self, "targets_touched")
            if isinstance(targets_touched, set):
                metadata["targets_touched"] = sorted(str(item) for item in targets_touched)
            else:
                metadata["targets_touched"] = self._json_safe(targets_touched)
        if metadata.get("objective_targets") is not None and "targets_touched" in metadata:
            touched = set(map(str, metadata.get("targets_touched") or []))
            targets = [str(target) for target in metadata.get("objective_targets") or []]
            metadata["targets_remaining"] = [
                target for target in targets if target not in touched
            ]
        return metadata

    def _objective_profile_to_ground_truth(self) -> dict[str, Any]:
        """Return the generated environment's objective profile, with defaults."""

        profile = getattr(self, "objective_profile", None)
        if isinstance(profile, Mapping):
            normalized = dict(profile)
        else:
            normalized = {}

        objective_type = getattr(self, "objective_type", None)
        objective_targets = list(getattr(self, "objective_targets", []) or [])
        normalized.setdefault("objective_type", objective_type)
        normalized.setdefault("objective_description", "")
        normalized.setdefault("success_predicate", "")
        normalized.setdefault("targets", objective_targets)
        normalized.setdefault("required_capabilities", [])
        normalized.setdefault("progress_metrics", [])
        normalized.setdefault("subgoals", [])
        normalized.setdefault("validator_skills", [])
        normalized.setdefault("failure_modes", [])
        normalized.setdefault(
            "minimum_acceptance_tier",
            self._default_minimum_acceptance_tier(str(objective_type)),
        )
        return self._json_safe(normalized)

    def _gameplay_profile_to_ground_truth(self) -> dict[str, Any]:
        """Return optional game-feel contract metadata."""

        profile = getattr(self, "gameplay_profile", None)
        if isinstance(profile, Mapping):
            return self._json_safe(dict(profile))
        return {}

    def _physics_relations_to_ground_truth(self) -> dict[str, Any]:
        """Return optional compositional physics-relation metadata."""

        graph = getattr(self, "physics_relations", None)
        if isinstance(graph, Mapping):
            return self._json_safe(dict(graph))
        return {}

    def _layout_plan_to_ground_truth(self) -> dict[str, Any]:
        """Return optional route-aware spatial planning metadata."""

        plan = getattr(self, "layout_plan", None)
        if isinstance(plan, Mapping):
            return self._json_safe(dict(plan))
        return {}

    def _semantic_requirements_to_ground_truth(self) -> list[dict[str, Any]]:
        """Return optional prompt-fidelity dynamics requirements."""

        requirements = getattr(self, "semantic_requirements", [])
        if not isinstance(requirements, Sequence) or isinstance(requirements, (str, bytes)):
            return []
        normalized: list[dict[str, Any]] = []
        for requirement in requirements:
            if isinstance(requirement, Mapping):
                normalized.append(self._json_safe(dict(requirement)))
        return normalized

    def _anti_cheat_profile_to_ground_truth(self) -> list[dict[str, Any]]:
        """Return optional prompt-fidelity anti-cheat requirements."""

        profile = getattr(self, "anti_cheat_profile", [])
        if not isinstance(profile, Sequence) or isinstance(profile, (str, bytes)):
            return []
        normalized: list[dict[str, Any]] = []
        for item in profile:
            if isinstance(item, Mapping):
                normalized.append(self._json_safe(dict(item)))
        return normalized

    def _capability_profile_to_ground_truth(self) -> dict[str, Any]:
        """Return the generated environment's capability profile, with defaults."""

        profile = getattr(self, "capability_profile", None)
        if isinstance(profile, Mapping):
            normalized = dict(profile)
        else:
            normalized = {}

        gravity = self._to_vec2d(self.config.gravity).length
        normalized.setdefault("movement", "ground_force" if gravity > 1e-6 else "thrust_2d")
        normalized.setdefault("interaction", ["touch_contact"])
        normalized.setdefault("gravity", "normal" if gravity > 1e-6 else "zero_g")
        normalized.setdefault("allowed_controls", ["apply_force_x", "apply_force_y", "brake"])
        normalized.setdefault(
            "forbidden_controls",
            [
                "teleport",
                "direct_object_move",
                "direct_object_rotation",
                "direct_goal_state_write",
            ],
        )
        normalized.setdefault("notes", "")
        return self._json_safe(normalized)

    @staticmethod
    def _default_minimum_acceptance_tier(objective_type: str) -> int:
        default_tiers = {
            "navigation_goal": 5,
            "single_target_touch": 5,
            "multi_target_touch": 5,
            "push_object": 5,
            "seesaw_balance": 4,
            "mechanism_activation": 4,
            "survival": 5,
            "custom_physics": 4,
        }
        return default_tiers.get(objective_type, 4)

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
        if body.body_type == pymunk.Body.DYNAMIC:
            self._configure_dynamic_body(body)

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
        if role == "agent":
            self.agent = record
        self._update_agent_strength()
        return record

    def create_dynamic_circle(
        self,
        name: str,
        *,
        pos: VectorLike | None = None,
        position: VectorLike | None = None,
        radius: float = 0.0,
        mass: float = 1.0,
        kind: str = "dynamic_circle",
        role: str | None = None,
        elasticity: float = 0.0,
        friction: float = 0.8,
        sensor: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> ObjectRecord:
        """Create and register a dynamic circular body."""

        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if mass <= 0.0:
            raise ValueError("mass must be positive")

        spawn_position = pos if pos is not None else position
        if spawn_position is None:
            raise ValueError("create_dynamic_circle requires pos")

        moment = pymunk.moment_for_circle(mass, 0.0, radius)
        body = pymunk.Body(mass, moment)
        body.position = self._to_vec2d(spawn_position)
        shape = pymunk.Circle(body, radius)
        shape.elasticity = elasticity
        shape.friction = friction
        shape.sensor = bool(sensor)
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
        sensor: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> ObjectRecord:
        """Create and register a static line segment."""

        if radius < 0.0:
            raise ValueError("radius cannot be negative")

        body = self.space.static_body
        shape = pymunk.Segment(body, self._to_vec2d(a), self._to_vec2d(b), radius)
        shape.elasticity = elasticity
        shape.friction = friction
        shape.sensor = bool(sensor)
        return self.register_object(
            name,
            body,
            shape,
            kind=kind,
            role=role,
            metadata=metadata,
        )

    def create_dynamic_box(
        self,
        name: str,
        *,
        center: VectorLike,
        size: VectorLike,
        mass: float = 1.0,
        angle: float = 0.0,
        kind: str = "dynamic_box",
        role: str | None = None,
        elasticity: float = 0.0,
        friction: float = 0.8,
        sensor: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> ObjectRecord:
        """Create and register a dynamic rectangular body."""

        width, height = self._to_pair(size)
        if width <= 0.0 or height <= 0.0:
            raise ValueError("box size values must be positive")
        if mass <= 0.0:
            raise ValueError("mass must be positive")

        moment = pymunk.moment_for_box(mass, (width, height))
        body = pymunk.Body(mass, moment)
        body.position = self._to_vec2d(center)
        body.angle = angle
        shape = pymunk.Poly.create_box(body, (width, height))
        shape.elasticity = elasticity
        shape.friction = friction
        shape.sensor = bool(sensor)
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
        sensor: bool = False,
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
        shape.sensor = bool(sensor)
        return self.register_object(
            name,
            body,
            shape,
            kind=kind,
            role=role,
            metadata=metadata,
        )

    def register_constraint(
        self,
        name: str | None = None,
        constraint: pymunk.Constraint | None = None,
        *,
        type: str | None = None,
        body_a: pymunk.Body | ObjectRecord | str | None = None,
        body_b: pymunk.Body | ObjectRecord | str | None = None,
        anchor_a: VectorLike | None = None,
        anchor_b: VectorLike | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ConstraintRecord:
        """Register a Pymunk joint/constraint for mechanics and telemetry."""

        if constraint is None:
            constraint = self._build_constraint_from_spec(
                type=type,
                body_a=body_a,
                body_b=body_b,
                anchor_a=anchor_a,
                anchor_b=anchor_b,
            )
        if name is None:
            stem = type or constraint.__class__.__name__
            name = f"{stem}_{len(self._constraints) + 1}"
        if not name:
            raise ValueError("registered constraint name cannot be empty")
        if name in self._constraints:
            raise ValueError(f"duplicate registered constraint name: {name}")
        if constraint not in self.space.constraints:
            self.space.add(constraint)
        constraint.collide_bodies = False
        self._apply_constraint_collision_group(constraint, group=1)
        record = ConstraintRecord(
            name=name,
            constraint=constraint,
            metadata=dict(metadata or {}),
        )
        self._constraints[name] = record
        return record

    def register_force_zone(
        self,
        name: str,
        *,
        center: VectorLike,
        size: VectorLike,
        force: VectorLike = (0.0, 0.0),
        mode: str = "constant",
        strength: float | None = None,
        affected_names: Sequence[str] | None = None,
        affected_roles: Sequence[str] | None = None,
        falloff: float = 0.0,
        role: str | None = "force_zone",
        metadata: Mapping[str, Any] | None = None,
    ) -> ObjectRecord:
        """Create a non-blocking force region and register its deterministic effect.

        ``mode="constant"`` applies force in the provided vector direction,
        scaled by ``strength`` when supplied. ``"attract"`` and ``"repel"`` use
        the zone center and ``strength`` to compute a radial force.
        """

        if not name:
            raise ValueError("force zone name cannot be empty")
        zone_mode = mode.replace("-", "_").lower()
        if zone_mode not in {"constant", "attract", "repel"}:
            raise ValueError(f"unsupported force zone mode: {mode}")
        force_vector = self._to_vec2d(force)
        zone_strength = (
            float(strength)
            if strength is not None
            else float(force_vector.length)
        )
        zone_metadata = {
            "kind": "force_zone",
            "force_zone": True,
            "mode": zone_mode,
            "force": self._vec_to_list(force_vector),
            "strength": self._finite_or_none(zone_strength),
            "affected_names": [str(item) for item in affected_names or ()],
            "affected_roles": [str(item) for item in affected_roles or ()],
            "falloff": self._finite_or_none(float(falloff)),
        }
        zone_metadata.update(dict(metadata or {}))
        record = self.create_static_box(
            name,
            center=center,
            size=size,
            role=role,
            sensor=True,
            metadata=zone_metadata,
        )
        self._force_zones[name] = ForceZoneRecord(
            name=name,
            object_name=record.name,
            mode=zone_mode,
            force=force_vector,
            strength=zone_strength,
            affected_names=tuple(str(item) for item in affected_names or ()),
            affected_roles=tuple(str(item) for item in affected_roles or ()),
            falloff=float(falloff),
            metadata=zone_metadata,
        )
        return record

    def create_pressure_plate(
        self,
        name: str,
        *,
        center: VectorLike,
        size: VectorLike,
        role: str | None = "trigger",
        elasticity: float = 0.0,
        friction: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> ObjectRecord:
        """Create a non-blocking trigger region for plate/button mechanisms."""

        plate_metadata = {
            "kind": "pressure_plate",
            "trigger": True,
            "mechanism_trigger": True,
        }
        plate_metadata.update(dict(metadata or {}))
        return self.create_static_box(
            name,
            center=center,
            size=size,
            role=role,
            elasticity=elasticity,
            friction=friction,
            sensor=True,
            metadata=plate_metadata,
        )

    def create_horizontal_push_lane(
        self,
        name: str,
        *,
        agent_name: str = "agent",
        object_name: str = "push_object",
        target_name: str = "target_region",
        agent_x: float,
        object_x: float,
        target_x: float,
        lane_y: float,
        agent_radius: float | None = None,
        object_size: VectorLike = (36.0, 36.0),
        object_mass: float = 1.25,
        object_friction: float = 0.35,
        target_size: VectorLike = (80.0, 56.0),
        target_role: str | None = "trigger",
        target_kind: str = "region",
        support_thickness: float = 28.0,
        support_margin: float = 120.0,
        support_friction: float = 0.9,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, ObjectRecord]:
        """Create a validator-friendly horizontal contact-push affordance.

        This primitive is intentionally small: it stages agent -> object ->
        target on one horizontal lane, adds stable support under the movable
        object, and keeps the target as a non-blocking sensor. Generated worlds
        can compose it with gates, hazards, or other structures without
        hand-tuning the fragile contact geometry every time.
        """

        if not name:
            raise ValueError("push lane name cannot be empty")
        radius = float(agent_radius if agent_radius is not None else getattr(self, "agent_radius", 15.0))
        if radius <= 0.0:
            raise ValueError("agent_radius must be positive")
        object_width, object_height = self._to_pair(object_size)
        target_width, target_height = self._to_pair(target_size)
        if object_mass <= 0.0:
            raise ValueError("object_mass must be positive")
        if support_thickness <= 0.0:
            raise ValueError("support_thickness must be positive")
        if (float(agent_x) - float(object_x)) * (float(target_x) - float(object_x)) >= 0.0:
            raise ValueError("horizontal push lane must stage the object between agent_x and target_x")

        lane_metadata = {"kind": "horizontal_push_lane", "push_lane": name}
        lane_metadata.update(dict(metadata or {}))

        object_center = (float(object_x), float(lane_y))
        support_top = float(lane_y) - object_height * 0.5 - 2.0
        support_min_x = min(float(agent_x), float(object_x), float(target_x)) - float(support_margin)
        support_max_x = max(float(agent_x), float(object_x), float(target_x)) + float(support_margin)
        support_width = max(1.0, support_max_x - support_min_x)
        support_center = (
            (support_min_x + support_max_x) * 0.5,
            support_top - float(support_thickness) * 0.5,
        )

        support = self.create_static_box(
            f"{name}_support",
            center=support_center,
            size=(support_width, float(support_thickness)),
            role="support",
            friction=float(support_friction),
            metadata={**lane_metadata, "support": True},
        )

        target_metadata = {**lane_metadata, "target_region": True}
        target_kind_normalized = target_kind.replace("-", "_").lower()
        if target_kind_normalized in {"pressure_plate", "plate", "trigger", "button", "switch"}:
            target = self.create_pressure_plate(
                target_name,
                center=(float(target_x), float(lane_y)),
                size=(target_width, target_height),
                role=target_role,
                metadata=target_metadata,
            )
        else:
            target = self.create_static_box(
                target_name,
                center=(float(target_x), float(lane_y)),
                size=(target_width, target_height),
                role=target_role,
                sensor=True,
                friction=0.0,
                metadata=target_metadata,
            )

        push_object = self.create_dynamic_box(
            object_name,
            center=object_center,
            size=(object_width, object_height),
            mass=float(object_mass),
            friction=float(object_friction),
            elasticity=0.0,
            metadata={**lane_metadata, "push_object": True},
        )

        agent = self.create_dynamic_circle(
            agent_name,
            pos=(float(agent_x), support_top + radius + 2.0),
            radius=radius,
            mass=1.0,
            friction=0.75,
            elasticity=0.0,
            role="agent",
            metadata={**lane_metadata, "lane_agent": True},
        )

        return {
            "support": support,
            "target": target,
            "object": push_object,
            "agent": agent,
        }

    def create_strike_shot_lane(
        self,
        name: str,
        *,
        agent_name: str = "agent",
        object_name: str = "ball",
        target_name: str = "goal_line",
        agent_x: float,
        object_x: float,
        target_x: float,
        lane_y: float,
        agent_radius: float | None = None,
        object_radius: float = 18.0,
        object_mass: float = 0.55,
        object_friction: float = 0.08,
        object_elasticity: float = 0.24,
        target_size: VectorLike = (48.0, 132.0),
        support_thickness: float = 28.0,
        support_margin: float = 180.0,
        support_friction: float = 0.72,
        goal_post_thickness: float = 14.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, ObjectRecord]:
        """Create a validator-friendly strike/kick affordance.

        This is for soccer, hockey, pinball-style shots, and other tasks where
        the agent should transfer a crisp impulse into a light dynamic object
        that travels through a generous target sensor. It is not a crate-push
        lane: the object is round, low-friction, and placed close enough to the
        goal for a direct shot validator to prove the objective.
        """

        if not name:
            raise ValueError("strike lane name cannot be empty")
        radius = float(agent_radius if agent_radius is not None else getattr(self, "agent_radius", 15.0))
        ball_radius = float(object_radius)
        if radius <= 0.0 or ball_radius <= 0.0:
            raise ValueError("agent_radius and object_radius must be positive")
        if object_mass <= 0.0:
            raise ValueError("object_mass must be positive")
        if support_thickness <= 0.0:
            raise ValueError("support_thickness must be positive")
        if (float(agent_x) - float(object_x)) * (float(target_x) - float(object_x)) >= 0.0:
            raise ValueError("strike shot lane must stage the ball between agent_x and target_x")

        target_width, target_height = self._to_pair(target_size)
        lane_metadata = {"kind": "strike_shot_lane", "shot_lane": name}
        lane_metadata.update(dict(metadata or {}))

        support_top = float(lane_y) - max(radius, ball_radius) - 2.0
        support_min_x = min(float(agent_x), float(object_x), float(target_x)) - float(support_margin)
        support_max_x = max(float(agent_x), float(object_x), float(target_x)) + float(support_margin)
        support = self.create_static_box(
            f"{name}_support",
            center=((support_min_x + support_max_x) * 0.5, support_top - float(support_thickness) * 0.5),
            size=(support_max_x - support_min_x, float(support_thickness)),
            role="support",
            friction=float(support_friction),
            metadata={**lane_metadata, "support": True},
        )

        target = self.create_static_box(
            target_name,
            center=(float(target_x), float(lane_y)),
            size=(target_width, target_height),
            role="goal",
            sensor=True,
            friction=0.0,
            metadata={**lane_metadata, "target_region": True, "goal_line": True},
        )

        # Visual/physical posts sit outside the direct shot corridor. They make
        # the goal readable without blocking the validator's center-line shot.
        post_offset = target_height * 0.5 + float(goal_post_thickness) * 0.5
        upper_post = self.create_static_box(
            f"{name}_goal_post_upper",
            center=(float(target_x), float(lane_y) + post_offset),
            size=(target_width, float(goal_post_thickness)),
            role="terrain",
            friction=0.8,
            metadata={**lane_metadata, "goal_post": True},
        )
        lower_post = self.create_static_box(
            f"{name}_goal_post_lower",
            center=(float(target_x), float(lane_y) - post_offset),
            size=(target_width, float(goal_post_thickness)),
            role="terrain",
            friction=0.8,
            metadata={**lane_metadata, "goal_post": True},
        )

        ball = self.create_dynamic_circle(
            object_name,
            pos=(float(object_x), float(lane_y)),
            radius=ball_radius,
            mass=float(object_mass),
            friction=float(object_friction),
            elasticity=float(object_elasticity),
            role="object",
            metadata={**lane_metadata, "strike_object": True, "ball": True},
        )
        ball.body.angular_damping = 0.78

        agent = self.create_dynamic_circle(
            agent_name,
            pos=(float(agent_x), float(lane_y)),
            radius=radius,
            mass=1.0,
            friction=0.68,
            elasticity=0.02,
            role="agent",
            metadata={**lane_metadata, "lane_agent": True, "striker": True},
        )

        self.set_solvability_hint(
            start=(float(agent_x), float(lane_y)),
            goal=(float(target_x), float(lane_y)),
            agent_radius=agent_radius,
            notes=f"Strike {object_name} through {target_name}.",
            metadata={"affordance": "strike_shot_lane", "name": name},
        )
        return {
            "support": support,
            "target": target,
            "object": ball,
            "agent": agent,
            "upper_post": upper_post,
            "lower_post": lower_post,
        }

    def create_ballistic_hoop_challenge(
        self,
        name: str,
        *,
        agent_name: str = "agent",
        object_name: str = "basketball",
        target_name: str = "hoop",
        agent_pos: VectorLike = (160.0, 150.0),
        object_pos: VectorLike = (220.0, 150.0),
        target_center: VectorLike = (440.0, 250.0),
        agent_radius: float | None = None,
        object_radius: float = 18.0,
        object_mass: float = 0.45,
        object_friction: float = 0.04,
        object_elasticity: float = 0.18,
        target_size: VectorLike = (110.0, 90.0),
        support_thickness: float = 28.0,
        support_margin: float = 120.0,
        support_friction: float = 0.65,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, ObjectRecord]:
        """Create a stable throw/lob/hoop affordance.

        This is a relation constructor for ``ballistic_object_to_region``. It
        keeps the projectile close to a generous non-blocking target sensor and
        adds readable hoop/backboard visuals without closing the arc path.
        """

        if not name:
            raise ValueError("ballistic hoop challenge name cannot be empty")
        radius = float(agent_radius if agent_radius is not None else getattr(self, "agent_radius", 15.0))
        ball_radius = float(object_radius)
        if radius <= 0.0 or ball_radius <= 0.0:
            raise ValueError("agent_radius and object_radius must be positive")
        if object_mass <= 0.0:
            raise ValueError("object_mass must be positive")

        agent_x, agent_y = self._to_pair(agent_pos)
        object_x, object_y = self._to_pair(object_pos)
        target_x, target_y = self._to_pair(target_center)
        target_width, target_height = self._to_pair(target_size)
        dx = target_x - object_x
        dy = target_y - object_y
        distance = math.hypot(dx, dy)
        if distance > 320.0:
            scale = 320.0 / distance
            target_x = object_x + dx * scale
            target_y = object_y + dy * scale

        challenge_metadata = {"kind": "ballistic_hoop_challenge", "ballistic_challenge": name}
        challenge_metadata.update(dict(metadata or {}))
        support_top = min(agent_y - radius - 2.0, object_y - ball_radius - 2.0)
        support_min_x = min(agent_x, object_x) - float(support_margin)
        support_max_x = max(agent_x, object_x, target_x) + float(support_margin) * 0.35
        support = self.create_static_box(
            f"{name}_support",
            center=((support_min_x + support_max_x) * 0.5, support_top - float(support_thickness) * 0.5),
            size=(max(1.0, support_max_x - support_min_x), float(support_thickness)),
            role="support",
            friction=float(support_friction),
            metadata={**challenge_metadata, "support": True},
        )

        target = self.create_static_box(
            target_name,
            center=(target_x, target_y),
            size=(target_width, target_height),
            role="goal",
            sensor=True,
            friction=0.0,
            metadata={**challenge_metadata, "target_region": True, "hoop_sensor": True},
        )
        backboard = self.create_static_box(
            f"{name}_backboard",
            center=(target_x + target_width * 0.55, target_y + target_height * 0.08),
            size=(12.0, target_height * 1.25),
            role="terrain",
            sensor=True,
            friction=0.0,
            metadata={**challenge_metadata, "visual_only": True, "backboard": True},
        )
        rim = self.create_static_box(
            f"{name}_rim_visual",
            center=(target_x, target_y - target_height * 0.45),
            size=(target_width * 0.78, 8.0),
            role="goal",
            sensor=True,
            friction=0.0,
            metadata={**challenge_metadata, "visual_only": True, "rim": True},
        )
        ball = self.create_dynamic_circle(
            object_name,
            pos=(object_x, object_y),
            radius=ball_radius,
            mass=float(object_mass),
            friction=float(object_friction),
            elasticity=float(object_elasticity),
            role="object",
            metadata={**challenge_metadata, "ballistic_object": True, "ball": True},
        )
        ball.body.angular_damping = 0.72
        agent = self.create_dynamic_circle(
            agent_name,
            pos=(agent_x, agent_y),
            radius=radius,
            mass=1.0,
            friction=0.65,
            elasticity=0.02,
            role="agent",
            metadata={**challenge_metadata, "throw_agent": True},
        )
        self.set_solvability_hint(
            start=(agent_x, agent_y),
            goal=(target_x, target_y),
            agent_radius=agent_radius,
            notes=f"Throw {object_name} into {target_name}.",
            metadata={"affordance": "ballistic_hoop_challenge", "name": name},
        )
        return {
            "support": support,
            "target": target,
            "object": ball,
            "agent": agent,
            "backboard": backboard,
            "rim": rim,
        }

    def create_ballistic_barrier_goal_challenge(
        self,
        name: str,
        *,
        agent_name: str = "agent",
        object_name: str = "soccer_ball",
        barrier_name: str = "wall",
        target_name: str = "goal_line",
        agent_pos: VectorLike = (190.0, 145.0),
        object_pos: VectorLike = (255.0, 145.0),
        barrier_center: VectorLike = (390.0, 205.0),
        barrier_size: VectorLike = (30.0, 105.0),
        target_center: VectorLike = (540.0, 195.0),
        agent_radius: float | None = None,
        object_radius: float = 17.0,
        object_mass: float = 0.42,
        object_friction: float = 0.035,
        object_elasticity: float = 0.16,
        target_size: VectorLike = (180.0, 160.0),
        support_thickness: float = 28.0,
        support_margin: float = 140.0,
        support_friction: float = 0.66,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, ObjectRecord]:
        """Create a stable lob/kick-over-barrier scoring affordance.

        This is a relation constructor for prompts like "kick the ball over a
        wall into a goal." It keeps the physical obstacle real, but makes the
        scoring region generous and non-blocking so validation measures the
        intended ballistic relation instead of fragile goal-post collisions.
        """

        if not name:
            raise ValueError("ballistic barrier goal challenge name cannot be empty")
        radius = float(agent_radius if agent_radius is not None else getattr(self, "agent_radius", 15.0))
        ball_radius = float(object_radius)
        if radius <= 0.0 or ball_radius <= 0.0:
            raise ValueError("agent_radius and object_radius must be positive")
        if object_mass <= 0.0:
            raise ValueError("object_mass must be positive")

        agent_x, agent_y = self._to_pair(agent_pos)
        object_x, object_y = self._to_pair(object_pos)
        barrier_x, barrier_y = self._to_pair(barrier_center)
        barrier_w, barrier_h = self._to_pair(barrier_size)
        target_x, target_y = self._to_pair(target_center)
        target_w, target_h = self._to_pair(target_size)
        # Keep the sanctioned first-pass relation deliberately solvable. The
        # visible wall still forces a lob, while the target remains generous
        # enough for deterministic validation rather than sports-rule scoring.
        barrier_w = min(float(barrier_w), 34.0)
        target_w = max(float(target_w), 180.0)
        target_h = max(float(target_h), 160.0)
        dx = target_x - object_x
        dy = target_y - object_y
        distance = math.hypot(dx, dy)
        if distance > 300.0:
            scale = 300.0 / distance
            target_x = object_x + dx * scale
            target_y = object_y + dy * scale
        metadata_base = {
            "kind": "ballistic_barrier_goal_challenge",
            "ballistic_challenge": name,
            "target_should_be_sensor": True,
        }
        metadata_base.update(dict(metadata or {}))

        support_top = min(agent_y - radius - 2.0, object_y - ball_radius - 2.0)
        support_min_x = min(agent_x, object_x, barrier_x, target_x) - float(support_margin)
        support_max_x = max(agent_x, object_x, barrier_x, target_x) + float(support_margin)
        support = self.create_static_box(
            f"{name}_support",
            center=((support_min_x + support_max_x) * 0.5, support_top - float(support_thickness) * 0.5),
            size=(max(1.0, support_max_x - support_min_x), float(support_thickness)),
            role="support",
            friction=float(support_friction),
            metadata={**metadata_base, "support": True},
        )
        barrier = self.create_static_box(
            barrier_name,
            center=(barrier_x, barrier_y),
            size=(barrier_w, barrier_h),
            role="obstacle",
            friction=0.78,
            metadata={**metadata_base, "barrier": True, "requires_clearance": True},
        )
        target = self.create_static_box(
            target_name,
            center=(target_x, target_y),
            size=(target_w, target_h),
            role="goal",
            sensor=True,
            friction=0.0,
            metadata={**metadata_base, "target_region": True, "goal_sensor": True},
        )
        post_offset = target_w * 0.54
        left_post = self.create_static_box(
            f"{name}_goal_post_left",
            center=(target_x - post_offset, target_y),
            size=(8.0, target_h * 1.15),
            role="goal",
            sensor=True,
            friction=0.0,
            metadata={**metadata_base, "visual_only": True, "goal_post": True},
        )
        right_post = self.create_static_box(
            f"{name}_goal_post_right",
            center=(target_x + post_offset, target_y),
            size=(8.0, target_h * 1.15),
            role="goal",
            sensor=True,
            friction=0.0,
            metadata={**metadata_base, "visual_only": True, "goal_post": True},
        )
        ball = self.create_dynamic_circle(
            object_name,
            pos=(object_x, object_y),
            radius=ball_radius,
            mass=float(object_mass),
            friction=float(object_friction),
            elasticity=float(object_elasticity),
            role="object",
            metadata={**metadata_base, "ballistic_object": True, "ball": True},
        )
        ball.body.angular_damping = 0.68
        agent = self.create_dynamic_circle(
            agent_name,
            pos=(agent_x, agent_y),
            radius=radius,
            mass=1.0,
            friction=0.64,
            elasticity=0.02,
            role="agent",
            metadata={**metadata_base, "kick_agent": True},
        )
        self.set_solvability_hint(
            start=(agent_x, agent_y),
            goal=(target_x, target_y),
            agent_radius=radius,
            notes=f"Kick/lob {object_name} over {barrier_name} into {target_name}.",
            metadata={"affordance": "ballistic_barrier_goal_challenge", "name": name},
        )
        return {
            "support": support,
            "barrier": barrier,
            "target": target,
            "object": ball,
            "agent": agent,
            "left_post": left_post,
            "right_post": right_post,
        }

    def create_support_exit_freefall_challenge(
        self,
        name: str,
        *,
        agent_name: str = "agent",
        object_name: str = "rock",
        boundary_name: str = "cliff_edge_boundary",
        drop_zone_name: str = "open_air_drop_zone",
        agent_x: float = 300.0,
        object_x: float = 405.0,
        edge_x: float = 540.0,
        lane_y: float = 360.0,
        agent_radius: float | None = None,
        object_radius: float = 22.0,
        object_mass: float = 1.65,
        object_friction: float = 0.18,
        support_thickness: float = 32.0,
        support_margin: float = 120.0,
        drop_height: float = 260.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, ObjectRecord]:
        """Create a stable push-over-edge/freefall affordance.

        This is a relation constructor for ``support_exit_freefall``. It stages
        the rock close enough to a real edge boundary for the validator while
        leaving open air below the edge for visible falling motion.
        """

        if not name:
            raise ValueError("support-exit challenge name cannot be empty")
        radius = float(agent_radius if agent_radius is not None else getattr(self, "agent_radius", 15.0))
        rock_radius = float(object_radius)
        if radius <= 0.0 or rock_radius <= 0.0:
            raise ValueError("agent_radius and object_radius must be positive")
        if object_mass <= 0.0:
            raise ValueError("object_mass must be positive")
        agent_x = float(agent_x)
        object_x = float(object_x)
        edge_x = float(edge_x)
        if object_x >= edge_x or edge_x - object_x > 200.0:
            object_x = edge_x - 150.0
        if agent_x >= object_x or object_x - agent_x < radius + rock_radius + 20.0:
            agent_x = object_x - max(95.0, radius + rock_radius + 45.0)

        challenge_metadata = {"kind": "support_exit_freefall_challenge", "support_exit_challenge": name}
        challenge_metadata.update(dict(metadata or {}))
        support_top = float(lane_y) - rock_radius - 2.0
        support_min_x = float(agent_x) - float(support_margin)
        support_width = max(1.0, float(edge_x) - support_min_x)
        support = self.create_static_box(
            f"{name}_support",
            center=(support_min_x + support_width * 0.5, support_top - float(support_thickness) * 0.5),
            size=(support_width, float(support_thickness)),
            role="support",
            friction=0.72,
            metadata={**challenge_metadata, "support": True, "stable_shelf": True},
        )
        boundary = self.create_static_box(
            boundary_name,
            center=(float(edge_x), float(lane_y) - rock_radius * 0.4),
            size=(28.0, max(120.0, rock_radius * 4.0)),
            role="trigger",
            sensor=True,
            friction=0.0,
            metadata={**challenge_metadata, "boundary": True, "cliff_edge": True},
        )
        drop_zone = self.create_static_box(
            drop_zone_name,
            center=(float(edge_x) + 80.0, float(lane_y) - float(drop_height) * 0.55),
            size=(240.0, max(160.0, float(drop_height))),
            role="goal",
            sensor=True,
            friction=0.0,
            metadata={**challenge_metadata, "drop_zone": True, "open_air": True},
        )
        rock = self.create_dynamic_circle(
            object_name,
            pos=(float(object_x), float(lane_y)),
            radius=rock_radius,
            mass=float(object_mass),
            friction=float(object_friction),
            elasticity=0.02,
            role="object",
            metadata={**challenge_metadata, "support_exit_object": True, "rock": True},
        )
        rock.body.angular_damping = 0.82
        agent = self.create_dynamic_circle(
            agent_name,
            pos=(float(agent_x), support_top + radius + 2.0),
            radius=radius,
            mass=1.0,
            friction=0.78,
            elasticity=0.0,
            role="agent",
            metadata={**challenge_metadata, "push_agent": True},
        )
        self.set_solvability_hint(
            start=(float(agent_x), support_top + radius + 2.0),
            goal=(float(edge_x), float(lane_y) - float(drop_height) * 0.5),
            agent_radius=agent_radius,
            notes=f"Push {object_name} across {boundary_name} so it falls through {drop_zone_name}.",
            metadata={"affordance": "support_exit_freefall_challenge", "name": name},
        )
        return {
            "support": support,
            "boundary": boundary,
            "drop_zone": drop_zone,
            "object": rock,
            "agent": agent,
        }

    def create_recurring_falling_hazards(
        self,
        name: str,
        *,
        count: int = 4,
        lane_xs: Sequence[float] | None = None,
        spawn_y: float | None = None,
        bottom_y: float | None = None,
        radius: float = 14.0,
        mass: float = 0.8,
        speed_y: float = -280.0,
        phase_gap_steps: int = 35,
        role: str = "hazard",
        name_prefix: str = "fireball",
        elasticity: float = 0.08,
        friction: float = 0.05,
        sensor: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, ObjectRecord]:
        """Create staggered, continuous falling hazards.

        This helper gives generated worlds a durable primitive for "falling
        fireballs", "raining rocks", and similar game-feel hazards without
        requiring fragile custom timer code in each generated environment.
        Hazards are released in phases, fall downward, and reset to their lane
        after exiting below the world.
        """

        if not name:
            raise ValueError("recurring hazard name cannot be empty")
        if name in self._recurring_hazards:
            raise ValueError(f"duplicate recurring hazard name: {name}")
        if count <= 0:
            raise ValueError("count must be positive")
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if mass <= 0.0:
            raise ValueError("mass must be positive")

        lanes = [float(value) for value in (lane_xs or [])]
        if not lanes:
            margin = max(float(radius) * 3.0, self.width * 0.16)
            if count == 1:
                lanes = [self.width * 0.5]
            else:
                span = max(1.0, self.width - margin * 2.0)
                lanes = [margin + span * index / (count - 1) for index in range(count)]
        while len(lanes) < count:
            lanes.append(lanes[len(lanes) % max(1, len(lanes))])
        lanes = lanes[:count]

        top_y = float(spawn_y if spawn_y is not None else self.height - float(radius) - 14.0)
        reset_bottom_y = float(bottom_y if bottom_y is not None else -float(radius) - 32.0)
        release_gap = max(1, int(phase_gap_steps))
        hazard_metadata = {
            "kind": "recurring_falling_hazard",
            "recurring_hazard": name,
            "falling": True,
            **dict(metadata or {}),
        }

        objects: dict[str, ObjectRecord] = {}
        object_names: list[str] = []
        spawn_lanes: list[tuple[float, float]] = []
        next_release_steps: dict[str, int] = {}
        active_names: set[str] = set()
        for index, lane_x in enumerate(lanes):
            object_name = f"{name_prefix}_{index + 1}"
            if object_name in self._objects:
                object_name = f"{name}_{object_name}"
            record = self.create_dynamic_circle(
                object_name,
                pos=(lane_x, top_y),
                radius=float(radius),
                mass=float(mass),
                friction=float(friction),
                elasticity=float(elasticity),
                sensor=bool(sensor),
                role=role,
                metadata={
                    **hazard_metadata,
                    "phase_index": index,
                    "phase_gap_steps": release_gap,
                    "spawn_lane": [lane_x, top_y],
                    "bottom_y": reset_bottom_y,
                    "speed_y": float(speed_y),
                },
            )
            record.body.velocity = (0.0, 0.0)
            record.body.angular_velocity = 0.0
            objects[object_name] = record
            object_names.append(object_name)
            spawn_lanes.append((lane_x, top_y))
            next_release_steps[object_name] = self._step_count + index * release_gap
            if index == 0:
                active_names.add(object_name)
                record.body.velocity = (0.0, float(speed_y))

        self._recurring_hazards[name] = RecurringHazardRecord(
            name=name,
            object_names=tuple(object_names),
            spawn_lanes=tuple(spawn_lanes),
            bottom_y=reset_bottom_y,
            speed_y=float(speed_y),
            phase_gap_steps=release_gap,
            active_names=active_names,
            next_release_steps=next_release_steps,
            metadata=hazard_metadata,
        )
        return objects

    def create_recurring_lateral_hazards(
        self,
        name: str,
        *,
        count: int = 4,
        lane_y: float,
        spawn_x: float | None = None,
        exit_x: float | None = None,
        size: VectorLike = (58.0, 28.0),
        shape: str = "box",
        mass: float = 1.0,
        speed_x: float = -220.0,
        phase_gap_steps: int = 45,
        role: str = "hazard",
        name_prefix: str = "car",
        elasticity: float = 0.0,
        friction: float = 0.2,
        sensor: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, ObjectRecord]:
        """Create staggered, continuous lateral hazards locked to one lane.

        This is the side-scroller counterpart to ``create_recurring_falling_hazards``.
        It is meant for prompts like "cars come endlessly", "trains cross the
        road", or "rolling obstacles pass through the lane". Hazards move
        horizontally at a controlled speed, reset after crossing the arena, and
        are pinned to ``lane_y`` so gravity or accidental wall contact cannot
        turn the challenge into a falling/off-screen fake.
        """

        if not name:
            raise ValueError("recurring lateral hazard name cannot be empty")
        if name in self._recurring_lateral_hazards:
            raise ValueError(f"duplicate recurring lateral hazard name: {name}")
        if count <= 0:
            raise ValueError("count must be positive")
        if mass <= 0.0:
            raise ValueError("mass must be positive")

        width, height = self._to_pair(size)
        if width <= 0.0 or height <= 0.0:
            raise ValueError("lateral hazard size values must be positive")
        speed = float(speed_x)
        if abs(speed) <= 1e-6:
            raise ValueError("speed_x must be nonzero")

        if spawn_x is None:
            spawn = self.width + width * 1.2 if speed < 0.0 else -width * 1.2
        else:
            spawn = float(spawn_x)
        if exit_x is None:
            exit_boundary = -width * 1.4 if speed < 0.0 else self.width + width * 1.4
        else:
            exit_boundary = float(exit_x)

        release_gap = max(1, int(phase_gap_steps))
        hazard_metadata = {
            "kind": "recurring_lateral_hazard",
            "recurring_lateral_hazard": name,
            "lateral": True,
            "ground_locked": True,
            "shape": str(shape or "box").lower(),
            **dict(metadata or {}),
        }
        shape_mode = str(shape or "box").lower().strip()
        use_circle = shape_mode in {"circle", "round", "ball", "boulder", "rock", "log", "barrel"}
        radius = max(width, height) * 0.5
        angular_speed = -speed / max(radius, 1.0) if use_circle else 0.0

        objects: dict[str, ObjectRecord] = {}
        object_names: list[str] = []
        spawn_lanes: list[tuple[float, float]] = []
        next_release_steps: dict[str, int] = {}
        active_names: set[str] = set()
        for index in range(count):
            object_name = f"{name_prefix}_{index + 1}"
            if object_name in self._objects:
                object_name = f"{name}_{object_name}"
            metadata_payload = {
                **hazard_metadata,
                "requested_sensor": bool(sensor),
                "phase_index": index,
                "phase_gap_steps": release_gap,
                "spawn_lane": [spawn, float(lane_y)],
                "exit_x": exit_boundary,
                "speed_x": speed,
                "angular_speed": angular_speed,
            }
            if use_circle:
                record = self.create_dynamic_circle(
                    object_name,
                    pos=(spawn, float(lane_y)),
                    radius=radius,
                    mass=float(mass),
                    friction=float(friction),
                    elasticity=float(elasticity),
                    sensor=True,
                    role=role,
                    metadata=metadata_payload,
                )
            else:
                record = self.create_dynamic_box(
                    object_name,
                    center=(spawn, float(lane_y)),
                    size=(width, height),
                    mass=float(mass),
                    friction=float(friction),
                    elasticity=float(elasticity),
                    # Recurring lane hazards need to pass through world bounds and
                    # decorative rails. Treat them as game hitboxes; objective code
                    # can still latch failure using overlap/proximity telemetry.
                    sensor=True,
                    role=role,
                    metadata=metadata_payload,
                )
            record.body.velocity = (0.0, 0.0)
            record.body.angular_velocity = 0.0
            objects[object_name] = record
            object_names.append(object_name)
            spawn_lanes.append((spawn, float(lane_y)))
            next_release_steps[object_name] = self._step_count + index * release_gap
            if index == 0:
                active_names.add(object_name)
                record.body.velocity = (speed, 0.0)
                record.body.angular_velocity = angular_speed

        self._recurring_lateral_hazards[name] = RecurringLateralHazardRecord(
            name=name,
            object_names=tuple(object_names),
            spawn_lanes=tuple(spawn_lanes),
            exit_x=exit_boundary,
            speed_x=speed,
            angular_speed=angular_speed,
            phase_gap_steps=release_gap,
            active_names=active_names,
            next_release_steps=next_release_steps,
            metadata=hazard_metadata,
        )
        return objects

    def create_readable_chaser(
        self,
        name: str,
        *,
        pos: VectorLike,
        target_name: str = "agent",
        radius: float = 16.0,
        mass: float = 0.9,
        force_strength: float = 900.0,
        max_speed: float = 135.0,
        stop_radius: float = 26.0,
        axis: str = "x",
        role: str = "chaser",
        elasticity: float = 0.0,
        friction: float = 0.2,
        sensor: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> ObjectRecord:
        """Create a dynamic enemy/hazard that visibly pursues a target.

        This is intentionally simple and deterministic: it applies bounded
        force toward the target after each high-level step and caps velocity.
        Generated worlds use it for "bear chases", "enemy squares pursue", and
        similar prompt-fidelity dynamics without hand-writing fragile
        ``after_step`` pursuit code.
        """

        record = self.create_dynamic_circle(
            name,
            pos=pos,
            radius=radius,
            mass=mass,
            role=role,
            elasticity=elasticity,
            friction=friction,
            sensor=sensor,
            metadata={
                "kind": "readable_chaser",
                "chaser": True,
                "target_name": target_name,
                "force_strength": float(force_strength),
                "max_speed": float(max_speed),
                "axis": axis,
                **dict(metadata or {}),
            },
        )
        self.register_readable_chaser(
            f"{name}_pursuit",
            chaser=name,
            target=target_name,
            force_strength=force_strength,
            max_speed=max_speed,
            stop_radius=stop_radius,
            axis=axis,
            metadata={"created_by": "create_readable_chaser"},
        )
        return record

    def register_readable_chaser(
        self,
        name: str,
        *,
        chaser: str,
        target: str = "agent",
        force_strength: float = 900.0,
        max_speed: float = 135.0,
        stop_radius: float = 26.0,
        axis: str = "x",
        metadata: Mapping[str, Any] | None = None,
    ) -> ReadableChaserRecord:
        """Register an existing dynamic object for deterministic pursuit."""

        if not name:
            raise ValueError("readable chaser name cannot be empty")
        if name in self._readable_chasers:
            raise ValueError(f"duplicate readable chaser name: {name}")
        if chaser not in self._objects:
            raise ValueError(f"unknown chaser object: {chaser}")
        if target not in self._objects:
            raise ValueError(f"unknown chaser target: {target}")
        record = self._objects[chaser]
        if record.body.body_type != pymunk.Body.DYNAMIC:
            raise ValueError("readable chaser must reference a dynamic object")
        chase_axis = str(axis or "x").lower()
        if chase_axis not in {"x", "xy"}:
            raise ValueError("readable chaser axis must be 'x' or 'xy'")
        chase_record = ReadableChaserRecord(
            name=name,
            chaser_name=str(chaser),
            target_name=str(target),
            force_strength=float(force_strength),
            max_speed=float(max_speed),
            stop_radius=float(stop_radius),
            axis=chase_axis,
            anchor_y=float(record.body.position.y) if chase_axis == "x" else None,
            metadata=dict(metadata or {}),
        )
        self._readable_chasers[name] = chase_record
        return chase_record

    def create_pressure_plate_gate_corridor(
        self,
        name: str,
        *,
        agent_name: str = "agent",
        object_name: str = "blue_box",
        plate_name: str = "pressure_plate",
        gate_name: str = "sliding_gate",
        goal_name: str = "goal",
        mechanism_name: str = "gate_mechanism",
        agent_x: float,
        object_x: float,
        plate_x: float,
        gate_x: float,
        goal_x: float,
        lane_y: float,
        agent_radius: float | None = None,
        object_size: VectorLike = (36.0, 36.0),
        object_mass: float = 1.2,
        object_friction: float = 0.3,
        plate_size: VectorLike = (86.0, 58.0),
        gate_size: VectorLike = (24.0, 132.0),
        goal_size: VectorLike = (128.0, 128.0),
        support_margin: float = 150.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a robust push-plate-gate-goal corridor affordance."""

        lane = self.create_horizontal_push_lane(
            name,
            agent_name=agent_name,
            object_name=object_name,
            target_name=plate_name,
            agent_x=agent_x,
            object_x=object_x,
            target_x=plate_x,
            lane_y=lane_y,
            agent_radius=agent_radius,
            object_size=object_size,
            object_mass=object_mass,
            object_friction=object_friction,
            target_size=plate_size,
            target_role="trigger",
            target_kind="pressure_plate",
            support_margin=max(
                float(support_margin),
                abs(float(goal_x) - float(agent_x)) + 80.0,
            ),
            metadata={"corridor": name, **dict(metadata or {})},
        )
        gate = self.create_static_box(
            gate_name,
            center=(float(gate_x), float(lane_y)),
            size=gate_size,
            role="gate",
            sensor=False,
            friction=0.8,
            metadata={"kind": "sliding_gate", "corridor": name},
        )
        goal = self.create_static_box(
            goal_name,
            center=(float(goal_x), float(lane_y)),
            size=goal_size,
            role="goal",
            sensor=True,
            friction=0.0,
            metadata={"kind": "goal_region", "corridor": name},
        )
        mechanism = self.register_pressure_plate_gate(
            mechanism_name,
            trigger=plate_name,
            gate=gate_name,
            activator=object_name,
            open_mode="sensorize",
            metadata={"corridor": name},
        )
        self.set_solvability_hint(
            start=(float(agent_x), float(lane_y)),
            goal=(float(goal_x), float(lane_y)),
            agent_radius=agent_radius,
            notes=f"Push {object_name} onto {plate_name}, open {gate_name}, then reach {goal_name}.",
            metadata={"affordance": "pressure_plate_gate_corridor", "name": name},
        )
        return {**lane, "gate": gate, "goal": goal, "mechanism": mechanism}

    def create_field_push_lane(
        self,
        name: str,
        *,
        agent_name: str = "agent",
        object_name: str = "charged_ball",
        field_name: str = "force_zone",
        target_name: str = "target",
        agent_x: float,
        object_x: float,
        field_x: float,
        target_x: float,
        lane_y: float,
        agent_radius: float | None = None,
        object_radius: float = 18.0,
        object_mass: float = 1.0,
        object_friction: float = 0.25,
        field_size: VectorLike = (110.0, 90.0),
        target_size: VectorLike = (100.0, 100.0),
        force: VectorLike = (1.0, 0.0),
        mode: str = "constant",
        strength: float = 3800.0,
        support_thickness: float = 28.0,
        support_margin: float = 150.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, ObjectRecord]:
        """Create a stable push lane feeding a dynamic object into a force zone."""

        radius = float(agent_radius if agent_radius is not None else getattr(self, "agent_radius", 15.0))
        if (float(agent_x) - float(object_x)) * (float(field_x) - float(object_x)) >= 0.0:
            raise ValueError("field push lane must stage the object between agent_x and field_x")
        lane_metadata = {"kind": "field_push_lane", "push_lane": name}
        lane_metadata.update(dict(metadata or {}))
        support_top = float(lane_y) - float(object_radius) - 2.0
        support_min_x = min(float(agent_x), float(object_x), float(field_x), float(target_x)) - float(support_margin)
        support_max_x = max(float(agent_x), float(object_x), float(field_x), float(target_x)) + float(support_margin)
        support = self.create_static_box(
            f"{name}_support",
            center=((support_min_x + support_max_x) * 0.5, support_top - float(support_thickness) * 0.5),
            size=(support_max_x - support_min_x, float(support_thickness)),
            role="support",
            friction=0.9,
            metadata={**lane_metadata, "support": True},
        )
        field = self.register_force_zone(
            field_name,
            center=(float(field_x), float(lane_y)),
            size=field_size,
            force=force,
            mode=mode,
            strength=strength,
            affected_names=[object_name],
            metadata={**lane_metadata, "field": True},
        )
        target = self.create_static_box(
            target_name,
            center=(float(target_x), float(lane_y)),
            size=target_size,
            role="goal",
            sensor=True,
            friction=0.0,
            metadata={**lane_metadata, "target_region": True},
        )
        obj = self.create_dynamic_circle(
            object_name,
            pos=(float(object_x), float(lane_y)),
            radius=float(object_radius),
            mass=float(object_mass),
            friction=float(object_friction),
            elasticity=0.0,
            metadata={**lane_metadata, "field_object": True},
        )
        agent = self.create_dynamic_circle(
            agent_name,
            pos=(float(agent_x), support_top + radius + 2.0),
            radius=radius,
            mass=1.0,
            friction=0.75,
            elasticity=0.0,
            role="agent",
            metadata={**lane_metadata, "lane_agent": True},
        )
        self.set_solvability_hint(
            start=(float(agent_x), float(lane_y)),
            goal=(float(target_x), float(lane_y)),
            agent_radius=agent_radius,
            notes=f"Push {object_name} into {field_name}; field carries it toward {target_name}.",
            metadata={"affordance": "field_push_lane", "name": name},
        )
        return {"support": support, "field": field, "target": target, "object": obj, "agent": agent}

    def register_pressure_plate_gate(
        self,
        name: str,
        *,
        trigger: str,
        gate: str,
        activator: str | Sequence[str],
        activation_distance: float | None = None,
        open_mode: str = "sensorize",
        metadata: Mapping[str, Any] | None = None,
    ) -> MechanismRecord:
        """Register deterministic pressure-plate semantics for a passable gate.

        Generated environments should use this instead of inventing collision
        callbacks. The mechanism opens once any activator enters the trigger
        sensor or falls within ``activation_distance``.
        """

        if not name:
            raise ValueError("mechanism name cannot be empty")
        if name in self._mechanisms:
            raise ValueError(f"duplicate registered mechanism name: {name}")
        if trigger not in self._objects:
            raise KeyError(f"unknown mechanism trigger: {trigger}")
        if gate not in self._objects:
            raise KeyError(f"unknown mechanism gate: {gate}")
        if isinstance(activator, str):
            activator_names = (activator,)
        else:
            activator_names = tuple(str(item) for item in activator)
        if not activator_names:
            raise ValueError("register_pressure_plate_gate requires at least one activator")
        missing = [item for item in activator_names if item not in self._objects]
        if missing:
            raise KeyError(f"unknown mechanism activator(s): {missing}")

        normalized_mode = open_mode.replace("-", "_").lower()
        if normalized_mode not in {"sensorize", "sensor", "passable", "none"}:
            raise ValueError(f"unsupported gate open mode: {open_mode}")
        record = MechanismRecord(
            name=name,
            trigger_name=str(trigger),
            gate_name=str(gate),
            activator_names=activator_names,
            activation_distance=activation_distance,
            open_mode=normalized_mode,
            metadata=dict(metadata or {}),
        )
        self._mechanisms[name] = record
        self._update_mechanisms()
        return record

    def is_mechanism_activated(self, name: str) -> bool:
        """Return whether a named deterministic mechanism has activated."""

        self._update_mechanisms()
        record = self._mechanisms.get(name)
        if record is None:
            return False
        return bool(record.activated)

    def is_object_on_trigger(
        self,
        trigger: str,
        activator: str,
        *,
        activation_distance: float | None = None,
    ) -> bool:
        """Return whether an activator currently satisfies a trigger region."""

        trigger_record = self.get_object(trigger)
        activator_record = self.get_object(activator)
        return self._trigger_contains_activator(
            trigger_record,
            activator_record,
            activation_distance=activation_distance,
        )

    def _build_constraint_from_spec(
        self,
        *,
        type: str | None,
        body_a: pymunk.Body | ObjectRecord | str | None,
        body_b: pymunk.Body | ObjectRecord | str | None,
        anchor_a: VectorLike | None,
        anchor_b: VectorLike | None,
    ) -> pymunk.Constraint:
        """Build common joints from the environment_spec.json shorthand."""

        if not type:
            raise ValueError("register_constraint requires type when constraint is omitted")
        resolved_a = self._resolve_constraint_body(body_a)
        resolved_b = self._resolve_constraint_body(body_b)
        if anchor_a is None:
            raise ValueError("register_constraint requires anchor_a")

        joint_type = type.replace("_", "").replace("-", "").lower()
        if joint_type in {"pivot", "pivotjoint"}:
            if anchor_b is None:
                return pymunk.PivotJoint(resolved_a, resolved_b, self._to_vec2d(anchor_a))
            return pymunk.PivotJoint(
                resolved_a,
                resolved_b,
                self._to_vec2d(anchor_a),
                self._to_vec2d(anchor_b),
            )
        if joint_type in {"pin", "pinjoint"}:
            if anchor_b is None:
                raise ValueError("PinJoint requires anchor_b")
            return pymunk.PinJoint(
                resolved_a,
                resolved_b,
                self._to_vec2d(anchor_a),
                self._to_vec2d(anchor_b),
            )
        if joint_type in {"slide", "slidejoint"}:
            if anchor_b is None:
                raise ValueError("SlideJoint requires anchor_b")
            return pymunk.SlideJoint(
                resolved_a,
                resolved_b,
                self._to_vec2d(anchor_a),
                self._to_vec2d(anchor_b),
                0.0,
                120.0,
            )
        raise ValueError(f"unsupported constraint type: {type}")

    def _resolve_constraint_body(
        self,
        body_ref: pymunk.Body | ObjectRecord | str | None,
    ) -> pymunk.Body:
        if body_ref is None:
            return self.space.static_body
        if isinstance(body_ref, pymunk.Body):
            return body_ref
        if isinstance(body_ref, ObjectRecord):
            return body_ref.body
        if isinstance(body_ref, str):
            return self.get_object(body_ref).body
        raise TypeError(f"unsupported constraint body reference: {body_ref!r}")

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

    def get_agent_record(self) -> ObjectRecord | None:
        """Return the registered dynamic agent object, when present."""

        for record in self._objects.values():
            if record.role == "agent" and record.body.body_type == pymunk.Body.DYNAMIC:
                return record
        return None

    def apply_agent_force(
        self,
        direction: VectorLike,
        *,
        strength: float | None = None,
    ) -> None:
        """Apply adaptive movement force to the registered agent."""

        agent = self.get_agent_record()
        if agent is None:
            return
        vector = self._to_vec2d(direction)
        if vector.length <= 0.0:
            return
        applied_strength = self.agent_strength if strength is None else strength
        vector = self._govern_agent_control_vector(agent, vector)
        if vector.length <= 0.0:
            return
        force = vector.normalized() * applied_strength
        agent.body.apply_force_at_world_point(force, agent.body.position)

    def _agent_uses_freeflight_controls(self) -> bool:
        """Return whether the agent is allowed continuous vertical thrust."""

        gravity = self._to_vec2d(self.config.gravity).length
        if gravity <= 30.0:
            return True
        profile = self._capability_profile_to_ground_truth()
        text = " ".join(
            str(profile.get(key, "")).lower()
            for key in ("movement", "gravity", "notes")
        )
        freeflight_tokens = (
            "thrust",
            "freeflight",
            "free_flight",
            "zero_g",
            "zero-g",
            "flying",
            "flight",
            "spaceship",
            "hover",
        )
        return any(token in text for token in freeflight_tokens)

    def _govern_agent_control_vector(
        self,
        agent: ObjectRecord,
        vector: pymunk.Vec2d,
    ) -> pymunk.Vec2d:
        """Keep controls consistent with the world's declared physics law."""

        if self._agent_uses_freeflight_controls():
            return vector

        # In normal-gravity ground worlds, continuous upward force reads as
        # hover/thruster motion. Convert upward input into an occasional
        # grounded jump impulse and keep sustained movement horizontal.
        horizontal = pymunk.Vec2d(float(vector.x), 0.0)
        if vector.y > 0.35 and self._agent_is_grounded(agent):
            if self._step_count - self._last_agent_jump_step >= 12:
                gravity = max(self._to_vec2d(self.config.gravity).length, 981.0)
                jump_height = 84.0
                impulse_speed = math.sqrt(2.0 * gravity * jump_height)
                agent.body.apply_impulse_at_world_point(
                    pymunk.Vec2d(0.0, agent.body.mass * impulse_speed),
                    agent.body.position,
                )
                self._last_agent_jump_step = self._step_count
        return horizontal

    def _agent_is_grounded(self, agent: ObjectRecord, *, tolerance: float = 7.0) -> bool:
        """Approximate whether an agent is standing on static support."""

        agent_bbs = [shape.cache_bb() for shape in agent.shapes]
        if not agent_bbs:
            return False
        agent_left = min(bb.left for bb in agent_bbs)
        agent_right = max(bb.right for bb in agent_bbs)
        agent_bottom = min(bb.bottom for bb in agent_bbs)
        for record in self._objects.values():
            if record is agent or record.body.body_type != pymunk.Body.STATIC:
                continue
            for shape in record.shapes:
                if getattr(shape, "sensor", False):
                    continue
                bb = shape.cache_bb()
                horizontal_overlap = min(agent_right, bb.right) - max(agent_left, bb.left)
                if horizontal_overlap <= 1.0:
                    continue
                vertical_gap = agent_bottom - bb.top
                if -1.5 <= vertical_gap <= tolerance:
                    return True
        return False

    def _new_space(self) -> pymunk.Space:
        space = pymunk.Space(threaded=False)
        space.gravity = self.config.gravity
        space.damping = self.config.damping
        space.iterations = self.config.iterations
        space.collision_slop = 0.05
        return space

    def _configure_dynamic_body(self, body: pymunk.Body) -> None:
        body.angular_damping = self.config.angular_damping

    def _update_agent_strength(self) -> None:
        max_mass = 1.0
        for record in self._objects.values():
            body = record.body
            if body.body_type != pymunk.Body.DYNAMIC:
                continue
            if math.isfinite(body.mass):
                max_mass = max(max_mass, float(body.mass))
        gravity = self._to_vec2d(self.config.gravity).length
        effective_acceleration = max(gravity, 981.0)
        self.agent_strength = max_mass * effective_acceleration * 2.0

    def _apply_dynamic_angular_damping(self, dt: float) -> None:
        exponent = max(dt * 60.0, 0.0)
        for body in self.space.bodies:
            if body.body_type != pymunk.Body.DYNAMIC:
                continue
            angular_damping = float(getattr(body, "angular_damping", self.config.angular_damping))
            angular_damping = max(0.0, min(1.0, angular_damping))
            body.angular_velocity *= angular_damping**exponent

    def _apply_force_zones(self) -> None:
        if not self._force_zones:
            return
        for zone in self._force_zones.values():
            zone_record = self._objects.get(zone.object_name)
            if zone_record is None:
                continue
            for record in self._objects.values():
                if record is zone_record or record.body.body_type != pymunk.Body.DYNAMIC:
                    continue
                if not self._force_zone_affects_record(zone, record):
                    continue
                if not self._record_inside_zone(zone_record, record):
                    continue
                force = self._force_for_zone(zone, zone_record, record)
                if force.length > 0.0:
                    record.body.apply_force_at_world_point(force, record.body.position)

    def _update_recurring_falling_hazards(self) -> None:
        if not self._recurring_hazards:
            return
        for hazard in self._recurring_hazards.values():
            for index, object_name in enumerate(hazard.object_names):
                record = self._objects.get(object_name)
                if record is None:
                    continue
                spawn_x, spawn_y = hazard.spawn_lanes[index]
                body = record.body
                if object_name not in hazard.active_names:
                    if self._step_count >= hazard.next_release_steps.get(object_name, 0):
                        hazard.active_names.add(object_name)
                        body.position = (spawn_x, spawn_y)
                        body.velocity = (0.0, hazard.speed_y)
                        body.angular_velocity = 0.0
                    else:
                        body.position = (spawn_x, spawn_y)
                        body.velocity = (0.0, 0.0)
                        body.angular_velocity = 0.0
                    continue
                if float(body.position.y) <= hazard.bottom_y:
                    hazard.active_names.discard(object_name)
                    hazard.next_release_steps[object_name] = (
                        self._step_count
                        + hazard.phase_gap_steps * max(1, len(hazard.object_names))
                    )
                    body.position = (spawn_x, spawn_y)
                    body.velocity = (0.0, 0.0)
                    body.angular_velocity = 0.0
                    continue
                if abs(float(body.velocity.y)) < abs(hazard.speed_y) * 0.35:
                    body.velocity = (float(body.velocity.x), hazard.speed_y)

    def _update_recurring_lateral_hazards(self) -> None:
        if not self._recurring_lateral_hazards:
            return
        for hazard in self._recurring_lateral_hazards.values():
            moving_left = hazard.speed_x < 0.0
            for index, object_name in enumerate(hazard.object_names):
                record = self._objects.get(object_name)
                if record is None:
                    continue
                spawn_x, lane_y = hazard.spawn_lanes[index]
                body = record.body
                if object_name not in hazard.active_names:
                    if self._step_count >= hazard.next_release_steps.get(object_name, 0):
                        hazard.active_names.add(object_name)
                        body.position = (spawn_x, lane_y)
                        body.velocity = (hazard.speed_x, 0.0)
                        body.angular_velocity = hazard.angular_speed
                        body.angle = 0.0
                    else:
                        body.position = (spawn_x, lane_y)
                        body.velocity = (0.0, 0.0)
                        body.angular_velocity = 0.0
                        body.angle = 0.0
                    continue

                crossed_exit = (
                    float(body.position.x) <= hazard.exit_x
                    if moving_left
                    else float(body.position.x) >= hazard.exit_x
                )
                if crossed_exit:
                    hazard.active_names.discard(object_name)
                    hazard.next_release_steps[object_name] = (
                        self._step_count
                        + hazard.phase_gap_steps * max(1, len(hazard.object_names))
                    )
                    body.position = (spawn_x, lane_y)
                    body.velocity = (0.0, 0.0)
                    body.angular_velocity = 0.0
                    body.angle = 0.0
                    continue

                body.position = (float(body.position.x), lane_y)
                body.velocity = (hazard.speed_x, 0.0)
                body.angular_velocity = hazard.angular_speed
                if abs(hazard.angular_speed) <= 1e-6:
                    body.angle = 0.0

    def _update_readable_chasers(self) -> None:
        if not self._readable_chasers:
            return
        for chase in self._readable_chasers.values():
            chaser = self._objects.get(chase.chaser_name)
            target = self._objects.get(chase.target_name)
            if chaser is None or target is None:
                continue
            body = chaser.body
            if body.body_type != pymunk.Body.DYNAMIC:
                continue
            if chase.axis == "x" and chase.anchor_y is not None:
                body.position = (float(body.position.x), chase.anchor_y)
                body.velocity = (float(body.velocity.x), 0.0)
                target_position = pymunk.Vec2d(float(target.body.position.x), chase.anchor_y)
            else:
                target_position = target.body.position
            delta = target_position - body.position
            distance = float(delta.length)
            if distance <= max(0.0, chase.stop_radius) or distance <= 1e-6:
                body.velocity *= 0.92
                continue
            direction = delta.normalized()
            body.apply_force_at_world_point(
                direction * max(0.0, chase.force_strength),
                body.position,
            )
            max_speed = max(1.0, chase.max_speed)
            if float(body.velocity.length) > max_speed:
                body.velocity = body.velocity.normalized() * max_speed

    def _force_zone_affects_record(
        self,
        zone: ForceZoneRecord,
        record: ObjectRecord,
    ) -> bool:
        if zone.affected_names and record.name not in zone.affected_names:
            return False
        if zone.affected_roles and str(record.role or "") not in zone.affected_roles:
            return False
        return True

    def _record_inside_zone(
        self,
        zone_record: ObjectRecord,
        record: ObjectRecord,
    ) -> bool:
        point = record.body.position
        for shape in zone_record.shapes:
            if shape.point_query(point).distance <= 0.0:
                return True
        return False

    def _force_for_zone(
        self,
        zone: ForceZoneRecord,
        zone_record: ObjectRecord,
        record: ObjectRecord,
    ) -> pymunk.Vec2d:
        if zone.mode == "constant":
            if zone.force.length <= 1e-6:
                return pymunk.Vec2d(0.0, 0.0)
            return zone.force.normalized() * max(float(zone.strength), 0.0)
        delta = zone_record.body.position - record.body.position
        if delta.length <= 1e-6:
            return pymunk.Vec2d(0.0, 0.0)
        direction = delta.normalized()
        if zone.mode == "repel":
            direction = -direction
        strength = max(float(zone.strength), 0.0)
        if zone.falloff > 0.0:
            strength /= max(1.0, 1.0 + zone.falloff * float(delta.length))
        return direction * strength

    def _update_mechanisms(self) -> None:
        if not self._mechanisms:
            return
        for mechanism in self._mechanisms.values():
            if not mechanism.activated and self._mechanism_triggered(mechanism):
                mechanism.activated = True
            if mechanism.activated:
                self._apply_mechanism_open_state(mechanism)

    def _mechanism_triggered(self, mechanism: MechanismRecord) -> bool:
        trigger = self._objects.get(mechanism.trigger_name)
        if trigger is None:
            return False
        for activator_name in mechanism.activator_names:
            activator = self._objects.get(activator_name)
            if activator is None:
                continue
            if self._trigger_contains_activator(
                trigger,
                activator,
                activation_distance=mechanism.activation_distance,
            ):
                return True
        return False

    def _trigger_contains_activator(
        self,
        trigger: ObjectRecord,
        activator: ObjectRecord,
        *,
        activation_distance: float | None = None,
    ) -> bool:
        point = activator.body.position
        tolerance = 3.0 if activation_distance is None else max(float(activation_distance), 0.0)
        for shape in trigger.shapes:
            if shape.point_query(point).distance <= tolerance:
                return True
        if activation_distance is not None:
            return (
                float(trigger.body.position.get_distance(point))
                <= max(float(activation_distance), 0.0)
            )
        return False

    def _apply_mechanism_open_state(self, mechanism: MechanismRecord) -> None:
        if mechanism.open_mode == "none":
            return
        gate = self._objects.get(mechanism.gate_name)
        if gate is None:
            return
        if mechanism.open_mode in {"sensorize", "sensor", "passable"}:
            for shape in gate.shapes:
                shape.sensor = True
                shape.friction = 0.0
            gate.metadata["mechanism_open"] = True
            gate.metadata["opened_by"] = mechanism.name
            gate.metadata["passable"] = True

    def _apply_constraint_collision_group(
        self,
        constraint: pymunk.Constraint,
        *,
        group: int,
    ) -> None:
        joint_bodies = {constraint.a, constraint.b}
        for record in self._objects.values():
            if record.body not in joint_bodies:
                continue
            for shape in record.shapes:
                shape.filter = pymunk.ShapeFilter(group=group)

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

    def _constraint_to_ground_truth(self, record: ConstraintRecord) -> dict[str, Any]:
        constraint = record.constraint
        return {
            "type": constraint.__class__.__name__,
            "metadata": self._json_safe(record.metadata),
            "max_force": self._finite_or_none(constraint.max_force),
            "error_bias": self._finite_or_none(constraint.error_bias),
            "max_bias": self._finite_or_none(constraint.max_bias),
            "impulse": self._finite_or_none(constraint.impulse),
        }

    def _force_zone_to_ground_truth(self, record: ForceZoneRecord) -> dict[str, Any]:
        return {
            "object_name": record.object_name,
            "mode": record.mode,
            "force": self._vec_to_list(record.force),
            "strength": self._finite_or_none(record.strength),
            "affected_names": list(record.affected_names),
            "affected_roles": list(record.affected_roles),
            "falloff": self._finite_or_none(record.falloff),
            "metadata": self._json_safe(record.metadata),
        }

    def _mechanism_to_ground_truth(self, record: MechanismRecord) -> dict[str, Any]:
        return {
            "trigger": record.trigger_name,
            "gate": record.gate_name,
            "activators": list(record.activator_names),
            "activation_distance": self._finite_or_none(record.activation_distance)
            if record.activation_distance is not None
            else None,
            "open_mode": record.open_mode,
            "activated": bool(record.activated),
            "metadata": self._json_safe(record.metadata),
        }

    def _recurring_hazard_to_ground_truth(
        self,
        record: RecurringHazardRecord,
    ) -> dict[str, Any]:
        return {
            "objects": list(record.object_names),
            "spawn_lanes": [
                [self._finite_or_none(x), self._finite_or_none(y)]
                for x, y in record.spawn_lanes
            ],
            "bottom_y": self._finite_or_none(record.bottom_y),
            "speed_y": self._finite_or_none(record.speed_y),
            "phase_gap_steps": int(record.phase_gap_steps),
            "active_names": sorted(record.active_names),
            "next_release_steps": {
                name: int(step) for name, step in record.next_release_steps.items()
            },
            "metadata": self._json_safe(record.metadata),
        }

    def _recurring_lateral_hazard_to_ground_truth(
        self,
        record: RecurringLateralHazardRecord,
    ) -> dict[str, Any]:
        return {
            "objects": list(record.object_names),
            "spawn_lanes": [
                [self._finite_or_none(x), self._finite_or_none(y)]
                for x, y in record.spawn_lanes
            ],
            "exit_x": self._finite_or_none(record.exit_x),
            "speed_x": self._finite_or_none(record.speed_x),
            "angular_speed": self._finite_or_none(record.angular_speed),
            "phase_gap_steps": int(record.phase_gap_steps),
            "active_names": sorted(record.active_names),
            "next_release_steps": {
                name: int(step) for name, step in record.next_release_steps.items()
            },
            "metadata": self._json_safe(record.metadata),
        }

    def _readable_chaser_to_ground_truth(
        self,
        record: ReadableChaserRecord,
    ) -> dict[str, Any]:
        return {
            "chaser": record.chaser_name,
            "target": record.target_name,
            "force_strength": self._finite_or_none(record.force_strength),
            "max_speed": self._finite_or_none(record.max_speed),
            "stop_radius": self._finite_or_none(record.stop_radius),
            "axis": record.axis,
            "anchor_y": self._finite_or_none(record.anchor_y)
            if record.anchor_y is not None
            else None,
            "metadata": self._json_safe(record.metadata),
        }

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
        if isinstance(value, set):
            return sorted((cls._json_safe(item) for item in value), key=str)
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
