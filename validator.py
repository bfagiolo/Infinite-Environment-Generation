"""Headless solvability oracle for Harness Alpha generated environments."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, deque
from dataclasses import dataclass, field, replace
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any, Iterable

import pymunk

from base_env import BaseEnv
from validator_probes.containment import object_inside_region_probe
from validator_probes.contact import contact_or_proximity_probe
from validator_probes.contract import ProbeResult
from validator_probes.field import field_effect_probe
from validator_probes.lever import launch_progress_probe, pivot_mechanism_probe, plank_rotation_probe
from validator_probes.motion import agent_target_motion_probe, object_region_motion_probe
from validator_probes.plans import probe_plan_for_subgoal
from validator_probes.push import (
    agent_motion_under_force_probe,
    box_blockage_probe,
    box_motion_probe,
    collision_filter_probe,
    push_contact_probe,
    push_force_probe,
    push_impulse_probe,
)
from validator_probes.reachability import path_reachability_probe
from validator_probes.spatial import concrete_target_probe, object_region_affordance_probe
from validator_probes.stability import passive_stability_probe


GENERATED_ENVS_DIR = Path("generated_envs")
SOLID_ROLES = {"terrain", "obstacle", "hazard"}
PASS_THROUGH_ROLES = {"agent", "goal"}
TIER_NAMES = {
    0: "invalid",
    1: "structurally_valid",
    2: "physically_plausible",
    3: "agent_actionable",
    4: "progress_verified",
    5: "solved_verified",
}
DEFAULT_MINIMUM_ACCEPTANCE_TIER = {
    "navigation_goal": 5,
    "single_target_touch": 5,
    "multi_target_touch": 5,
    "push_object": 5,
    "seesaw_balance": 4,
    "mechanism_activation": 4,
    "survival": 5,
    "custom_physics": 4,
}


Point = tuple[float, float]
GridCell = tuple[int, int]


def _probe_evidence_tier(details: dict[str, Any]) -> int:
    """Return the strongest non-solving tier supported by structured probes."""

    tiers: list[int] = []
    for probe in _collect_probe_dicts(details):
        try:
            tier = int(probe.get("tier_evidence", 0))
        except (TypeError, ValueError):
            continue
        if tier > 0:
            tiers.append(tier)
    return max(tiers, default=0)


def _collect_probe_dicts(value: Any, *, _depth: int = 0) -> list[dict[str, Any]]:
    """Recursively collect serialized ProbeResult dictionaries."""

    if _depth > 8:
        return []
    if isinstance(value, dict):
        if isinstance(value.get("name"), str) and "passed" in value and "tier_evidence" in value:
            return [value]
        probes: list[dict[str, Any]] = []
        for child in value.values():
            probes.extend(_collect_probe_dicts(child, _depth=_depth + 1))
        return probes
    if isinstance(value, list | tuple):
        probes: list[dict[str, Any]] = []
        for child in value:
            probes.extend(_collect_probe_dicts(child, _depth=_depth + 1))
        return probes
    return []


@dataclass(frozen=True)
class ValidatorConfig:
    """Configuration for headless rollout and grid search."""

    grid_size: float = 24.0
    agent_radius: float = 12.0
    simulation_steps: int = 1
    substeps: int = 1
    include_dynamic_blockers: bool = False
    max_cells: int = 250_000
    kinetic_validation: bool = True
    kinetic_steps: int = 300
    kinetic_displacement_threshold: float = 5.0
    kinetic_substeps: int = 5
    tier_policy: dict[str, Any] = field(default_factory=dict)
    expected_physics_relations: dict[str, Any] | None = None
    expected_layout_plan: dict[str, Any] | None = None


@dataclass(frozen=True)
class Blocker:
    """A solid shape projected from get_ground_truth() telemetry."""

    object_name: str
    role: str | None
    shape_type: str
    bb: dict[str, float]
    data: dict[str, Any]


@dataclass(frozen=True)
class ValidationResult:
    """Outcome returned by the headless oracle."""

    solvable: bool
    reason: str
    env_class: str
    path: tuple[Point, ...] = ()
    visited_cells: int = 0
    blocking_object: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    contract_valid: bool = False
    structurally_valid: bool = False
    objective_valid: bool = False
    kinetic_progress: bool = False
    kinetic_solved: bool = False

    @property
    def achieved_tier(self) -> int:
        """Return the strongest verification tier supported by this result."""

        if self.kinetic_solved:
            return 5
        probe_tier = _probe_evidence_tier(self.details)
        probe_tier = min(probe_tier, 4)
        if self.kinetic_progress:
            return max(4, probe_tier)
        if self.details.get("agent_actionable"):
            return max(3, probe_tier)
        if self.objective_valid or self.details.get("physically_plausible"):
            return max(2, probe_tier)
        if self.structurally_valid:
            return max(1, probe_tier)
        return 0

    @property
    def tier_name(self) -> str:
        """Return the human-readable verification tier name."""

        return _tier_name(self.achieved_tier)

    @property
    def minimum_acceptance_tier(self) -> int:
        """Return the required tier for this environment's objective profile."""

        return _minimum_acceptance_tier(self.details)

    @property
    def objective_tier(self) -> int:
        """Return the ideal proof tier this task deserves in principle."""

        return _objective_tier(self.details)

    @property
    def operational_acceptance_tier(self) -> int:
        """Return the current enforceable acceptance tier."""

        return self.minimum_acceptance_tier

    @property
    def verification_gap(self) -> int:
        """Return how far the run is below the ideal objective tier."""

        return max(0, self.objective_tier - self.achieved_tier)

    @property
    def accepted(self) -> bool:
        """Return whether the result meets the objective's acceptance tier."""

        return self.achieved_tier >= self.minimum_acceptance_tier

    def to_dict(self) -> dict[str, Any]:
        return {
            "solvable": self.solvable,
            "accepted": self.accepted,
            "achieved_tier": self.achieved_tier,
            "tier_name": self.tier_name,
            "objective_tier": self.objective_tier,
            "operational_acceptance_tier": self.operational_acceptance_tier,
            "minimum_acceptance_tier": self.minimum_acceptance_tier,
            "verification_gap": self.verification_gap,
            "contract_valid": self.contract_valid,
            "structurally_valid": self.structurally_valid,
            "objective_valid": self.objective_valid,
            "kinetic_progress": self.kinetic_progress,
            "kinetic_solved": self.kinetic_solved,
            "reason": self.reason,
            "env_class": self.env_class,
            "path": [list(point) for point in self.path],
            "visited_cells": self.visited_cells,
            "blocking_object": self.blocking_object,
            "details": self.details,
        }

    def with_layers(self, **layers: bool) -> "ValidationResult":
        """Return a copy with updated validation-layer booleans."""

        return replace(self, **layers)


def validate_generated_env(
    env_path: str | Path,
    *,
    class_name: str | None = None,
    config: ValidatorConfig | None = None,
    tier_policy: dict[str, Any] | None = None,
) -> ValidationResult:
    """Load a generated environment, run headless physics, and check reachability."""

    validator_config = config or ValidatorConfig()
    env_class = load_env_class(env_path, class_name=class_name)
    env = env_class()
    if validator_config.expected_physics_relations:
        env.physics_relations = dict(validator_config.expected_physics_relations)
    if validator_config.expected_layout_plan:
        env.layout_plan = dict(validator_config.expected_layout_plan)

    active_tier_policy = tier_policy or validator_config.tier_policy or None

    if callable(getattr(env, "check_objective", None)):
        return verify_objective_trial(env, validator_config, tier_policy=active_tier_policy)

    for _ in range(validator_config.simulation_steps):
        env.step(substeps=validator_config.substeps)

    ground_truth = env.get_ground_truth()
    geometric_result = validate_ground_truth(ground_truth, config=validator_config)
    if not geometric_result.solvable:
        return geometric_result
    if not validator_config.kinetic_validation:
        return geometric_result

    kinetic_result = verify_physical_interaction(env_class(), geometric_result, validator_config)
    if kinetic_result is not None:
        return kinetic_result
    return geometric_result.with_layers(
        kinetic_progress=geometric_result.solvable,
        kinetic_solved=geometric_result.solvable,
    )


def verify_objective_trial(
    env: BaseEnv,
    config: ValidatorConfig | None = None,
    *,
    tier_policy: dict[str, Any] | None = None,
) -> ValidationResult:
    """Run a headless kinetic trial and pass only if check_objective becomes True."""

    validator_config = config or ValidatorConfig()
    env_class = env.__class__.__name__
    structurally_valid, structural_reason, structural_details = _objective_structural_check(env)
    if not structurally_valid:
        if tier_policy:
            structural_details = {**structural_details, "tier_policy": tier_policy}
        contract_valid = not bool(structural_details.get("contract_errors"))
        return ValidationResult(
            False,
            structural_reason,
            env_class,
            details=structural_details,
            contract_valid=contract_valid,
            structurally_valid=False,
            objective_valid=False,
            kinetic_progress=False,
            kinetic_solved=False,
        )

    if tier_policy:
        structural_details = {**structural_details, "tier_policy": tier_policy}
    validator_route = _select_validator_route(structural_details)
    structural_details = {**structural_details, "validator_route": validator_route}

    semantic_failure = _semantic_validation_result(env, validator_config, structural_details)
    if semantic_failure is not None:
        return semantic_failure
    gameplay_failure = _gameplay_validation_result(env, validator_config, structural_details)
    if gameplay_failure is not None:
        return gameplay_failure
    anti_cheat_failure = _anti_cheat_validation_result(env, validator_config, structural_details)
    if anti_cheat_failure is not None:
        return anti_cheat_failure

    target = _find_kinetic_target(env, None)
    agent = env.get_agent_record()
    start_position = _copy_vec2d(target.body.position) if target is not None else None
    agent_start_position = _copy_vec2d(agent.body.position) if agent is not None else None

    objective_state, objective_error = _evaluate_objective(env)
    if objective_error is not None:
        return ValidationResult(
            False,
            objective_error,
            env_class,
            details={"objective": "check_objective", **structural_details},
            contract_valid=True,
            structurally_valid=structurally_valid,
            objective_valid=False,
            kinetic_progress=False,
            kinetic_solved=False,
        )
    if objective_state:
        return ValidationResult(
            True,
            "Code-level objective satisfied at initial state",
            env_class,
            details={"objective": "check_objective", **structural_details},
            contract_valid=True,
            structurally_valid=structurally_valid,
            objective_valid=True,
            kinetic_progress=True,
            kinetic_solved=True,
        )

    if validator_route["oracle"] == "subgoal_plan":
        if not _route_allows_force_control(validator_route):
            return _capability_mismatch_result(
                env_class,
                structural_details,
                "Capability profile does not allow force controls required for subgoal validation.",
            )
        return verify_subgoal_plan(env, validator_config, structural_details)

    if validator_route["oracle"] in {"multi_target_touch", "single_target_touch"}:
        if not _route_allows_force_control(validator_route):
            return _capability_mismatch_result(
                env_class,
                structural_details,
                "Capability profile does not allow force controls required for touch validation.",
            )
        return verify_multi_target_touch(env, validator_config, structural_details)

    if validator_route["oracle"] == "navigation_bfs":
        navigation_result = validate_ground_truth(env.get_ground_truth(), config=validator_config)
        if not navigation_result.structurally_valid:
            return replace(
                navigation_result,
                reason=f"Navigation precheck failed: {navigation_result.reason}",
                details={**structural_details, **navigation_result.details},
                objective_valid=True,
            )

    for step_index in range(validator_config.kinetic_steps):
        action = None
        if agent is not None and _route_allows_force_control(validator_route):
            direction = _objective_action_direction(env, agent, target)
            action = {"move": (float(direction.x), float(direction.y))}
        try:
            env.step(action=action, substeps=validator_config.kinetic_substeps)
        except Exception as exc:
            return ValidationResult(
                False,
                f"Simulation/objective runtime failed: {type(exc).__name__}: {exc}",
                env_class,
                blocking_object=target.name if target else None,
                details={
                    **structural_details,
                    "objective": "check_objective",
                    "kinetic_target": target.name if target else None,
                    "failed_step": step_index + 1,
                },
                contract_valid=True,
                structurally_valid=structurally_valid,
                objective_valid=False,
                kinetic_progress=_kinetic_progress_observed(
                    agent,
                    agent_start_position,
                    target,
                    start_position,
                    validator_config,
                ),
                kinetic_solved=False,
            )
        objective_state, objective_error = _evaluate_objective(env)
        if objective_error is not None:
            return ValidationResult(
                False,
                objective_error,
                env_class,
                blocking_object=target.name if target else None,
                details={
                    **structural_details,
                    "objective": "check_objective",
                    "kinetic_target": target.name if target else None,
                    "failed_step": step_index + 1,
                },
                contract_valid=True,
                structurally_valid=structurally_valid,
                objective_valid=False,
                kinetic_progress=_kinetic_progress_observed(
                    agent,
                    agent_start_position,
                    target,
                    start_position,
                    validator_config,
                ),
                kinetic_solved=False,
            )
        if objective_state:
            return ValidationResult(
                True,
                "Code-level objective satisfied during kinetic trial",
                env_class,
                details={
                    **structural_details,
                    "objective": "check_objective",
                    "objective_step": step_index + 1,
                    "kinetic_target": target.name if target else None,
                },
                contract_valid=True,
                structurally_valid=structurally_valid,
                objective_valid=True,
                kinetic_progress=True,
                kinetic_solved=True,
            )

    progress_observed = _kinetic_progress_observed(
        agent,
        agent_start_position,
        target,
        start_position,
        validator_config,
    )
    if target is not None and start_position is not None:
        displacement = float(target.body.position.get_distance(start_position))
        if displacement < validator_config.kinetic_displacement_threshold:
            return ValidationResult(
                False,
                "Inadequate mechanical leverage: Agent too weak or object too heavy.",
                env_class,
                blocking_object=target.name,
                details={
                    **structural_details,
                    "objective": "check_objective",
                    "kinetic_target": target.name,
                    "target_displacement": displacement,
                    "required_displacement": validator_config.kinetic_displacement_threshold,
                    "agent_strength": env.agent_strength,
                    "steps": validator_config.kinetic_steps,
                },
                contract_valid=True,
                structurally_valid=structurally_valid,
                objective_valid=True,
                kinetic_progress=progress_observed,
                kinetic_solved=False,
            )

    return ValidationResult(
        False,
        "Objective not satisfied during 300-step kinetic trial.",
        env_class,
        blocking_object=target.name if target else None,
        details={
            **structural_details,
            "objective": "check_objective",
            "kinetic_target": target.name if target else None,
            "steps": validator_config.kinetic_steps,
        },
        contract_valid=True,
        structurally_valid=structurally_valid,
        objective_valid=True,
        kinetic_progress=progress_observed,
        kinetic_solved=False,
    )


def verify_physical_interaction(
    env: BaseEnv,
    geometric_result: ValidationResult | None = None,
    config: ValidatorConfig | None = None,
) -> ValidationResult | None:
    """Verify that the agent can physically move a relevant dynamic mechanism."""

    validator_config = config or ValidatorConfig()
    target = _find_kinetic_target(env, geometric_result)
    agent = env.get_agent_record()
    if target is None or agent is None:
        return None

    start_position = _copy_vec2d(target.body.position)
    for _ in range(validator_config.kinetic_steps):
        direction = target.body.position - agent.body.position
        if direction.length <= 1.0:
            direction = (1.0, 0.0)
        env.step(
            action={"move": (float(direction.x), float(direction.y))},
            substeps=validator_config.kinetic_substeps,
        )

    displacement = float(target.body.position.get_distance(start_position))
    if displacement < validator_config.kinetic_displacement_threshold:
        return ValidationResult(
            False,
            "Inadequate mechanical leverage: Agent too weak or object too heavy.",
            env.__class__.__name__,
            path=geometric_result.path if geometric_result else (),
            visited_cells=geometric_result.visited_cells if geometric_result else 0,
            blocking_object=target.name,
            details={
                "kinetic_target": target.name,
                "target_displacement": displacement,
                "required_displacement": validator_config.kinetic_displacement_threshold,
                "agent_strength": env.agent_strength,
                "steps": validator_config.kinetic_steps,
            },
            contract_valid=True,
            structurally_valid=geometric_result.structurally_valid if geometric_result else True,
            objective_valid=geometric_result.objective_valid if geometric_result else False,
            kinetic_progress=False,
            kinetic_solved=False,
        )
    return None


def verify_multi_target_touch(
    env: BaseEnv,
    config: ValidatorConfig,
    structural_details: dict[str, Any],
) -> ValidationResult:
    """Oracle for objectives where the agent must touch several named targets."""

    env_class = env.__class__.__name__
    agent = env.get_agent_record()
    if agent is None:
        return ValidationResult(
            False,
            "Structural validation failed: missing dynamic agent with role='agent'.",
            env_class,
            details=structural_details,
            contract_valid=True,
            structurally_valid=False,
            objective_valid=False,
            kinetic_progress=False,
            kinetic_solved=False,
        )

    target_names = _valid_objective_target_names(env, structural_details)
    if not target_names:
        return ValidationResult(
            False,
            "Structural validation failed: multi_target_touch has no valid objective_targets.",
            env_class,
            details={**structural_details, "oracle": "multi_target_touch"},
            contract_valid=True,
            structurally_valid=False,
            objective_valid=False,
            kinetic_progress=False,
            kinetic_solved=False,
        )

    agent_start_position = _copy_vec2d(agent.body.position)
    oracle_touched: set[str] = set()
    target_order: list[str] = []
    target_radius_by_name = {
        name: _record_radius(env.get_object(name), fallback=getattr(env, "target_radius", 12.0))
        for name in target_names
    }
    agent_radius = _record_radius(agent, fallback=getattr(env, "agent_radius", config.agent_radius))
    touch_threshold = float(getattr(env, "touch_threshold", 0.0))

    objective_state, objective_error = _evaluate_objective(env)
    if objective_error is not None:
        return ValidationResult(
            False,
            objective_error,
            env_class,
            details={**structural_details, "oracle": "multi_target_touch"},
            contract_valid=True,
            structurally_valid=True,
            objective_valid=False,
            kinetic_progress=False,
            kinetic_solved=False,
        )
    if objective_state:
        return ValidationResult(
            True,
            "Code-level objective satisfied at initial state",
            env_class,
            details={**structural_details, "oracle": "multi_target_touch"},
            contract_valid=True,
            structurally_valid=True,
            objective_valid=True,
            kinetic_progress=True,
            kinetic_solved=True,
        )

    for step_index in range(config.kinetic_steps):
        _update_touched_targets(
            env,
            agent,
            target_names,
            target_radius_by_name,
            agent_radius,
            touch_threshold,
            oracle_touched,
        )
        if len(oracle_touched) == len(target_names):
            objective_state, objective_error = _evaluate_objective(env)
            if objective_error is not None:
                return _multi_target_failure_result(
                    env_class,
                    objective_error,
                    structural_details,
                    oracle_touched,
                    target_names,
                    target_order,
                    step_index,
                    agent,
                    agent_start_position,
                    config,
                    objective_valid=False,
                )
            if objective_state:
                return _multi_target_success_result(
                    env_class,
                    structural_details,
                    oracle_touched,
                    target_names,
                    target_order,
                    step_index,
                )
            return _multi_target_failure_result(
                env_class,
                (
                    "Oracle touched all objective targets, but check_objective did not "
                    "return True. Multi-target objectives should persist touched "
                    "targets instead of requiring simultaneous contact."
                ),
                structural_details,
                oracle_touched,
                target_names,
                target_order,
                step_index,
                agent,
                agent_start_position,
                config,
                objective_valid=False,
            )

        current_target = _nearest_untouched_target(env, agent, target_names, oracle_touched)
        if current_target is None:
            current_target = env.get_object(target_names[-1])
        if not target_order or target_order[-1] != current_target.name:
            target_order.append(current_target.name)
        try:
            _step_agent_toward(env, agent, current_target.body.position, config)
        except Exception as exc:
            return _multi_target_failure_result(
                env_class,
                f"Simulation/objective runtime failed: {type(exc).__name__}: {exc}",
                structural_details,
                oracle_touched,
                target_names,
                target_order,
                step_index + 1,
                agent,
                agent_start_position,
                config,
                objective_valid=False,
            )

        objective_state, objective_error = _evaluate_objective(env)
        if objective_error is not None:
            return _multi_target_failure_result(
                env_class,
                objective_error,
                structural_details,
                oracle_touched,
                target_names,
                target_order,
                step_index + 1,
                agent,
                agent_start_position,
                config,
                objective_valid=False,
            )
        if objective_state:
            _update_touched_targets(
                env,
                agent,
                target_names,
                target_radius_by_name,
                agent_radius,
                touch_threshold,
                oracle_touched,
            )
            return _multi_target_success_result(
                env_class,
                structural_details,
                oracle_touched,
                target_names,
                target_order,
                step_index + 1,
            )

    _update_touched_targets(
        env,
        agent,
        target_names,
        target_radius_by_name,
        agent_radius,
        touch_threshold,
        oracle_touched,
    )
    if oracle_touched:
        reason = (
            "Partial multi-target touch progress: "
            f"touched {len(oracle_touched)}/{len(target_names)} targets, "
            "but check_objective did not complete."
        )
    else:
        reason = "No multi-target touch progress during kinetic trial."
    return _multi_target_failure_result(
        env_class,
        reason,
        structural_details,
        oracle_touched,
        target_names,
        target_order,
        config.kinetic_steps,
        agent,
        agent_start_position,
        config,
        objective_valid=True,
    )


def _objective_structural_check(env: BaseEnv) -> tuple[bool, str, dict[str, Any]]:
    object_count = len(getattr(env, "_objects", {}))
    agent = env.get_agent_record()
    objective_metadata = _objective_metadata_from_env(env)
    objective_profile = _profile_dict(objective_metadata.get("objective_profile"))
    capability_profile = _profile_dict(objective_metadata.get("capability_profile"))
    gameplay_profile = _profile_dict(objective_metadata.get("gameplay_profile"))
    physics_relations = _profile_dict(objective_metadata.get("physics_relations"))
    layout_plan = _profile_dict(objective_metadata.get("layout_plan"))
    semantic_requirements = _semantic_requirements_from_env(env, objective_metadata)
    anti_cheat_profile = _anti_cheat_profile_from_env(env, objective_metadata)
    details = {
        "object_count": object_count,
        "has_agent": agent is not None,
        "objective": "check_objective",
        "objective_type": objective_metadata.get("objective_type"),
        "objective_targets": objective_metadata.get("objective_targets"),
        "objective_profile": objective_profile,
        "capability_profile": capability_profile,
        "gameplay_profile": gameplay_profile,
        "physics_relations": physics_relations,
        "layout_plan": layout_plan,
        "semantic_requirements": semantic_requirements,
        "anti_cheat_profile": anti_cheat_profile,
        "minimum_acceptance_tier": _minimum_acceptance_tier_from_profile(
            objective_profile,
            objective_metadata.get("objective_type"),
        ),
    }
    details["physics_parameter_probes"] = _physics_parameter_probe_results(
        env,
        physics_relations,
    )
    if agent is None:
        return False, "Structural validation failed: missing dynamic agent with role='agent'.", details
    if object_count <= 0:
        return False, "Structural validation failed: no registered objects.", details
    if not callable(getattr(env, "check_objective", None)):
        return False, "Objective validation failed: missing check_objective().", details
    if not objective_metadata.get("objective_type"):
        return False, "Structural validation failed: missing objective_type metadata.", details
    if objective_metadata.get("objective_targets") is None:
        return False, "Structural validation failed: missing objective_targets metadata.", details
    if not objective_profile:
        return False, "Structural validation failed: missing objective_profile metadata.", details
    if not capability_profile:
        return False, "Structural validation failed: missing capability_profile metadata.", details
    contract_errors = _semantic_contract_errors(env, objective_profile)
    if contract_errors:
        details["contract_errors"] = contract_errors
        return False, "Contract validation failed: " + "; ".join(contract_errors), details
    return True, "Structural validation passed.", details


def _semantic_contract_errors(
    env: BaseEnv,
    objective_profile: dict[str, Any],
) -> list[str]:
    """Hard semantic gates that follow from declared objective subgoals."""

    errors: list[str] = []
    force_zones = getattr(env, "_force_zones", {})
    for subgoal in _subgoal_list(objective_profile):
        kind = str(subgoal.get("kind") or "")
        if kind != "field_force_interaction":
            continue
        if not force_zones:
            errors.append(
                "field_force_interaction requires at least one BaseEnv force zone "
                "registered with self.register_force_zone(...)"
            )
            continue
        field_name = subgoal.get("field")
        if field_name is None:
            errors.append(
                "field_force_interaction subgoal must include a 'field' name that "
                "matches a registered force zone"
            )
            continue
        if str(field_name) not in force_zones:
            errors.append(
                f"field_force_interaction references field {field_name!r}, but that "
                "name is not present in env._force_zones"
            )
    return errors


def _semantic_requirements_from_env(
    env: BaseEnv,
    objective_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return explicit or prompt-inferred physical-behavior requirements."""

    explicit = objective_metadata.get("semantic_requirements")
    if isinstance(explicit, list):
        requirements = [dict(item) for item in explicit if isinstance(item, dict)]
        return _normalize_semantic_requirements_for_prompt(requirements, _source_prompt_from_env(env))

    direct = getattr(env, "semantic_requirements", None)
    if isinstance(direct, list):
        requirements = [dict(item) for item in direct if isinstance(item, dict)]
        return _normalize_semantic_requirements_for_prompt(requirements, _source_prompt_from_env(env))

    return _infer_semantic_requirements_from_prompt(_source_prompt_from_env(env))


def _normalize_semantic_requirements_for_prompt(
    requirements: list[dict[str, Any]],
    prompt: str,
) -> list[dict[str, Any]]:
    if not _prompt_requests_agent_projectile_impact(prompt):
        return requirements
    normalized: list[dict[str, Any]] = []
    for requirement in requirements:
        item = dict(requirement)
        text = json.dumps(item, sort_keys=True, default=str).lower()
        if any(token in text for token in ("agent_bullet", "bullet", "projectile", "agent-fired", "agent fired")):
            if str(item.get("role") or "").lower() == "hazard":
                item["role"] = "projectile"
            if str(item.get("kind") or "") == "dynamic_hazard_motion":
                item["kind"] = "object_motion"
            item.setdefault("source", "prompt_normalized_agent_projectile")
        normalized.append(item)
    return normalized


def _source_prompt_from_env(env: BaseEnv) -> str:
    module = sys.modules.get(env.__class__.__module__)
    if module is None:
        return ""
    return _original_user_prompt_fragment(str(getattr(module, "SOURCE_PROMPT", "") or ""))


def _original_user_prompt_fragment(source_prompt: str) -> str:
    """Strip generated repair/context payloads from SOURCE_PROMPT.

    Harness repairs pass an enhanced request to the Architect containing
    simulation briefs, memory examples, relation graphs, and prior failures.
    Prompt-fidelity inference must use only the user's original sentence;
    otherwise a remembered "car" example can contaminate a bullet prompt.
    """

    text = " ".join(str(source_prompt or "").split())
    if not text:
        return ""
    for marker in (
        " SIMULATION BRIEF ",
        " GAMEPLAY ARCHITECT PROFILE ",
        " PHYSICS RELATION GRAPH ",
        " ROUTE-AWARE LAYOUT PLAN ",
        " SEMANTIC MEMORY RETRIEVAL ",
        " VISUAL RECIPE ",
        " Repair pass ",
    ):
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx].strip()
    text = re.sub(r"\s+Seed\s+\d+\s+diversity directive:.*$", "", text, flags=re.IGNORECASE).strip()
    return text


def _infer_semantic_requirements_from_prompt(prompt: str) -> list[dict[str, Any]]:
    """Infer high-confidence dynamic semantics from the original prompt."""

    text = prompt.lower()
    requirements: list[dict[str, Any]] = []
    falling_terms = ("falling", "raining", "rain down", "dropping", "drop down", "fall from")
    hazard_terms = ("fire", "fireball", "fire ball", "rock", "boulder", "meteor", "spike", "hazard")
    if any(term in text for term in falling_terms) and any(term in text for term in hazard_terms):
        name_terms = [term for term in ("fire", "fireball", "rock", "boulder", "meteor", "spike", "hazard") if term in text]
        if "fire ball" in text and "fireball" not in name_terms:
            name_terms.append("fireball")
        requirements.append(
            {
                "kind": "dynamic_hazard_motion",
                "description": "Prompt describes falling/raining hazards; at least one hazard must move downward during passive simulation.",
                "role": "hazard",
                "name_contains": name_terms or ["hazard"],
                "motion": "falling",
                "axis": "y",
                "direction": "down",
                "min_displacement_y": 60,
                "min_objects": 1,
                "steps": 150,
                "source": "prompt_inferred",
            }
        )

    drifting_terms = ("drifting", "floating", "float", "zero-gravity", "zero gravity", "zero_g")
    collectible_terms = ("crystal", "artifact", "core", "gem", "plate", "sock", "target")
    if any(term in text for term in drifting_terms) and any(term in text for term in collectible_terms):
        name_terms = [term for term in collectible_terms if term in text]
        requirements.append(
            {
                "kind": "object_motion",
                "description": "Prompt describes drifting/floating objects; at least one target-like object should move visibly.",
                "name_contains": name_terms,
                "motion": "drifting",
                "axis": "any",
                "direction": "any",
                "min_displacement": 12,
                "min_objects": 1,
                "steps": 150,
                "source": "prompt_inferred",
            }
        )
    chase_terms = (
        "chased",
        "chasing",
        "pursued",
        "pursuing",
        "pursuit",
        "followed by",
        "hunt",
        "hunter",
        "angry agent",
        "angry agents",
    )
    enemy_terms = (
        "square",
        "squares",
        "enemy",
        "enemies",
        "chaser",
        "chasers",
        "monster",
        "monsters",
        "hazard",
        "angry agent",
        "angry agents",
    )
    if any(term in text for term in chase_terms) and any(term in text for term in enemy_terms):
        name_terms = [
            term
            for term in ("square", "enemy", "chaser", "monster", "hazard", "angry", "angry_agent")
            if term in text
        ]
        requirements.append(
            {
                "kind": "dynamic_hazard_motion",
                "description": "Prompt describes chasing hazards; at least one enemy must move closer to the agent during passive simulation.",
                "role": "hazard",
                "name_contains": name_terms or ["enemy", "chaser", "angry_agent", "angry"],
                "motion": "pursuit",
                "axis": "any",
                "direction": "toward_agent",
                "min_distance_reduction": 30,
                "min_objects": 1,
                "steps": 180,
                "source": "prompt_inferred",
            }
        )
    lateral_terms = ("car", "cars", "truck", "trucks", "train", "trains", "traffic", "vehicle", "vehicles")
    incoming_terms = ("come", "coming", "incoming", "endless", "endlessly", "sequential", "sequentially", "cross", "crossing", "drive", "driving", "jump over")
    if any(term in text for term in lateral_terms) and any(term in text for term in incoming_terms):
        name_terms = [term for term in ("car", "truck", "train", "traffic", "vehicle") if term in text]
        requirements.append(
            {
                "kind": "dynamic_hazard_motion",
                "description": "Prompt describes incoming lateral hazards; at least one vehicle must move horizontally through the agent lane.",
                "role": "hazard",
                "name_contains": name_terms or ["car", "vehicle"],
                "motion": "moving",
                "axis": "x",
                "direction": "any",
                "min_displacement_x": 120,
                "min_objects": 1,
                "steps": 240,
                "source": "prompt_inferred",
            }
        )
    return requirements


def _anti_cheat_profile_from_env(
    env: BaseEnv,
    objective_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return explicit or prompt-inferred checks for degenerate prompt solutions."""

    prompt = _source_prompt_from_env(env)
    profile: list[dict[str, Any]] = []
    explicit = objective_metadata.get("anti_cheat_profile")
    if isinstance(explicit, list) and explicit:
        profile.extend(
            _normalize_anti_cheat_check_for_prompt(dict(item), prompt)
            for item in explicit
            if isinstance(item, dict)
        )

    direct = getattr(env, "anti_cheat_profile", None)
    if isinstance(direct, list) and direct:
        profile.extend(
            _normalize_anti_cheat_check_for_prompt(dict(item), prompt)
            for item in direct
            if isinstance(item, dict)
        )

    profile.extend(
        _normalize_anti_cheat_check_for_prompt(dict(item), prompt)
        for item in _infer_anti_cheat_profile_from_prompt(prompt, objective_metadata)
        if isinstance(item, dict)
    )
    return _dedupe_anti_cheat_profile(profile)


def _normalize_anti_cheat_check_for_prompt(check: dict[str, Any], prompt: str) -> dict[str, Any]:
    text = prompt.lower()
    vehicle_prompt = any(
        term in text
        for term in ("car", "cars", "truck", "trucks", "train", "trains", "traffic", "vehicle", "vehicles")
    ) and any(
        term in text
        for term in ("come", "coming", "incoming", "endless", "endlessly", "sequential", "sequentially", "cross", "crossing", "drive", "driving", "jump over")
    )
    rolling_lane_prompt = any(term in text for term in ("rolling", "rolls", "roll ")) and any(
        term in text
        for term in ("boulder", "boulders", "rock", "rocks", "stone", "stones", "barrel", "barrels", "log", "logs")
    ) and any(term in text for term in ("avoid", "dodge", "jump over", "escape", "survive", "incoming", "come", "coming"))
    if (vehicle_prompt or rolling_lane_prompt) and str(check.get("kind") or "") == "active_threat_engagement":
        normalized = dict(check)
        normalized["motion"] = "lateral"
        normalized["role"] = normalized.get("role") or "hazard"
        normalized["name_contains"] = (
            ["boulder", "rock", "stone", "barrel", "log"]
            if rolling_lane_prompt and not vehicle_prompt
            else ["car", "truck", "train", "traffic", "vehicle"]
        )
        normalized["min_displacement"] = max(_safe_float(normalized.get("min_displacement"), 0.0), 120.0)
        normalized["max_y_drift"] = min(_safe_float(normalized.get("max_y_drift"), 52.0), 52.0)
        normalized["must_enter_agent_lane"] = True
        normalized["must_threaten_agent_column"] = True
        normalized.setdefault(
            "description",
            "Incoming lateral threats must remain grounded and cross the agent lane.",
        )
        return normalized
    return check


def _dedupe_anti_cheat_profile(profile: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in profile:
        key = (
            str(item.get("kind") or ""),
            str(item.get("motion") or ""),
            ",".join(sorted(_string_list(item.get("name_contains")))),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _infer_anti_cheat_profile_from_prompt(
    prompt: str,
    objective_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Infer high-confidence prompt-fidelity anti-cheat probes.

    These are not task templates. They are generic checks for cases where the
    code-level objective can pass while the core challenge from the text never
    actually engages the agent.
    """

    text = prompt.lower()
    profile: list[dict[str, Any]] = []
    challenge_terms = (
        "avoid",
        "dodge",
        "jump over",
        "survive",
        "escape",
        "evade",
        "run from",
        "last",
    )
    incoming_terms = (
        "come",
        "coming",
        "incoming",
        "endless",
        "endlessly",
        "sequential",
        "sequentially",
        "cross",
        "crossing",
        "drive",
        "driving",
        "roll",
        "rolling",
        "fall",
        "falling",
        "rain",
        "raining",
        "shoot",
        "shooting",
        "shot",
        "shots",
        "laser",
        "projectile",
    )
    threat_terms = (
        "car",
        "cars",
        "truck",
        "trucks",
        "train",
        "trains",
        "vehicle",
        "traffic",
        "fire",
        "fireball",
        "fire ball",
        "ball",
        "balls",
        "rock",
        "rocks",
        "boulder",
        "meteor",
        "laser",
        "lasers",
        "shot",
        "shots",
        "projectile",
        "enemy",
        "enemies",
        "bear",
        "monster",
        "chaser",
        "angry",
    )
    if (
        any(term in text for term in challenge_terms)
        and any(term in text for term in threat_terms)
        and (
            any(term in text for term in incoming_terms + ("chase", "chased", "chasing", "pursue", "pursued", "pursuing"))
            or ("escape" in text and any(term in text for term in ("enemy", "enemies", "agent", "agents", "angry", "bear", "monster")))
        )
    ):
        name_terms = [
            term
            for term in (
                "car",
                "truck",
                "train",
                "vehicle",
                "traffic",
                "fire",
                "fireball",
                "ball",
                "rock",
                "boulder",
                "meteor",
                "laser",
                "shot",
                "projectile",
                "enemy",
                "bear",
                "monster",
                "chaser",
                "angry",
            )
            if term in text
        ]
        motion = "any"
        if any(term in text for term in ("car", "truck", "train", "traffic", "vehicle")):
            motion = "lateral"
        if any(term in text for term in ("rolling", "rolls", "roll ")) and any(
            term in text for term in ("boulder", "rock", "stone", "barrel", "log")
        ):
            motion = "lateral"
        if any(term in text for term in ("falling", "raining", "drops", "dropping", "fall from")):
            motion = "falling"
        if any(term in text for term in ("shot", "shoot", "laser", "projectile")):
            motion = "projectile"
        if any(term in text for term in ("chase", "chased", "chasing", "pursue", "pursued", "pursuing", "followed by")) or (
            "escape" in text and any(term in text for term in ("enemy", "enemies", "agent", "agents", "angry", "bear", "monster"))
        ):
            motion = "pursuit"
        profile.append(
            {
                "kind": "active_threat_engagement",
                "description": (
                    "The requested hazard/threat must actually enter the playable "
                    "challenge lane; a static, blocked, or off-route threat is a degenerate solution."
                ),
                "role": "hazard",
                "name_contains": name_terms or ["hazard", "enemy"],
                "motion": motion,
                "min_displacement": 95 if motion != "falling" else 70,
                "min_distance_reduction": 24,
                "max_y_drift": 52,
                "must_enter_agent_lane": motion in {"lateral", "projectile", "pursuit", "any"},
                "must_threaten_agent_column": motion in {"lateral", "projectile", "any"},
                "steps": 300,
                "source": "prompt_inferred",
            }
        )

    objective_type = str((objective_metadata or {}).get("objective_type") or "").lower()
    targets = _string_list((objective_metadata or {}).get("objective_targets"))
    if not _prompt_requests_agent_projectile_impact(prompt) and (
        objective_type in {"push_object", "mechanism_activation"} or any(
        term in text for term in ("push", "shove", "kick", "throw", "launch", "knock")
        )
    ):
        profile.append(
            {
                "kind": "nontrivial_agent_interaction",
                "description": "Manipulation tasks must require visible object motion rather than starting solved.",
                "object_names": [target for target in targets if target != "agent"],
                "name_contains": ["box", "ball", "rock", "crate", "weight", "object"],
                "min_displacement": 35,
                "steps": 120,
                "source": "prompt_inferred",
            }
        )
    return profile


def _prompt_requests_agent_projectile_impact(prompt: str) -> bool:
    text = prompt.lower()
    return bool(
        re.search(
            r"\b(agent|player|person|character|robot)\b.{0,45}\b(shoots|shoot|fires|fire|firing)\b.{0,60}\b(bullet|projectile|missile|laser|blaster)\b",
            text,
        )
    ) and any(
        term in text
        for term in (
            "knock",
            "knocks",
            "knock over",
            "topple",
            "break",
            "hit",
            "hits",
            "pile",
            "stack",
            "tower",
            "target",
            "squares",
            "blocks",
        )
    ) and not any(
        term in text
        for term in (
            "avoid",
            "avoiding",
            "dodge",
            "dodging",
            "survive",
            "enemy",
            "enemies",
            "incoming",
            "at the agent",
            "toward the agent",
        )
    )


def _anti_cheat_validation_result(
    env: BaseEnv,
    config: ValidatorConfig,
    structural_details: dict[str, Any],
) -> ValidationResult | None:
    profile = structural_details.get("anti_cheat_profile")
    if not isinstance(profile, list) or not profile:
        return None

    probes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for check in profile:
        if not isinstance(check, dict):
            continue
        kind = str(check.get("kind") or "")
        if kind == "active_threat_engagement":
            probe = _probe_active_threat_engagement(env, check, config)
        elif kind == "nontrivial_agent_interaction":
            probe = _probe_nontrivial_agent_interaction(env, check, config)
        else:
            continue
        probes.append(probe)
        if not probe.get("passed"):
            failures.append(probe)

    structural_details["anti_cheat_probes"] = probes
    if not failures:
        return None

    first = failures[0]
    return ValidationResult(
        False,
        f"Prompt anti-cheat validation failed: {first.get('repair') or first.get('diagnosis')}",
        env.__class__.__name__,
        blocking_object=_first_probe_object(first),
        details={
            **structural_details,
            "oracle": "semantic_anticheat",
            "anti_cheat_failures": failures,
        },
        contract_valid=True,
        structurally_valid=True,
        objective_valid=True,
        kinetic_progress=False,
        kinetic_solved=False,
    )


def _semantic_validation_result(
    env: BaseEnv,
    config: ValidatorConfig,
    structural_details: dict[str, Any],
) -> ValidationResult | None:
    requirements = structural_details.get("semantic_requirements")
    if not isinstance(requirements, list) or not requirements:
        return None

    probes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for requirement in requirements:
        if not isinstance(requirement, dict):
            continue
        probe = _run_semantic_requirement_probe(env, requirement, config)
        probes.append(probe)
        if not probe.get("passed"):
            failures.append(probe)

    if not failures:
        structural_details["semantic_probes"] = probes
        return None

    first = failures[0]
    return ValidationResult(
        False,
        f"Semantic dynamics validation failed: {first.get('repair') or first.get('diagnosis')}",
        env.__class__.__name__,
        blocking_object=_first_probe_object(first),
        details={
            **structural_details,
            "oracle": "semantic_dynamics",
            "semantic_probes": probes,
            "semantic_failures": failures,
        },
        contract_valid=True,
        structurally_valid=True,
        objective_valid=True,
        kinetic_progress=False,
        kinetic_solved=False,
    )


def _gameplay_validation_result(
    env: BaseEnv,
    config: ValidatorConfig,
    structural_details: dict[str, Any],
) -> ValidationResult | None:
    profile = _profile_dict(structural_details.get("gameplay_profile"))
    dynamics = profile.get("dynamics")
    if not isinstance(dynamics, list) or not dynamics:
        return None
    probes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for dynamic in dynamics:
        if not isinstance(dynamic, dict):
            continue
        dynamic_type = str(dynamic.get("type") or "")
        effective_dynamic = _merged_gameplay_dynamic(dynamic)
        if dynamic_type in {"recurring_falling_hazard", "recurring_hazard", "falling_hazard"}:
            probe = _probe_recurring_falling_hazards(env, effective_dynamic, config)
        elif dynamic_type in {"readable_hazard_stream", "recurring_lateral_hazard", "lateral_hazard_stream", "crossing_hazard"}:
            probe = _probe_recurring_lateral_hazards(env, effective_dynamic, config)
        elif dynamic_type == "heavy_but_movable_push":
            probe = _probe_heavy_but_movable_push(env, effective_dynamic, config)
        else:
            continue
        probes.append(probe)
        if not probe.get("passed"):
            failures.append(probe)
    structural_details["gameplay_probes"] = probes
    if not failures:
        return None
    first = failures[0]
    return ValidationResult(
        False,
        f"Gameplay dynamics validation failed: {first.get('repair') or first.get('diagnosis')}",
        env.__class__.__name__,
        blocking_object=(first.get("objects") or [None])[0],
        details={
            **structural_details,
            "oracle": "gameplay_dynamics",
            "gameplay_failures": failures,
        },
        contract_valid=True,
        structurally_valid=True,
        objective_valid=False,
        kinetic_progress=False,
        kinetic_solved=False,
    )


def _merged_gameplay_dynamic(dynamic: dict[str, Any]) -> dict[str, Any]:
    parameters = _profile_dict(dynamic.get("parameters"))
    merged = {**parameters, **dynamic}
    if "spawn_lanes" not in merged and "emitters" in parameters:
        merged["spawn_lanes"] = parameters.get("emitters")
    if "minimum_visible_fall_px" not in merged:
        speed_range = parameters.get("fireball_speed_y_px_s_range")
        if isinstance(speed_range, list) and speed_range:
            merged["minimum_visible_fall_px"] = min(220.0, max(140.0, _safe_float(speed_range[0], 260.0) * 0.6))
    return merged


def _probe_recurring_lateral_hazards(
    env: BaseEnv,
    dynamic: dict[str, Any],
    config: ValidatorConfig,
) -> dict[str, Any]:
    requirement = {
        "role": "hazard",
        "name_contains": ["car", "truck", "train", "vehicle", "traffic", "hazard"],
    }
    try:
        probe_env = env.__class__()
    except Exception as exc:
        return {
            "name": "recurring_lateral_hazard",
            "passed": False,
            "objects": [],
            "metrics": {"dynamic": dynamic},
            "diagnosis": "gameplay_probe_env_rebuild_failed",
            "repair": f"Could not rebuild environment for lateral hazard probe: {type(exc).__name__}: {exc}",
            "severity": "error",
            "tier_evidence": 1,
        }
    records = _anti_cheat_matching_records(probe_env, requirement)
    if not records:
        return {
            "name": "recurring_lateral_hazard",
            "passed": False,
            "objects": [],
            "metrics": {"dynamic": dynamic},
            "diagnosis": "no_lateral_hazard_objects",
            "repair": "Create dynamic lane hazards named like car_* or vehicle_* and register them with role='hazard'. Prefer create_recurring_lateral_hazards(...).",
            "severity": "error",
            "tier_evidence": 1,
        }

    agent = probe_env.get_agent_record()
    agent_start = _copy_vec2d(agent.body.position) if agent is not None else pymunk.Vec2d(probe_env.width * 0.5, probe_env.height * 0.5)
    steps = max(300, _safe_int(dynamic.get("probe_steps"), 420))
    substeps = max(1, int(config.kinetic_substeps))
    min_x_span = _safe_float(dynamic.get("minimum_lateral_travel_px"), 160.0)
    max_y_drift = _safe_float(dynamic.get("max_ground_y_drift_px"), 42.0)
    lane_half_height = max(52.0, _record_radius(agent, fallback=18.0) * 2.4) if agent is not None else 64.0
    column_half_width = max(72.0, _record_radius(agent, fallback=18.0) * 3.2) if agent is not None else 90.0

    state: dict[str, dict[str, Any]] = {}
    for record in records:
        pos = _copy_vec2d(record.body.position)
        state[record.name] = {
            "start": pos,
            "end": pos,
            "min_x": float(pos.x),
            "max_x": float(pos.x),
            "min_y": float(pos.y),
            "max_y": float(pos.y),
            "entered_agent_lane": abs(float(pos.y - agent_start.y)) <= lane_half_height,
            "threatened_agent_column": abs(float(pos.x - agent_start.x)) <= column_half_width,
            "reset_like_jumps": 0,
            "previous": pos,
            "first_motion_step": None,
        }

    for step_index in range(1, steps + 1):
        probe_env.step(substeps=substeps)
        for record in records:
            item = state[record.name]
            previous = item["previous"]
            pos = _copy_vec2d(record.body.position)
            step_dx = abs(float(pos.x - previous.x))
            if step_dx > 120.0:
                item["reset_like_jumps"] += 1
            if step_dx > 2.0 and item["first_motion_step"] is None:
                item["first_motion_step"] = step_index
            item["previous"] = pos
            item["end"] = pos
            item["min_x"] = min(float(item["min_x"]), float(pos.x))
            item["max_x"] = max(float(item["max_x"]), float(pos.x))
            item["min_y"] = min(float(item["min_y"]), float(pos.y))
            item["max_y"] = max(float(item["max_y"]), float(pos.y))
            if abs(float(pos.y - agent_start.y)) <= lane_half_height:
                item["entered_agent_lane"] = True
            if abs(float(pos.x - agent_start.x)) <= column_half_width:
                item["threatened_agent_column"] = True

    per_object: list[dict[str, Any]] = []
    moving_lane_objects: list[str] = []
    threat_objects: list[str] = []
    for record in records:
        item = state[record.name]
        start = item["start"]
        end = item["end"]
        x_span = float(item["max_x"] - item["min_x"])
        y_span = float(item["max_y"] - item["min_y"])
        lane_locked = y_span <= max_y_drift
        moved = x_span >= min_x_span
        engaged = bool(item["entered_agent_lane"]) and bool(item["threatened_agent_column"])
        if record.body.body_type == pymunk.Body.DYNAMIC and moved and lane_locked:
            moving_lane_objects.append(record.name)
        if record.body.body_type == pymunk.Body.DYNAMIC and moved and lane_locked and engaged:
            threat_objects.append(record.name)
        per_object.append(
            {
                "name": record.name,
                "dynamic": record.body.body_type == pymunk.Body.DYNAMIC,
                "x_span": round(x_span, 3),
                "y_span": round(y_span, 3),
                "lane_locked": lane_locked,
                "entered_agent_lane": bool(item["entered_agent_lane"]),
                "threatened_agent_column": bool(item["threatened_agent_column"]),
                "reset_like_jumps": int(item["reset_like_jumps"]),
                "first_motion_step": item["first_motion_step"],
                "start": _vec_list(start),
                "end": _vec_list(end),
            }
        )

    required_movers = min(2, max(1, len(records)))
    passed = len(moving_lane_objects) >= required_movers and bool(threat_objects)
    return {
        "name": "recurring_lateral_hazard",
        "passed": passed,
        "objects": [record.name for record in records],
        "metrics": {
            "dynamic": dynamic,
            "agent_start": _vec_list(agent_start),
            "moving_lane_objects": moving_lane_objects,
            "threat_objects": threat_objects,
            "required_moving_lane_objects": required_movers,
            "min_x_span": min_x_span,
            "max_y_drift": max_y_drift,
            "per_object": per_object,
        },
        "diagnosis": "recurring_lateral_hazards_observed" if passed else "recurring_lateral_hazards_missing_or_falling",
        "repair": (
            "Recurring lateral hazard gameplay satisfied."
            if passed
            else "Implement a grounded sequential hazard stream. Prefer create_recurring_lateral_hazards(...): spawn cars/vehicles offscreen, keep them lane-locked with near-zero vertical drift, move them horizontally through the agent lane, and reset them after exit. Do not let cars fall off-screen or get blocked by outer walls."
        ),
        "severity": "info" if passed else "error",
        "tier_evidence": 3 if passed else 1,
    }


def _probe_recurring_falling_hazards(
    env: BaseEnv,
    dynamic: dict[str, Any],
    config: ValidatorConfig,
) -> dict[str, Any]:
    requirement = {
        "kind": "dynamic_hazard_motion",
        "role": "hazard",
        "name_contains": ["fire", "fireball", "hazard", "meteor", "rock"],
        "motion": "falling",
        "direction": "down",
        "min_displacement_y": dynamic.get("minimum_visible_fall_px", 180),
        "min_objects": min(3, max(1, _safe_int(dynamic.get("spawn_lanes"), 3))),
    }
    try:
        probe_env = env.__class__()
    except Exception as exc:
        return {
            "name": "recurring_falling_hazard",
            "passed": False,
            "objects": [],
            "metrics": {"dynamic": dynamic},
            "diagnosis": "gameplay_probe_env_rebuild_failed",
            "repair": f"Could not rebuild environment for gameplay probe: {type(exc).__name__}: {exc}",
            "severity": "error",
            "tier_evidence": 1,
        }
    records = _semantic_matching_records(probe_env, requirement)
    if not records:
        return {
            "name": "recurring_falling_hazard",
            "passed": False,
            "objects": [],
            "metrics": {"dynamic": dynamic},
            "diagnosis": "no_recurring_hazard_objects",
            "repair": "Create dynamic hazard objects named like fireball_* or hazard_* and register them with role='hazard'.",
            "severity": "error",
            "tier_evidence": 1,
        }

    substeps = max(1, int(config.kinetic_substeps))
    steps = max(240, _safe_int(dynamic.get("probe_steps"), 360))
    previous_positions = {record.name: _copy_vec2d(record.body.position) for record in records}
    current_visible_drop: dict[str, float] = {record.name: 0.0 for record in records}
    first_fall_step: dict[str, int] = {}
    resets: dict[str, int] = {record.name: 0 for record in records}
    max_visible_drop: dict[str, float] = {record.name: 0.0 for record in records}
    for step_index in range(1, steps + 1):
        action = {"move": [1.0, 0.0]} if probe_env.get_agent_record() is not None else None
        probe_env.step(action=action, substeps=substeps)
        for record in records:
            previous = previous_positions[record.name]
            current = record.body.position
            downward_delta = _visible_downward_delta(probe_env, previous, current)
            if downward_delta > 0.0:
                current_visible_drop[record.name] += downward_delta
            elif float(current.y - previous.y) > 80.0:
                resets[record.name] += 1
                current_visible_drop[record.name] = 0.0
            else:
                current_visible_drop[record.name] = max(0.0, current_visible_drop[record.name] * 0.98)
            max_visible_drop[record.name] = max(
                max_visible_drop[record.name], current_visible_drop[record.name]
            )
            if current_visible_drop[record.name] >= 24.0 and record.name not in first_fall_step:
                first_fall_step[record.name] = step_index
            previous_positions[record.name] = _copy_vec2d(current)

    required_drop = _safe_float(dynamic.get("minimum_visible_fall_px"), 180.0)
    falling = [name for name, drop in max_visible_drop.items() if drop >= required_drop]
    unique_start_steps = sorted(set(first_fall_step.values()))
    staggered = len(unique_start_steps) >= min(2, len(records))
    recurring = (
        bool(dynamic.get("reset_when_below_world"))
        or bool(dynamic.get("despawn_on_out_of_bounds"))
        or str(dynamic.get("cadence")) == "staggered_continuous"
        or str(dynamic.get("type")) == "recurring_hazard"
    )
    recurrence_observed = any(count > 0 for count in resets.values())
    passed = len(falling) >= min(3, len(records)) and staggered and (not recurring or recurrence_observed)
    return {
        "name": "recurring_falling_hazard",
        "passed": passed,
        "objects": [record.name for record in records],
        "metrics": {
            "dynamic": dynamic,
            "falling_objects": falling,
            "first_fall_steps": first_fall_step,
            "unique_first_fall_steps": unique_start_steps,
            "resets": resets,
            "max_visible_drop": {key: round(value, 3) for key, value in max_visible_drop.items()},
            "required_visible_drop": required_drop,
        },
        "diagnosis": "recurring_falling_hazards_observed" if passed else "recurring_falling_hazards_missing",
        "repair": (
            "Recurring falling hazard gameplay satisfied."
            if passed
            else "Implement staggered continuous falling hazards: add phase offsets/timers, reset hazards to the top after they exit below the world, and avoid dropping every hazard at the same time."
        ),
        "severity": "info" if passed else "error",
        "tier_evidence": 3 if passed else 1,
    }


def _probe_heavy_but_movable_push(
    env: BaseEnv,
    dynamic: dict[str, Any],
    config: ValidatorConfig,
) -> dict[str, Any]:
    # The existing move_object_to_region oracle does deep push diagnostics. This
    # gameplay probe is intentionally lightweight: it checks that the declared
    # feel target is represented in the profile and leaves rollout proof to the
    # subgoal validator.
    required = _safe_float(dynamic.get("visible_displacement_required_px"), 80.0)
    return {
        "name": "heavy_but_movable_push",
        "passed": required >= 40.0,
        "objects": [],
        "metrics": {"visible_displacement_required_px": required},
        "diagnosis": "push_feel_target_declared",
        "repair": "Use the push rollout diagnostics to verify visible displacement under agent force.",
        "severity": "info",
        "tier_evidence": 1,
    }


def _probe_active_threat_engagement(
    env: BaseEnv,
    check: dict[str, Any],
    config: ValidatorConfig,
) -> dict[str, Any]:
    try:
        probe_env = env.__class__()
    except Exception as exc:
        return {
            "name": "active_threat_engagement",
            "passed": False,
            "objects": [],
            "metrics": {"check": check},
            "diagnosis": "anti_cheat_probe_env_rebuild_failed",
            "repair": f"Anti-cheat probe could not rebuild the environment: {type(exc).__name__}: {exc}",
            "severity": "error",
            "tier_evidence": 1,
        }

    agent = probe_env.get_agent_record()
    if agent is None:
        return {
            "name": "active_threat_engagement",
            "passed": False,
            "objects": [],
            "metrics": {"check": check},
            "diagnosis": "missing_agent_for_threat_probe",
            "repair": "Create and register a dynamic role='agent' object before adding threats.",
            "severity": "error",
            "tier_evidence": 1,
        }

    records = _anti_cheat_matching_records(probe_env, check)
    if not records:
        return {
            "name": "active_threat_engagement",
            "passed": False,
            "objects": [],
            "metrics": {"check": check},
            "diagnosis": "no_matching_threat_objects",
            "repair": (
                "Prompt anti-cheat failure: the objective mentions an active threat, but no matching "
                "registered threat bodies were found. Create dynamic role='hazard' objects with names/"
                f"metadata containing {check.get('name_contains')!r}."
            ),
            "severity": "error",
            "tier_evidence": 1,
        }

    agent_start = _copy_vec2d(agent.body.position)
    agent_radius = _record_radius(agent, fallback=18.0)
    lane_half_height = max(52.0, agent_radius * 2.4)
    column_half_width = max(70.0, agent_radius * 3.2)
    steps = max(90, _safe_int(check.get("steps"), 300))
    substeps = max(1, int(config.kinetic_substeps))
    min_displacement = _safe_float(check.get("min_displacement"), 95.0)
    min_distance_reduction = _safe_float(check.get("min_distance_reduction"), 24.0)
    motion = str(check.get("motion") or "any").lower()
    must_enter_lane = bool(check.get("must_enter_agent_lane", True))
    must_threaten_column = bool(check.get("must_threaten_agent_column", False))

    state: dict[str, dict[str, Any]] = {}
    for record in records:
        pos = _copy_vec2d(record.body.position)
        distance = float(pos.get_distance(agent_start))
        state[record.name] = {
            "start": pos,
            "end": pos,
            "min_x": float(pos.x),
            "max_x": float(pos.x),
            "min_y": float(pos.y),
            "max_y": float(pos.y),
            "start_agent_distance": distance,
            "min_agent_distance": distance,
            "entered_agent_lane": abs(float(pos.y - agent_start.y)) <= lane_half_height,
            "threatened_agent_column": abs(float(pos.x - agent_start.x)) <= column_half_width,
        }

    for _ in range(steps):
        probe_env.step(substeps=substeps)
        current_agent = probe_env.get_agent_record()
        agent_pos = _copy_vec2d(current_agent.body.position) if current_agent is not None else agent_start
        for record in records:
            item = state[record.name]
            pos = _copy_vec2d(record.body.position)
            item["end"] = pos
            item["min_x"] = min(float(item["min_x"]), float(pos.x))
            item["max_x"] = max(float(item["max_x"]), float(pos.x))
            item["min_y"] = min(float(item["min_y"]), float(pos.y))
            item["max_y"] = max(float(item["max_y"]), float(pos.y))
            item["min_agent_distance"] = min(
                float(item["min_agent_distance"]),
                float(pos.get_distance(agent_pos)),
            )
            if abs(float(pos.y - agent_start.y)) <= lane_half_height:
                item["entered_agent_lane"] = True
            if abs(float(pos.x - agent_start.x)) <= column_half_width:
                item["threatened_agent_column"] = True

    per_object: list[dict[str, Any]] = []
    passed_objects: list[str] = []
    for record in records:
        item = state[record.name]
        start = item["start"]
        end = item["end"]
        dx = float(end.x - start.x)
        dy = float(end.y - start.y)
        distance = float(end.get_distance(start))
        x_span = float(item["max_x"] - item["min_x"])
        y_span = float(item["max_y"] - item["min_y"])
        distance_reduction = float(item["start_agent_distance"] - item["min_agent_distance"])
        dynamic = record.body.body_type == pymunk.Body.DYNAMIC
        active_motion = distance >= min_displacement
        lane_stable = True
        if motion == "lateral":
            active_motion = x_span >= min_displacement
            lane_stable = y_span <= _safe_float(check.get("max_y_drift"), 52.0)
        elif motion == "falling":
            active_motion = y_span >= min_displacement
        elif motion == "pursuit":
            active_motion = distance_reduction >= min_distance_reduction and distance >= 8.0
        elif motion == "projectile":
            active_motion = distance >= max(120.0, min_displacement)
        lane_ok = (not must_enter_lane) or bool(item["entered_agent_lane"])
        column_ok = (not must_threaten_column) or bool(item["threatened_agent_column"])
        object_passed = dynamic and active_motion and lane_stable and lane_ok and column_ok
        if object_passed:
            passed_objects.append(record.name)
        per_object.append(
            {
                "name": record.name,
                "role": record.role,
                "kind": record.kind,
                "dynamic": dynamic,
                "dx": round(dx, 3),
                "dy": round(dy, 3),
                "distance": round(distance, 3),
                "x_span": round(x_span, 3),
                "y_span": round(y_span, 3),
                "distance_reduction": round(distance_reduction, 3),
                "entered_agent_lane": bool(item["entered_agent_lane"]),
                "threatened_agent_column": bool(item["threatened_agent_column"]),
                "lane_stable": bool(lane_stable),
                "start": _vec_list(start),
                "end": _vec_list(end),
            }
        )

    required = max(1, _safe_int(check.get("min_active_threats"), 1))
    passed = len(passed_objects) >= required
    return {
        "name": "active_threat_engagement",
        "passed": passed,
        "objects": [record.name for record in records],
        "metrics": {
            "check": check,
            "agent_start": _vec_list(agent_start),
            "lane_half_height": round(lane_half_height, 3),
            "column_half_width": round(column_half_width, 3),
            "steps": steps,
            "passed_objects": passed_objects,
            "required_passed_objects": required,
            "per_object": per_object,
        },
        "diagnosis": "active_threat_engaged" if passed else "active_threat_never_engaged",
        "repair": (
            "Prompt anti-cheat passed: active threats moved through the playable challenge lane."
            if passed
            else _active_threat_repair(check, per_object)
        ),
        "severity": "info" if passed else "error",
        "tier_evidence": 3 if passed else 1,
    }


def _probe_nontrivial_agent_interaction(
    env: BaseEnv,
    check: dict[str, Any],
    config: ValidatorConfig,
) -> dict[str, Any]:
    objective_state, objective_error = _evaluate_objective(env)
    passed = objective_error is None and not bool(objective_state)
    return {
        "name": "nontrivial_agent_interaction",
        "passed": passed,
        "objects": [],
        "metrics": {"check": check, "objective_initially_satisfied": bool(objective_state)},
        "diagnosis": "interaction_not_initially_solved" if passed else "interaction_starts_solved",
        "repair": (
            "Manipulation objective is not initially solved."
            if passed
            else "Prompt anti-cheat failure: the manipulation objective starts solved. Stage the object outside the target/sensor and require visible contact-driven motion before success."
        ),
        "severity": "info" if passed else "error",
        "tier_evidence": 1 if passed else 0,
    }


def _anti_cheat_matching_records(env: BaseEnv, check: dict[str, Any]) -> list[Any]:
    tokens = [item.lower() for item in _string_list(check.get("name_contains")) if str(item).strip()]
    explicit_names = {str(name) for name in _string_list(check.get("object_names"))}
    role = str(check.get("role") or "").lower().strip()
    matches = []
    fallback = []
    for record in getattr(env, "_objects", {}).values():
        text = " ".join(
            [
                str(record.name).lower(),
                str(record.role or "").lower(),
                str(record.kind or "").lower(),
                json.dumps(record.metadata, sort_keys=True, default=str).lower(),
            ]
        )
        if explicit_names and record.name not in explicit_names:
            continue
        token_match = not tokens or any(token in text for token in tokens)
        if not token_match:
            continue
        role_match = not role or str(record.role or "").lower() == role
        threat_role_match = str(record.role or "").lower() in {
            "hazard",
            "enemy",
            "chaser",
            "pursuer",
            "opponent",
            "obstacle",
        }
        if role_match or threat_role_match:
            matches.append(record)
        else:
            fallback.append(record)
    return matches or fallback


def _active_threat_repair(check: dict[str, Any], per_object: list[dict[str, Any]]) -> str:
    metrics = "; ".join(
        f"{item['name']}: dist={item['distance']}, x_span={item['x_span']}, "
        f"y_span={item['y_span']}, lane={item['entered_agent_lane']}, "
        f"column={item['threatened_agent_column']}, lane_stable={item.get('lane_stable')}, dynamic={item['dynamic']}, "
        f"start={item['start']}, end={item['end']}"
        for item in per_object[:5]
    )
    motion = str(check.get("motion") or "active").lower()
    if motion == "lateral":
        action = (
            "Open the side walls or spawn lane so cars/trucks/rolling threats can enter and cross the agent lane. "
            "Keep them lane-locked with near-zero vertical drift; do not let cars fall off-screen, drop through the floor, "
            "or get trapped behind world boundaries, guard rails, or decorative walls."
        )
    elif motion == "falling":
        action = (
            "Spawn hazards above the playable route with clear vertical travel and no shelves/ceilings catching them."
        )
    elif motion == "pursuit":
        action = (
            "Give chasers an open route to the agent and bounded pursuit force so distance actually decreases."
        )
    elif motion == "projectile":
        action = (
            "Activate projectiles immediately or within the first seconds and give them clear travel across the arena."
        )
    else:
        action = (
            "Make the named threats dynamic, give them velocity/force, and route them through the actual player challenge space."
        )
    return (
        "Prompt anti-cheat failure: the world technically has the requested threat, but the threat never "
        "engages the player challenge. "
        f"{action} The objective may not pass by leaving the main hazard static, blocked, off-route, or harmless. "
        f"Observed: {metrics}"
    )


def _run_semantic_requirement_probe(
    env: BaseEnv,
    requirement: dict[str, Any],
    config: ValidatorConfig,
) -> dict[str, Any]:
    kind = str(requirement.get("kind") or "object_motion")
    supported = {
        "dynamic_hazard_motion",
        "object_motion",
        "rotating_body",
        "bouncing_body",
        "field_motion",
    }
    if kind not in supported:
        return {
            "name": "semantic_dynamics",
            "passed": True,
            "tier_evidence": 1,
            "objects": [],
            "metrics": {"requirement": requirement, "supported": False},
            "diagnosis": "semantic_requirement_not_passively_probeable",
            "repair": "Requirement is recorded for telemetry but is not a passive semantic probe.",
            "severity": "info",
        }

    try:
        probe_env = env.__class__()
    except Exception as exc:
        return {
            "name": "semantic_dynamics",
            "passed": False,
            "tier_evidence": 1,
            "objects": [],
            "metrics": {"requirement": requirement},
            "diagnosis": "semantic_probe_env_rebuild_failed",
            "repair": f"Semantic probe could not rebuild the environment: {type(exc).__name__}: {exc}",
            "severity": "error",
        }

    effective_requirement = _effective_semantic_requirement(probe_env, requirement)
    records = _semantic_matching_records(probe_env, effective_requirement)
    if not records:
        return {
            "name": "semantic_dynamics",
            "passed": False,
            "tier_evidence": 1,
            "objects": [],
            "metrics": {"requirement": requirement},
            "diagnosis": "no_matching_semantic_objects",
            "repair": _semantic_no_match_repair(requirement),
            "severity": "error",
        }

    initial = {
        record.name: {
            "position": _copy_vec2d(record.body.position),
            "angle": float(record.body.angle),
            "body_type": int(record.body.body_type),
            "agent_distance": _distance_to_agent(probe_env, record),
        }
        for record in records
    }
    steps = _safe_int(effective_requirement.get("steps"), 150)
    substeps = max(1, int(config.kinetic_substeps))
    falling_required = _requires_visible_falling(effective_requirement)
    previous_positions = {
        record.name: _copy_vec2d(record.body.position) for record in records
    }
    observed_bounds = {
        record.name: {
            "min_x": float(record.body.position.x),
            "max_x": float(record.body.position.x),
            "min_y": float(record.body.position.y),
            "max_y": float(record.body.position.y),
        }
        for record in records
    }
    current_visible_drop: dict[str, float] = {record.name: 0.0 for record in records}
    max_visible_drop: dict[str, float] = {record.name: 0.0 for record in records}
    first_fall_step: dict[str, int] = {}
    reset_jumps: dict[str, int] = {record.name: 0 for record in records}
    for step_index in range(1, max(1, steps) + 1):
        action = {"move": [1.0, 0.0]} if falling_required and probe_env.get_agent_record() is not None else None
        probe_env.step(action=action, substeps=substeps)
        for record in records:
            previous = previous_positions[record.name]
            current = record.body.position
            bounds = observed_bounds[record.name]
            bounds["min_x"] = min(float(bounds["min_x"]), float(current.x))
            bounds["max_x"] = max(float(bounds["max_x"]), float(current.x))
            bounds["min_y"] = min(float(bounds["min_y"]), float(current.y))
            bounds["max_y"] = max(float(bounds["max_y"]), float(current.y))
            previous_positions[record.name] = _copy_vec2d(current)
            if not falling_required:
                continue
            downward_delta = _visible_downward_delta(probe_env, previous, current)
            if downward_delta > 0.0:
                current_visible_drop[record.name] += downward_delta
                if current_visible_drop[record.name] >= 24.0 and record.name not in first_fall_step:
                    first_fall_step[record.name] = step_index
            elif float(current.y - previous.y) > 80.0:
                reset_jumps[record.name] += 1
                current_visible_drop[record.name] = 0.0
            else:
                current_visible_drop[record.name] = max(0.0, current_visible_drop[record.name] * 0.98)
            max_visible_drop[record.name] = max(
                max_visible_drop[record.name], current_visible_drop[record.name]
            )

    per_object: list[dict[str, Any]] = []
    passed_objects: list[str] = []
    for record in records:
        start = initial[record.name]["position"]
        end = record.body.position
        dx = float(end.x - start.x)
        dy = float(end.y - start.y)
        bounds = observed_bounds[record.name]
        x_span = float(bounds["max_x"] - bounds["min_x"])
        y_span = float(bounds["max_y"] - bounds["min_y"])
        visible_drop_y = _visible_downward_drop(probe_env, start, end)
        if falling_required:
            visible_drop_y = max(visible_drop_y, max_visible_drop.get(record.name, 0.0))
        distance = float(end.get_distance(start))
        angle_delta = abs(float(record.body.angle) - float(initial[record.name]["angle"]))
        start_agent_distance = initial[record.name].get("agent_distance")
        end_agent_distance = _distance_to_agent(probe_env, record)
        agent_distance_reduction = (
            float(start_agent_distance - end_agent_distance)
            if start_agent_distance is not None and end_agent_distance is not None
            else 0.0
        )
        dynamic = record.body.body_type == pymunk.Body.DYNAMIC
        object_passed = dynamic and _semantic_motion_passed(
            effective_requirement,
            dx=dx,
            dy=dy,
            distance=distance,
            angle_delta=angle_delta,
            agent_distance_reduction=agent_distance_reduction,
            visible_drop_y=visible_drop_y,
            x_span=x_span,
            y_span=y_span,
        )
        if object_passed:
            passed_objects.append(record.name)
        per_object.append(
            {
                "name": record.name,
                "role": record.role,
                "kind": record.kind,
                "dynamic": dynamic,
                "dx": round(dx, 3),
                "dy": round(dy, 3),
                "distance": round(distance, 3),
                "x_span": round(x_span, 3),
                "y_span": round(y_span, 3),
                "visible_drop_y": round(visible_drop_y, 3),
                "max_visible_drop_segment": round(max_visible_drop.get(record.name, 0.0), 3),
                "first_fall_step": first_fall_step.get(record.name),
                "reset_jumps": reset_jumps.get(record.name, 0),
                "angle_delta": round(angle_delta, 4),
                "agent_distance_reduction": round(agent_distance_reduction, 3),
                "start": [round(float(start.x), 3), round(float(start.y), 3)],
                "end": [round(float(end.x), 3), round(float(end.y), 3)],
            }
        )

    min_objects = max(1, _safe_int(requirement.get("min_objects"), 1))
    passed = len(passed_objects) >= min_objects
    return {
        "name": "semantic_dynamics",
        "passed": passed,
        "tier_evidence": 2 if passed else 1,
        "objects": [record.name for record in records],
        "metrics": {
            "requirement": requirement,
            "effective_requirement": effective_requirement,
            "matched_objects": len(records),
            "passed_objects": passed_objects,
            "required_passed_objects": min_objects,
            "steps": steps,
            "per_object": per_object,
        },
        "diagnosis": "semantic_motion_observed" if passed else "semantic_motion_missing",
        "repair": "Semantic motion requirement satisfied."
        if passed
        else _semantic_motion_repair(effective_requirement, per_object),
        "severity": "info" if passed else "error",
    }


def _effective_semantic_requirement(env: BaseEnv, requirement: dict[str, Any]) -> dict[str, Any]:
    """Apply non-negotiable visual/semantic floors to LLM-authored requirements."""

    effective = dict(requirement)
    motion = str(effective.get("motion") or "").lower()
    axis = str(effective.get("axis") or "").lower()
    direction = str(effective.get("direction") or "").lower()
    text = " ".join(_string_list(effective.get("name_contains"))).lower()
    vehicle_lateral = (
        motion in {"lateral", "crossing", "driving", "rolling"} or axis == "x" or direction in {"left", "right"}
    ) and any(
        token in text
        for token in (
            "car",
            "truck",
            "train",
            "traffic",
            "vehicle",
            "rolling",
            "boulder",
            "rock",
            "stone",
            "barrel",
            "log",
        )
    )
    if vehicle_lateral:
        # Recurring lanes release objects in phases, so endpoint displacement
        # can understate real travel after resets. The span-aware probe below
        # proves visible horizontal travel without overfitting to exact timing.
        declared_x = max(
            _safe_float(effective.get("min_displacement_x"), 0.0),
            _safe_float(effective.get("min_displacement"), 0.0),
        )
        effective["motion"] = "lateral"
        effective["axis"] = "x"
        effective["min_displacement_x"] = 120.0
        effective["steps"] = max(_safe_int(effective.get("steps"), 150), 320)
        effective.setdefault(
            "visual_semantic_floor",
            "vehicle hazards must visibly travel horizontally through the lane; recurring resets are measured by x-span, not just final displacement",
        )
    if motion in {"falling", "dropping", "raining"} or direction == "down":
        height = float(getattr(env, "height", 640.0) or 640.0)
        visible_drop = min(220.0, max(140.0, height * 0.28))
        declared_drop = _safe_float(effective.get("min_displacement_y"), 0.0)
        effective["min_displacement_y"] = max(declared_drop, visible_drop)
        effective["steps"] = max(_safe_int(effective.get("steps"), 150), 180)
        effective.setdefault(
            "visual_semantic_floor",
            "falling hazards must visibly descend through open play space, not merely jiggle or settle onto a nearby shelf",
        )
    if motion in {"ballistic", "projectile", "shooting", "laser", "shot"}:
        declared_distance = _safe_float(effective.get("min_displacement"), 0.0)
        effective["min_displacement"] = max(declared_distance, 120.0)
        effective["steps"] = max(_safe_int(effective.get("steps"), 150), 180)
        effective.setdefault(
            "visual_semantic_floor",
            "projectile hazards must visibly travel across the arena, not remain parked or barely drift",
        )
    if motion in {"chasing", "pursuing", "pursuit", "patrolling"} or direction == "toward_agent":
        declared_reduction = _safe_float(effective.get("min_distance_reduction"), 0.0)
        effective["min_distance_reduction"] = max(declared_reduction, 30.0)
        effective["steps"] = max(_safe_int(effective.get("steps"), 150), 180)
        effective.setdefault(
            "visual_semantic_floor",
            "pursuers must visibly reduce distance to the agent; planar maze chasers should not be judged as falling/freeflight hazards",
        )
    return effective


def _visible_downward_drop(env: BaseEnv, start: pymunk.Vec2d, end: pymunk.Vec2d) -> float:
    """Measure downward travel that is actually visible inside the world bounds."""

    height = float(getattr(env, "height", 0.0) or 0.0)
    if height <= 0.0:
        return max(0.0, float(start.y - end.y))
    visible_start_y = min(max(float(start.y), 0.0), height)
    visible_end_y = min(max(float(end.y), 0.0), height)
    return max(0.0, visible_start_y - visible_end_y)


def _visible_downward_delta(env: BaseEnv, previous: pymunk.Vec2d, current: pymunk.Vec2d) -> float:
    """Per-step visible downward travel, robust to offscreen parking/spawn resets."""

    height = float(getattr(env, "height", 0.0) or 0.0)
    if height <= 0.0:
        return max(0.0, float(previous.y - current.y))
    previous_visible = 0.0 <= float(previous.y) <= height
    current_visible = 0.0 <= float(current.y) <= height
    crosses_visible_band = float(previous.y) > height and float(current.y) < height
    if not (previous_visible or current_visible or crosses_visible_band):
        return 0.0
    visible_previous_y = min(max(float(previous.y), 0.0), height)
    visible_current_y = min(max(float(current.y), 0.0), height)
    return max(0.0, visible_previous_y - visible_current_y)


def _requires_visible_falling(requirement: dict[str, Any]) -> bool:
    motion = str(requirement.get("motion") or "").lower()
    direction = str(requirement.get("direction") or "").lower()
    return motion in {"falling", "dropping", "raining"} or direction == "down"


def _semantic_matching_records(env: BaseEnv, requirement: dict[str, Any]) -> list[Any]:
    names = {str(name) for name in _string_list(requirement.get("object_names"))}
    role = str(requirement.get("role") or "").lower().strip()
    contains = [item.lower() for item in _string_list(requirement.get("name_contains")) if str(item).strip()]
    matches = []
    for record in getattr(env, "_objects", {}).values():
        record_name = str(record.name).lower()
        record_role = str(record.role or "").lower()
        metadata = json.dumps(record.metadata, sort_keys=True, default=str).lower()
        text = " ".join([record_name, record_role, str(record.kind).lower(), metadata])
        if names and record.name not in names:
            continue
        role_matches = not role or record_role == role
        if not role_matches and role == "hazard" and any(
            token in text for token in ("chaser", "pursuer", "angry", "enemy", "monster")
        ):
            role_matches = record_role in {"hazard", "enemy", "chaser", "pursuer", "opponent"}
        if not role_matches and role == "projectile" and any(
            token in text for token in ("projectile", "bullet", "missile", "laser", "fired_by")
        ):
            role_matches = record_role in {"projectile", "hazard", "tool", "object", ""}
        name_matches = not contains or any(token in text for token in contains)
        if role_matches and name_matches:
            matches.append(record)
    return matches


def _semantic_motion_passed(
    requirement: dict[str, Any],
    *,
    dx: float,
    dy: float,
    distance: float,
    angle_delta: float,
    agent_distance_reduction: float = 0.0,
    visible_drop_y: float | None = None,
    x_span: float | None = None,
    y_span: float | None = None,
) -> bool:
    motion = str(requirement.get("motion") or "").lower()
    axis = str(requirement.get("axis") or "any").lower()
    direction = str(requirement.get("direction") or "any").lower()
    min_distance = _safe_float(requirement.get("min_displacement"), 20.0)
    min_x = _safe_float(requirement.get("min_displacement_x"), min_distance)
    min_y = _safe_float(requirement.get("min_displacement_y"), min_distance)
    min_angle = _safe_float(requirement.get("min_angle_delta"), 0.15)
    min_distance_reduction = _safe_float(requirement.get("min_distance_reduction"), 30.0)

    if direction == "toward_agent" or motion in {"chasing", "pursuing", "pursuit", "patrolling"}:
        return agent_distance_reduction >= min_distance_reduction and distance >= min(8.0, min_distance)
    if axis == "angle" or motion in {"rotating", "spinning", "orbiting"}:
        return angle_delta >= min_angle
    if direction == "down" or motion in {"falling", "dropping", "raining"}:
        observed_drop = visible_drop_y if visible_drop_y is not None else -dy
        return observed_drop >= min_y
    if direction == "up":
        return dy >= min_y
    if direction == "left":
        return max(-dx, float(x_span or 0.0)) >= min_x
    if direction == "right":
        return max(dx, float(x_span or 0.0)) >= min_x
    if axis == "x":
        return max(abs(dx), float(x_span or 0.0)) >= min_x
    if axis == "y":
        return max(abs(dy), float(y_span or 0.0)) >= min_y
    return distance >= min_distance


def _semantic_no_match_repair(requirement: dict[str, Any]) -> str:
    role = requirement.get("role")
    name_contains = requirement.get("name_contains")
    return (
        "Prompt fidelity failure: no registered objects matched the semantic requirement. "
        f"Create/register objects with role={role!r} and name/metadata containing {name_contains!r}, "
        "then implement the requested physical behavior."
    )


def _semantic_motion_repair(requirement: dict[str, Any], per_object: list[dict[str, Any]]) -> str:
    motion = str(requirement.get("motion") or "motion")
    direction = str(requirement.get("direction") or "any")
    metrics = "; ".join(
        f"{item['name']}: dx={item['dx']}, dy={item['dy']}, "
        f"visible_drop_y={item.get('visible_drop_y')}, "
        f"max_visible_drop_segment={item.get('max_visible_drop_segment')}, "
        f"start={item.get('start')}, end={item.get('end')}, "
        f"agent_distance_reduction={item.get('agent_distance_reduction')}, "
        f"dist={item['distance']}, dynamic={item['dynamic']}"
        for item in per_object[:4]
    )
    if motion in {"falling", "dropping", "raining"} or direction == "down":
        return (
            "Prompt fidelity failure: hazards exist but did not fall downward enough. "
            "Make them dynamic hazard bodies above the route, remove supporting shelves/platforms, "
            "enable gravity or downward velocity, and ensure they travel through the play area. "
            "If a hazard stops near the top boundary, spawn it below the ceiling or cut ceiling gaps above drop lanes. "
            f"Observed: {metrics}"
        )
    if motion in {"drifting", "floating"}:
        return (
            "Prompt fidelity failure: objects exist but did not visibly drift. "
            "Use dynamic bodies in zero/low gravity with initial velocity or a force zone. "
            f"Observed: {metrics}"
        )
    if motion in {"chasing", "pursuing", "pursuit", "patrolling"} or requirement.get("direction") == "toward_agent":
        return (
            "Prompt fidelity failure: chasing hazards exist but did not move closer to the agent. "
            "Use dynamic hazard bodies and implement bounded pursuit in after_step() by applying force toward self.agent.body.position. "
            "For top-down chase worlds, use planar/top_down_flat movement with zero vertical gravity or stable support so chasers do not fall away from the agent. "
            "Each chaser must have an open route/branch mouth into the maze corridor, receive force every step, and reduce its distance to the agent by the required amount. "
            "Do not teleport or directly mutate positions. "
            f"Observed: {metrics}"
        )
    if motion in {"ballistic", "projectile", "shooting", "laser", "shot"}:
        role_text = str(requirement.get("role") or "").lower()
        noun = "projectile hazards" if role_text == "hazard" else "projectile objects"
        return (
            f"Prompt fidelity failure: {noun} exist but did not travel far enough to read as shots. "
            "Make at least one projectile active immediately or within the first 30 simulation steps, "
            "give it initial velocity or bounded force so it travels at least 120 px during passive validation, "
            "and avoid parked inactive projectile pool bodies matching semantic_requirements. "
            f"Observed: {metrics}"
        )
    return (
        f"Prompt fidelity failure: requested {motion!r} behavior was not observed. "
        "Make the referenced objects dynamic and ensure their position/angle changes during headless simulation. "
        f"Observed: {metrics}"
    )


def _first_probe_object(probe: dict[str, Any]) -> str | None:
    objects = probe.get("objects")
    if isinstance(objects, list) and objects:
        return str(objects[0])
    return None


def _distance_to_agent(env: BaseEnv, record: Any) -> float | None:
    agent = env.get_agent_record()
    if agent is None:
        return None
    try:
        return float(record.body.position.get_distance(agent.body.position))
    except Exception:
        return None


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _valid_objective_target_names(
    env: BaseEnv,
    structural_details: dict[str, Any],
) -> list[str]:
    names = structural_details.get("objective_targets") or []
    valid_names: list[str] = []
    for name in names:
        name = str(name)
        if name == "agent":
            continue
        if name in env._objects:
            valid_names.append(name)
    return valid_names


def verify_subgoal_plan(
    env: BaseEnv,
    config: ValidatorConfig,
    structural_details: dict[str, Any],
) -> ValidationResult:
    """Execute generic objective subgoals before falling back to type-specific oracles."""

    env_class = env.__class__.__name__
    agent = env.get_agent_record()
    if agent is None:
        return ValidationResult(
            False,
            "Structural validation failed: missing dynamic agent with role='agent'.",
            env_class,
            details={**structural_details, "oracle": "subgoal_plan"},
            contract_valid=True,
            structurally_valid=False,
            objective_valid=False,
            kinetic_progress=False,
            kinetic_solved=False,
        )

    route = _profile_dict(structural_details.get("validator_route"))
    subgoals = route.get("subgoals") or _subgoal_list(
        _profile_dict(structural_details.get("objective_profile"))
    )
    if not isinstance(subgoals, list) or not subgoals:
        return _generic_objective_trial(env, config, structural_details)

    probe_plan = [probe_plan_for_subgoal(subgoal).to_dict() for subgoal in subgoals]
    affordance_failures = _subgoal_affordance_failures(env, subgoals, config)
    if affordance_failures:
        first_failure = affordance_failures[0]
        failed_subgoal = first_failure.get("subgoal")
        failed_index = int(first_failure.get("subgoal_index") or 1)
        failure_text = "; ".join(str(item.get("message")) for item in affordance_failures[:3])
        return ValidationResult(
            False,
            f"Affordance check failed before rollout: {failure_text}",
            env_class,
            blocking_object=_subgoal_blocking_object(failed_subgoal)
            if isinstance(failed_subgoal, dict)
            else None,
            details={
                **structural_details,
                "oracle": "subgoal_affordance",
                "failed_subgoal": failed_subgoal,
                "failed_subgoal_index": failed_index,
                "probe_plan": probe_plan,
                "affordance_failures": affordance_failures,
                "affordance_probes": _probe_results_from_failures(affordance_failures),
                "affordance_summary": _format_affordance_summary(affordance_failures),
            },
            contract_valid=True,
            structurally_valid=True,
            objective_valid=True,
            kinetic_progress=False,
            kinetic_solved=False,
        )

    pre_rollout_probes = _subgoal_affordance_probe_results(env, subgoals, config)
    agent = env.get_agent_record()
    if agent is None:
        return ValidationResult(
            False,
            "Structural validation failed after pre-rollout probes: missing dynamic agent with role='agent'.",
            env_class,
            details={
                **structural_details,
                "oracle": "subgoal_plan",
                "probe_plan": probe_plan,
                "pre_rollout_probes": pre_rollout_probes,
            },
            contract_valid=True,
            structurally_valid=False,
            objective_valid=False,
            kinetic_progress=False,
            kinetic_solved=False,
        )
    agent_start = _copy_vec2d(agent.body.position)
    initial_distances = _subgoal_distances(env, subgoals)
    completed: list[dict[str, Any]] = []
    progress_events: list[dict[str, Any]] = []
    mechanism_events: list[dict[str, Any]] = []
    last_error: str | None = None

    budget_per_subgoal = max(120, config.kinetic_steps * 2)
    total_steps = 0

    for subgoal_index, subgoal in enumerate(subgoals, start=1):
        kind = str(subgoal.get("kind") or "").strip()
        if not kind:
            last_error = f"Subgoal {subgoal_index} is missing kind."
            break
        subgoal_diagnostics = _start_subgoal_diagnostics(env, subgoal, config)
        subgoal_budget = _subgoal_rollout_budget(
            config,
            kind=kind,
            completed_subgoals=completed,
            base_budget=budget_per_subgoal,
            subgoal=subgoal,
        )
        for local_step in range(subgoal_budget):
            total_steps += 1
            if _subgoal_satisfied(env, subgoal, config):
                # Let check_objective observe the exact contact/entry frame.
                # Generated objectives often persist flags such as
                # ball_touched/crossed_boundary from state predicates.
                objective_state, _ = _evaluate_objective(env)
                if subgoal_index == len(subgoals) and not objective_state:
                    _record_subgoal_diagnostics(
                        subgoal_diagnostics,
                        env,
                        subgoal,
                        local_step,
                        config,
                    )
                    env.step(substeps=config.kinetic_substeps)
                    continue
                completed.append(subgoal)
                if kind == "activate_mechanism":
                    next_subgoal = subgoals[subgoal_index] if subgoal_index < len(subgoals) else None
                    mechanism_events.append(
                        _mechanism_state_diagnostics(env, subgoal, next_subgoal, config)
                    )
                break
            try:
                _step_subgoal(env, agent, subgoal, config)
                _record_subgoal_diagnostics(
                    subgoal_diagnostics,
                    env,
                    subgoal,
                    local_step,
                    config,
                )
            except Exception as exc:
                return ValidationResult(
                    False,
                    f"Subgoal simulation failed: {type(exc).__name__}: {exc}",
                    env_class,
                    blocking_object=_subgoal_blocking_object(subgoal),
                    details={
                        **structural_details,
                        "oracle": "subgoal_plan",
                        "failed_subgoal": subgoal,
                        "failed_subgoal_index": subgoal_index,
                        "probe_plan": probe_plan,
                        "pre_rollout_probes": pre_rollout_probes,
                        "completed_subgoals": completed,
                        "mechanism_diagnostics": mechanism_events,
                        "total_steps": total_steps,
                        "subgoal_diagnostics": _finish_subgoal_diagnostics(
                            subgoal_diagnostics,
                            env,
                            subgoal,
                            config,
                        ),
                    },
                    contract_valid=True,
                    structurally_valid=True,
                    objective_valid=False,
                    kinetic_progress=bool(completed)
                    or _subgoal_progress_observed(env, subgoals, initial_distances, config),
                    kinetic_solved=False,
                )

            objective_state, objective_error = _evaluate_objective(env)
            if objective_error is not None:
                return ValidationResult(
                    False,
                    objective_error,
                    env_class,
                    blocking_object=_subgoal_blocking_object(subgoal),
                    details={
                        **structural_details,
                        "oracle": "subgoal_plan",
                        "failed_subgoal": subgoal,
                        "failed_subgoal_index": subgoal_index,
                        "probe_plan": probe_plan,
                        "pre_rollout_probes": pre_rollout_probes,
                        "completed_subgoals": completed,
                        "mechanism_diagnostics": mechanism_events,
                        "total_steps": total_steps,
                        "subgoal_diagnostics": _finish_subgoal_diagnostics(
                            subgoal_diagnostics,
                            env,
                            subgoal,
                            config,
                        ),
                    },
                    contract_valid=True,
                    structurally_valid=True,
                    objective_valid=False,
                    kinetic_progress=bool(completed)
                    or _subgoal_progress_observed(env, subgoals, initial_distances, config),
                    kinetic_solved=False,
                )
            if objective_state:
                return ValidationResult(
                    True,
                    "Code-level objective satisfied during subgoal validation.",
                    env_class,
                    details={
                        **structural_details,
                        "oracle": "subgoal_plan",
                        "completed_subgoals": [*completed, subgoal],
                        "probe_plan": probe_plan,
                        "pre_rollout_probes": pre_rollout_probes,
                        "objective_step": total_steps,
                        "mechanism_diagnostics": mechanism_events,
                        "subgoal_progress": _subgoal_progress_report(
                            env,
                            subgoals,
                            initial_distances,
                        ),
                    },
                    contract_valid=True,
                    structurally_valid=True,
                    objective_valid=True,
                    kinetic_progress=True,
                    kinetic_solved=True,
                )
        else:
            if _subgoal_satisfied(env, subgoal, config):
                objective_state, _ = _evaluate_objective(env)
                if objective_state or subgoal_index < len(subgoals):
                    completed.append(subgoal)
                    continue
            objective_state, objective_error = _evaluate_objective(env)
            if objective_error is not None:
                return ValidationResult(
                    False,
                    objective_error,
                    env_class,
                    blocking_object=_subgoal_blocking_object(subgoal),
                    details={
                        **structural_details,
                        "oracle": "subgoal_plan",
                        "failed_subgoal": subgoal,
                        "failed_subgoal_index": subgoal_index,
                        "probe_plan": probe_plan,
                        "pre_rollout_probes": pre_rollout_probes,
                        "completed_subgoals": completed,
                        "mechanism_diagnostics": mechanism_events,
                        "total_steps": total_steps,
                        "subgoal_diagnostics": _finish_subgoal_diagnostics(
                            subgoal_diagnostics,
                            env,
                            subgoal,
                            config,
                        ),
                    },
                    contract_valid=True,
                    structurally_valid=True,
                    objective_valid=False,
                    kinetic_progress=bool(completed)
                    or _subgoal_progress_observed(env, subgoals, initial_distances, config),
                    kinetic_solved=False,
                )
            if objective_state:
                return ValidationResult(
                    True,
                    "Code-level objective satisfied during subgoal validation.",
                    env_class,
                    details={
                        **structural_details,
                        "oracle": "subgoal_plan",
                        "completed_subgoals": [*completed, subgoal],
                        "probe_plan": probe_plan,
                        "pre_rollout_probes": pre_rollout_probes,
                        "objective_step": total_steps,
                        "mechanism_diagnostics": mechanism_events,
                        "subgoal_progress": _subgoal_progress_report(
                            env,
                            subgoals,
                            initial_distances,
                        ),
                    },
                    contract_valid=True,
                    structurally_valid=True,
                    objective_valid=True,
                    kinetic_progress=True,
                    kinetic_solved=True,
                )
            diagnostics = _finish_subgoal_diagnostics(
                subgoal_diagnostics,
                env,
                subgoal,
                config,
            )
            last_error = (
                f"Subgoal {subgoal_index} ({kind}) did not complete within "
                f"{subgoal_budget} steps."
            )
            if diagnostics is not None:
                last_error = _diagnostic_failure_reason(last_error, diagnostics)
            progress_events.append(
                {
                    "subgoal_index": subgoal_index,
                    "subgoal": subgoal,
                    "progress": _subgoal_progress_report(env, [subgoal], initial_distances),
                    "diagnostics": diagnostics,
                }
            )
            break

    objective_state, objective_error = _evaluate_objective(env)
    if objective_error is not None:
        last_error = objective_error
        objective_valid = False
    else:
        objective_valid = True
        if objective_state:
            return ValidationResult(
                True,
                "Code-level objective satisfied after subgoal validation.",
                env_class,
                details={
                    **structural_details,
                    "oracle": "subgoal_plan",
                    "completed_subgoals": completed,
                    "probe_plan": probe_plan,
                    "pre_rollout_probes": pre_rollout_probes,
                    "objective_step": total_steps,
                    "mechanism_diagnostics": mechanism_events,
                    "subgoal_progress": _subgoal_progress_report(
                        env,
                        subgoals,
                        initial_distances,
                    ),
                },
                contract_valid=True,
                structurally_valid=True,
                objective_valid=True,
                kinetic_progress=True,
                kinetic_solved=True,
            )

    if objective_valid and len(completed) == len(subgoals):
        for grace_step in range(45):
            total_steps += 1
            try:
                env.step(None, substeps=config.kinetic_substeps)
            except Exception as exc:
                last_error = f"Post-subgoal objective grace step failed: {type(exc).__name__}: {exc}"
                objective_valid = False
                break
            objective_state, objective_error = _evaluate_objective(env)
            if objective_error is not None:
                last_error = objective_error
                objective_valid = False
                break
            if objective_state:
                return ValidationResult(
                    True,
                    "Code-level objective satisfied during post-subgoal grace validation.",
                    env_class,
                    details={
                        **structural_details,
                        "oracle": "subgoal_plan",
                        "completed_subgoals": completed,
                        "probe_plan": probe_plan,
                        "pre_rollout_probes": pre_rollout_probes,
                        "objective_step": total_steps,
                        "objective_grace_steps": grace_step + 1,
                        "mechanism_diagnostics": mechanism_events,
                        "subgoal_progress": _subgoal_progress_report(
                            env,
                            subgoals,
                            initial_distances,
                        ),
                    },
                    contract_valid=True,
                    structurally_valid=True,
                    objective_valid=True,
                    kinetic_progress=True,
                    kinetic_solved=True,
                )

    progress = bool(completed) or _subgoal_progress_observed(
        env,
        subgoals,
        initial_distances,
        config,
    )
    return ValidationResult(
        False,
        last_error or "Objective not satisfied during subgoal validation.",
        env_class,
        blocking_object=_subgoal_blocking_object(subgoals[len(completed)])
        if len(completed) < len(subgoals)
        else None,
        details={
            **structural_details,
            "oracle": "subgoal_plan",
            "completed_subgoals": completed,
            "probe_plan": probe_plan,
            "pre_rollout_probes": pre_rollout_probes,
            "failed_subgoal": subgoals[len(completed)]
            if len(completed) < len(subgoals)
            else None,
            "subgoal_progress": _subgoal_progress_report(env, subgoals, initial_distances),
            "progress_events": progress_events,
            "mechanism_diagnostics": mechanism_events,
            "subgoal_diagnostics": progress_events[-1].get("diagnostics")
            if progress_events
            else None,
            "total_steps": total_steps,
            "agent_displacement": float(agent.body.position.get_distance(agent_start)),
        },
        contract_valid=True,
        structurally_valid=True,
        objective_valid=objective_valid,
        kinetic_progress=progress,
        kinetic_solved=False,
    )


def _generic_objective_trial(
    env: BaseEnv,
    config: ValidatorConfig,
    structural_details: dict[str, Any],
) -> ValidationResult:
    target = _find_kinetic_target(env, None)
    if target is None:
        return ValidationResult(
            False,
            "No subgoals or kinetic target available for generic validation.",
            env.__class__.__name__,
            details={**structural_details, "oracle": "generic_objective_trial"},
            contract_valid=True,
            structurally_valid=True,
            objective_valid=True,
            kinetic_progress=False,
            kinetic_solved=False,
        )
    return verify_multi_target_touch(env, config, structural_details)


def _select_validator_route(structural_details: dict[str, Any]) -> dict[str, Any]:
    """Choose the current oracle from objective and capability profiles."""

    objective_profile = _profile_dict(structural_details.get("objective_profile"))
    capability_profile = _profile_dict(structural_details.get("capability_profile"))
    objective_type = str(
        objective_profile.get("objective_type")
        or structural_details.get("objective_type")
        or "custom_physics"
    )
    movement = str(capability_profile.get("movement") or "ground_force")
    gravity = str(capability_profile.get("gravity") or "normal")
    interaction = _string_list(capability_profile.get("interaction"))
    validator_skills = _string_list(objective_profile.get("validator_skills"))
    required_capabilities = _string_list(objective_profile.get("required_capabilities"))
    progress_metrics = _string_list(objective_profile.get("progress_metrics"))
    allowed_controls = _string_list(capability_profile.get("allowed_controls"))
    forbidden_controls = _string_list(capability_profile.get("forbidden_controls"))
    subgoals = _subgoal_list(objective_profile)
    physics_relations = _profile_dict(structural_details.get("physics_relations"))
    relation_subgoals = _relation_graph_subgoals(physics_relations)
    if relation_subgoals and _relation_subgoals_align_with_objective_names(
        relation_subgoals,
        subgoals,
        structural_details,
    ):
        # The relation graph is generated by the harness before codegen and is
        # the mechanic source of truth. Use it to refine matching mechanical
        # subgoals, but preserve any downstream objective steps the relation
        # graph did not project, such as "then reach the exit".
        subgoals = _merge_relation_and_objective_subgoals(
            relation_subgoals,
            subgoals,
        )

    signal_tokens = {
        objective_type,
        movement,
        gravity,
        *interaction,
        *validator_skills,
        *required_capabilities,
        *progress_metrics,
    }
    lowered = {token.lower() for token in signal_tokens}

    if subgoals:
        oracle = "subgoal_plan"
    elif objective_type in {"multi_target_touch", "single_target_touch"}:
        oracle = "multi_target_touch" if objective_type != "single_target_touch" else "single_target_touch"
    elif objective_type == "navigation_goal" or any(
        token in lowered for token in {"navigation_probe", "navigation", "distance_to_goal"}
    ):
        oracle = "navigation_bfs"
    elif objective_type in {"mechanism_activation", "seesaw_balance", "push_object"} or any(
        token in lowered
        for token in {
            "push_contact",
            "ride_platform",
            "contact_trigger",
            "mechanism_progress_probe",
            "lever_activation",
            "plank_angle",
            "object_displacement",
        }
    ):
        oracle = "mechanism_progress_probe"
    elif (
        "touch_contact" in lowered
        and any("target" in token or "touch" in token for token in lowered)
    ):
        oracle = "multi_target_touch"
    else:
        oracle = "generic_progress_probe"

    return {
        "oracle": oracle,
        "objective_type": objective_type,
        "movement": movement,
        "gravity": gravity,
        "interaction": interaction,
        "validator_skills": validator_skills,
        "required_capabilities": required_capabilities,
        "progress_metrics": progress_metrics,
        "subgoals": subgoals,
        "physics_relations": physics_relations,
        "relation_subgoals": relation_subgoals,
        "allowed_controls": allowed_controls,
        "forbidden_controls": forbidden_controls,
    }


def _subgoal_list(objective_profile: dict[str, Any]) -> list[dict[str, Any]]:
    subgoals = objective_profile.get("subgoals")
    if not isinstance(subgoals, list):
        return []
    return [dict(item) for item in subgoals if isinstance(item, dict)]


def _relation_graph_subgoals(physics_relations: dict[str, Any]) -> list[dict[str, Any]]:
    """Return executable subgoals projected by the relation graph, if present."""

    subgoals = physics_relations.get("suggested_subgoals")
    if not isinstance(subgoals, list):
        return []
    return [dict(item) for item in subgoals if isinstance(item, dict)]


def _merge_relation_and_objective_subgoals(
    relation_subgoals: list[dict[str, Any]],
    objective_subgoals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Refine objective subgoals with relation projections without dropping goals.

    The deterministic relation graph often describes the core mechanic, e.g.
    "push crate to trigger -> activate mechanism". The LLM-authored objective
    profile may then add a final semantic step, e.g. "agent reaches exit". If we
    replace the whole objective with only the relation projection, validation can
    stop before the actual win condition. Keep relation subgoals first and append
    objective subgoals that are not already represented.
    """

    merged = [dict(item) for item in relation_subgoals]
    seen = {_subgoal_identity(item) for item in merged}
    for item in objective_subgoals:
        identity = _subgoal_identity(item)
        if identity in seen:
            continue
        merged.append(dict(item))
        seen.add(identity)
    return merged


def _subgoal_identity(subgoal: dict[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
    stable_keys = (
        "kind",
        "interaction",
        "object",
        "region",
        "target",
        "trigger",
        "mechanism",
        "effect",
        "boundary",
        "weight",
        "plank",
    )
    return (
        str(subgoal.get("kind") or ""),
        tuple((key, str(subgoal.get(key) or "")) for key in stable_keys if key in subgoal),
    )


def _relation_subgoals_align_with_objective_names(
    relation_subgoals: list[dict[str, Any]],
    objective_subgoals: list[dict[str, Any]],
    structural_details: dict[str, Any],
) -> bool:
    """Avoid stale placeholder relation names overriding concrete objective names."""

    objective_names = {
        str(item)
        for item in structural_details.get("objective_targets") or []
        if str(item)
    }
    objective_names.update(_subgoal_referenced_names(objective_subgoals))
    relation_names = _subgoal_referenced_names(relation_subgoals)
    if not relation_names:
        return True
    if not objective_names:
        return True
    generic_placeholders = {"object", "target", "targets", "ball", "target_zone", "region"}
    missing = relation_names - objective_names
    if missing & generic_placeholders:
        return False
    return len(missing) == 0 or len(relation_names & objective_names) >= max(1, len(relation_names) // 2)


def _subgoal_referenced_names(subgoals: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for subgoal in subgoals:
        for key in ("target", "object", "region", "field", "trigger", "effect", "mechanism", "boundary"):
            value = subgoal.get(key)
            if isinstance(value, str) and value:
                names.add(value)
    return names


def _physics_parameter_probe_results(
    env: BaseEnv,
    physics_relations: dict[str, Any],
) -> list[dict[str, Any]]:
    """Inspect relation-level mass/friction/gravity/sensor/clearance contracts."""

    if not isinstance(physics_relations, dict):
        return []
    raw_relations = physics_relations.get("relations")
    if not isinstance(raw_relations, list):
        return []

    probes: list[dict[str, Any]] = []
    for relation_index, raw_relation in enumerate(raw_relations, start=1):
        if not isinstance(raw_relation, dict):
            continue
        constraints = raw_relation.get("parameter_constraints")
        if not isinstance(constraints, dict) or not constraints:
            continue
        relation_type = str(raw_relation.get("type") or f"relation_{relation_index}")
        object_names = [str(item) for item in raw_relation.get("objects") or [] if item]
        region_names = [str(item) for item in raw_relation.get("regions") or [] if item]
        primary_object = _first_existing_record(env, object_names)
        primary_region = _first_existing_record(env, region_names)

        for key, constraint in constraints.items():
            if not isinstance(constraint, dict):
                continue
            if key == "object_mass":
                probes.append(
                    _numeric_parameter_probe(
                        name="object_mass_bounds",
                        value=float(primary_object.body.mass) if primary_object is not None else None,
                        constraint=constraint,
                        relation_type=relation_type,
                        objects=(primary_object.name,) if primary_object is not None else tuple(object_names),
                        repair="Tune the object's mass through self.layout so the declared relation is physically possible.",
                    )
                )
            elif key in {"object_friction", "affected_object_friction"}:
                probes.append(
                    _numeric_parameter_probe(
                        name="object_friction_bounds",
                        value=_record_max_shape_attr(primary_object, "friction") if primary_object is not None else None,
                        constraint=constraint,
                        relation_type=relation_type,
                        objects=(primary_object.name,) if primary_object is not None else tuple(object_names),
                        repair="Tune dynamic object friction through self.layout before redesigning the task.",
                    )
                )
            elif key in {"object_restitution", "object_elasticity"}:
                probes.append(
                    _numeric_parameter_probe(
                        name="object_elasticity_bounds",
                        value=_record_max_shape_attr(primary_object, "elasticity") if primary_object is not None else None,
                        constraint=constraint,
                        relation_type=relation_type,
                        objects=(primary_object.name,) if primary_object is not None else tuple(object_names),
                        repair="Tune object elasticity/restitution through self.layout so impacts are lively but stable.",
                    )
                )
            elif key == "target_sensor":
                required = bool(constraint.get("required", True))
                observed = _record_is_sensor(primary_region) if primary_region is not None else None
                passed = observed is not None and bool(observed) == required
                probes.append(
                    ProbeResult(
                        name="target_sensor_contract",
                        passed=passed,
                        tier_evidence=2 if passed else 1,
                        objects=(primary_region.name,) if primary_region is not None else tuple(region_names),
                        metrics={
                            "relation_type": relation_type,
                            "required_sensor": required,
                            "observed_sensor": observed,
                            "constraint": constraint,
                        },
                        diagnosis="target_sensor_matches_relation" if passed else "target_sensor_mismatch",
                        repair=(
                            "Make the target/goal/region a non-blocking sensor=True shape at the physical success location."
                            if required
                            else "Remove sensor=True from the target only if the prompt requires a solid blocker."
                        ),
                        severity="info" if passed else "warning",
                    ).to_dict()
                )
            elif key == "object_to_region_distance":
                distance = None
                if primary_object is not None and primary_region is not None:
                    distance = float(primary_object.body.position.get_distance(primary_region.body.position))
                probes.append(
                    _numeric_parameter_probe(
                        name="object_to_region_distance_bounds",
                        value=distance,
                        constraint=constraint,
                        relation_type=relation_type,
                        objects=tuple(record.name for record in (primary_object, primary_region) if record is not None),
                        repair="Move object_start or target_center closer through self.layout while preserving the requested relation.",
                    )
                )
            elif key == "clearance_margin":
                clearance = _ballistic_clearance_margin(env, primary_object, primary_region)
                if clearance is None:
                    probes.append(
                        ProbeResult(
                            name="barrier_clearance_margin",
                            passed=True,
                            tier_evidence=2,
                            objects=tuple(record.name for record in (primary_object, primary_region) if record is not None),
                            metrics={
                                "relation_type": relation_type,
                                "value": None,
                                "constraint": constraint,
                                "blocking_barrier_detected": False,
                            },
                            diagnosis="no_static_barrier_clearance_needed",
                            repair="No solid barrier was detected between object and region.",
                            severity="info",
                        ).to_dict()
                    )
                else:
                    probes.append(
                        _numeric_parameter_probe(
                            name="barrier_clearance_margin",
                            value=clearance,
                            constraint=constraint,
                            relation_type=relation_type,
                            objects=tuple(record.name for record in (primary_object, primary_region) if record is not None),
                            repair="Lower barrier_height, raise target_center, increase target_size, or tune impulse/elasticity so the arc clears the blocker.",
                        )
                    )
            elif key == "gravity":
                probes.append(_gravity_parameter_probe(env, constraint, relation_type))
            elif key == "field_strength":
                probes.append(_field_strength_parameter_probe(env, constraint, relation_type, raw_relation))

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for probe in probes:
        key = json.dumps(probe, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(probe)
    return deduped


def _first_existing_record(env: BaseEnv, names: list[str]):
    for name in names:
        record = getattr(env, "_objects", {}).get(name)
        if record is not None:
            return record
    return None


def _record_max_shape_attr(record, attr: str) -> float | None:
    values: list[float] = []
    for shape in getattr(record, "shapes", ()) or ():
        value = getattr(shape, attr, None)
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return max(values)


def _numeric_parameter_probe(
    *,
    name: str,
    value: float | None,
    constraint: dict[str, Any],
    relation_type: str,
    objects: tuple[str, ...],
    repair: str,
) -> dict[str, Any]:
    passed = value is not None
    min_value = _safe_float(constraint.get("min"), None)
    max_value = _safe_float(constraint.get("max"), None)
    if value is not None and min_value is not None:
        passed = passed and value >= min_value
    if value is not None and max_value is not None:
        passed = passed and value <= max_value
    return ProbeResult(
        name=name,
        passed=bool(passed),
        tier_evidence=2 if passed else 1,
        objects=objects,
        metrics={
            "relation_type": relation_type,
            "value": None if value is None else round(float(value), 3),
            "min": min_value,
            "max": max_value,
            "constraint": constraint,
        },
        diagnosis=f"{name}_ok" if passed else f"{name}_outside_contract",
        repair=repair,
        severity="info" if passed else "warning",
    ).to_dict()


def _gravity_parameter_probe(
    env: BaseEnv,
    constraint: dict[str, Any],
    relation_type: str,
) -> dict[str, Any]:
    gravity = getattr(env.space, "gravity", pymunk.Vec2d(0.0, 0.0))
    gy = float(gravity.y)
    observed = "zero_g" if abs(float(gravity.x)) < 1.0 and abs(gy) < 1.0 else ("normal" if gy < -100.0 else "custom")
    allowed = [str(item) for item in constraint.get("allowed") or []]
    passed = not allowed or observed in allowed
    return ProbeResult(
        name="gravity_model_contract",
        passed=passed,
        tier_evidence=2 if passed else 1,
        objects=(),
        metrics={
            "relation_type": relation_type,
            "observed_model": observed,
            "gravity": [round(float(gravity.x), 3), round(float(gravity.y), 3)],
            "allowed": allowed,
            "constraint": constraint,
        },
        diagnosis="gravity_matches_relation" if passed else "gravity_mismatch_for_relation",
        repair="Align EnvConfig.gravity and capability_profile['gravity'] with the relation graph and explicit prompt physics.",
        severity="info" if passed else "warning",
    ).to_dict()


def _field_strength_parameter_probe(
    env: BaseEnv,
    constraint: dict[str, Any],
    relation_type: str,
    relation: dict[str, Any],
) -> dict[str, Any]:
    force_zones = getattr(env, "_force_zones", {})
    fields = [str(item) for item in relation.get("fields") or [] if item]
    zone_records = [
        zone
        for name, zone in force_zones.items()
        if not fields or str(name) in fields or str(getattr(zone, "name", "")) in fields
    ]
    strengths: list[float] = []
    for zone in zone_records:
        strength = getattr(zone, "strength", None)
        if strength is not None:
            strengths.append(abs(float(strength)))
        force = getattr(zone, "force", None)
        if force is not None:
            try:
                strengths.append(float(force.length))
            except Exception:
                pass
    value = max(strengths) if strengths else None
    return _numeric_parameter_probe(
        name="field_strength_bounds",
        value=value,
        constraint=constraint,
        relation_type=relation_type,
        objects=tuple(fields),
        repair="Increase field_strength or registered force magnitude through self.layout, and ensure affected_names/roles include the target object.",
    )


def _ballistic_clearance_margin(env: BaseEnv, object_record, region_record) -> float | None:
    if object_record is None or region_record is None:
        return None
    required_apex = _ballistic_clearance_apex_y(env, object_record, region_record)
    if required_apex is None:
        return None
    object_y = float(object_record.body.position.y)
    region_y = float(region_record.body.position.y)
    return required_apex - max(object_y, region_y)


def _subgoal_affordance_failures(
    env: BaseEnv,
    subgoals: list[dict[str, Any]],
    config: ValidatorConfig,
) -> list[dict[str, Any]]:
    """Return pre-rollout physical affordance failures for executable subgoals."""

    failures: list[dict[str, Any]] = []
    for index, subgoal in enumerate(subgoals, start=1):
        kind = str(subgoal.get("kind") or "")
        if kind in {"move_object_to_region", "low_friction_slide_to_region", "strike_object_to_region", "ballistic_object_to_region"}:
            failures.extend(
                _move_object_affordance_failures(
                    env,
                    subgoal,
                    index,
                    config,
                    sequential_after_object_motion=_is_sequential_object_motion_subgoal(
                        subgoals,
                        index,
                        subgoal,
                    ),
                )
            )
        elif kind == "support_exit_freefall":
            failures.extend(_support_exit_affordance_failures(env, subgoal, index, config))
        elif kind in {"agent_reach_region", "agent_touch_object"}:
            failures.extend(_agent_target_affordance_failures(env, subgoal, index, config))
        elif kind == "field_force_interaction":
            failures.extend(_field_affordance_failures(env, subgoal, index, config))
        elif kind == "lever_launch":
            failures.extend(_lever_affordance_failures(env, subgoal, index, config))
    return failures


def _subgoal_affordance_probe_results(
    env: BaseEnv,
    subgoals: list[dict[str, Any]],
    config: ValidatorConfig,
) -> list[dict[str, Any]]:
    """Return pre-rollout probes even when they pass, so repair can preserve them."""

    probes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for subgoal in subgoals:
        kind = str(subgoal.get("kind") or "")
        if kind in {"move_object_to_region", "low_friction_slide_to_region", "strike_object_to_region", "ballistic_object_to_region"}:
            object_record = _subgoal_record(env, subgoal, "object")
            region_record = _subgoal_record(env, subgoal, "region")
            agent = env.get_agent_record()
            if object_record is not None and not _is_agent_fired_projectile_subgoal(env, subgoal, object_record):
                probes.append(
                    passive_stability_probe(
                        env,
                        object_record.name,
                        steps=60,
                        substeps=config.kinetic_substeps,
                        max_displacement=8.0,
                    ).to_dict()
                )
                agent = env.get_agent_record()
                object_record = _subgoal_record(env, subgoal, "object")
                region_record = _subgoal_record(env, subgoal, "region")
            if agent is not None and object_record is not None and region_record is not None:
                probes.append(
                    object_region_affordance_probe(
                        env,
                        agent_name=agent.name,
                        object_name=object_record.name,
                        region_name=region_record.name,
                        agent_radius=config.agent_radius,
                        max_object_to_region=340.0 if kind == "ballistic_object_to_region" else (260.0 if _is_strike_subgoal(subgoal) else 140.0),
                        max_agent_to_object=190.0 if kind == "ballistic_object_to_region" else (170.0 if _is_strike_subgoal(subgoal) else 220.0),
                        max_alignment_degrees=75.0 if kind == "ballistic_object_to_region" else (25.0 if _is_strike_subgoal(subgoal) else 35.0),
                    ).to_dict()
                )
        elif kind == "support_exit_freefall":
            object_record = _subgoal_record(env, subgoal, "object")
            boundary_record = _subgoal_record(env, subgoal, "boundary") or _subgoal_record(env, subgoal, "region")
            if object_record is not None:
                probes.append(
                    passive_stability_probe(
                        env,
                        object_record.name,
                        steps=45,
                        substeps=config.kinetic_substeps,
                        max_displacement=10.0,
                    ).to_dict()
                )
            if object_record is not None and boundary_record is not None:
                distance = float(object_record.body.position.get_distance(boundary_record.body.position))
                probes.append(
                    ProbeResult(
                        name="support_exit_affordance",
                        passed=distance <= 220.0,
                        tier_evidence=3 if distance <= 220.0 else 2,
                        objects=(object_record.name, boundary_record.name),
                        metrics={"object_to_boundary_distance": round(distance, 3), "recommended_max": 220.0},
                        diagnosis="support_exit_staged" if distance <= 220.0 else "boundary_too_far",
                        repair="Stage the object near the true support edge/boundary and keep open space beyond it.",
                        severity="info" if distance <= 220.0 else "warning",
                    ).to_dict()
                )
        elif kind in {"agent_reach_region", "agent_touch_object"}:
            target_name = (
                subgoal.get("target")
                if kind == "agent_reach_region"
                else subgoal.get("object") or subgoal.get("target")
            )
            probes.append(
                concrete_target_probe(
                    env,
                    subgoal_kind=kind,
                    target_name=str(target_name) if target_name is not None else None,
                ).to_dict()
            )
        elif kind == "field_force_interaction":
            object_record = _subgoal_record(env, subgoal, "object")
            field_record = _subgoal_record(env, subgoal, "field")
            if object_record is not None and field_record is not None:
                registered = field_record.name in getattr(env, "_force_zones", {})
                probes.append(
                    ProbeResult(
                        name="force_zone_registered",
                        passed=registered,
                        tier_evidence=2 if registered else 1,
                        objects=(object_record.name, field_record.name),
                        metrics={
                            "registered": registered,
                            "field_name": field_record.name,
                            "object_name": object_record.name,
                        },
                        diagnosis="force_zone_available" if registered else "force_zone_missing",
                        repair=(
                            "Force zone is registered."
                            if registered
                            else "Use self.register_force_zone(...) for field-force interactions."
                        ),
                        severity="info" if registered else "error",
                    ).to_dict()
                )
        elif kind == "lever_launch":
            plank_record = _subgoal_record(env, subgoal, "plank")
            weight_record = _subgoal_record(env, subgoal, "weight") or _subgoal_record(
                env,
                subgoal,
                "object",
            )
            if plank_record is not None:
                probes.append(
                    pivot_mechanism_probe(
                        plank_name=plank_record.name,
                        weight_name=weight_record.name if weight_record is not None else None,
                        has_pivot_constraint=_has_pivot_constraint(env, plank_record),
                        plank_is_dynamic=plank_record.body.body_type == pymunk.Body.DYNAMIC,
                        weight_is_dynamic=None
                        if weight_record is None
                        else weight_record.body.body_type == pymunk.Body.DYNAMIC,
                    ).to_dict()
                )

    deduped: list[dict[str, Any]] = []
    for probe in probes:
        key = json.dumps(probe, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(probe)
    return deduped


def _is_sequential_object_motion_subgoal(
    subgoals: list[dict[str, Any]],
    subgoal_index: int,
    subgoal: dict[str, Any],
) -> bool:
    """Return True when a later subgoal depends on prior motion of the same object."""

    object_name = str(subgoal.get("object") or "")
    if not object_name or subgoal_index <= 1:
        return False
    motion_kinds = {
        "move_object_to_region",
        "low_friction_slide_to_region",
        "strike_object_to_region",
        "field_force_interaction",
        "lever_launch",
    }
    for previous in subgoals[: subgoal_index - 1]:
        if str(previous.get("kind") or "") not in motion_kinds:
            continue
        if str(previous.get("object") or previous.get("weight") or "") == object_name:
            return True
    return False


def _is_agent_fired_projectile_subgoal(
    env: BaseEnv,
    subgoal: dict[str, Any],
    object_record: Any | None = None,
) -> bool:
    if str(subgoal.get("kind") or "") != "ballistic_object_to_region":
        return False
    interaction = str(subgoal.get("interaction") or "").lower()
    if interaction == "agent_fired_projectile_impact":
        return True
    name = str(subgoal.get("object") or "").lower()
    record = object_record or _subgoal_record(env, subgoal, "object")
    metadata = getattr(record, "metadata", {}) if record is not None else {}
    metadata_text = json.dumps(metadata, sort_keys=True, default=str).lower()
    role = str(getattr(record, "role", "") or "").lower() if record is not None else ""
    kind = str(getattr(record, "kind", "") or "").lower() if record is not None else ""
    object_text = " ".join([name, role, kind, metadata_text])
    if any(token in object_text for token in ("agent_bullet", "agent_projectile", "fired_by\": \"agent", "agent_fired_projectile")):
        return True
    if any(token in name for token in ("bullet", "projectile", "missile", "laser")) and _prompt_requests_agent_projectile_impact(
        _source_prompt_from_env(env)
    ):
        return True
    return False


def _record_is_agent_fired_projectile(env: BaseEnv, record: Any) -> bool:
    name = str(getattr(record, "name", "") or "").lower()
    role = str(getattr(record, "role", "") or "").lower()
    kind = str(getattr(record, "kind", "") or "").lower()
    metadata = getattr(record, "metadata", {}) or {}
    metadata_text = json.dumps(metadata, sort_keys=True, default=str).lower()
    object_text = " ".join([name, role, kind, metadata_text])
    if any(token in object_text for token in ("agent_bullet", "agent_projectile", "agent_fired_projectile", "fired_by\": \"agent")):
        return True
    return any(token in name for token in ("bullet", "projectile", "missile", "laser")) and _prompt_requests_agent_projectile_impact(
        _source_prompt_from_env(env)
    )


def _move_object_affordance_failures(
    env: BaseEnv,
    subgoal: dict[str, Any],
    subgoal_index: int,
    config: ValidatorConfig,
    *,
    sequential_after_object_motion: bool = False,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    agent = env.get_agent_record()
    object_record = _subgoal_record(env, subgoal, "object")
    region_record = _subgoal_record(env, subgoal, "region")
    object_name = str(subgoal.get("object") or "<missing object>")
    region_name = str(subgoal.get("region") or "<missing region>")

    if agent is None:
        return [
            _affordance_failure(
                subgoal,
                subgoal_index,
                "missing_agent",
                "move_object_to_region has no dynamic role='agent' body to apply force.",
            )
        ]
    if object_record is None:
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "missing_object",
                f"move_object_to_region references {object_name!r}, but no registered object exists.",
            )
        )
    if region_record is None:
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "missing_region",
                f"move_object_to_region references {region_name!r}, but no registered region exists.",
            )
        )
    if object_record is None or region_record is None:
        return failures

    agent_projectile = _is_agent_fired_projectile_subgoal(env, subgoal, object_record)
    if not agent_projectile:
        stability_probe = passive_stability_probe(
            env,
            object_record.name,
            steps=60,
            substeps=config.kinetic_substeps,
            max_displacement=8.0,
        )
        stability = dict(stability_probe.metrics)
        if not stability_probe.passed:
            failures.append(
                _affordance_failure(
                    subgoal,
                    subgoal_index,
                    stability_probe.diagnosis or "object_not_passively_stable",
                    (
                        f"{object_record.name} drifts {stability['object_displacement']:.1f} px "
                        "with no agent action before the push starts; add a stable floor, rail, "
                        "or low-friction guide so the object is controllable."
                    ),
                    {**stability, "probe_result": stability_probe.to_dict()},
                )
            )
            agent = env.get_agent_record()
            object_record = _subgoal_record(env, subgoal, "object")
            region_record = _subgoal_record(env, subgoal, "region")
            if agent is None or object_record is None or region_record is None:
                return failures

    metrics = _move_object_affordance_metrics(agent, object_record, region_record, config)
    strike_subgoal = _is_strike_subgoal(subgoal)
    ballistic_subgoal = str(subgoal.get("kind") or "") == "ballistic_object_to_region"
    max_object_to_region = 760.0 if agent_projectile else (340.0 if ballistic_subgoal else (260.0 if strike_subgoal else 140.0))
    max_agent_to_object = 190.0 if ballistic_subgoal else (170.0 if strike_subgoal else 220.0)
    max_alignment = 75.0 if ballistic_subgoal else (25.0 if strike_subgoal else 35.0)
    spatial_probe = object_region_affordance_probe(
        env,
        agent_name=agent.name,
        object_name=object_record.name,
        region_name=region_record.name,
        agent_radius=config.agent_radius,
        max_object_to_region=max_object_to_region,
        max_agent_to_object=max_agent_to_object,
        max_alignment_degrees=max_alignment,
    )
    metrics = {**metrics, "probe_result": spatial_probe.to_dict()}
    if object_record.body.body_type != pymunk.Body.DYNAMIC:
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "object_not_dynamic",
                f"{object_record.name} must be dynamic for move_object_to_region, but it is static/kinematic.",
                metrics,
            )
        )
    if _record_looks_like_region(region_record) and not _record_is_sensor(region_record):
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "region_not_sensor",
                f"{region_record.name} is a target region but is not sensor=True; it can physically block {object_record.name}.",
                metrics,
            )
        )
    if not sequential_after_object_motion and metrics["object_to_region_distance"] > max_object_to_region + 2.0:
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "object_too_far_from_region",
                (
                    f"{object_record.name} starts {metrics['object_to_region_distance']:.1f} px "
                    f"from {region_record.name}; recommended <= {max_object_to_region:.1f} px for "
                    f"{'ballistic/throw' if ballistic_subgoal else ('strike/shot' if strike_subgoal else 'generic push')} validation."
                ),
                metrics,
            )
        )
    if not sequential_after_object_motion and metrics["agent_to_object_distance"] > max_agent_to_object:
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "agent_too_far_from_object",
                (
                    f"agent starts {metrics['agent_to_object_distance']:.1f} px from "
                    f"{object_record.name}; recommended <= {max_agent_to_object:.1f} px for the "
                    f"{'shot setup' if strike_subgoal else 'first push'}."
                ),
                metrics,
            )
        )
    if not sequential_after_object_motion and metrics["alignment_angle_degrees"] > max_alignment:
        label = "throw arc staging" if ballistic_subgoal else (
            "shot lane alignment" if strike_subgoal else "push alignment"
        )
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "poor_ballistic_staging" if ballistic_subgoal else "poor_push_alignment",
                (
                    f"agent -> {object_record.name} -> {region_record.name} alignment is "
                    f"{metrics['alignment_angle_degrees']:.1f} degrees; recommended <= {max_alignment:.1f} "
                    f"degrees for {label}."
                ),
                metrics,
            )
        )

    blocker = None if (sequential_after_object_motion or ballistic_subgoal) else _solid_blocker_between(
        env,
        object_record,
        region_record,
        config,
    )
    if blocker is not None:
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "solid_blocker_between_object_and_region",
                f"solid blocker {blocker.name} lies between {object_record.name} and {region_record.name}.",
                metrics,
            )
        )

    mass_ratio = metrics["object_mass"] / max(metrics["agent_mass"], 0.001)
    if mass_ratio > 5.0:
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "object_too_heavy",
                (
                    f"{object_record.name} mass ratio is {mass_ratio:.1f}x agent mass; "
                    "recommended <= 5.0x unless a mechanical advantage is present."
                ),
                metrics,
            )
        )
    return failures


def _field_affordance_failures(
    env: BaseEnv,
    subgoal: dict[str, Any],
    subgoal_index: int,
    config: ValidatorConfig,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    object_record = _subgoal_record(env, subgoal, "object")
    field_record = _subgoal_record(env, subgoal, "field")
    object_name = str(subgoal.get("object") or "<missing object>")
    field_name = str(subgoal.get("field") or "<missing field>")
    if object_record is None:
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "missing_field_object",
                f"field_force_interaction references {object_name!r}, but no registered dynamic object exists.",
            )
        )
    if field_record is None:
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "missing_force_zone",
                f"field_force_interaction references {field_name!r}, but no registered field/zone exists.",
            )
        )
    if object_record is None or field_record is None:
        return failures

    if object_record.body.body_type != pymunk.Body.DYNAMIC:
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "field_object_not_dynamic",
                f"{object_record.name} must be dynamic so the force zone can influence it.",
            )
        )
    if field_record.name not in getattr(env, "_force_zones", {}):
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "field_not_registered_as_force_zone",
                (
                    f"{field_record.name} is not a BaseEnv force zone. Use "
                    "self.register_force_zone(...) instead of overriding step() or "
                    "making a plain sensor box for magnetic/wind/current forces."
                ),
            )
        )

    field_distance = float(object_record.body.position.get_distance(field_record.body.position))
    metrics = {
        "object_to_field_distance": round(field_distance, 3),
        "recommended_object_to_field_max": 220.0,
        "field_is_sensor": _record_is_sensor(field_record),
        "force_zone_registered": field_record.name in getattr(env, "_force_zones", {}),
    }
    if field_distance > 260.0:
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "object_too_far_from_force_zone",
                (
                    f"{object_record.name} starts {field_distance:.1f} px from "
                    f"{field_record.name}; place it closer or add a preceding "
                    "move_object_to_region subgoal that pushes it into the force zone."
                ),
                metrics,
            )
        )
    if not _record_is_sensor(field_record):
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "force_zone_not_sensor",
                f"{field_record.name} should be sensor=True so it applies force without blocking.",
                metrics,
            )
        )
    return failures


def _support_exit_affordance_failures(
    env: BaseEnv,
    subgoal: dict[str, Any],
    subgoal_index: int,
    config: ValidatorConfig,
) -> list[dict[str, Any]]:
    """Pre-rollout checks for object-leaves-support/freefall relations."""

    failures: list[dict[str, Any]] = []
    object_record = _subgoal_record(env, subgoal, "object")
    boundary_record = _subgoal_record(env, subgoal, "boundary") or _subgoal_record(env, subgoal, "region")
    if object_record is None:
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "missing_object",
                "support_exit_freefall requires a concrete dynamic object name.",
            )
        )
        return failures
    if object_record.body.body_type != pymunk.Body.DYNAMIC:
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "object_not_dynamic",
                f"{object_record.name} must be dynamic before it can exit support/freefall.",
            )
        )
    if boundary_record is None:
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "missing_boundary",
                "support_exit_freefall requires a registered boundary/region sensor aligned with the real support edge.",
            )
        )
        return failures

    stability_probe = passive_stability_probe(
        env,
        object_record.name,
        steps=30,
        substeps=config.kinetic_substeps,
        max_displacement=14.0,
    )
    metrics = dict(stability_probe.metrics)
    metrics.update(
        {
            "object_position": _vec_list(object_record.body.position),
            "boundary_position": _vec_list(boundary_record.body.position),
            "object_to_boundary_distance": round(
                float(object_record.body.position.get_distance(boundary_record.body.position)),
                3,
            ),
            "recommended_object_to_boundary_max": 220.0,
            "probe_result": stability_probe.to_dict(),
        }
    )
    if not stability_probe.passed:
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "object_not_passively_stable",
                (
                    f"{object_record.name} drifts {metrics.get('object_displacement', 0.0):.1f} px "
                    "before contact; stage it on a stable support before the edge."
                ),
                metrics,
            )
        )
    if metrics["object_to_boundary_distance"] > 220.0:
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "boundary_too_far",
                (
                    f"{object_record.name} starts {metrics['object_to_boundary_distance']:.1f} px "
                    f"from {boundary_record.name}; recommended <= 220 px for support-exit validation."
                ),
                metrics,
            )
        )
    return failures


def _lever_affordance_failures(
    env: BaseEnv,
    subgoal: dict[str, Any],
    subgoal_index: int,
    config: ValidatorConfig,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    plank_record = _subgoal_record(env, subgoal, "plank")
    weight_record = _subgoal_record(env, subgoal, "weight") or _subgoal_record(
        env,
        subgoal,
        "object",
    )
    target_record = _subgoal_record(env, subgoal, "target")
    impact_record = _subgoal_record(env, subgoal, "impact_region") or _subgoal_record(
        env,
        subgoal,
        "region",
    )
    plank_name = str(subgoal.get("plank") or "<missing plank>")
    if plank_record is None:
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "missing_plank",
                f"lever_launch references {plank_name!r}, but no registered plank exists.",
            )
        )
        return failures

    mechanism_probe = pivot_mechanism_probe(
        plank_name=plank_record.name,
        weight_name=weight_record.name if weight_record is not None else None,
        has_pivot_constraint=_has_pivot_constraint(env, plank_record),
        plank_is_dynamic=plank_record.body.body_type == pymunk.Body.DYNAMIC,
        weight_is_dynamic=None
        if weight_record is None
        else weight_record.body.body_type == pymunk.Body.DYNAMIC,
    )
    if not mechanism_probe.passed:
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                mechanism_probe.diagnosis or "pivot_mechanism_invalid",
                mechanism_probe.repair,
                {"probe_result": mechanism_probe.to_dict()},
            )
        )

    if weight_record is None:
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "missing_weight",
                "lever_launch should name a dynamic weight/object that loads the lever.",
            )
        )
    elif weight_record.body.body_type != pymunk.Body.DYNAMIC:
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "weight_not_dynamic",
                f"{weight_record.name} must be dynamic so it can transfer energy into the lever.",
            )
        )

    if impact_record is not None and weight_record is not None:
        distance = float(weight_record.body.position.get_distance(impact_record.body.position))
        metrics = {
            "weight_to_impact_distance": round(distance, 3),
            "recommended_weight_to_impact_max": 160.0,
            "impact_region_is_sensor": _record_is_sensor(impact_record),
        }
        if distance > 180.0:
            failures.append(
                _affordance_failure(
                    subgoal,
                    subgoal_index,
                    "weight_too_far_from_impact_region",
                    (
                        f"{weight_record.name} starts {distance:.1f} px from "
                        f"{impact_record.name}; recommended <= 160 px for lever launch validation."
                    ),
                    metrics,
                )
            )
        if _record_looks_like_region(impact_record) and not _record_is_sensor(impact_record):
            failures.append(
                _affordance_failure(
                    subgoal,
                    subgoal_index,
                    "impact_region_not_sensor",
                    f"{impact_record.name} should be sensor=True so it marks the load side without blocking the weight.",
                    metrics,
                )
            )

    if target_record is None and subgoal.get("target") is not None:
        failures.append(
            _affordance_failure(
                subgoal,
                subgoal_index,
                "missing_launch_target",
                f"lever_launch target {subgoal.get('target')!r} is not registered.",
            )
        )
    return failures


def _agent_target_affordance_failures(
    env: BaseEnv,
    subgoal: dict[str, Any],
    subgoal_index: int,
    config: ValidatorConfig,
) -> list[dict[str, Any]]:
    kind = str(subgoal.get("kind") or "")
    if kind not in {"agent_reach_region", "agent_touch_object"}:
        return []
    target_field = "target" if kind == "agent_reach_region" else "object"
    target_name = subgoal.get(target_field) or subgoal.get("target") or subgoal.get("object")
    target_probe = concrete_target_probe(
        env,
        subgoal_kind=kind,
        target_name=str(target_name) if target_name is not None else None,
    )
    if target_name is None:
        return [
            _affordance_failure(
                subgoal,
                subgoal_index,
                "missing_target_name",
                f"{kind} must name a concrete registered target/object, not an implicit target.",
                {"probe_result": target_probe.to_dict()},
            )
        ]
    target_record = _subgoal_record(env, subgoal, "target") or _subgoal_record(
        env,
        subgoal,
        "object",
    )
    if target_record is None:
        return [
            _affordance_failure(
                subgoal,
                subgoal_index,
                "target_not_registered",
                (
                    f"{kind} references {str(target_name)!r}, but no registered object/region "
                    "with that exact name exists. Use explicit concrete names, not aliases like any_bumper."
                ),
                {"probe_result": target_probe.to_dict()},
            )
        ]
    return []


def _move_object_affordance_metrics(agent, object_record, region_record, config: ValidatorConfig) -> dict[str, Any]:
    agent_pos = agent.body.position
    object_pos = object_record.body.position
    region_pos = region_record.body.position
    agent_to_object = object_pos - agent_pos
    object_to_region = region_pos - object_pos
    return {
        "agent_to_object_distance": round(float(agent_to_object.length), 3),
        "object_to_region_distance": round(float(object_to_region.length), 3),
        "alignment_angle_degrees": round(_angle_between_degrees(agent_to_object, object_to_region), 3),
        "agent_mass": round(float(agent.body.mass), 3),
        "object_mass": round(float(object_record.body.mass), 3),
        "object_radius": round(_record_radius(object_record, config.agent_radius), 3),
        "region_is_sensor": _record_is_sensor(region_record),
        "recommended_object_to_region_max": 140.0,
        "recommended_agent_to_object_max": 220.0,
        "recommended_alignment_angle_max": 35.0,
    }


def _passive_object_stability(
    env: BaseEnv,
    object_name: str,
    config: ValidatorConfig,
) -> dict[str, Any]:
    record = env._objects.get(object_name)
    if record is None:
        return {"object_displacement": 0.0, "reason": "missing_object"}
    start = _copy_vec2d(record.body.position)
    steps = 60
    try:
        for _ in range(steps):
            env.step(substeps=config.kinetic_substeps)
        refreshed = env._objects.get(object_name)
        end = _copy_vec2d(refreshed.body.position) if refreshed is not None else start
        displacement = float(start.get_distance(end))
        return {
            "object_displacement": round(displacement, 3),
            "start_position": [round(float(start.x), 3), round(float(start.y), 3)],
            "end_position": [round(float(end.x), 3), round(float(end.y), 3)],
            "passive_steps": steps,
            "recommended_max_displacement": 8.0,
        }
    finally:
        env.reset()


def _angle_between_degrees(first: pymunk.Vec2d, second: pymunk.Vec2d) -> float:
    if first.length <= 1e-6 or second.length <= 1e-6:
        return 0.0
    cosine = max(-1.0, min(1.0, float(first.normalized().dot(second.normalized()))))
    return math.degrees(math.acos(cosine))


def _record_is_sensor(record) -> bool:
    return bool(record.shapes) and all(bool(shape.sensor) for shape in record.shapes)


def _record_looks_like_region(record) -> bool:
    role = str(record.role or "").lower()
    name = str(record.name or "").lower()
    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    metadata_role = str(metadata.get("role") or metadata.get("kind") or "").lower()
    tokens = ("plate", "switch", "goal", "zone", "target", "trigger", "checkpoint", "region", "pad")
    return role in {"goal", "trigger", "region"} or metadata_role in {"goal", "trigger", "region"} or any(
        token in name for token in tokens
    )


def _record_point_inside_record(container_record, point_record) -> bool:
    point = point_record.body.position
    for shape in container_record.shapes:
        if shape.point_query(point).distance <= 0.0:
            return True
    return False


def _has_pivot_constraint(env: BaseEnv, record) -> bool:
    for constraint_record in getattr(env, "_constraints", {}).values():
        constraint = constraint_record.constraint
        if not isinstance(constraint, pymunk.PivotJoint):
            continue
        if constraint.a is record.body or constraint.b is record.body:
            return True
    return False


def _solid_blocker_between(env: BaseEnv, source, target, config: ValidatorConfig):
    start = source.body.position
    end = target.body.position
    expansion = max(_record_radius(source, config.agent_radius), config.agent_radius) + 4.0
    samples = 16
    for record in env._objects.values():
        if record is source or record is target or record.role == "agent" or _record_is_sensor(record):
            continue
        if record.body.body_type != pymunk.Body.STATIC:
            continue
        for shape in record.shapes:
            bb = shape.bb
            if (
                any(
                    token in record.name.lower()
                    for token in ("floor", "ground", "shelf", "support", "ice", "rink", "lane")
                )
                and float(bb.top) <= min(float(start.y), float(end.y)) + 4.0
            ):
                continue
            if float(bb.top) < min(float(start.y), float(end.y)) - expansion * 0.5:
                continue
            if float(bb.bottom) > max(float(start.y), float(end.y)) + expansion * 0.5:
                continue
            for sample_index in range(1, samples):
                alpha = sample_index / samples
                point = start + (end - start) * alpha
                if _expanded_bb_contains(bb, point, expansion):
                    if _static_guide_only_bounds_path(record, bb, point):
                        continue
                    return record
    return None


def _static_guide_only_bounds_path(record, bb, point: pymunk.Vec2d) -> bool:
    """Treat tangent guide rails as lane boundaries, not centerline blockers."""
    name = str(record.name or "").lower()
    role = str(record.role or "").lower()
    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    metadata_kind = str(metadata.get("kind") or metadata.get("role") or "").lower()
    looks_like_guide = (
        any(token in name for token in ("rail", "guide", "lane", "bumper", "stop"))
        or any(token in role for token in ("rail", "guide", "lane", "bumper", "stop"))
        or any(token in metadata_kind for token in ("rail", "guide", "lane", "bumper", "stop"))
    )
    if not looks_like_guide:
        return False
    return not _expanded_bb_contains(bb, point, 0.0)


def _expanded_bb_contains(bb, point: pymunk.Vec2d, expansion: float) -> bool:
    return (
        float(bb.left) - expansion <= float(point.x) <= float(bb.right) + expansion
        and float(bb.bottom) - expansion <= float(point.y) <= float(bb.top) + expansion
    )


def _affordance_failure(
    subgoal: dict[str, Any],
    subgoal_index: int,
    code: str,
    message: str,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "subgoal_index": subgoal_index,
        "subgoal": dict(subgoal),
        "code": code,
        "message": message,
        "metrics": dict(metrics or {}),
    }


def _format_affordance_summary(failures: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- subgoal {failure.get('subgoal_index')}: {failure.get('code')} - {failure.get('message')}"
        for failure in failures
    )


def _probe_results_from_failures(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for failure in failures:
        metrics = failure.get("metrics")
        if not isinstance(metrics, dict):
            continue
        probe = metrics.get("probe_result")
        if not isinstance(probe, dict):
            continue
        key = json.dumps(probe, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        probes.append(probe)
    return probes


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _route_allows_force_control(route: dict[str, Any]) -> bool:
    allowed = {item.lower() for item in _string_list(route.get("allowed_controls"))}
    forbidden = {item.lower() for item in _string_list(route.get("forbidden_controls"))}
    force_controls = {
        "apply_force",
        "apply_force_x",
        "apply_force_y",
        "force",
        "force_x",
        "force_y",
        "move",
        "move_left",
        "move_right",
        "move_up",
        "move_down",
        "left",
        "right",
        "up",
        "down",
        "arrow_left",
        "arrow_right",
        "arrow_up",
        "arrow_down",
        "w",
        "a",
        "s",
        "d",
        "wasd",
        "arrow_keys",
        "thrust",
        "thrust_2d",
        "thrust_left",
        "thrust_right",
        "thrust_up",
        "thrust_down",
        "thrust_forward",
        "thrust_backward",
        "rotate_left",
        "rotate_right",
        "ground_force",
    }
    return bool((allowed & force_controls) - forbidden)


def _capability_mismatch_result(
    env_class: str,
    structural_details: dict[str, Any],
    reason: str,
) -> ValidationResult:
    return ValidationResult(
        False,
        reason,
        env_class,
        details={**structural_details, "failure_category": "capability_mismatch"},
        contract_valid=True,
        structurally_valid=True,
        objective_valid=True,
        kinetic_progress=False,
        kinetic_solved=False,
    )


def _record_radius(record, fallback: float = 12.0) -> float:
    for shape in record.shapes:
        if isinstance(shape, pymunk.Circle):
            return float(shape.radius)
    for shape in record.shapes:
        bb = shape.bb
        width = float(bb.right - bb.left)
        height = float(bb.top - bb.bottom)
        if width > 0.0 and height > 0.0:
            return max(width, height) / 2.0
    return float(fallback)


def _record_min_radius(record, fallback: float = 12.0) -> float:
    for shape in record.shapes:
        if isinstance(shape, pymunk.Circle):
            return float(shape.radius)
    for shape in record.shapes:
        bb = shape.bb
        width = float(bb.right - bb.left)
        height = float(bb.top - bb.bottom)
        if width > 0.0 and height > 0.0:
            return min(width, height) / 2.0
    return float(fallback)


def _start_subgoal_diagnostics(
    env: BaseEnv,
    subgoal: dict[str, Any],
    config: ValidatorConfig,
) -> dict[str, Any] | None:
    kind = str(subgoal.get("kind") or "")
    agent = env.get_agent_record()
    if agent is None:
        return None
    if kind in {"move_object_to_region", "low_friction_slide_to_region", "strike_object_to_region", "ballistic_object_to_region"}:
        object_record = _subgoal_record(env, subgoal, "object")
        region_record = _subgoal_record(env, subgoal, "region")
        if object_record is None or region_record is None:
            return None
        initial = _push_rollout_state(env, agent, object_record, region_record, config)
        return {
            "kind": kind,
            "object": object_record.name,
            "region": region_record.name,
            "initial": initial,
            "samples": [initial],
        }
    if kind == "support_exit_freefall":
        object_record = _subgoal_record(env, subgoal, "object")
        boundary_record = _subgoal_record(env, subgoal, "boundary") or _subgoal_record(
            env,
            subgoal,
            "region",
        )
        if object_record is None or boundary_record is None:
            return None
        initial = _push_rollout_state(env, agent, object_record, boundary_record, config)
        return {
            "kind": kind,
            "object": object_record.name,
            "boundary": boundary_record.name,
            "initial": initial,
            "samples": [initial],
        }
    if kind in {"agent_reach_region", "agent_touch_object"}:
        target_record = _subgoal_record(env, subgoal, "target") or _subgoal_record(
            env,
            subgoal,
            "object",
        )
        if target_record is None:
            return None
        initial = _agent_target_rollout_state(env, agent, target_record)
        path_diagnostics = (
            _current_reachability_diagnostics(env, target_record, config)
            if kind == "agent_reach_region"
            else None
        )
        return {
            "kind": kind,
            "target": target_record.name,
            "initial": initial,
            "samples": [initial],
            "initial_path_diagnostics": path_diagnostics,
        }
    if kind == "field_force_interaction":
        object_record = _subgoal_record(env, subgoal, "object")
        field_record = _subgoal_record(env, subgoal, "field")
        target_record = _subgoal_record(env, subgoal, "target") or _subgoal_record(
            env,
            subgoal,
            "region",
        )
        if object_record is None or field_record is None:
            return None
        initial = _field_rollout_state(env, object_record, field_record, target_record)
        return {
            "kind": kind,
            "object": object_record.name,
            "field": field_record.name,
            "target": target_record.name if target_record is not None else None,
            "initial": initial,
            "samples": [initial],
        }
    if kind == "lever_launch":
        plank_record = _subgoal_record(env, subgoal, "plank")
        weight_record = _subgoal_record(env, subgoal, "weight") or _subgoal_record(
            env,
            subgoal,
            "object",
        )
        target_record = _subgoal_record(env, subgoal, "target")
        if plank_record is None:
            return None
        initial = _lever_rollout_state(env, agent, plank_record, weight_record, target_record)
        env._validator_lever_initial = initial
        return {
            "kind": kind,
            "plank": plank_record.name,
            "weight": weight_record.name if weight_record is not None else None,
            "target": target_record.name if target_record is not None else None,
            "initial": initial,
            "samples": [initial],
        }
    return None


def _record_subgoal_diagnostics(
    diagnostics: dict[str, Any] | None,
    env: BaseEnv,
    subgoal: dict[str, Any],
    local_step: int,
    config: ValidatorConfig,
) -> None:
    if diagnostics is None:
        return
    kind = str(diagnostics.get("kind") or "")
    sample_interval = 20 if kind in {"move_object_to_region", "low_friction_slide_to_region", "strike_object_to_region", "ballistic_object_to_region", "support_exit_freefall"} else 60
    if local_step % sample_interval != 0 and local_step < config.kinetic_steps * 2 - 1:
        return
    agent = env.get_agent_record()
    if agent is None:
        return
    if kind in {"move_object_to_region", "low_friction_slide_to_region", "strike_object_to_region", "ballistic_object_to_region"}:
        object_record = _subgoal_record(env, subgoal, "object")
        region_record = _subgoal_record(env, subgoal, "region")
        if object_record is None or region_record is None:
            return
        if kind == "ballistic_object_to_region":
            _update_ballistic_observation(env, object_record, region_record, subgoal, config)
        sample = _push_rollout_state(env, agent, object_record, region_record, config)
    elif kind == "support_exit_freefall":
        object_record = _subgoal_record(env, subgoal, "object")
        boundary_record = _subgoal_record(env, subgoal, "boundary") or _subgoal_record(
            env,
            subgoal,
            "region",
        )
        if object_record is None or boundary_record is None:
            return
        sample = _push_rollout_state(env, agent, object_record, boundary_record, config)
    elif kind == "field_force_interaction":
        object_record = _subgoal_record(env, subgoal, "object")
        field_record = _subgoal_record(env, subgoal, "field")
        target_record = _subgoal_record(env, subgoal, "target") or _subgoal_record(
            env,
            subgoal,
            "region",
        )
        if object_record is None or field_record is None:
            return
        sample = _field_rollout_state(env, object_record, field_record, target_record)
    elif kind == "lever_launch":
        plank_record = _subgoal_record(env, subgoal, "plank")
        weight_record = _subgoal_record(env, subgoal, "weight") or _subgoal_record(
            env,
            subgoal,
            "object",
        )
        target_record = _subgoal_record(env, subgoal, "target")
        if plank_record is None:
            return
        sample = _lever_rollout_state(env, agent, plank_record, weight_record, target_record)
    elif kind in {"agent_reach_region", "agent_touch_object"}:
        target_record = _subgoal_record(env, subgoal, "target") or _subgoal_record(
            env,
            subgoal,
            "object",
        )
        if target_record is None:
            return
        sample = _agent_target_rollout_state(env, agent, target_record)
    else:
        return
    sample["local_step"] = local_step + 1
    diagnostics.setdefault("samples", []).append(sample)


def _finish_subgoal_diagnostics(
    diagnostics: dict[str, Any] | None,
    env: BaseEnv,
    subgoal: dict[str, Any],
    config: ValidatorConfig,
) -> dict[str, Any] | None:
    if diagnostics is None:
        return None
    kind = str(diagnostics.get("kind") or "")
    if kind in {"agent_reach_region", "agent_touch_object"}:
        return _finish_agent_target_diagnostics(diagnostics, env, subgoal, config)
    if kind == "field_force_interaction":
        return _finish_field_diagnostics(diagnostics, env, subgoal, config)
    if kind == "lever_launch":
        return _finish_lever_diagnostics(diagnostics, env, subgoal, config)
    agent = env.get_agent_record()
    object_record = _subgoal_record(env, subgoal, "object")
    region_record = _subgoal_record(env, subgoal, "region") or _subgoal_record(
        env,
        subgoal,
        "boundary",
    )
    if agent is None or object_record is None or region_record is None:
        return diagnostics
    final = _push_rollout_state(env, agent, object_record, region_record, config)
    diagnostics["final"] = final
    samples = [item for item in diagnostics.get("samples", []) if isinstance(item, dict)]
    initial = diagnostics.get("initial") if isinstance(diagnostics.get("initial"), dict) else {}
    start_distance = _float_or_none(initial.get("object_to_region_distance"))
    final_distance = _float_or_none(final.get("object_to_region_distance"))
    distance_reduced = None
    if start_distance is not None and final_distance is not None:
        distance_reduced = round(start_distance - final_distance, 3)
    object_displacement = _position_distance(
        initial.get("object_position"),
        final.get("object_position"),
    )
    max_velocity_delta = _max_velocity_delta(samples, "object_velocity_toward_region")
    contact_count = sum(
        1
        for sample in samples
        if _float_or_none(sample.get("agent_object_surface_gap")) is not None
        and float(sample["agent_object_surface_gap"]) <= 2.0
    )
    blocker = _solid_blocker_between(env, object_record, region_record, config)
    overlap_names = _static_overlaps(env, object_record)
    diagnostics["summary"] = {
        "start_distance": start_distance,
        "final_distance": final_distance,
        "distance_reduced": distance_reduced,
        "object_displacement": object_displacement,
        "best_distance": min(
            (
                float(sample["object_to_region_distance"])
                for sample in samples
                if sample.get("object_to_region_distance") is not None
            ),
            default=final_distance,
        ),
        "max_lateral_error": max(
            (
                abs(float(sample["agent_lateral_offset"]))
                for sample in samples
                if sample.get("agent_lateral_offset") is not None
            ),
            default=None,
        ),
        "min_agent_longitudinal": min(
            (
                float(sample["agent_longitudinal_offset"])
                for sample in samples
                if sample.get("agent_longitudinal_offset") is not None
            ),
            default=None,
        ),
        "max_object_velocity_toward_region": max(
            (
                float(sample["object_velocity_toward_region"])
                for sample in samples
                if sample.get("object_velocity_toward_region") is not None
            ),
            default=None,
        ),
        "max_object_velocity_delta": max_velocity_delta,
        "min_surface_gap": min(
            (
                float(sample["agent_object_surface_gap"])
                for sample in samples
                if sample.get("agent_object_surface_gap") is not None
            ),
            default=final.get("agent_object_surface_gap"),
        ),
        "contact_sample_count": contact_count,
        "max_applied_force": max(
            (
                float(sample["applied_force_magnitude"])
                for sample in samples
                if sample.get("applied_force_magnitude") is not None
            ),
            default=final.get("applied_force_magnitude"),
        ),
        "max_applied_force_toward_region": max(
            (
                float(sample["applied_force_toward_region"])
                for sample in samples
                if sample.get("applied_force_toward_region") is not None
            ),
            default=final.get("applied_force_toward_region"),
        ),
        "agent_displacement": _position_distance(
            initial.get("agent_position"),
            final.get("agent_position"),
        ),
        "max_agent_velocity_toward_region": max(
            (
                abs(float(sample["agent_velocity_toward_region"]))
                for sample in samples
                if sample.get("agent_velocity_toward_region") is not None
            ),
            default=final.get("agent_velocity_toward_region"),
        ),
        "collision_filter_blocking_reasons": final.get("collision_filter_blocking_reasons"),
        "collision_allowed": final.get("agent_object_collision_allowed"),
        "blockers": [blocker.name] if blocker is not None else [],
        "static_overlaps": overlap_names,
        "static_overlap_count": len(overlap_names),
        "final_phase": final.get("controller_phase"),
    }
    if kind == "ballistic_object_to_region":
        _add_ballistic_summary(diagnostics["summary"], samples, final)
    diagnostics["failure_modes"] = _push_failure_modes(diagnostics, config)
    collision_summary = {
        "allowed": diagnostics["summary"].get("collision_allowed"),
        "blocking_reasons": diagnostics["summary"].get("collision_filter_blocking_reasons"),
    }
    diagnostics["probes"] = [
        collision_filter_probe(
            agent_name=agent.name,
            object_name=object_record.name,
            filter_summary=collision_summary,
        ).to_dict(),
        push_force_probe(
            agent_name=agent.name,
            object_name=object_record.name,
            summary=diagnostics["summary"],
        ).to_dict(),
        agent_motion_under_force_probe(
            agent_name=agent.name,
            summary=diagnostics["summary"],
        ).to_dict(),
        push_contact_probe(
            agent_name=agent.name,
            object_name=object_record.name,
            summary=diagnostics["summary"],
        ).to_dict(),
        push_impulse_probe(
            object_name=object_record.name,
            summary=diagnostics["summary"],
        ).to_dict(),
        box_motion_probe(
            object_name=object_record.name,
            region_name=region_record.name,
            summary=diagnostics["summary"],
        ).to_dict(),
        box_blockage_probe(
            object_name=object_record.name,
            region_name=region_record.name,
            summary=diagnostics["summary"],
        ).to_dict(),
        object_region_motion_probe(
            object_name=str(diagnostics.get("object") or "object"),
            region_name=str(diagnostics.get("region") or "region"),
            summary=diagnostics["summary"],
            failure_modes=diagnostics["failure_modes"],
        ).to_dict(),
        object_inside_region_probe(
            env,
            object_name=object_record.name,
            region_name=region_record.name,
            threshold=_subgoal_threshold(env, subgoal, config),
        ).to_dict(),
    ]
    diagnostics["repair_instruction"] = _push_repair_instruction(diagnostics)
    return diagnostics


def _push_rollout_state(
    env: BaseEnv,
    agent,
    object_record,
    region_record,
    config: ValidatorConfig,
) -> dict[str, Any]:
    observation = _update_ballistic_observation(env, object_record, region_record, None, config)
    control = _push_contact_state(agent, object_record, region_record, config)
    distance = float(object_record.body.position.get_distance(region_record.body.position))
    force_state = getattr(env, "_validator_last_push_force", {})
    if not isinstance(force_state, dict):
        force_state = {}
    collision_summary = _agent_object_collision_filter_summary(agent, object_record)
    state = {
        "object_to_region_distance": round(distance, 3),
        "agent_to_object_distance": round(
            float(agent.body.position.get_distance(object_record.body.position)),
            3,
        ),
        "agent_longitudinal_offset": round(control["agent_longitudinal_offset"], 3),
        "agent_lateral_offset": round(control["agent_lateral_offset"], 3),
        "staging_gap": round(control["staging_gap"], 3),
        "object_velocity_toward_region": round(control["object_velocity_toward_region"], 3),
        "agent_velocity_toward_region": round(control["agent_velocity_toward_region"], 3),
        "agent_object_surface_gap": round(control["agent_object_surface_gap"], 3),
        "agent_object_center_distance": round(control["agent_object_center_distance"], 3),
        "agent_object_collision_allowed": collision_summary["allowed"],
        "collision_filter_blocking_reasons": collision_summary["blocking_reasons"],
        "applied_force_magnitude": _rounded_or_none(force_state.get("magnitude")),
        "applied_force_toward_region": _rounded_or_none(force_state.get("toward_region")),
        "applied_force_vector": force_state.get("vector"),
        "controller_phase": control["phase"],
        "object_position": [
            round(float(object_record.body.position.x), 3),
            round(float(object_record.body.position.y), 3),
        ],
        "region_position": [
            round(float(region_record.body.position.x), 3),
            round(float(region_record.body.position.y), 3),
        ],
        "agent_position": [
            round(float(agent.body.position.x), 3),
            round(float(agent.body.position.y), 3),
        ],
    }
    if observation is not None:
        state.update(
            {
                "ballistic_barrier_name": observation.get("barrier_name"),
                "ballistic_barrier_top_y": _rounded_or_none(observation.get("barrier_top_y")),
                "ballistic_required_apex_y": _rounded_or_none(observation.get("required_apex_y")),
                "ballistic_max_y": _rounded_or_none(observation.get("max_y")),
                "ballistic_clearance_margin_observed": _rounded_or_none(
                    observation.get("clearance_margin_observed")
                ),
                "ballistic_crossed_barrier": bool(observation.get("crossed_barrier")),
            }
        )
    return state


def _field_rollout_state(
    env: BaseEnv,
    object_record,
    field_record,
    target_record,
) -> dict[str, Any]:
    field_zone = getattr(env, "_force_zones", {}).get(field_record.name)
    object_position = object_record.body.position
    target_distance = None
    velocity_toward_target = None
    if target_record is not None:
        target_delta = target_record.body.position - object_position
        target_distance = float(target_delta.length)
        if target_delta.length > 1.0:
            velocity_toward_target = float(object_record.body.velocity.dot(target_delta.normalized()))
        else:
            velocity_toward_target = 0.0
    return {
        "object_position": [
            round(float(object_position.x), 3),
            round(float(object_position.y), 3),
        ],
        "field_position": [
            round(float(field_record.body.position.x), 3),
            round(float(field_record.body.position.y), 3),
        ],
        "target_position": None
        if target_record is None
        else [
            round(float(target_record.body.position.x), 3),
            round(float(target_record.body.position.y), 3),
        ],
        "object_to_field_distance": round(
            float(object_position.get_distance(field_record.body.position)),
            3,
        ),
        "object_to_target_distance": None
        if target_distance is None
        else round(target_distance, 3),
        "object_velocity_magnitude": round(float(object_record.body.velocity.length), 3),
        "object_velocity_toward_target": None
        if velocity_toward_target is None
        else round(velocity_toward_target, 3),
        "object_inside_field": _record_point_inside_record(field_record, object_record),
        "force_zone_registered": field_zone is not None,
        "force_zone_mode": getattr(field_zone, "mode", None),
        "force_zone_strength": _rounded_or_none(getattr(field_zone, "strength", None)),
        "force_zone_force": None
        if field_zone is None
        else [round(float(field_zone.force.x), 3), round(float(field_zone.force.y), 3)],
    }


def _finish_field_diagnostics(
    diagnostics: dict[str, Any],
    env: BaseEnv,
    subgoal: dict[str, Any],
    config: ValidatorConfig,
) -> dict[str, Any] | None:
    object_record = _subgoal_record(env, subgoal, "object")
    field_record = _subgoal_record(env, subgoal, "field")
    target_record = _subgoal_record(env, subgoal, "target") or _subgoal_record(
        env,
        subgoal,
        "region",
    )
    if object_record is None or field_record is None:
        return diagnostics
    final = _field_rollout_state(env, object_record, field_record, target_record)
    diagnostics["final"] = final
    samples = [item for item in diagnostics.get("samples", []) if isinstance(item, dict)]
    initial = diagnostics.get("initial") if isinstance(diagnostics.get("initial"), dict) else {}
    start_target_distance = _float_or_none(initial.get("object_to_target_distance"))
    final_target_distance = _float_or_none(final.get("object_to_target_distance"))
    progress_delta = None
    if start_target_distance is not None and final_target_distance is not None:
        progress_delta = round(start_target_distance - final_target_distance, 3)
    displacement = _position_distance(initial.get("object_position"), final.get("object_position"))
    velocity_values = [
        _float_or_none(sample.get("object_velocity_magnitude"))
        for sample in samples
    ]
    velocity_values = [value for value in velocity_values if value is not None]
    velocity_delta = None
    if velocity_values:
        velocity_delta = round(max(velocity_values) - min(velocity_values), 3)
    inside_count = sum(1 for sample in samples if sample.get("object_inside_field"))
    diagnostics["summary"] = {
        "start_distance": start_target_distance or _float_or_none(initial.get("object_to_field_distance")),
        "final_distance": final_target_distance or _float_or_none(final.get("object_to_field_distance")),
        "distance_reduced": progress_delta,
        "displacement": displacement,
        "progress_delta": progress_delta,
        "velocity_delta": velocity_delta,
        "inside_field_sample_count": inside_count,
        "force_zone_registered": final.get("force_zone_registered"),
        "force_zone_mode": final.get("force_zone_mode"),
        "force_zone_strength": final.get("force_zone_strength"),
    }
    diagnostics["failure_modes"] = _field_failure_modes(diagnostics)
    diagnostics["probes"] = [
        field_effect_probe(
            object_name=object_record.name,
            field_name=field_record.name,
            summary=diagnostics["summary"],
        ).to_dict()
    ]
    diagnostics["repair_instruction"] = _field_repair_instruction(diagnostics)
    return diagnostics


def _field_failure_modes(diagnostics: dict[str, Any]) -> list[str]:
    summary = diagnostics.get("summary") if isinstance(diagnostics.get("summary"), dict) else {}
    modes: list[str] = []
    if not summary.get("force_zone_registered"):
        modes.append("field_not_registered_as_force_zone")
    if int(summary.get("inside_field_sample_count") or 0) <= 0:
        modes.append("object_never_entered_field")
    displacement = _float_or_none(summary.get("displacement"))
    progress_delta = _float_or_none(summary.get("progress_delta"))
    velocity_delta = _float_or_none(summary.get("velocity_delta"))
    if displacement is not None and displacement < 5.0:
        modes.append("field_object_barely_moved")
    if progress_delta is not None and progress_delta < 5.0:
        modes.append("field_did_not_improve_target_progress")
    if velocity_delta is not None and velocity_delta < 1.0:
        modes.append("field_velocity_change_too_small")
    return modes or ["field_effect_insufficient"]


def _field_repair_instruction(diagnostics: dict[str, Any]) -> str:
    modes = set(_string_list(diagnostics.get("failure_modes")))
    object_name = str(diagnostics.get("object") or "affected object")
    field_name = str(diagnostics.get("field") or "force zone")
    target_name = str(diagnostics.get("target") or "target")
    summary = diagnostics.get("summary") if isinstance(diagnostics.get("summary"), dict) else {}
    base = (
        f"Repair field_force_interaction for {object_name} through {field_name}. "
        f"Displacement: {summary.get('displacement')} px; "
        f"progress_delta: {summary.get('progress_delta')} px. "
    )
    instructions: list[str] = []
    if "field_not_registered_as_force_zone" in modes:
        instructions.append("Use self.register_force_zone(...) and do not override step().")
    if "object_never_entered_field" in modes:
        instructions.append(
            "Place the object inside or just before the force zone, or add a preceding move_object_to_region subgoal that pushes it into the zone."
        )
    if "field_object_barely_moved" in modes or "field_velocity_change_too_small" in modes:
        instructions.append(
            "Increase bounded field strength, reduce affected object mass/friction, or enlarge the zone so the force acts for more simulation steps."
        )
    if "field_did_not_improve_target_progress" in modes:
        instructions.append(
            f"Orient the field force toward {target_name}, or use mode='attract' with the field/target placed in the desired direction."
        )
    if not instructions:
        instructions.append("Keep the field deterministic, stronger, and aligned with the declared progress metric.")
    return base + " ".join(instructions)


def _lever_rollout_state(
    env: BaseEnv,
    agent,
    plank_record,
    weight_record,
    target_record,
) -> dict[str, Any]:
    agent_pos = agent.body.position
    target_distance = None
    velocity_toward_target = None
    target_delta_y = None
    if target_record is not None:
        target_delta = target_record.body.position - agent_pos
        target_distance = float(target_delta.length)
        target_delta_y = float(target_record.body.position.y - agent_pos.y)
        if target_delta.length > 1.0:
            velocity_toward_target = float(agent.body.velocity.dot(target_delta.normalized()))
        else:
            velocity_toward_target = 0.0
    return {
        "plank_angle": round(float(plank_record.body.angle), 4),
        "plank_angular_velocity": round(float(plank_record.body.angular_velocity), 4),
        "plank_position": [
            round(float(plank_record.body.position.x), 3),
            round(float(plank_record.body.position.y), 3),
        ],
        "weight_position": None
        if weight_record is None
        else [
            round(float(weight_record.body.position.x), 3),
            round(float(weight_record.body.position.y), 3),
        ],
        "agent_position": [
            round(float(agent_pos.x), 3),
            round(float(agent_pos.y), 3),
        ],
        "agent_velocity": [
            round(float(agent.body.velocity.x), 3),
            round(float(agent.body.velocity.y), 3),
        ],
        "agent_to_target_distance": None
        if target_distance is None
        else round(target_distance, 3),
        "agent_velocity_toward_target": None
        if velocity_toward_target is None
        else round(velocity_toward_target, 3),
        "target_delta_y": None if target_delta_y is None else round(target_delta_y, 3),
        "has_pivot_constraint": _has_pivot_constraint(env, plank_record),
    }


def _finish_lever_diagnostics(
    diagnostics: dict[str, Any],
    env: BaseEnv,
    subgoal: dict[str, Any],
    config: ValidatorConfig,
) -> dict[str, Any] | None:
    agent = env.get_agent_record()
    plank_record = _subgoal_record(env, subgoal, "plank")
    weight_record = _subgoal_record(env, subgoal, "weight") or _subgoal_record(
        env,
        subgoal,
        "object",
    )
    target_record = _subgoal_record(env, subgoal, "target")
    if agent is None or plank_record is None:
        return diagnostics
    final = _lever_rollout_state(env, agent, plank_record, weight_record, target_record)
    diagnostics["final"] = final
    samples = [item for item in diagnostics.get("samples", []) if isinstance(item, dict)]
    initial = diagnostics.get("initial") if isinstance(diagnostics.get("initial"), dict) else {}
    initial_angle = _float_or_none(initial.get("plank_angle"))
    final_angle = _float_or_none(final.get("plank_angle"))
    angle_delta = None
    if initial_angle is not None and final_angle is not None:
        angle_delta = round(final_angle - initial_angle, 4)
    start_distance = _float_or_none(initial.get("agent_to_target_distance"))
    final_distance = _float_or_none(final.get("agent_to_target_distance"))
    distance_reduced = None
    if start_distance is not None and final_distance is not None:
        distance_reduced = round(start_distance - final_distance, 3)
    initial_agent_position = initial.get("agent_position")
    final_agent_position = final.get("agent_position")
    agent_lift = _agent_lift_toward_target(initial, final)
    max_angular_velocity = max(
        (
            abs(float(sample["plank_angular_velocity"]))
            for sample in samples
            if sample.get("plank_angular_velocity") is not None
        ),
        default=abs(float(final.get("plank_angular_velocity") or 0.0)),
    )
    max_agent_velocity_toward_target = max(
        (
            float(sample["agent_velocity_toward_target"])
            for sample in samples
            if sample.get("agent_velocity_toward_target") is not None
        ),
        default=final.get("agent_velocity_toward_target"),
    )
    diagnostics["summary"] = {
        "start_distance": start_distance,
        "final_distance": final_distance,
        "distance_reduced": distance_reduced,
        "agent_target_distance_reduced": distance_reduced,
        "agent_displacement": _position_distance(initial_agent_position, final_agent_position),
        "agent_lift_toward_target": agent_lift,
        "plank_angle_delta": angle_delta,
        "plank_angle_delta_abs": None if angle_delta is None else round(abs(angle_delta), 4),
        "max_plank_angular_velocity_abs": round(float(max_angular_velocity), 4),
        "max_agent_velocity_toward_target": _rounded_or_none(max_agent_velocity_toward_target),
        "has_pivot_constraint": _has_pivot_constraint(env, plank_record),
    }
    min_angle_delta = _float_subgoal_value(subgoal, "min_angle_delta", 0.08)
    min_agent_lift = _float_subgoal_value(subgoal, "min_agent_lift", 45.0)
    diagnostics["failure_modes"] = _lever_failure_modes(diagnostics, min_angle_delta, min_agent_lift)
    diagnostics["probes"] = [
        pivot_mechanism_probe(
            plank_name=plank_record.name,
            weight_name=weight_record.name if weight_record is not None else None,
            has_pivot_constraint=_has_pivot_constraint(env, plank_record),
            plank_is_dynamic=plank_record.body.body_type == pymunk.Body.DYNAMIC,
            weight_is_dynamic=None
            if weight_record is None
            else weight_record.body.body_type == pymunk.Body.DYNAMIC,
        ).to_dict(),
        plank_rotation_probe(
            plank_name=plank_record.name,
            summary=diagnostics["summary"],
            min_angle_delta=min_angle_delta,
        ).to_dict(),
        launch_progress_probe(
            agent_name=agent.name,
            target_name=target_record.name if target_record is not None else None,
            summary=diagnostics["summary"],
            min_agent_lift=min_agent_lift,
        ).to_dict(),
    ]
    diagnostics["repair_instruction"] = _lever_repair_instruction(diagnostics)
    return diagnostics


def _agent_lift_toward_target(initial: dict[str, Any], final: dict[str, Any]) -> float | None:
    first = initial.get("agent_position")
    second = final.get("agent_position")
    if not isinstance(first, list | tuple) or not isinstance(second, list | tuple):
        return None
    if len(first) < 2 or len(second) < 2:
        return None
    target_delta_y = _float_or_none(initial.get("target_delta_y"))
    try:
        dy = float(second[1]) - float(first[1])
    except (TypeError, ValueError):
        return None
    if target_delta_y is None or abs(target_delta_y) <= 1.0:
        return round(abs(dy), 3)
    return round(dy if target_delta_y > 0 else -dy, 3)


def _lever_failure_modes(
    diagnostics: dict[str, Any],
    min_angle_delta: float,
    min_agent_lift: float,
) -> list[str]:
    summary = diagnostics.get("summary") if isinstance(diagnostics.get("summary"), dict) else {}
    modes: list[str] = []
    if not summary.get("has_pivot_constraint"):
        modes.append("missing_pivot_constraint")
    angle_delta = _float_or_none(summary.get("plank_angle_delta_abs"))
    agent_lift = _float_or_none(summary.get("agent_lift_toward_target"))
    target_progress = _float_or_none(summary.get("agent_target_distance_reduced"))
    if angle_delta is not None and angle_delta < min_angle_delta:
        modes.append("plank_rotation_insufficient")
    if agent_lift is not None and agent_lift < min_agent_lift:
        modes.append("launch_progress_insufficient")
    if target_progress is not None and target_progress < 10.0:
        modes.append("agent_did_not_approach_launch_target")
    return modes or ["lever_launch_progress_observed"]


def _lever_repair_instruction(diagnostics: dict[str, Any]) -> str:
    modes = set(_string_list(diagnostics.get("failure_modes")))
    plank_name = str(diagnostics.get("plank") or "plank")
    weight_name = str(diagnostics.get("weight") or "weight")
    target_name = str(diagnostics.get("target") or "target")
    summary = diagnostics.get("summary") if isinstance(diagnostics.get("summary"), dict) else {}
    base = (
        f"Repair lever_launch for {plank_name} using {weight_name}. "
        f"plank_angle_delta_abs={summary.get('plank_angle_delta_abs')}; "
        f"agent_lift_toward_target={summary.get('agent_lift_toward_target')}. "
    )
    instructions: list[str] = []
    if "missing_pivot_constraint" in modes:
        instructions.append("Register a PivotJoint on the dynamic plank using register_constraint; do not use a static visual-only plank.")
    if "plank_rotation_insufficient" in modes:
        instructions.append("Increase torque transfer: move the weight closer to the load side, reduce plank mass/friction, lengthen the plank, or ensure the weight contacts the plank.")
    if "launch_progress_insufficient" in modes:
        instructions.append(f"Stage the agent on the launch side and align/enlarge {target_name} with the launch arc.")
    if "agent_did_not_approach_launch_target" in modes:
        instructions.append("Lower or move the high goal toward the reachable launch arc without changing the task.")
    if not instructions:
        instructions.append("Preserve the pivot mechanism but make the energy transfer clearer and less delicate.")
    return base + " ".join(instructions)


def _float_subgoal_value(subgoal: dict[str, Any], key: str, fallback: float) -> float:
    try:
        return float(subgoal.get(key, fallback))
    except (TypeError, ValueError):
        return float(fallback)


def _rounded_or_none(value: Any, digits: int = 3) -> float | None:
    number = _float_or_none(value)
    return None if number is None else round(number, digits)


def _position_distance(first: Any, second: Any) -> float | None:
    if not isinstance(first, list | tuple) or not isinstance(second, list | tuple):
        return None
    if len(first) < 2 or len(second) < 2:
        return None
    try:
        return round(math.hypot(float(second[0]) - float(first[0]), float(second[1]) - float(first[1])), 3)
    except (TypeError, ValueError):
        return None


def _max_velocity_delta(samples: list[dict[str, Any]], key: str) -> float | None:
    values = [_float_or_none(sample.get(key)) for sample in samples]
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return None
    return round(max(abs(values[index] - values[index - 1]) for index in range(1, len(values))), 3)


def _agent_object_collision_filter_summary(agent, object_record) -> dict[str, Any]:
    blocking_reasons: list[str] = []
    allowed_pair = False
    for agent_shape in agent.shapes:
        for object_shape in object_record.shapes:
            reasons = _shape_pair_blocking_reasons(agent_shape, object_shape)
            if not reasons:
                allowed_pair = True
            blocking_reasons.extend(reasons)
    return {
        "allowed": allowed_pair,
        "blocking_reasons": sorted(set(blocking_reasons)),
        "agent_shape_count": len(agent.shapes),
        "object_shape_count": len(object_record.shapes),
    }


def _shape_pair_blocking_reasons(first: pymunk.Shape, second: pymunk.Shape) -> list[str]:
    reasons: list[str] = []
    if bool(first.sensor):
        reasons.append("agent_shape_is_sensor")
    if bool(second.sensor):
        reasons.append("object_shape_is_sensor")
    first_filter = first.filter
    second_filter = second.filter
    if first_filter.group != 0 and first_filter.group == second_filter.group:
        reasons.append("same_nonzero_collision_group")
    if int(first_filter.categories) & int(second_filter.mask) == 0:
        reasons.append("agent_category_not_in_object_mask")
    if int(second_filter.categories) & int(first_filter.mask) == 0:
        reasons.append("object_category_not_in_agent_mask")
    return reasons


def _static_overlaps(env: BaseEnv, object_record) -> list[str]:
    names: list[str] = []
    object_center_y = float(object_record.body.position.y)
    for record in env._objects.values():
        if record is object_record or record.role == "agent" or _record_is_sensor(record):
            continue
        if record.body.body_type != pymunk.Body.STATIC:
            continue
        static_shapes = [
            static_shape
            for static_shape in record.shapes
            if not _looks_like_support_contact(record, static_shape, object_center_y)
        ]
        if any(
            _bb_overlap(object_shape.bb, static_shape.bb)
            for object_shape in object_record.shapes
            for static_shape in static_shapes
        ):
            names.append(record.name)
    return sorted(set(names))


def _looks_like_support_contact(record, shape: pymunk.Shape, object_center_y: float) -> bool:
    name = str(getattr(record, "name", "") or "").lower()
    role = str(getattr(record, "role", "") or "").lower()
    support_tokens = ("floor", "ground", "support", "ice", "rink", "lane", "platform")
    if not any(token in name for token in support_tokens) and role != "terrain":
        return False
    return float(shape.bb.top) <= object_center_y + 4.0


def _bb_overlap(first, second) -> bool:
    return not (
        float(first.right) < float(second.left)
        or float(first.left) > float(second.right)
        or float(first.top) < float(second.bottom)
        or float(first.bottom) > float(second.top)
    )


def _agent_target_rollout_state(env: BaseEnv, agent, target_record) -> dict[str, Any]:
    distance = float(agent.body.position.get_distance(target_record.body.position))
    direction = target_record.body.position - agent.body.position
    if direction.length <= 1.0:
        velocity_toward_target = 0.0
    else:
        velocity_toward_target = float(agent.body.velocity.dot(direction.normalized()))
    return {
        "agent_to_target_distance": round(distance, 3),
        "agent_to_target_dx": round(float(target_record.body.position.x - agent.body.position.x), 3),
        "agent_to_target_dy": round(float(target_record.body.position.y - agent.body.position.y), 3),
        "agent_velocity_toward_target": round(velocity_toward_target, 3),
        "agent_position": [
            round(float(agent.body.position.x), 3),
            round(float(agent.body.position.y), 3),
        ],
        "target_position": [
            round(float(target_record.body.position.x), 3),
            round(float(target_record.body.position.y), 3),
        ],
    }


def _finish_agent_target_diagnostics(
    diagnostics: dict[str, Any],
    env: BaseEnv,
    subgoal: dict[str, Any],
    config: ValidatorConfig,
) -> dict[str, Any]:
    agent = env.get_agent_record()
    target_record = _subgoal_record(env, subgoal, "target") or _subgoal_record(
        env,
        subgoal,
        "object",
    )
    if agent is None or target_record is None:
        return diagnostics
    final = _agent_target_rollout_state(env, agent, target_record)
    diagnostics["final"] = final
    samples = [item for item in diagnostics.get("samples", []) if isinstance(item, dict)]
    initial = diagnostics.get("initial") if isinstance(diagnostics.get("initial"), dict) else {}
    start_distance = _float_or_none(initial.get("agent_to_target_distance"))
    final_distance = _float_or_none(final.get("agent_to_target_distance"))
    distance_reduced = None
    if start_distance is not None and final_distance is not None:
        distance_reduced = round(start_distance - final_distance, 3)
    best_distance = min(
        (
            float(sample["agent_to_target_distance"])
            for sample in samples
            if sample.get("agent_to_target_distance") is not None
        ),
        default=final_distance,
    )
    max_velocity = max(
        (
            float(sample["agent_velocity_toward_target"])
            for sample in samples
            if sample.get("agent_velocity_toward_target") is not None
        ),
        default=None,
    )
    diagnostics["summary"] = {
        "start_distance": start_distance,
        "final_distance": final_distance,
        "distance_reduced": distance_reduced,
        "best_distance": best_distance,
        "threshold": round(_subgoal_threshold(env, subgoal, config), 3),
        "final_dx": final.get("agent_to_target_dx"),
        "final_dy": final.get("agent_to_target_dy"),
        "max_agent_velocity_toward_target": max_velocity,
        "agent_displacement": _position_distance(
            initial.get("agent_position"),
            final.get("agent_position"),
        ),
    }
    if str(diagnostics.get("kind") or "") == "agent_reach_region":
        diagnostics["path_diagnostics"] = _current_reachability_diagnostics(
            env,
            target_record,
            config,
        )
        initial_path = diagnostics.get("initial_path_diagnostics")
        if isinstance(initial_path, dict):
            diagnostics["summary"]["initial_path_found"] = initial_path.get("path_found")
            diagnostics["summary"]["initial_blocking_object"] = initial_path.get(
                "blocking_object"
            )
            diagnostics["summary"]["initial_path_points"] = len(
                initial_path.get("path_points") or []
            )
        diagnostics["summary"]["final_path_found"] = diagnostics["path_diagnostics"].get(
            "path_found"
        )
        diagnostics["summary"]["final_blocking_object"] = diagnostics["path_diagnostics"].get(
            "blocking_object"
        )
        diagnostics["summary"]["final_path_points"] = len(
            diagnostics["path_diagnostics"].get("path_points") or []
        )
    diagnostics["failure_modes"] = _agent_target_failure_modes(diagnostics)
    probes = [
        agent_target_motion_probe(
            target_name=str(diagnostics.get("target") or target_record.name),
            summary=diagnostics["summary"],
            failure_modes=diagnostics["failure_modes"],
        ).to_dict(),
        contact_or_proximity_probe(
            env,
            first_name=agent.name,
            second_name=target_record.name,
            threshold=_subgoal_threshold(env, subgoal, config),
        ).to_dict(),
    ]
    if isinstance(diagnostics.get("path_diagnostics"), dict):
        probes.append(
            path_reachability_probe(
                start_name="agent",
                target_name=str(diagnostics.get("target") or target_record.name),
                path_diagnostics=diagnostics.get("path_diagnostics"),
            ).to_dict()
        )
    diagnostics["probes"] = probes
    diagnostics["repair_instruction"] = _agent_target_repair_instruction(diagnostics)
    return diagnostics


def _agent_target_failure_modes(diagnostics: dict[str, Any]) -> list[str]:
    summary = diagnostics.get("summary") if isinstance(diagnostics.get("summary"), dict) else {}
    path_diagnostics = (
        diagnostics.get("path_diagnostics")
        if isinstance(diagnostics.get("path_diagnostics"), dict)
        else {}
    )
    distance_reduced = _float_or_none(summary.get("distance_reduced"))
    final_distance = _float_or_none(summary.get("final_distance"))
    best_distance = _float_or_none(summary.get("best_distance"))
    threshold = _float_or_none(summary.get("threshold"))
    final_dx = _float_or_none(summary.get("final_dx"))
    final_dy = _float_or_none(summary.get("final_dy"))
    max_velocity = _float_or_none(summary.get("max_agent_velocity_toward_target"))
    modes: list[str] = []
    if path_diagnostics and not path_diagnostics.get("path_found"):
        modes.append("post_subgoal_path_blocked")
    if distance_reduced is not None and distance_reduced < 5.0:
        modes.append("agent_made_insufficient_target_progress")
    if best_distance is not None and final_distance is not None and final_distance - best_distance > 40.0:
        modes.append("agent_overshot_or_bounced_away_from_target")
    if threshold is not None and final_distance is not None and final_distance <= threshold * 4.0:
        modes.append("target_threshold_or_sensor_too_tight")
    if (
        threshold is not None
        and final_dx is not None
        and final_dy is not None
        and abs(final_dx) <= threshold
        and abs(final_dy) > threshold
    ):
        modes.append("target_offset_from_reachable_lane")
    if max_velocity is not None and max_velocity < 20.0:
        modes.append("agent_never_gained_target_velocity")
    return modes or ["agent_reach_controller_failed"]


def _agent_target_repair_instruction(diagnostics: dict[str, Any]) -> str:
    modes = set(_string_list(diagnostics.get("failure_modes")))
    target_name = str(diagnostics.get("target") or "target")
    summary = diagnostics.get("summary") if isinstance(diagnostics.get("summary"), dict) else {}
    base = (
        f"Repair agent reach/touch subgoal for {target_name}. "
        f"Distance reduced: {summary.get('distance_reduced')} px; "
        f"final distance: {summary.get('final_distance')} px; "
        f"threshold: {summary.get('threshold')} px. "
    )
    instructions: list[str] = []
    if "post_subgoal_path_blocked" in modes or "post_mechanism_path_blocked" in modes:
        path_diagnostics = diagnostics.get("path_diagnostics")
        blocker = None
        if isinstance(path_diagnostics, dict):
            blocker = path_diagnostics.get("blocking_object")
        instructions.append(
            f"The post-subgoal path is still blocked by {blocker or 'solid geometry'}; clear/sensorize that blocker, move the agent into the open route after the earlier subgoal, or place the target on the reachable lane."
        )
    if "agent_made_insufficient_target_progress" in modes:
        instructions.append("Move the target closer, clear the path, or increase agent_strength.")
    if "agent_overshot_or_bounced_away_from_target" in modes:
        instructions.append("Make the target a larger non-blocking sensor region and reduce bounce/elasticity near it.")
    if "target_threshold_or_sensor_too_tight" in modes:
        instructions.append("The agent got close but did not satisfy the predicate; enlarge the goal sensor/threshold or align check_objective with the visible goal radius.")
    if "target_offset_from_reachable_lane" in modes:
        instructions.append("The agent aligned horizontally but remained vertically offset from the target; move the goal onto the reachable lane, add vertical access, or enlarge the goal region so the visible lane counts as success.")
    if "agent_never_gained_target_velocity" in modes:
        instructions.append("Remove blockers or friction traps that prevent acceleration toward the target.")
    if not instructions:
        instructions.append("Shorten and straighten the final path without changing the objective.")
    return base + " ".join(instructions)


def _push_failure_modes(diagnostics: dict[str, Any], config: ValidatorConfig) -> list[str]:
    summary = diagnostics.get("summary") if isinstance(diagnostics.get("summary"), dict) else {}
    modes: list[str] = []
    distance_reduced = _float_or_none(summary.get("distance_reduced"))
    max_lateral = _float_or_none(summary.get("max_lateral_error"))
    min_longitudinal = _float_or_none(summary.get("min_agent_longitudinal"))
    max_velocity = _float_or_none(summary.get("max_object_velocity_toward_region"))
    min_gap = _float_or_none(summary.get("min_surface_gap"))
    contact_count = int(summary.get("contact_sample_count") or 0)
    max_force = _float_or_none(summary.get("max_applied_force"))
    max_forward_force = _float_or_none(summary.get("max_applied_force_toward_region"))
    object_displacement = _float_or_none(summary.get("object_displacement"))
    agent_displacement = _float_or_none(summary.get("agent_displacement"))
    clearance_margin = _float_or_none(summary.get("ballistic_clearance_margin_observed"))
    if summary.get("collision_allowed") is False:
        modes.append("agent_object_collision_filtered")
    if summary.get("ballistic_required_apex_y") is not None and clearance_margin is not None and clearance_margin < 0.0:
        modes.append("ballistic_apex_below_barrier_clearance")
    if summary.get("ballistic_barrier_name") and summary.get("ballistic_crossed_barrier") is False:
        modes.append("ballistic_object_never_crossed_barrier")
    initial = diagnostics.get("initial") if isinstance(diagnostics.get("initial"), dict) else {}
    staging_gap = _float_or_none(initial.get("staging_gap")) or (config.agent_radius * 2.0)
    if max_force is not None and max_force <= 1.0:
        modes.append("agent_force_not_applied")
    if max_forward_force is not None and max_forward_force <= 1.0:
        modes.append("agent_force_not_aligned_with_push_axis")
    if max_force is not None and max_force > 1.0 and agent_displacement is not None and agent_displacement <= 1.0:
        modes.append("agent_pinned_or_immobile")
    if contact_count <= 0 and min_gap is not None and min_gap > 2.0:
        modes.append("agent_never_contacted_object")
    if object_displacement is not None and object_displacement <= 1.0:
        modes.append("object_stationary")
    if distance_reduced is not None and distance_reduced < -config.kinetic_displacement_threshold:
        modes.append("object_moved_away_from_region")
    elif distance_reduced is not None and distance_reduced < config.kinetic_displacement_threshold:
        modes.append("object_made_insufficient_forward_progress")
    if min_longitudinal is not None and min_longitudinal > -staging_gap * 0.45:
        modes.append("agent_never_reached_push_side")
    if max_lateral is not None and max_lateral > staging_gap * 0.75:
        modes.append("agent_or_object_drifted_sideways")
    if max_velocity is not None and max_velocity < 20.0:
        modes.append("object_never_gained_forward_velocity")
    if summary.get("blockers") or summary.get("static_overlap_count"):
        modes.append("object_blocked_or_pinned")
    return modes or ["generic_push_controller_failed"]


def _push_repair_instruction(diagnostics: dict[str, Any]) -> str:
    modes = set(_string_list(diagnostics.get("failure_modes")))
    summary = diagnostics.get("summary") if isinstance(diagnostics.get("summary"), dict) else {}
    object_name = str(diagnostics.get("object") or "moved object")
    region_name = str(diagnostics.get("region") or "target region")
    base = (
        f"Repair {_diagnostic_relation_name(diagnostics)} for {object_name} -> {region_name}. "
        f"Distance reduced: {summary.get('distance_reduced')} px; "
        f"final distance: {summary.get('final_distance')} px. "
    )
    instructions: list[str] = []
    if "object_moved_away_from_region" in modes:
        instructions.append(
            "The object moved away from the target; place the agent directly behind the object on the object-to-region axis and add short guide rails/walls that prevent reverse or sideways escape."
        )
    if "agent_never_reached_push_side" in modes:
        instructions.append(
            "The agent did not reliably reach the pushing side; stage the agent closer behind the object with clear free space before contact."
        )
    if "agent_or_object_drifted_sideways" in modes:
        instructions.append(
            "Sideways drift was high; align the corridor tightly and remove angled contacts that deflect the box."
        )
    if "agent_object_collision_filtered" in modes:
        instructions.append(
            "Agent and object collision filters or sensor flags appear to prevent physical contact; make both shapes non-sensor and ensure ShapeFilter groups/categories/masks allow agent-box collision."
        )
    if "agent_force_not_applied" in modes or "agent_force_not_aligned_with_push_axis" in modes:
        instructions.append(
            "The controller did not command useful forward force; increase agent_strength and place the agent so the target position lies through the box toward the region."
        )
    if "agent_pinned_or_immobile" in modes:
        instructions.append(
            "Force is being applied but the agent body barely moves; reduce agent/floor friction, remove pinning overlaps, or ensure the agent starts on a traversable low-friction floor."
        )
    if "agent_never_contacted_object" in modes:
        instructions.append(
            "The agent never physically contacted the object; place it closer behind the object or reduce the staging gap so sustained force presses into the object."
        )
    if "object_stationary" in modes:
        instructions.append(
            "The object stayed effectively stationary; check collision filtering, pinning geometry, mass/friction, and whether the agent is actually overlapping/contacting the box."
        )
    if "object_never_gained_forward_velocity" in modes:
        instructions.append(
            "The object did not gain useful forward velocity; reduce object friction/mass or increase agent_strength while keeping physical force control."
        )
    if "ballistic_apex_below_barrier_clearance" in modes:
        instructions.append(
            f"The ballistic arc is too low: observed apex margin was {summary.get('ballistic_clearance_margin_observed')} px against required apex {summary.get('ballistic_required_apex_y')} px. Lower the barrier height through the registry, increase upward impulse/agent_strength, reduce object mass/friction, or move the target closer while preserving the over-wall relation."
        )
    if "ballistic_object_never_crossed_barrier" in modes:
        instructions.append(
            f"The object never crossed the required barrier {summary.get('ballistic_barrier_name')}; stage ball, barrier, and goal on one clear axis with the barrier between object and goal, and keep posts/rails sensor-only."
        )
    if "object_blocked_or_pinned" in modes:
        if str(diagnostics.get("kind") or "") == "ballistic_object_to_region":
            instructions.append(
                "The projectile appears blocked or pinned by static geometry. Keep the required wall/barrier, but make decorative posts/rails sensor-only, widen clear arc space around the ball, and lower/shorten only the barrier if apex telemetry says the arc is impossible."
            )
        else:
            instructions.append(
                "The object appears blocked or pinned by static geometry; remove/sensorize overlapping rails/walls or widen the lane around the movable object."
            )
    if not instructions:
        instructions.append(
            "Make the push path shorter, straighter, and mechanically guided without changing the objective or lowering Tier 5."
        )
    return base + " ".join(instructions)


def _current_reachability_diagnostics(
    env: BaseEnv,
    target_record,
    config: ValidatorConfig,
) -> dict[str, Any]:
    agent = env.get_agent_record()
    if agent is None:
        return {"path_found": False, "reason": "missing agent"}
    start = (float(agent.body.position.x), float(agent.body.position.y))
    goal = (float(target_record.body.position.x), float(target_record.body.position.y))
    return _grid_reachability_between(env.get_ground_truth(), start, goal, config)


def _mechanism_state_diagnostics(
    env: BaseEnv,
    subgoal: dict[str, Any],
    next_subgoal: dict[str, Any] | None,
    config: ValidatorConfig,
) -> dict[str, Any]:
    trigger = _subgoal_record(env, subgoal, "trigger")
    effect = str(subgoal.get("effect") or "")
    mechanism_candidates = _mechanism_candidates(env, effect)
    registered_mechanisms = _registered_mechanism_states(env)
    diagnostics: dict[str, Any] = {
        "subgoal": dict(subgoal),
        "trigger": trigger.name if trigger is not None else subgoal.get("trigger"),
        "effect": effect,
        "registered_mechanisms": registered_mechanisms,
        "mechanism_candidates": [
            {
                "name": record.name,
                "role": record.role,
                "body_type": _body_type_name(record.body),
                "sensor": _record_is_sensor(record),
                "static_solid": _record_is_static_solid(record),
            }
            for record in mechanism_candidates
        ],
    }
    if trigger is not None:
        activators = [
            record.name
            for record in env._objects.values()
            if record is not trigger
            and record.role != "agent"
            and record.body.body_type == pymunk.Body.DYNAMIC
            and float(record.body.position.get_distance(trigger.body.position))
            <= max(
                _subgoal_threshold(env, subgoal, config),
                _record_radius(record, config.agent_radius)
                + _record_radius(trigger, 24.0) * 0.65,
            )
        ]
        diagnostics["activators_on_trigger"] = activators
        diagnostics["trigger_has_activator"] = bool(activators)
    if next_subgoal and str(next_subgoal.get("kind") or "") == "agent_reach_region":
        target = _subgoal_record(env, next_subgoal, "target")
        if target is not None:
            diagnostics["post_activation_path"] = _current_reachability_diagnostics(
                env,
                target,
                config,
            )
    blocking_mechanisms = [
        item["name"]
        for item in diagnostics["mechanism_candidates"]
        if item.get("static_solid")
    ]
    if blocking_mechanisms:
        diagnostics["repair_instruction"] = (
            "Mechanism activation occurred, but the likely gate/door object still appears "
            f"as solid static geometry: {blocking_mechanisms}. Move it out of the path, "
            "convert it to sensor=True, or make check_objective/path layout match the opened state."
        )
    return diagnostics


def _registered_mechanism_satisfied(env: BaseEnv, subgoal: dict[str, Any]) -> bool:
    update = getattr(env, "_update_mechanisms", None)
    if callable(update):
        update()
    requested = str(subgoal.get("mechanism") or "").strip()
    trigger = str(subgoal.get("trigger") or "").strip()
    effect = str(subgoal.get("effect") or "").strip().lower()
    for name, record in getattr(env, "_mechanisms", {}).items():
        if requested and name != requested:
            continue
        if trigger and getattr(record, "trigger_name", "") != trigger:
            continue
        if effect and "gate" in effect:
            gate_name = str(getattr(record, "gate_name", "")).lower()
            if not any(token in gate_name for token in ("gate", "door", "barrier", "blocker")):
                continue
        if bool(getattr(record, "activated", False)):
            return True
    return False


def _registered_mechanism_states(env: BaseEnv) -> list[dict[str, Any]]:
    update = getattr(env, "_update_mechanisms", None)
    if callable(update):
        update()
    states: list[dict[str, Any]] = []
    for name, record in getattr(env, "_mechanisms", {}).items():
        states.append(
            {
                "name": name,
                "trigger": getattr(record, "trigger_name", None),
                "gate": getattr(record, "gate_name", None),
                "activators": list(getattr(record, "activator_names", ()) or ()),
                "activated": bool(getattr(record, "activated", False)),
                "open_mode": getattr(record, "open_mode", None),
            }
        )
    return states


def _mechanism_candidates(env: BaseEnv, effect: str) -> list[Any]:
    tokens = {"gate", "door", "barrier", "blocker", "lock", "mechanism"}
    effect_lower = effect.lower()
    if "gate" in effect_lower:
        tokens.add("gate")
    candidates = []
    for record in env._objects.values():
        text = " ".join(
            [
                record.name,
                record.role or "",
                record.kind,
                json.dumps(record.metadata, sort_keys=True, default=str),
            ]
        ).lower()
        if any(token in text for token in tokens):
            candidates.append(record)
    return candidates


def _record_is_static_solid(record) -> bool:
    return record.body.body_type == pymunk.Body.STATIC and not _record_is_sensor(record)


def _body_type_name(body: pymunk.Body) -> str:
    if body.body_type == pymunk.Body.STATIC:
        return "static"
    if body.body_type == pymunk.Body.KINEMATIC:
        return "kinematic"
    return "dynamic"


def _diagnostic_failure_reason(prefix: str, diagnostics: dict[str, Any]) -> str:
    summary = diagnostics.get("summary") if isinstance(diagnostics.get("summary"), dict) else {}
    modes = ", ".join(_string_list(diagnostics.get("failure_modes"))) or "unknown_push_failure"
    repair = str(diagnostics.get("repair_instruction") or "").strip()
    kind = str(diagnostics.get("kind") or "")
    distance_reduced = summary.get("distance_reduced")
    final_distance = summary.get("final_distance")
    if kind == "ballistic_object_to_region":
        label = "Ballistic relation diagnostics"
    elif kind == "support_exit_freefall":
        label = "Support-exit/freefall diagnostics"
    elif kind in {"move_object_to_region", "low_friction_slide_to_region", "strike_object_to_region"}:
        label = "Object-region diagnostics"
    else:
        label = "Subgoal diagnostics"
    evidence = (
        f"{prefix} {label}: modes={modes}; "
        f"distance_reduced={distance_reduced} px; final_distance={final_distance} px."
    )
    if repair:
        evidence = f"{evidence} Repair: {repair}"
    return evidence


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_strike_subgoal(subgoal: dict[str, Any]) -> bool:
    kind = str(subgoal.get("kind") or "").lower()
    interaction = str(subgoal.get("interaction") or "").lower()
    action = str(subgoal.get("action") or "").lower()
    return kind == "strike_object_to_region" or any(
        token in f"{interaction} {action}"
        for token in ("kick", "strike", "shot", "shoot", "slam", "hit_contact")
    )


def _diagnostic_relation_name(diagnostics: dict[str, Any]) -> str:
    kind = str(diagnostics.get("kind") or "")
    if kind in {"strike_object_to_region", "ballistic_object_to_region", "support_exit_freefall"}:
        return kind
    return "move_object_to_region"


def _action_for_subgoal(
    env: BaseEnv,
    agent,
    subgoal: dict[str, Any],
    config: ValidatorConfig,
) -> dict[str, tuple[float, float]] | None:
    kind = str(subgoal.get("kind") or "")
    if kind in {"move_object_to_region", "low_friction_slide_to_region", "strike_object_to_region", "ballistic_object_to_region"}:
        object_record = _subgoal_record(env, subgoal, "object")
        region_record = _subgoal_record(env, subgoal, "region")
        if object_record is None or region_record is None:
            return None
        return {
            "move": _push_object_direction(
                agent.body.position,
                object_record.body.position,
                region_record.body.position,
            )
        }
    if kind == "support_exit_freefall":
        object_record = _subgoal_record(env, subgoal, "object")
        boundary_record = _subgoal_record(env, subgoal, "boundary") or _subgoal_record(
            env,
            subgoal,
            "region",
        )
        if object_record is None or boundary_record is None:
            return None
        return {
            "move": _push_object_direction(
                agent.body.position,
                object_record.body.position,
                boundary_record.body.position,
            )
        }
    if kind in {"agent_reach_region", "agent_touch_object"}:
        target_record = _subgoal_record(env, subgoal, "target") or _subgoal_record(
            env, subgoal, "object"
        )
        if target_record is None:
            return None
        direction = target_record.body.position - agent.body.position
        if direction.length <= 1.0:
            return None
        return {"move": (float(direction.x), float(direction.y))}
    if kind == "survive_duration":
        return _survival_avoidance_action(env, agent, subgoal, config)
    if kind == "activate_mechanism":
        return None
    return {"move": (1.0, 0.0)}


def _survival_avoidance_action(
    env: BaseEnv,
    agent,
    subgoal: dict[str, Any],
    config: ValidatorConfig,
) -> dict[str, tuple[float, float]] | None:
    """Simple deterministic dodge policy for survival validation."""

    agent_pos = agent.body.position
    hazards = _hazard_records_for_survival(env, subgoal)
    lateral_jump = _lateral_hazard_jump_action(env, agent, hazards, config)
    if lateral_jump is not None:
        return lateral_jump
    sampled = _sample_survival_action(env, agent, hazards, config)
    if sampled is not None:
        return sampled

    desired = pymunk.Vec2d(0.0, 0.0)
    for hazard in hazards:
        offset = agent_pos - hazard.body.position
        distance = max(float(offset.length), 1.0)
        if distance > 360.0:
            continue
        away = offset.normalized() if distance > 1e-6 else pymunk.Vec2d(1.0, 0.0)
        hazard_velocity = hazard.body.velocity
        speed = float(hazard_velocity.length)
        closing = 0.0
        if speed > 1e-6:
            closing = max(0.0, float(hazard_velocity.normalized().dot(away)))
            tangent = pymunk.Vec2d(-hazard_velocity.y, hazard_velocity.x).normalized()
            if float(tangent.dot(away)) < 0.0:
                tangent = -tangent
        else:
            tangent = pymunk.Vec2d(-away.y, away.x)
        urgency = (1.0 / distance) * (1.0 + closing * min(speed / 220.0, 2.0))
        desired += away * (urgency * 180.0)
        if closing > 0.2:
            desired += tangent * (urgency * 130.0)

    margin = max(float(config.agent_radius) * 3.0, 54.0)
    width = float(getattr(env, "width", 960.0) or 960.0)
    height = float(getattr(env, "height", 640.0) or 640.0)
    if float(agent_pos.x) < margin:
        desired += pymunk.Vec2d(1.0, 0.0) * ((margin - float(agent_pos.x)) / margin) * 2.0
    if float(agent_pos.x) > width - margin:
        desired += pymunk.Vec2d(-1.0, 0.0) * ((float(agent_pos.x) - (width - margin)) / margin) * 2.0
    if float(agent_pos.y) < margin:
        desired += pymunk.Vec2d(0.0, 1.0) * ((margin - float(agent_pos.y)) / margin) * 2.0
    if float(agent_pos.y) > height - margin:
        desired += pymunk.Vec2d(0.0, -1.0) * ((float(agent_pos.y) - (height - margin)) / margin) * 2.0

    if float(desired.length) <= 1e-6:
        desired = pymunk.Vec2d(width * 0.5, height * 0.5) - agent_pos
    if float(desired.length) <= 1e-6:
        return None
    desired = desired.normalized()
    return {"move": (float(desired.x), float(desired.y))}


def _lateral_hazard_jump_action(
    env: BaseEnv,
    agent,
    hazards: list[Any],
    config: ValidatorConfig,
) -> dict[str, tuple[float, float]] | None:
    """Time a grounded jump over lane-locked horizontal hazards.

    This handles side-view prompts like "jump over endless cars" without
    relaxing the objective. It is still a generic relation oracle: a dynamic
    hazard must travel horizontally through the agent's lane, and the agent
    responds with an upward jump when time-to-contact enters a safe window.
    """

    if not hazards:
        return None
    agent_pos = agent.body.position
    agent_radius = _record_radius(agent, fallback=float(config.agent_radius))
    lane_half_height = max(46.0, agent_radius * 2.4)
    best_time: float | None = None
    best_hazard = None
    saw_lateral_hazard = False
    for hazard in hazards:
        text = " ".join(
            [
                str(getattr(hazard, "name", "")).lower(),
                str(getattr(hazard, "role", "")).lower(),
                str(getattr(hazard, "kind", "")).lower(),
                json.dumps(getattr(hazard, "metadata", {}), sort_keys=True, default=str).lower(),
            ]
        )
        if not any(
            token in text
            for token in (
                "car",
                "truck",
                "train",
                "traffic",
                "vehicle",
                "rolling",
                "lateral",
                "recurring_lateral",
            )
        ):
            continue
        saw_lateral_hazard = True
        if hazard.body.body_type != pymunk.Body.DYNAMIC:
            continue
        pos = hazard.body.position
        vel = hazard.body.velocity
        if abs(float(pos.y - agent_pos.y)) > lane_half_height:
            continue
        vx = float(vel.x)
        if abs(vx) < 35.0:
            continue
        dx = float(pos.x - agent_pos.x)
        # Positive time means the object is moving toward the agent's x column.
        time_to_column = -dx / vx
        if time_to_column < -0.08 or time_to_column > 1.2:
            continue
        if best_time is None or abs(time_to_column - 0.42) < abs(best_time - 0.42):
            best_time = float(time_to_column)
            best_hazard = hazard

    if best_hazard is None or best_time is None:
        return {"move": (0.0, 0.0)} if saw_lateral_hazard else None

    # Jump a little before the object reaches the agent. While airborne, a
    # repeated upward command is harmless in BaseEnv's normal-gravity controller.
    if -0.08 <= best_time <= 0.58:
        return {"move": (0.0, 1.0)}
    return {"move": (0.0, 0.0)}


def _sample_survival_action(
    env: BaseEnv,
    agent,
    hazards: list[Any],
    config: ValidatorConfig,
) -> dict[str, tuple[float, float]] | None:
    if not hazards:
        return None
    width = float(getattr(env, "width", 960.0) or 960.0)
    height = float(getattr(env, "height", 640.0) or 640.0)
    margin = max(float(config.agent_radius) * 3.0, 54.0)
    agent_pos = agent.body.position
    agent_vel = agent.body.velocity
    strength = max(float(getattr(env, "agent_strength", 2500.0) or 2500.0), 1.0)
    mass = max(float(getattr(agent.body, "mass", 1.0) or 1.0), 1.0)
    accel = min(900.0, max(180.0, strength / mass))
    directions = [pymunk.Vec2d(0.0, 0.0)]
    for index in range(16):
        angle = (math.tau * index) / 16.0
        directions.append(pymunk.Vec2d(math.cos(angle), math.sin(angle)))

    best_direction = None
    best_score = -float("inf")
    horizons = (0.25, 0.45, 0.7, 1.0, 1.35)
    for direction in directions:
        unit = direction.normalized() if float(direction.length) > 1e-6 else pymunk.Vec2d(0.0, 0.0)
        score = 0.0
        min_clearance = float("inf")
        wall_penalty = 0.0
        for horizon in horizons:
            predicted_agent = agent_pos + agent_vel * horizon + unit * (0.5 * accel * horizon * horizon)
            if predicted_agent.x < margin:
                wall_penalty += (margin - predicted_agent.x) * 2.5
            if predicted_agent.x > width - margin:
                wall_penalty += (predicted_agent.x - (width - margin)) * 2.5
            if predicted_agent.y < margin:
                wall_penalty += (margin - predicted_agent.y) * 2.5
            if predicted_agent.y > height - margin:
                wall_penalty += (predicted_agent.y - (height - margin)) * 2.5
            for hazard in hazards:
                predicted_hazard = hazard.body.position + hazard.body.velocity * horizon
                clearance = float(predicted_agent.get_distance(predicted_hazard))
                min_clearance = min(min_clearance, clearance)
                if clearance < 150.0:
                    score -= (150.0 - clearance) ** 2 * 0.025
        center = pymunk.Vec2d(width * 0.5, height * 0.5)
        center_distance = float((agent_pos + unit * 25.0).get_distance(center))
        velocity_alignment = float(agent_vel.dot(unit)) if float(unit.length) > 1e-6 else 0.0
        score += min_clearance * 1.6
        score -= wall_penalty
        score -= center_distance * 0.03
        score -= max(0.0, velocity_alignment) * 0.015
        if score > best_score:
            best_score = score
            best_direction = unit

    if best_direction is None or float(best_direction.length) <= 1e-6:
        return None
    return {"move": (float(best_direction.x), float(best_direction.y))}


def _hazard_records_for_survival(env: BaseEnv, subgoal: dict[str, Any]) -> list[Any]:
    avoid_role = str(subgoal.get("avoid_role") or "hazard").lower()
    records = []
    for record in getattr(env, "_objects", {}).values():
        role = str(record.role or "").lower()
        name = str(record.name or "").lower()
        metadata = json.dumps(record.metadata, sort_keys=True, default=str).lower()
        text = f"{name} {metadata}"
        if role == avoid_role:
            records.append(record)
        elif avoid_role == "hazard" and role in {"enemy", "chaser"} and any(
            token in text for token in ("shot", "laser", "projectile", "bolt", "bullet", "missile")
        ):
            records.append(record)
    return records


def _step_subgoal(
    env: BaseEnv,
    agent,
    subgoal: dict[str, Any],
    config: ValidatorConfig,
) -> None:
    kind = str(subgoal.get("kind") or "")
    if kind in {"move_object_to_region", "low_friction_slide_to_region", "strike_object_to_region", "ballistic_object_to_region"}:
        object_record = _subgoal_record(env, subgoal, "object")
        region_record = _subgoal_record(env, subgoal, "region")
        if object_record is not None and region_record is not None:
            if kind == "ballistic_object_to_region":
                _step_ballistic_object_to_region(env, agent, object_record, region_record, config)
            elif _is_strike_subgoal(subgoal):
                _step_strike_object_to_region(env, agent, object_record, region_record, config)
            else:
                _step_push_object_to_region(env, agent, object_record, region_record, config)
            return
    if kind == "support_exit_freefall":
        object_record = _subgoal_record(env, subgoal, "object")
        boundary_record = _subgoal_record(env, subgoal, "boundary") or _subgoal_record(env, subgoal, "region")
        if object_record is not None and boundary_record is not None:
            if not _support_exit_motion_observed(object_record, boundary_record, subgoal):
                _step_support_exit_freefall(env, agent, object_record, boundary_record, config)
            else:
                env.step(substeps=config.kinetic_substeps)
            return
    if kind in {"agent_reach_region", "agent_touch_object"}:
        target_record = _subgoal_record(env, subgoal, "target") or _subgoal_record(
            env, subgoal, "object"
        )
        if target_record is not None:
            target_position = _navigation_target_position(env, agent, target_record, config)
            _step_agent_toward(env, agent, target_position, config)
            return
    if kind == "field_force_interaction":
        object_record = _subgoal_record(env, subgoal, "object")
        field_record = _subgoal_record(env, subgoal, "field")
        if object_record is not None and field_record is not None:
            if not _record_point_inside_record(field_record, object_record):
                _step_push_object_to_region(env, agent, object_record, field_record, config)
            else:
                env.step(substeps=config.kinetic_substeps)
            return
    if kind == "lever_launch":
        weight_record = _subgoal_record(env, subgoal, "weight") or _subgoal_record(
            env,
            subgoal,
            "object",
        )
        impact_record = _subgoal_record(env, subgoal, "impact_region") or _subgoal_record(
            env,
            subgoal,
            "region",
        )
        if weight_record is not None and impact_record is not None:
            if float(weight_record.body.position.get_distance(impact_record.body.position)) > _subgoal_threshold(env, subgoal, config):
                _step_push_object_to_region(env, agent, weight_record, impact_record, config)
            else:
                env.step(substeps=config.kinetic_substeps)
            return
        target_record = _subgoal_record(env, subgoal, "target")
        if target_record is not None:
            _step_agent_toward(env, agent, target_record.body.position, config)
            return
    action = _action_for_subgoal(env, agent, subgoal, config)
    env.step(action=action, substeps=config.kinetic_substeps)


def _step_push_object_to_region(
    env: BaseEnv,
    agent,
    object_record,
    region_record,
    config: ValidatorConfig,
) -> None:
    state = _push_contact_state(agent, object_record, region_record, config)
    if state["distance"] <= 1.0:
        env._validator_last_push_force = {
            "magnitude": 0.0,
            "toward_region": 0.0,
            "vector": [0.0, 0.0],
        }
        env.step(substeps=config.kinetic_substeps)
        return

    object_pos = object_record.body.position
    push_axis = state["axis"]
    perpendicular = state["perpendicular"]
    staging_gap = state["staging_gap"]
    distance = state["distance"]
    longitudinal = state["agent_longitudinal_offset"]
    lateral = state["agent_lateral_offset"]
    object_velocity_toward_region = state["object_velocity_toward_region"]

    desired_stage = object_pos - push_axis * staging_gap
    lateral_error = perpendicular * lateral

    if state["phase"] == "stage_behind":
        target_position = desired_stage
    elif state["phase"] == "correct_lateral":
        target_position = desired_stage - lateral_error * 0.8
    elif state["phase"] == "recover_reverse_motion":
        target_position = desired_stage
    elif state["phase"] == "brake_near_region":
        target_position = object_pos - push_axis * (staging_gap * 0.9)
    else:
        push_distance = min(72.0, max(28.0, distance + 8.0))
        target_position = object_pos + push_axis * push_distance

    if longitudinal > staging_gap * 0.2:
        # Never intentionally stage in front of the object; that commonly
        # reverses the push and increases object-to-region distance.
        target_position = desired_stage
    if object_velocity_toward_region < -25.0:
        target_position = desired_stage

    force_metrics = _apply_agent_pd_force(env, agent, target_position, push_axis)
    env._validator_last_push_force = force_metrics
    env.step(substeps=config.kinetic_substeps)


def _step_strike_object_to_region(
    env: BaseEnv,
    agent,
    object_record,
    region_record,
    config: ValidatorConfig,
) -> None:
    state = _push_contact_state(agent, object_record, region_record, config)
    if state["distance"] <= 1.0:
        env._validator_last_push_force = {
            "magnitude": 0.0,
            "toward_region": 0.0,
            "vector": [0.0, 0.0],
        }
        env.step(substeps=config.kinetic_substeps)
        return

    object_pos = object_record.body.position
    push_axis = state["axis"]
    perpendicular = state["perpendicular"]
    object_radius = _record_radius(object_record, config.agent_radius)
    agent_radius = _record_radius(agent, config.agent_radius)
    staging_gap = max(6.0, agent_radius + object_radius - 1.0)
    desired_stage = object_pos - push_axis * staging_gap
    lateral = state["agent_lateral_offset"]
    longitudinal = state["agent_longitudinal_offset"]
    surface_gap = state["agent_object_surface_gap"]

    if longitudinal > -staging_gap * 0.25 or longitudinal < -staging_gap * 4.5:
        target_position = desired_stage - perpendicular * lateral * 0.65
    elif abs(lateral) > max(6.0, object_radius * 0.26):
        target_position = desired_stage - perpendicular * lateral
    else:
        # Drive through the ball, like a kick/follow-through, instead of
        # creeping behind it like a crate push.
        target_position = object_pos + push_axis * min(220.0, max(120.0, state["distance"] + 35.0))

    # Once surface contact is happening, keep force committed through the shot
    # axis so the ball gains useful velocity rather than being side-scrubbed.
    if surface_gap <= 3.0:
        target_position = object_pos + push_axis * min(240.0, max(150.0, state["distance"] + 55.0))

    force_metrics = _apply_agent_pd_force(
        env,
        agent,
        target_position,
        push_axis,
        desired_speed_cap=560.0,
        force_multiplier=5.5,
    )
    env._validator_last_push_force = force_metrics
    env.step(substeps=config.kinetic_substeps)


def _step_ballistic_object_to_region(
    env: BaseEnv,
    agent,
    object_record,
    region_record,
    config: ValidatorConfig,
) -> None:
    """Validator controller for throw/lob/hoop-style relations."""

    object_pos = object_record.body.position
    region_pos = region_record.body.position
    to_region = region_pos - object_pos
    if _record_is_agent_fired_projectile(env, object_record):
        if to_region.length > 1.0:
            axis = to_region.normalized()
            toward_speed = float(object_record.body.velocity.dot(axis))
            if toward_speed < 180.0 and _ballistic_impulse_allowed(env, object_record, region_record):
                impulse = axis * max(420.0, float(getattr(object_record.body, "mass", 1.0) or 1.0) * 620.0)
                object_record.body.apply_impulse_at_world_point(impulse, object_record.body.position)
                _record_ballistic_impulse(env, object_record, region_record)
                env._validator_last_push_force = {
                    "magnitude": round(float(impulse.length), 3),
                    "toward_region": round(float(impulse.dot(axis)), 3),
                    "vector": [round(float(impulse.x), 3), round(float(impulse.y), 3)],
                    "mode": "agent_fired_projectile_assist",
                }
            else:
                env._validator_last_push_force = {
                    "magnitude": 0.0,
                    "toward_region": round(toward_speed, 3),
                    "vector": [0.0, 0.0],
                    "mode": "agent_projectile_already_in_flight",
                }
        env.step(substeps=config.kinetic_substeps)
        return
    horizontal = pymunk.Vec2d(1.0 if to_region.x >= 0 else -1.0, 0.0)
    object_radius = _record_radius(object_record, config.agent_radius)
    agent_radius = _record_radius(agent, config.agent_radius)
    staging_gap = max(8.0, agent_radius + object_radius - 2.0)
    desired_stage = object_pos - horizontal * staging_gap
    agent_distance = float(agent.body.position.get_distance(object_pos))
    surface_gap = agent_distance - (agent_radius + object_radius)

    if agent_distance > staging_gap + 5.0:
        force_metrics = _apply_agent_pd_force(
            env,
            agent,
            desired_stage,
            horizontal,
            desired_speed_cap=520.0,
            force_multiplier=4.8,
        )
        env._validator_last_push_force = force_metrics
        env.step(substeps=config.kinetic_substeps)
        return

    if surface_gap <= 18.0 and _ballistic_impulse_allowed(env, object_record, region_record):
        impulse = _ballistic_impulse_to_region(env, object_record, region_record)
        object_record.body.apply_impulse_at_world_point(impulse, object_record.body.position)
        _record_ballistic_impulse(env, object_record, region_record)
        env._validator_last_push_force = {
            "magnitude": round(float(impulse.length), 3),
            "toward_region": round(float(impulse.dot(to_region.normalized())) if to_region.length > 1.0 else 0.0, 3),
            "vector": [round(float(impulse.x), 3), round(float(impulse.y), 3)],
        }
        env.step(substeps=config.kinetic_substeps)
        return

    gravity_y = float(getattr(env.space, "gravity", (0.0, -981.0)).y)
    upward = 0.65 if gravity_y < -1.0 else 0.2
    shot_axis = (to_region.normalized() + pymunk.Vec2d(0.0, upward)).normalized() if to_region.length > 1.0 else pymunk.Vec2d(horizontal.x, upward).normalized()
    target_position = object_pos + shot_axis * 260.0
    force_metrics = _apply_agent_pd_force(
        env,
        agent,
        target_position,
        shot_axis,
        desired_speed_cap=720.0,
        force_multiplier=7.5,
    )
    env._validator_last_push_force = force_metrics
    env.step(substeps=config.kinetic_substeps)


def _ballistic_impulse_allowed(env: BaseEnv, object_record, region_record) -> bool:
    impulses = getattr(env, "_validator_ballistic_impulses", {})
    key = f"{object_record.name}->{region_record.name}"
    if not isinstance(impulses, dict):
        return True
    record = impulses.get(key)
    if not isinstance(record, dict):
        return True
    count = int(record.get("count") or 0)
    if count >= 3:
        return False
    last_step = int(record.get("last_step") or -10_000)
    if int(getattr(env, "step_count", 0)) - last_step < 45:
        return False
    object_pos = object_record.body.position
    region_pos = region_record.body.position
    to_region = region_pos - object_pos
    if to_region.length <= _record_radius(region_record, 32.0):
        return False
    axis = to_region.normalized() if to_region.length > 1.0 else pymunk.Vec2d(1.0, 0.0)
    toward_velocity = float(object_record.body.velocity.dot(axis))
    return toward_velocity < 90.0


def _record_ballistic_impulse(env: BaseEnv, object_record, region_record) -> None:
    impulses = getattr(env, "_validator_ballistic_impulses", {})
    if not isinstance(impulses, dict):
        impulses = {}
    key = f"{object_record.name}->{region_record.name}"
    existing = impulses.get(key) if isinstance(impulses.get(key), dict) else {}
    impulses[key] = {
        "count": int(existing.get("count") or 0) + 1,
        "last_step": int(getattr(env, "step_count", 0)),
    }
    setattr(env, "_validator_ballistic_impulses", impulses)


def _ballistic_impulse_to_region(env: BaseEnv, object_record, region_record) -> pymunk.Vec2d:
    object_pos = object_record.body.position
    region_pos = region_record.body.position
    dx = float(region_pos.x - object_pos.x)
    dy = float(region_pos.y - object_pos.y)
    gravity_y = float(getattr(env.space, "gravity", pymunk.Vec2d(0.0, -981.0)).y)
    clearance_y = _ballistic_clearance_apex_y(env, object_record, region_record)
    horizontal_distance = abs(dx)
    travel_time = max(0.62, min(1.35, horizontal_distance / 330.0 if horizontal_distance > 1.0 else 0.75))
    vx = dx / travel_time
    vy = (dy - 0.5 * gravity_y * travel_time * travel_time) / travel_time
    if gravity_y < -1.0 and clearance_y is not None and clearance_y > float(object_pos.y):
        min_clearance_vy = math.sqrt(max(0.0, 2.0 * abs(gravity_y) * (clearance_y - float(object_pos.y))))
        vy = max(vy, min_clearance_vy)
    desired_velocity = pymunk.Vec2d(vx, vy)
    max_speed = 980.0 if clearance_y is not None else 820.0
    if desired_velocity.length > max_speed:
        desired_velocity = desired_velocity.normalized() * max_speed
    delta_v = desired_velocity - object_record.body.velocity
    return delta_v * max(float(object_record.body.mass), 0.1)


def _ballistic_clearance_apex_y(env: BaseEnv, object_record, region_record) -> float | None:
    """Return a safe apex height when solid blockers sit between object and target."""

    start_x = float(object_record.body.position.x)
    end_x = float(region_record.body.position.x)
    min_x = min(start_x, end_x)
    max_x = max(start_x, end_x)
    highest_top: float | None = None
    for record in getattr(env, "_objects", {}).values():
        if record is object_record or record is region_record:
            continue
        if record.body.body_type != pymunk.Body.STATIC:
            continue
        if _record_is_sensor(record) or record.role in {"goal", "support"}:
            continue
        name_role = f"{record.name} {record.role}".lower()
        if any(token in name_role for token in ("ceiling", "bound", "floor", "ground", "support")):
            continue
        try:
            bounds = _record_bounds(record)
        except Exception:
            continue
        left, bottom, right, top = bounds
        if right < min_x or left > max_x:
            continue
        if top <= min(float(object_record.body.position.y), float(region_record.body.position.y)) + 20.0:
            continue
        highest_top = top if highest_top is None else max(highest_top, top)
    if highest_top is None:
        return None
    return highest_top + max(55.0, _record_radius(object_record, 18.0) * 2.4)


def _ballistic_observation_key(object_record, region_record) -> str:
    return f"{object_record.name}->{region_record.name}"


def _ballistic_clearance_requirement_satisfied(
    env: BaseEnv,
    object_record,
    region_record,
    subgoal: dict[str, Any],
    config: ValidatorConfig,
) -> bool:
    """Require target entry plus explicit arc evidence for over-barrier tasks."""

    observation = _update_ballistic_observation(
        env,
        object_record,
        region_record,
        subgoal,
        config,
    )
    if observation is None or not observation.get("requires_clearance"):
        return True
    clearance = bool(observation.get("clearance_satisfied"))
    crossed = bool(observation.get("crossed_barrier"))
    if observation.get("barrier_name"):
        return clearance and crossed
    return clearance


def _update_ballistic_observation(
    env: BaseEnv,
    object_record,
    region_record,
    subgoal: dict[str, Any] | None,
    config: ValidatorConfig | None,
) -> dict[str, Any] | None:
    """Track barrier crossing and apex height for ballistic relation validation."""

    observations = getattr(env, "_validator_ballistic_observations", None)
    if not isinstance(observations, dict):
        observations = {}
    key = _ballistic_observation_key(object_record, region_record)
    previous = observations.get(key) if isinstance(observations.get(key), dict) else {}
    requirement = _ballistic_clearance_requirement(
        env,
        object_record,
        region_record,
        subgoal,
        config,
        previous=previous,
    )
    if requirement is None and not previous:
        return None

    position = object_record.body.position
    start_position = previous.get("start_position")
    if not isinstance(start_position, list) or len(start_position) < 2:
        start_position = [float(position.x), float(position.y)]
    max_y = max(
        _float_or_none(previous.get("max_y")) or float(position.y),
        float(position.y),
    )

    required_apex_y = None
    barrier_name = None
    barrier_top_y = None
    crossed_barrier = bool(previous.get("crossed_barrier"))
    requires_clearance = bool(previous.get("requires_clearance"))
    if requirement is not None:
        required_apex_y = _float_or_none(requirement.get("required_apex_y"))
        barrier_name = str(requirement.get("barrier_name") or "") or None
        barrier_top_y = _float_or_none(requirement.get("barrier_top_y"))
        requires_clearance = True
        barrier_record = requirement.get("barrier_record")
        if barrier_record is not None:
            bounds = _record_bounds(barrier_record)
            region_x = float(region_record.body.position.x)
            barrier_x = float(barrier_record.body.position.x)
            direction = 1.0 if region_x >= barrier_x else -1.0
            barrier_edge = bounds[2] if direction > 0.0 else bounds[0]
            crossed_barrier = crossed_barrier or (float(position.x) - barrier_edge) * direction >= 0.0
    else:
        required_apex_y = _float_or_none(previous.get("required_apex_y"))
        barrier_name = str(previous.get("barrier_name") or "") or None
        barrier_top_y = _float_or_none(previous.get("barrier_top_y"))

    clearance_margin_observed = None
    clearance_satisfied = bool(previous.get("clearance_satisfied"))
    if required_apex_y is not None:
        clearance_margin_observed = max_y - required_apex_y
        clearance_satisfied = clearance_satisfied or clearance_margin_observed >= 0.0

    observation = {
        "object": object_record.name,
        "region": region_record.name,
        "start_position": [round(float(start_position[0]), 3), round(float(start_position[1]), 3)],
        "max_y": round(float(max_y), 3),
        "requires_clearance": requires_clearance,
        "required_apex_y": None if required_apex_y is None else round(float(required_apex_y), 3),
        "barrier_name": barrier_name,
        "barrier_top_y": None if barrier_top_y is None else round(float(barrier_top_y), 3),
        "clearance_margin_observed": None
        if clearance_margin_observed is None
        else round(float(clearance_margin_observed), 3),
        "clearance_satisfied": clearance_satisfied,
        "crossed_barrier": crossed_barrier,
    }
    observations[key] = observation
    setattr(env, "_validator_ballistic_observations", observations)
    return observation


def _ballistic_clearance_requirement(
    env: BaseEnv,
    object_record,
    region_record,
    subgoal: dict[str, Any] | None,
    config: ValidatorConfig | None,
    *,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    barrier_name = None
    margin = _float_or_none((subgoal or {}).get("clearance_margin"))
    for requirement in _semantic_requirements_from_env(env, {}):
        if not isinstance(requirement, dict):
            continue
        if str(requirement.get("kind") or "") != "ballistic_arc_required":
            continue
        req_object = str(requirement.get("object") or object_record.name)
        req_region = str(requirement.get("region") or region_record.name)
        if req_object != object_record.name or req_region != region_record.name:
            continue
        barrier_name = str(requirement.get("barrier") or "") or barrier_name
        margin = _float_or_none(requirement.get("clearance_margin")) or margin
        break

    if not barrier_name and previous:
        barrier_name = str(previous.get("barrier_name") or "") or None

    barrier_record = env._objects.get(barrier_name) if barrier_name else None
    if barrier_record is None and config is not None:
        barrier_record = _solid_blocker_between(env, object_record, region_record, config)
    if barrier_record is None:
        required_apex = _ballistic_clearance_apex_y(env, object_record, region_record)
        if required_apex is None:
            return None
        return {
            "required_apex_y": required_apex,
            "barrier_name": None,
            "barrier_top_y": None,
            "barrier_record": None,
        }

    _, _, _, top = _record_bounds(barrier_record)
    margin = 45.0 if margin is None else float(margin)
    return {
        "required_apex_y": float(top) + margin,
        "barrier_name": barrier_record.name,
        "barrier_top_y": float(top),
        "barrier_record": barrier_record,
    }


def _add_ballistic_summary(
    summary: dict[str, Any],
    samples: list[dict[str, Any]],
    final: dict[str, Any],
) -> None:
    max_y = max(
        (
            float(sample["ballistic_max_y"])
            for sample in samples
            if sample.get("ballistic_max_y") is not None
        ),
        default=_float_or_none(final.get("ballistic_max_y")),
    )
    required_apex = _float_or_none(final.get("ballistic_required_apex_y"))
    if max_y is not None:
        summary["ballistic_max_y"] = round(float(max_y), 3)
    if required_apex is not None:
        summary["ballistic_required_apex_y"] = round(float(required_apex), 3)
        if max_y is not None:
            summary["ballistic_clearance_margin_observed"] = round(float(max_y - required_apex), 3)
    for key in ("ballistic_barrier_name", "ballistic_barrier_top_y", "ballistic_crossed_barrier"):
        if final.get(key) is not None:
            summary[key] = final.get(key)


def _record_bounds(record) -> tuple[float, float, float, float]:
    """Return left, bottom, right, top bounds for a registered object."""

    boxes = []
    for shape in getattr(record, "shapes", ()) or ():
        try:
            bb = shape.cache_bb()
        except Exception:
            bb = getattr(shape, "bb", None)
        if bb is None:
            continue
        boxes.append((float(bb.left), float(bb.bottom), float(bb.right), float(bb.top)))
    if boxes:
        left = min(box[0] for box in boxes)
        bottom = min(box[1] for box in boxes)
        right = max(box[2] for box in boxes)
        top = max(box[3] for box in boxes)
        return left, bottom, right, top
    pos = record.body.position
    radius = _record_radius(record, 16.0)
    return float(pos.x - radius), float(pos.y - radius), float(pos.x + radius), float(pos.y + radius)


def _record_is_sensor(record) -> bool:
    return any(bool(getattr(shape, "sensor", False)) for shape in getattr(record, "shapes", ()) or ())


def _step_support_exit_freefall(
    env: BaseEnv,
    agent,
    object_record,
    boundary_record,
    config: ValidatorConfig,
) -> None:
    """Validator controller for pushing an object across a real support edge."""

    object_pos = object_record.body.position
    boundary_pos = boundary_record.body.position
    direction_sign = 1.0 if float(boundary_pos.x - object_pos.x) >= 0.0 else -1.0
    push_axis = pymunk.Vec2d(direction_sign, 0.0)
    object_radius = _record_radius(object_record, config.agent_radius)
    agent_radius = _record_radius(agent, config.agent_radius)
    staging_gap = max(6.0, agent_radius + object_radius - 1.0)
    desired_stage = object_pos - push_axis * staging_gap
    agent_delta = agent.body.position - object_pos
    longitudinal = float(agent_delta.dot(push_axis))
    lateral = float(agent_delta.y)
    surface_gap = float(agent.body.position.get_distance(object_pos)) - (agent_radius + object_radius)

    if longitudinal > -staging_gap * 0.2 or longitudinal < -staging_gap * 4.5:
        target_position = pymunk.Vec2d(float(desired_stage.x), float(object_pos.y))
    elif abs(lateral) > max(8.0, object_radius * 0.35):
        target_position = pymunk.Vec2d(float(desired_stage.x), float(object_pos.y))
    else:
        target_position = object_pos + push_axis * 120.0

    force_metrics = _apply_agent_pd_force(
        env,
        agent,
        target_position,
        push_axis,
        desired_speed_cap=560.0,
        force_multiplier=6.0,
    )
    if surface_gap <= 5.0:
        impulse = push_axis * max(float(object_record.body.mass), 0.1) * 32.0
        object_record.body.apply_impulse_at_world_point(impulse, object_record.body.position)
        force_metrics = {
            "magnitude": round(float(impulse.length), 3),
            "toward_region": round(float(impulse.dot(push_axis)), 3),
            "vector": [round(float(impulse.x), 3), round(float(impulse.y), 3)],
        }
    env._validator_last_push_force = force_metrics
    env.step(substeps=config.kinetic_substeps)


def _support_exit_motion_observed(
    object_record,
    boundary_record,
    subgoal: dict[str, Any],
) -> bool:
    """Return whether an object has left support and begun falling/exiting."""

    object_pos = object_record.body.position
    boundary_pos = boundary_record.body.position
    min_fall_distance = _float_subgoal_value(subgoal, "min_fall_distance", 64.0)
    min_downward_velocity = _float_subgoal_value(subgoal, "min_downward_velocity", 24.0)
    crossed_distance = _float_subgoal_value(subgoal, "crossed_distance", 42.0)

    # Pymunk coordinates are y-up in this harness; negative y velocity means
    # a visible fall under normal gravity.
    if (
        float(object_pos.y) <= float(boundary_pos.y) - min_fall_distance
        and float(object_record.body.velocity.y) <= -min_downward_velocity
    ):
        return True

    axis_hint = str(subgoal.get("axis") or "").lower()
    if axis_hint in {"x", "horizontal"}:
        return abs(float(object_pos.x) - float(boundary_pos.x)) <= crossed_distance
    if axis_hint in {"y", "vertical"}:
        return abs(float(object_pos.y) - float(boundary_pos.y)) <= crossed_distance
    return float(object_pos.get_distance(boundary_pos)) <= crossed_distance


def _push_contact_state(agent, object_record, region_record, config: ValidatorConfig) -> dict[str, Any]:
    object_pos = object_record.body.position
    region_pos = region_record.body.position
    object_to_region = region_pos - object_pos
    distance = float(object_to_region.length)
    if distance <= 1.0:
        axis = pymunk.Vec2d(1.0, 0.0)
    else:
        axis = object_to_region.normalized()
    perpendicular = pymunk.Vec2d(-axis.y, axis.x)
    object_radius = _record_radius(object_record, config.agent_radius)
    agent_radius = _record_radius(agent, config.agent_radius)
    center_distance = float(agent.body.position.get_distance(object_pos))
    surface_gap = center_distance - (agent_radius + object_radius)
    staging_gap = max(10.0, agent_radius + object_radius + 3.0)
    agent_from_object = agent.body.position - object_pos
    longitudinal = float(agent_from_object.dot(axis))
    lateral = float(agent_from_object.dot(perpendicular))
    object_velocity_toward_region = float(object_record.body.velocity.dot(axis))
    agent_velocity_toward_region = float(agent.body.velocity.dot(axis))

    phase = "push_forward"
    if longitudinal > -staging_gap * 0.45 or longitudinal < -staging_gap * 3.0:
        phase = "stage_behind"
    elif abs(lateral) > max(8.0, object_radius * 0.35):
        phase = "correct_lateral"
    elif object_velocity_toward_region < -25.0:
        phase = "recover_reverse_motion"
    elif distance < max(42.0, object_radius * 1.6) and object_velocity_toward_region > 90.0:
        phase = "brake_near_region"

    return {
        "axis": axis,
        "perpendicular": perpendicular,
        "distance": distance,
        "staging_gap": float(staging_gap),
        "agent_longitudinal_offset": longitudinal,
        "agent_lateral_offset": lateral,
        "agent_object_center_distance": center_distance,
        "agent_object_surface_gap": surface_gap,
        "object_velocity_toward_region": object_velocity_toward_region,
        "agent_velocity_toward_region": agent_velocity_toward_region,
        "phase": phase,
    }


def _apply_agent_pd_force(
    env: BaseEnv,
    agent,
    target_position: pymunk.Vec2d,
    push_axis: pymunk.Vec2d | None = None,
    *,
    desired_speed_cap: float = 380.0,
    force_multiplier: float = 3.25,
) -> dict[str, Any]:
    offset = target_position - agent.body.position
    if offset.length <= 1.0:
        return {"magnitude": 0.0, "toward_region": 0.0, "vector": [0.0, 0.0]}
    desired_speed = min(float(desired_speed_cap), max(80.0, offset.length * 4.0))
    desired_velocity = offset.normalized() * desired_speed
    velocity_error = desired_velocity - agent.body.velocity
    force_scale = max(float(agent.body.mass), 1.0) * 24.0
    max_force = max(float(getattr(env, "agent_strength", 1.0)), 1.0) * float(force_multiplier)
    force = velocity_error * force_scale
    if force.length > max_force:
        force = force.normalized() * max_force
    agent.body.apply_force_at_world_point(force, agent.body.position)
    toward_region = float(force.dot(push_axis)) if push_axis is not None else None
    return {
        "magnitude": round(float(force.length), 3),
        "toward_region": None if toward_region is None else round(toward_region, 3),
        "vector": [round(float(force.x), 3), round(float(force.y), 3)],
    }


def _push_object_direction(
    agent_position: pymunk.Vec2d,
    object_position: pymunk.Vec2d,
    region_position: pymunk.Vec2d,
) -> tuple[float, float]:
    object_to_region = region_position - object_position
    if object_to_region.length <= 1.0:
        return (0.0, 0.0)
    push_axis = object_to_region.normalized()
    desired_agent_position = object_position - push_axis * 34.0
    staging_error = desired_agent_position - agent_position
    if staging_error.length > 24.0:
        direction = staging_error
    else:
        direction = push_axis
    if direction.length <= 1.0:
        return (float(push_axis.x), float(push_axis.y))
    direction = direction.normalized()
    return (float(direction.x), float(direction.y))


def _subgoal_satisfied(env: BaseEnv, subgoal: dict[str, Any], config: ValidatorConfig) -> bool:
    kind = str(subgoal.get("kind") or "")
    threshold = _subgoal_threshold(env, subgoal, config)
    if kind in {"move_object_to_region", "low_friction_slide_to_region", "strike_object_to_region", "ballistic_object_to_region"}:
        object_record = _subgoal_record(env, subgoal, "object")
        region_record = _subgoal_record(env, subgoal, "region")
        if object_record is None or region_record is None:
            return False
        target_reached = (
            _record_point_inside_record(region_record, object_record)
            or float(object_record.body.position.get_distance(region_record.body.position)) <= threshold
        )
        if kind == "ballistic_object_to_region":
            _update_ballistic_observation(env, object_record, region_record, subgoal, config)
            return target_reached and _ballistic_clearance_requirement_satisfied(
                env,
                object_record,
                region_record,
                subgoal,
                config,
            )
        return target_reached
    if kind == "support_exit_freefall":
        object_record = _subgoal_record(env, subgoal, "object")
        boundary_record = _subgoal_record(env, subgoal, "boundary") or _subgoal_record(
            env,
            subgoal,
            "region",
        )
        if object_record is None or boundary_record is None:
            return False
        return _support_exit_motion_observed(object_record, boundary_record, subgoal)
    if kind == "agent_reach_region":
        agent = env.get_agent_record()
        target_record = _subgoal_record(env, subgoal, "target")
        if agent is None or target_record is None:
            return False
        return float(agent.body.position.get_distance(target_record.body.position)) <= threshold
    if kind == "agent_touch_object":
        agent = env.get_agent_record()
        object_record = _subgoal_record(env, subgoal, "object") or _subgoal_record(
            env, subgoal, "target"
        )
        if agent is None or object_record is None:
            return False
        return float(agent.body.position.get_distance(object_record.body.position)) <= threshold
    if kind == "activate_mechanism":
        if _registered_mechanism_satisfied(env, subgoal):
            return True
        trigger = _subgoal_record(env, subgoal, "trigger")
        if trigger is None:
            return bool(_evaluate_objective(env)[0])
        for record in env._objects.values():
            if record is trigger or record.role == "agent":
                continue
            if record.body.body_type != pymunk.Body.DYNAMIC:
                continue
            trigger_threshold = max(
                threshold,
                _record_radius(record, config.agent_radius)
                + _record_radius(trigger, 24.0) * 0.65,
            )
            if float(record.body.position.get_distance(trigger.body.position)) <= trigger_threshold:
                return True
        return bool(_evaluate_objective(env)[0])
    if kind == "field_force_interaction":
        object_record = _subgoal_record(env, subgoal, "object")
        target_record = _subgoal_record(env, subgoal, "target") or _subgoal_record(
            env,
            subgoal,
            "region",
        )
        field_record = _subgoal_record(env, subgoal, "field")
        if object_record is None:
            return False
        if target_record is not None:
            return float(object_record.body.position.get_distance(target_record.body.position)) <= threshold
        if field_record is not None and _record_point_inside_record(field_record, object_record):
            return True
        return bool(_evaluate_objective(env)[0])
    if kind == "lever_launch":
        if bool(_evaluate_objective(env)[0]):
            return True
        initial = getattr(env, "_validator_lever_initial", None)
        if not isinstance(initial, dict):
            return False
        agent = env.get_agent_record()
        plank_record = _subgoal_record(env, subgoal, "plank")
        target_record = _subgoal_record(env, subgoal, "target")
        if agent is None or plank_record is None:
            return False
        current = _lever_rollout_state(env, agent, plank_record, None, target_record)
        angle_start = _float_or_none(initial.get("plank_angle"))
        angle_now = _float_or_none(current.get("plank_angle"))
        angle_delta = 0.0 if angle_start is None or angle_now is None else abs(angle_now - angle_start)
        lift = _agent_lift_toward_target(initial, current) or 0.0
        min_angle = _float_subgoal_value(subgoal, "min_angle_delta", 0.08)
        min_lift = _float_subgoal_value(subgoal, "min_agent_lift", 45.0)
        target_distance = _float_or_none(current.get("agent_to_target_distance"))
        return angle_delta >= min_angle and (
            lift >= min_lift
            or (target_distance is not None and target_distance <= threshold)
        )
    if kind == "survive_duration":
        if (
            bool(getattr(env, "was_hit", False))
            or bool(getattr(env, "agent_hit", False))
            or float(getattr(env, "hits_taken", 0.0) or 0.0) > 0.0
        ):
            return False
        objective_state, _ = _evaluate_objective(env)
        if bool(objective_state):
            return True
        required_steps = _duration_steps_from_subgoal(subgoal)
        if required_steps > 0.0:
            observed_steps = max(
                float(getattr(env, "survival_steps", 0.0) or 0.0),
                float(getattr(env, "steps_survived", 0.0) or 0.0),
                float(getattr(env, "elapsed_steps", 0.0) or 0.0),
                float(getattr(env, "timer", 0.0) or 0.0),
            )
            if observed_steps >= required_steps:
                return True
        duration = _duration_seconds_from_subgoal(subgoal)
        return duration > 0.0 and float(getattr(env, "_time", 0.0)) >= duration
    return False


def _subgoal_threshold(env: BaseEnv, subgoal: dict[str, Any], config: ValidatorConfig) -> float:
    raw = subgoal.get("threshold") or subgoal.get("distance_threshold")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    kind = str(subgoal.get("kind") or "")
    if kind in {"move_object_to_region", "low_friction_slide_to_region", "strike_object_to_region", "ballistic_object_to_region"}:
        object_record = _subgoal_record(env, subgoal, "object")
        region_record = _subgoal_record(env, subgoal, "region")
        object_radius = _record_radius(object_record, config.agent_radius) if object_record else 12.0
        if kind == "ballistic_object_to_region":
            region_radius = _record_min_radius(region_record, 34.0) if region_record else 34.0
            return max(36.0, object_radius + region_radius * 0.8)
        if kind == "strike_object_to_region":
            region_radius = _record_min_radius(region_record, 24.0) if region_record else 24.0
            return max(24.0, object_radius + region_radius * 0.65)
        region_radius = _record_radius(region_record, 24.0) if region_record else 24.0
        return max(28.0, object_radius + region_radius * 0.65)
    if kind == "support_exit_freefall":
        boundary_record = _subgoal_record(env, subgoal, "boundary") or _subgoal_record(
            env,
            subgoal,
            "region",
        )
        return max(42.0, _record_radius(boundary_record, 30.0) if boundary_record else 42.0)
    if kind == "agent_touch_object":
        agent = env.get_agent_record()
        object_record = _subgoal_record(env, subgoal, "object") or _subgoal_record(
            env, subgoal, "target"
        )
        agent_radius = _record_radius(agent, config.agent_radius) if agent else config.agent_radius
        object_radius = _record_radius(object_record, 12.0) if object_record else 12.0
        # `touch_threshold` is often tuned for large goal sensors. Do not let a
        # generous scoring threshold make physical contact subgoals complete
        # before the agent actually reaches the object.
        contact_slop = min(float(getattr(env, "touch_threshold", 3.0) or 3.0), 3.0)
        return agent_radius + object_radius + max(2.0, contact_slop)
    if kind == "activate_mechanism":
        trigger_record = _subgoal_record(env, subgoal, "trigger")
        if trigger_record is not None:
            return max(
                32.0,
                _record_radius(trigger_record, 24.0) * 0.75 + config.agent_radius,
            )
    if kind == "agent_reach_region":
        target_record = _subgoal_record(env, subgoal, "target")
        if target_record is not None:
            return max(
                32.0,
                _record_radius(target_record, 24.0) * 0.65 + config.agent_radius,
            )
    if kind == "field_force_interaction":
        target_record = _subgoal_record(env, subgoal, "target") or _subgoal_record(
            env,
            subgoal,
            "region",
        )
        if target_record is not None:
            return max(32.0, _record_radius(target_record, 24.0) * 0.75)
    if kind == "lever_launch":
        target_record = _subgoal_record(env, subgoal, "target")
        if target_record is not None:
            return max(38.0, _record_radius(target_record, 28.0) * 0.75 + config.agent_radius)
    return max(28.0, float(getattr(env, "touch_threshold", 20.0)) + config.agent_radius)


def _duration_seconds_from_subgoal(subgoal: dict[str, Any]) -> float:
    for key in ("duration_seconds", "duration_s", "duration", "seconds"):
        value = _float_or_none(subgoal.get(key))
        if value is not None and value > 0.0:
            return value
    return 0.0


def _duration_steps_from_subgoal(subgoal: dict[str, Any]) -> float:
    value = _float_or_none(subgoal.get("duration_steps"))
    if value is not None and value > 0.0:
        return value
    seconds = _duration_seconds_from_subgoal(subgoal)
    if seconds > 0.0:
        return seconds * 60.0
    return 0.0


def _subgoal_rollout_budget(
    config: ValidatorConfig,
    *,
    kind: str,
    completed_subgoals: list[dict[str, Any]],
    base_budget: int,
    subgoal: dict[str, Any] | None = None,
) -> int:
    """Give post-mechanism traversal enough time without relaxing early objectives."""

    if kind == "survive_duration":
        required_steps = _duration_steps_from_subgoal(subgoal or {})
        if required_steps <= 0.0:
            seconds = _duration_seconds_from_subgoal(subgoal or {})
            required_steps = seconds * 60.0
        return max(base_budget, int(math.ceil(required_steps)) + 90)
    if kind != "agent_reach_region" or not completed_subgoals:
        return base_budget
    completed_kinds = {str(subgoal.get("kind") or "") for subgoal in completed_subgoals}
    if "activate_mechanism" in completed_kinds:
        return max(base_budget * 3, config.kinetic_steps * 6)
    if completed_kinds & {"move_object_to_region", "strike_object_to_region", "ballistic_object_to_region", "support_exit_freefall", "lever_launch", "field_force_interaction"}:
        return max(base_budget * 2, config.kinetic_steps * 4)
    return base_budget


def _subgoal_record(env: BaseEnv, subgoal: dict[str, Any], field: str):
    name = subgoal.get(field)
    if name is None and field == "target":
        name = subgoal.get("region") or subgoal.get("object")
    if name is None:
        return None
    name = str(name)
    return env._objects.get(name)


def _subgoal_blocking_object(subgoal: dict[str, Any] | None) -> str | None:
    if not isinstance(subgoal, dict):
        return None
    for field in ("object", "weight", "plank", "target", "region", "boundary", "impact_region", "trigger", "field"):
        value = subgoal.get(field)
        if value:
            return str(value)
    return None


def _subgoal_distances(
    env: BaseEnv,
    subgoals: list[dict[str, Any]],
) -> dict[int, float]:
    distances: dict[int, float] = {}
    for index, subgoal in enumerate(subgoals):
        distance = _subgoal_distance(env, subgoal)
        if distance is not None:
            distances[index] = distance
    return distances


def _subgoal_distance(env: BaseEnv, subgoal: dict[str, Any]) -> float | None:
    kind = str(subgoal.get("kind") or "")
    if kind in {"move_object_to_region", "low_friction_slide_to_region", "strike_object_to_region", "ballistic_object_to_region", "support_exit_freefall"}:
        object_record = _subgoal_record(env, subgoal, "object")
        region_record = _subgoal_record(env, subgoal, "region") or _subgoal_record(env, subgoal, "boundary")
        if object_record is None or region_record is None:
            return None
        return float(object_record.body.position.get_distance(region_record.body.position))
    if kind in {"agent_reach_region", "agent_touch_object"}:
        agent = env.get_agent_record()
        target_record = (
            _subgoal_record(env, subgoal, "target")
            or _subgoal_record(env, subgoal, "object")
        )
        if agent is None or target_record is None:
            return None
        return float(agent.body.position.get_distance(target_record.body.position))
    if kind == "activate_mechanism":
        trigger_record = _subgoal_record(env, subgoal, "trigger")
        if trigger_record is None:
            return None
        distances = [
            float(record.body.position.get_distance(trigger_record.body.position))
            for record in env._objects.values()
            if record is not trigger_record
            and record.role != "agent"
            and record.body.body_type == pymunk.Body.DYNAMIC
        ]
        return min(distances) if distances else None
    if kind == "field_force_interaction":
        object_record = _subgoal_record(env, subgoal, "object")
        target_record = _subgoal_record(env, subgoal, "target") or _subgoal_record(
            env,
            subgoal,
            "region",
        )
        field_record = _subgoal_record(env, subgoal, "field")
        if object_record is None:
            return None
        if target_record is not None:
            return float(object_record.body.position.get_distance(target_record.body.position))
        if field_record is not None:
            return float(object_record.body.position.get_distance(field_record.body.position))
    if kind == "lever_launch":
        agent = env.get_agent_record()
        target_record = _subgoal_record(env, subgoal, "target")
        if agent is not None and target_record is not None:
            return float(agent.body.position.get_distance(target_record.body.position))
        weight_record = _subgoal_record(env, subgoal, "weight") or _subgoal_record(
            env,
            subgoal,
            "object",
        )
        impact_record = _subgoal_record(env, subgoal, "impact_region") or _subgoal_record(
            env,
            subgoal,
            "region",
        )
        if weight_record is not None and impact_record is not None:
            return float(weight_record.body.position.get_distance(impact_record.body.position))
    return None


def _subgoal_progress_observed(
    env: BaseEnv,
    subgoals: list[dict[str, Any]],
    initial_distances: dict[int, float],
    config: ValidatorConfig,
) -> bool:
    for index, start_distance in initial_distances.items():
        if index >= len(subgoals):
            continue
        current = _subgoal_distance(env, subgoals[index])
        if current is None:
            continue
        if start_distance - current >= config.kinetic_displacement_threshold:
            return True
    return False


def _subgoal_progress_report(
    env: BaseEnv,
    subgoals: list[dict[str, Any]],
    initial_distances: dict[int, float],
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for index, subgoal in enumerate(subgoals):
        current = _subgoal_distance(env, subgoal)
        start = initial_distances.get(index)
        report.append(
            {
                "index": index + 1,
                "kind": subgoal.get("kind"),
                "start_distance": start,
                "current_distance": current,
                "distance_reduced": None
                if start is None or current is None
                else round(float(start - current), 3),
            }
        )
    return report


def _update_touched_targets(
    env: BaseEnv,
    agent,
    target_names: list[str],
    target_radius_by_name: dict[str, float],
    agent_radius: float,
    touch_threshold: float,
    touched: set[str],
) -> None:
    for target_name in target_names:
        if target_name in touched:
            continue
        distance = float(agent.body.position.get_distance(env.get_object(target_name).body.position))
        target_radius = target_radius_by_name.get(target_name, 0.0)
        if distance <= agent_radius + target_radius + touch_threshold:
            touched.add(target_name)


def _nearest_untouched_target(
    env: BaseEnv,
    agent,
    target_names: list[str],
    touched: set[str],
):
    candidates = [env.get_object(name) for name in target_names if name not in touched]
    if not candidates:
        return None
    candidates.sort(
        key=lambda record: (
            float(record.body.position.get_distance(agent.body.position)),
            record.name,
        )
    )
    return candidates[0]


def _step_agent_toward(
    env: BaseEnv,
    agent,
    target_position: pymunk.Vec2d,
    config: ValidatorConfig,
) -> None:
    """Apply a simple steering/braking controller and advance physics."""

    offset = target_position - agent.body.position
    if offset.length <= 1.0:
        desired_velocity = pymunk.Vec2d(0.0, 0.0)
    else:
        direction = offset.normalized()
        desired_speed = min(700.0, max(140.0, offset.length * 5.0))
        desired_velocity = direction * desired_speed

    velocity_error = desired_velocity - agent.body.velocity
    if velocity_error.length > 1.0:
        force_scale = max(float(agent.body.mass), 1.0) * 64.0
        max_force = max(float(getattr(env, "agent_strength", 1.0)), 1.0) * 4.0
        force = velocity_error * force_scale
        if force.length > max_force:
            force = force.normalized() * max_force
        agent.body.apply_force_at_world_point(force, agent.body.position)
    env.step(substeps=config.kinetic_substeps)


def _navigation_target_position(
    env: BaseEnv,
    agent,
    target_record,
    config: ValidatorConfig,
) -> pymunk.Vec2d:
    """Return a waypoint toward the target using the deterministic path oracle."""

    diagnostics = _current_reachability_diagnostics(env, target_record, config)
    path_points = diagnostics.get("path_points") if isinstance(diagnostics, dict) else None
    if not isinstance(path_points, list) or len(path_points) < 2:
        return target_record.body.position

    agent_position = agent.body.position
    lookahead = max(config.grid_size * 2.0, config.agent_radius * 3.0)
    fallback = path_points[-1]
    for point in path_points[1:]:
        if not isinstance(point, list | tuple) or len(point) < 2:
            continue
        waypoint = pymunk.Vec2d(float(point[0]), float(point[1]))
        if float(agent_position.get_distance(waypoint)) >= lookahead:
            return waypoint
    return pymunk.Vec2d(float(fallback[0]), float(fallback[1]))


def _multi_target_success_result(
    env_class: str,
    structural_details: dict[str, Any],
    touched: set[str],
    target_names: list[str],
    target_order: list[str],
    step_index: int,
) -> ValidationResult:
    return ValidationResult(
        True,
        "Multi-target touch objective satisfied during kinetic trial.",
        env_class,
        details={
            **structural_details,
            "oracle": "multi_target_touch",
            "objective": "check_objective",
            "objective_step": step_index,
            "targets_touched": sorted(touched),
            "targets_remaining": _targets_remaining(target_names, touched),
            "target_order": target_order,
        },
        contract_valid=True,
        structurally_valid=True,
        objective_valid=True,
        kinetic_progress=True,
        kinetic_solved=True,
    )


def _multi_target_failure_result(
    env_class: str,
    reason: str,
    structural_details: dict[str, Any],
    touched: set[str],
    target_names: list[str],
    target_order: list[str],
    step_index: int,
    agent,
    agent_start_position: pymunk.Vec2d,
    config: ValidatorConfig,
    *,
    objective_valid: bool,
) -> ValidationResult:
    progress = bool(touched) or _kinetic_progress_observed(
        agent,
        agent_start_position,
        None,
        None,
        config,
    )
    return ValidationResult(
        False,
        reason,
        env_class,
        blocking_object=_targets_remaining(target_names, touched)[0]
        if _targets_remaining(target_names, touched)
        else None,
        details={
            **structural_details,
            "oracle": "multi_target_touch",
            "objective": "check_objective",
            "failed_step": step_index,
            "targets_touched": sorted(touched),
            "targets_remaining": _targets_remaining(target_names, touched),
            "target_order": target_order,
            "steps": config.kinetic_steps,
        },
        contract_valid=True,
        structurally_valid=True,
        objective_valid=objective_valid,
        kinetic_progress=progress,
        kinetic_solved=False,
    )


def _targets_remaining(target_names: list[str], touched: set[str]) -> list[str]:
    return [target_name for target_name in target_names if target_name not in touched]


def _objective_metadata_from_env(env: BaseEnv) -> dict[str, Any]:
    try:
        ground_truth = env._json_safe(env.get_ground_truth())
    except Exception:
        return {
            "objective_type": getattr(env, "objective_type", None),
            "objective_targets": list(getattr(env, "objective_targets", []) or []),
        }
    objective = ground_truth.get("objective") or {}
    if not isinstance(objective, dict):
        return {}
    return objective


def _ground_truth_profile_details(ground_truth: dict[str, Any]) -> dict[str, Any]:
    objective = ground_truth.get("objective") or {}
    if not isinstance(objective, dict):
        return {}
    objective_profile = _profile_dict(objective.get("objective_profile"))
    capability_profile = _profile_dict(objective.get("capability_profile"))
    layout_plan = _profile_dict(objective.get("layout_plan"))
    objective_type = objective.get("objective_type")
    return {
        "objective_type": objective_type,
        "objective_targets": objective.get("objective_targets"),
        "objective_profile": objective_profile,
        "capability_profile": capability_profile,
        "layout_plan": layout_plan,
        "semantic_requirements": [
            dict(item)
            for item in objective.get("semantic_requirements", [])
            if isinstance(item, dict)
        ]
        if isinstance(objective.get("semantic_requirements"), list)
        else [],
        "anti_cheat_profile": [
            dict(item)
            for item in objective.get("anti_cheat_profile", [])
            if isinstance(item, dict)
        ]
        if isinstance(objective.get("anti_cheat_profile"), list)
        else [],
        "minimum_acceptance_tier": _minimum_acceptance_tier_from_profile(
            objective_profile,
            objective_type,
        ),
    }


def _profile_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _tier_name(tier: int) -> str:
    return TIER_NAMES.get(int(tier), "unknown")


def _minimum_acceptance_tier(details: dict[str, Any]) -> int:
    tier_policy = _profile_dict(details.get("tier_policy"))
    policy_tier = tier_policy.get(
        "operational_acceptance_tier",
        tier_policy.get("required_tier"),
    )
    if isinstance(policy_tier, int):
        return _clamp_tier(policy_tier)
    try:
        if isinstance(policy_tier, str) and policy_tier.strip():
            return _clamp_tier(int(policy_tier))
    except ValueError:
        pass

    explicit = details.get("minimum_acceptance_tier")
    if isinstance(explicit, int):
        return _clamp_tier(explicit)

    objective_profile = _profile_dict(details.get("objective_profile"))
    return _minimum_acceptance_tier_from_profile(
        objective_profile,
        details.get("objective_type"),
    )


def _objective_tier(details: dict[str, Any]) -> int:
    tier_policy = _profile_dict(details.get("tier_policy"))
    policy_tier = tier_policy.get("objective_tier")
    if isinstance(policy_tier, int):
        return _clamp_tier(policy_tier)
    try:
        if isinstance(policy_tier, str) and policy_tier.strip():
            return _clamp_tier(int(policy_tier))
    except ValueError:
        pass
    objective_profile = _profile_dict(details.get("objective_profile"))
    profile_tier = objective_profile.get("objective_tier")
    if isinstance(profile_tier, int):
        return _clamp_tier(profile_tier)
    try:
        if isinstance(profile_tier, str) and profile_tier.strip():
            return _clamp_tier(int(profile_tier))
    except ValueError:
        pass
    return max(
        _minimum_acceptance_tier(details),
        _ideal_tier_from_profile(objective_profile, details.get("objective_type")),
    )


def _minimum_acceptance_tier_from_profile(
    objective_profile: dict[str, Any],
    objective_type: Any,
) -> int:
    forced_minimum = 5 if _subgoals_require_solved_verification(objective_profile) else 0
    profile_tier = objective_profile.get("minimum_acceptance_tier")
    if isinstance(profile_tier, int):
        return _clamp_tier(max(profile_tier, forced_minimum))
    try:
        if isinstance(profile_tier, str) and profile_tier.strip():
            return _clamp_tier(max(int(profile_tier), forced_minimum))
    except ValueError:
        pass
    return _clamp_tier(max(DEFAULT_MINIMUM_ACCEPTANCE_TIER.get(str(objective_type), 4), forced_minimum))


def _ideal_tier_from_profile(
    objective_profile: dict[str, Any],
    objective_type: Any,
) -> int:
    if _subgoals_require_solved_verification(objective_profile):
        return 5
    objective_type_text = str(objective_type or objective_profile.get("objective_type") or "")
    if objective_type_text in {
        "navigation_goal",
        "single_target_touch",
        "multi_target_touch",
        "push_object",
        "mechanism_activation",
    }:
        return 5
    return max(4, DEFAULT_MINIMUM_ACCEPTANCE_TIER.get(objective_type_text, 4))


def _subgoals_require_solved_verification(objective_profile: dict[str, Any]) -> bool:
    subgoals = _subgoal_list(objective_profile)
    if not subgoals:
        return False
    executable_kinds = {
        "agent_reach_region",
        "agent_touch_object",
        "move_object_to_region",
        "low_friction_slide_to_region",
        "strike_object_to_region",
        "ballistic_object_to_region",
        "support_exit_freefall",
        "classify_objects_to_regions",
        "bounce_to_target",
        "field_force_interaction",
        "lever_launch",
        "activate_mechanism",
        "survive_duration",
        "maintain_balance",
    }
    kinds = {str(subgoal.get("kind") or "") for subgoal in subgoals}
    if not kinds or not kinds <= executable_kinds:
        return False
    return bool(
        kinds
        & {
            "agent_reach_region",
            "agent_touch_object",
            "move_object_to_region",
            "strike_object_to_region",
            "ballistic_object_to_region",
            "support_exit_freefall",
        }
    )


def _clamp_tier(tier: int) -> int:
    return max(0, min(5, int(tier)))


def _evaluate_objective(env: BaseEnv) -> tuple[bool | None, str | None]:
    try:
        return bool(env.check_objective()), None
    except Exception as exc:
        return None, f"Objective check failed: {type(exc).__name__}: {exc}"


def _copy_vec2d(vector) -> pymunk.Vec2d:
    return pymunk.Vec2d(float(vector.x), float(vector.y))


def _vec_list(vector) -> list[float]:
    copied = _copy_vec2d(vector)
    return [round(float(copied.x), 3), round(float(copied.y), 3)]


def _kinetic_progress_observed(
    agent,
    agent_start_position: pymunk.Vec2d | None,
    target,
    target_start_position: pymunk.Vec2d | None,
    config: ValidatorConfig,
) -> bool:
    threshold = config.kinetic_displacement_threshold
    if target is not None and target_start_position is not None:
        if float(target.body.position.get_distance(target_start_position)) >= threshold:
            return True
    if agent is not None and agent_start_position is not None:
        if float(agent.body.position.get_distance(agent_start_position)) >= threshold:
            return True
    return False


def _objective_action_direction(env: BaseEnv, agent, target) -> pymunk.Vec2d:
    if target is not None:
        direction = target.body.position - agent.body.position
        if direction.length > 1.0:
            return direction
    try:
        goal = _resolve_goal(env.get_ground_truth())
        direction = pymunk.Vec2d(goal[0], goal[1]) - agent.body.position
        if direction.length > 1.0:
            return direction
    except Exception:
        pass
    return pymunk.Vec2d(1.0, 0.0)


def validate_ground_truth(
    ground_truth: dict[str, Any],
    *,
    config: ValidatorConfig | None = None,
) -> ValidationResult:
    """Run grid BFS using only BaseEnv ground-truth telemetry."""

    validator_config = _resolve_config_from_ground_truth(ground_truth, config)
    env_class = str(ground_truth.get("env", "UnknownEnv"))
    profile_details = _ground_truth_profile_details(ground_truth)

    try:
        start = _resolve_start(ground_truth)
        goal = _resolve_goal(ground_truth)
    except ValueError as exc:
        return ValidationResult(
            False,
            str(exc),
            env_class,
            details=profile_details,
            contract_valid=True,
            structurally_valid=False,
            objective_valid=False,
            kinetic_progress=False,
            kinetic_solved=False,
        )

    bounds = _resolve_bounds(ground_truth, start, goal, validator_config)
    cols, rows = _grid_shape(bounds, validator_config.grid_size)
    cell_count = cols * rows
    if cell_count > validator_config.max_cells:
        return ValidationResult(
            False,
            f"Grid too large for validation: {cell_count} cells",
            env_class,
            details={**profile_details, "bounds": bounds, "grid_size": validator_config.grid_size},
            contract_valid=True,
            structurally_valid=False,
            objective_valid=False,
            kinetic_progress=False,
            kinetic_solved=False,
        )

    blockers = _filter_support_blockers(
        _extract_blockers(ground_truth, validator_config),
        start,
        goal,
        profile_details.get("layout_plan"),
    )
    start_cell = _point_to_cell(start, bounds, validator_config.grid_size)
    goal_cell = _point_to_cell(goal, bounds, validator_config.grid_size)

    start_blocker = _cell_blocker(start_cell, bounds, validator_config, blockers)
    if start_blocker is not None:
        return ValidationResult(
            False,
            f"Start blocked by {start_blocker.object_name}",
            env_class,
            blocking_object=start_blocker.object_name,
            details={**profile_details, "start": start, "start_cell": start_cell},
            contract_valid=True,
            structurally_valid=False,
            objective_valid=False,
            kinetic_progress=False,
            kinetic_solved=False,
        )

    goal_blocker = _cell_blocker(goal_cell, bounds, validator_config, blockers)
    if goal_blocker is not None:
        return ValidationResult(
            False,
            f"Goal blocked by {goal_blocker.object_name}",
            env_class,
            blocking_object=goal_blocker.object_name,
            details={**profile_details, "goal": goal, "goal_cell": goal_cell},
            contract_valid=True,
            structurally_valid=False,
            objective_valid=False,
            kinetic_progress=False,
            kinetic_solved=False,
        )

    path_cells, visited_count, blocker_counts = _bfs_path(
        start_cell,
        goal_cell,
        bounds,
        cols,
        rows,
        validator_config,
        blockers,
    )

    if path_cells:
        path = tuple(_cell_to_point(cell, bounds, validator_config.grid_size) for cell in path_cells)
        return ValidationResult(
            True,
            "Path found from agent start to goal",
            env_class,
            path=path,
            visited_cells=visited_count,
            details={
                **profile_details,
                "grid_size": validator_config.grid_size,
                "agent_radius": validator_config.agent_radius,
                "path_cells": len(path_cells),
                "physically_plausible": True,
                "agent_actionable": True,
            },
            contract_valid=True,
            structurally_valid=True,
            objective_valid=False,
            kinetic_progress=False,
            kinetic_solved=False,
        )

    blocker_name = _top_blocker_name(blocker_counts, blockers)
    if blocker_name is not None:
        return ValidationResult(
            False,
            f"Path blocked by {blocker_name}",
            env_class,
            visited_cells=visited_count,
            blocking_object=blocker_name,
            details={
                **profile_details,
                "grid_size": validator_config.grid_size,
                "agent_radius": validator_config.agent_radius,
                "visited_cells": visited_count,
            },
            contract_valid=True,
            structurally_valid=False,
            objective_valid=False,
            kinetic_progress=False,
            kinetic_solved=False,
        )

    return ValidationResult(
        False,
        "No path found; search exhausted without a dominant blocking object",
        env_class,
        visited_cells=visited_count,
        details={
            **profile_details,
            "grid_size": validator_config.grid_size,
            "agent_radius": validator_config.agent_radius,
            "visited_cells": visited_count,
        },
        contract_valid=True,
        structurally_valid=False,
        objective_valid=False,
        kinetic_progress=False,
        kinetic_solved=False,
    )


def _grid_reachability_between(
    ground_truth: dict[str, Any],
    start: Point,
    goal: Point,
    config: ValidatorConfig,
) -> dict[str, Any]:
    validator_config = _resolve_config_from_ground_truth(ground_truth, config)
    try:
        bounds = _resolve_bounds(ground_truth, start, goal, validator_config)
        cols, rows = _grid_shape(bounds, validator_config.grid_size)
        cell_count = cols * rows
        if cell_count > validator_config.max_cells:
            return {
                "path_found": False,
                "reason": "grid_too_large",
                "cell_count": cell_count,
                "max_cells": validator_config.max_cells,
            }
        blockers = _filter_support_blockers(
            _extract_blockers(ground_truth, validator_config),
            start,
            goal,
            _profile_dict((ground_truth.get("objective") or {}).get("layout_plan"))
            if isinstance(ground_truth.get("objective"), dict)
            else {},
        )
        start_cell = _point_to_cell(start, bounds, validator_config.grid_size)
        goal_cell = _point_to_cell(goal, bounds, validator_config.grid_size)
        start_blocker = _cell_blocker(start_cell, bounds, validator_config, blockers)
        if start_blocker is not None:
            return {
                "path_found": False,
                "reason": "start_blocked",
                "blocking_object": start_blocker.object_name,
                "start": list(start),
                "goal": list(goal),
                "start_cell": list(start_cell),
                "goal_cell": list(goal_cell),
            }
        goal_blocker = _cell_blocker(goal_cell, bounds, validator_config, blockers)
        if goal_blocker is not None:
            return {
                "path_found": False,
                "reason": "goal_blocked",
                "blocking_object": goal_blocker.object_name,
                "start": list(start),
                "goal": list(goal),
                "start_cell": list(start_cell),
                "goal_cell": list(goal_cell),
            }
        path_cells, visited_count, blocker_counts = _bfs_path(
            start_cell,
            goal_cell,
            bounds,
            cols,
            rows,
            validator_config,
            blockers,
        )
    except Exception as exc:
        return {
            "path_found": False,
            "reason": f"path_diagnostic_error:{type(exc).__name__}: {exc}",
        }
    if path_cells:
        path_points = [
            list(_cell_to_point(cell, bounds, validator_config.grid_size))
            for cell in _thin_path_cells(path_cells)
        ]
        return {
            "path_found": True,
            "path_cells": len(path_cells),
            "path_points": path_points,
            "visited_cells": visited_count,
            "grid_size": validator_config.grid_size,
            "start": list(start),
            "goal": list(goal),
        }
    return {
        "path_found": False,
        "reason": "path_blocked",
        "blocking_object": _top_blocker_name(blocker_counts, blockers),
        "visited_cells": visited_count,
        "grid_size": validator_config.grid_size,
        "start": list(start),
        "goal": list(goal),
    }


def _thin_path_cells(path_cells: tuple[GridCell, ...], *, max_points: int = 80) -> tuple[GridCell, ...]:
    """Return a compact path sample suitable for waypoint following/diagnostics."""

    if len(path_cells) <= max_points:
        return path_cells
    step = max(1, math.ceil(len(path_cells) / max_points))
    thinned = list(path_cells[::step])
    if thinned[-1] != path_cells[-1]:
        thinned.append(path_cells[-1])
    return tuple(thinned)


def load_env_class(env_path: str | Path, *, class_name: str | None = None) -> type[BaseEnv]:
    """Dynamically import a generated environment and return its BaseEnv class."""

    path = _resolve_env_path(env_path)
    module = _load_module_from_path(path)
    resolved_name = class_name or getattr(module, "GENERATED_ENV_CLASS", None)
    if resolved_name:
        env_class = getattr(module, resolved_name, None)
        if env_class is None:
            raise ValueError(f"{path} declares missing class {resolved_name!r}")
        if not isinstance(env_class, type) or not issubclass(env_class, BaseEnv):
            raise TypeError(f"{resolved_name!r} is not a BaseEnv subclass")
        return env_class

    candidates = [
        item
        for item in module.__dict__.values()
        if isinstance(item, type) and issubclass(item, BaseEnv) and item is not BaseEnv
    ]
    if len(candidates) != 1:
        raise ValueError(f"{path} must define exactly one BaseEnv subclass")
    return candidates[0]


def _resolve_config_from_ground_truth(
    ground_truth: dict[str, Any],
    config: ValidatorConfig | None,
) -> ValidatorConfig:
    hint = ground_truth.get("solvability_check") or {}
    base = config or ValidatorConfig()
    grid_size = float(hint.get("grid_size") or base.grid_size)
    agent_radius = hint.get("agent_radius")
    if agent_radius is None:
        agent_radius = _infer_agent_radius(ground_truth) or base.agent_radius
    return ValidatorConfig(
        grid_size=grid_size,
        agent_radius=float(agent_radius),
        simulation_steps=base.simulation_steps,
        substeps=base.substeps,
        include_dynamic_blockers=base.include_dynamic_blockers,
        max_cells=base.max_cells,
        kinetic_validation=base.kinetic_validation,
        kinetic_steps=base.kinetic_steps,
        kinetic_displacement_threshold=base.kinetic_displacement_threshold,
        kinetic_substeps=base.kinetic_substeps,
        tier_policy=dict(base.tier_policy),
    )


def _resolve_start(ground_truth: dict[str, Any]) -> Point:
    hint = ground_truth.get("solvability_check") or {}
    if hint.get("start") is not None:
        return _as_point(hint["start"])
    agent = _find_object_by_role(ground_truth, "agent")
    if agent is not None:
        return _as_point(agent["body"]["position"])
    raise ValueError("Missing start: provide solvability_check.start or role='agent'")


def _resolve_goal(ground_truth: dict[str, Any]) -> Point:
    hint = ground_truth.get("solvability_check") or {}
    if hint.get("goal") is not None:
        return _as_point(hint["goal"])
    goal = _find_object_by_role(ground_truth, "goal")
    if goal is not None:
        return _as_point(goal["body"]["position"])
    raise ValueError("Missing goal: provide solvability_check.goal or role='goal'")


def _find_object_by_role(ground_truth: dict[str, Any], role: str) -> dict[str, Any] | None:
    for data in ground_truth.get("objects", {}).values():
        if data.get("role") == role:
            return data
    return None


def _infer_agent_radius(ground_truth: dict[str, Any]) -> float | None:
    agent = _find_object_by_role(ground_truth, "agent")
    if agent is None:
        return None
    for shape in agent.get("shapes", []):
        if shape.get("type") == "Circle" and shape.get("radius") is not None:
            return float(shape["radius"])
    return None


def _extract_blockers(
    ground_truth: dict[str, Any],
    config: ValidatorConfig,
) -> tuple[Blocker, ...]:
    blockers: list[Blocker] = []
    for object_name, data in ground_truth.get("objects", {}).items():
        role = data.get("role")
        body_type = data.get("body", {}).get("type")
        if role in PASS_THROUGH_ROLES:
            continue
        if role not in SOLID_ROLES and body_type != "static":
            continue
        if body_type == "dynamic" and not config.include_dynamic_blockers:
            continue

        for shape in data.get("shapes", []):
            if bool(shape.get("sensor")):
                continue
            bb = shape.get("bb") or {}
            if not _has_valid_bb(bb):
                continue
            blockers.append(
                Blocker(
                    object_name=object_name,
                    role=role,
                    shape_type=str(shape.get("type", "Unknown")),
                    bb={key: float(bb[key]) for key in ("left", "right", "bottom", "top")},
                    data=shape,
                )
            )
    return tuple(blockers)


def _filter_support_blockers(
    blockers: tuple[Blocker, ...],
    start: Point,
    goal: Point,
    layout_plan: dict[str, Any] | None = None,
) -> tuple[Blocker, ...]:
    """Remove intended floor/ramp support from 2D reachability blockers.

    Grid reachability is an oracle over occupied space, not a platformer contact
    solver. For side-view worlds, route support segments are necessary terrain,
    so treating every ramp/floor segment as an obstacle produces false "path
    blocked by support_seg_*" failures. We keep walls/overhangs/blockers, but
    ignore thin/segment support surfaces that are meant to carry the route.
    """

    filtered: list[Blocker] = []
    support_ceiling = min(float(start[1]), float(goal[1])) + 4.0
    route_points = _layout_plan_route_points(layout_plan)
    for blocker in blockers:
        name = blocker.object_name.lower()
        role = str(blocker.role or "").lower()
        support_like = role in {"terrain", "support"} or any(
            token in name
            for token in ("floor", "ground", "support", "platform", "ice", "rink", "lane", "ramp", "stair")
        )
        if support_like and float(blocker.bb["top"]) <= support_ceiling:
            continue
        if support_like and _is_route_support_surface(blocker, route_points):
            continue
        filtered.append(blocker)
    return tuple(filtered)


def _layout_plan_route_points(layout_plan: dict[str, Any] | None) -> list[Point]:
    if not isinstance(layout_plan, dict):
        return []
    raw_points = layout_plan.get("critical_path_points")
    if not isinstance(raw_points, list):
        support_plan = layout_plan.get("support_plan")
        if isinstance(support_plan, dict):
            raw_points = support_plan.get("waypoints")
    points: list[Point] = []
    if isinstance(raw_points, list):
        for point in raw_points:
            try:
                points.append(_as_point(point))
            except Exception:
                continue
    return points


def _is_route_support_surface(blocker: Blocker, route_points: list[Point]) -> bool:
    width = float(blocker.bb["right"] - blocker.bb["left"])
    height = float(blocker.bb["top"] - blocker.bb["bottom"])
    if blocker.shape_type == "Segment":
        try:
            a = _as_point(blocker.data.get("world_a"))
            b = _as_point(blocker.data.get("world_b"))
        except Exception:
            return False
        dx = abs(b[0] - a[0])
        dy = abs(b[1] - a[1])
        if dx < 8.0:
            return False
        slope_degrees = math.degrees(math.atan2(dy, max(dx, 1e-6)))
        if slope_degrees > 48.0:
            return False
        if not route_points:
            return True
        return any(_distance_point_to_segment(point, a, b) <= 90.0 for point in route_points)

    thin_surface = width >= 2.5 * max(height, 1.0) and height <= 44.0
    if not thin_surface:
        return False
    if not route_points:
        return True
    return any(
        blocker.bb["left"] - 80.0 <= point[0] <= blocker.bb["right"] + 80.0
        and blocker.bb["bottom"] - 80.0 <= point[1] <= blocker.bb["top"] + 120.0
        for point in route_points
    )


def _find_kinetic_target(
    env: BaseEnv,
    geometric_result: ValidationResult | None,
):
    candidates = [
        record
        for record in env._objects.values()
        if record.body.body_type == pymunk.Body.DYNAMIC
        and record.role != "agent"
        and _is_mechanically_relevant(record)
    ]
    if not candidates:
        return None

    path = geometric_result.path if geometric_result else ()
    if path:
        candidates.sort(key=lambda record: (_distance_to_path(record.body.position, path), record.name))
    else:
        agent = env.get_agent_record()
        if agent is not None:
            candidates.sort(
                key=lambda record: (
                    float(record.body.position.get_distance(agent.body.position)),
                    record.name,
                )
            )
    return candidates[0]


def _is_mechanically_relevant(record) -> bool:
    text = " ".join(
        [
            record.name,
            record.kind,
            record.role or "",
            json.dumps(record.metadata, sort_keys=True),
        ]
    ).lower()
    keywords = {
        "ball",
        "weight",
        "weighted",
        "heavy",
        "crate",
        "box",
        "key",
        "door",
        "gate",
        "lever",
        "plank",
        "see-saw",
        "seesaw",
        "mechanism",
        "push",
        "move",
        "launch",
    }
    return record.role in {"obstacle", "hazard"} or any(keyword in text for keyword in keywords)


def _distance_to_path(point, path: tuple[Point, ...]) -> float:
    if not path:
        return 0.0
    return min(math.hypot(float(point.x) - x, float(point.y) - y) for x, y in path)


def _resolve_bounds(
    ground_truth: dict[str, Any],
    start: Point,
    goal: Point,
    config: ValidatorConfig,
) -> dict[str, float]:
    env_config = ground_truth.get("config") or {}
    width = float(env_config.get("width") or 1024.0)
    height = float(env_config.get("height") or 768.0)

    left = min(0.0, start[0], goal[0])
    bottom = min(0.0, start[1], goal[1])
    right = max(width, start[0], goal[0])
    top = max(height, start[1], goal[1])

    for data in ground_truth.get("objects", {}).values():
        for shape in data.get("shapes", []):
            bb = shape.get("bb") or {}
            if not _has_valid_bb(bb):
                continue
            left = min(left, float(bb["left"]))
            right = max(right, float(bb["right"]))
            bottom = min(bottom, float(bb["bottom"]))
            top = max(top, float(bb["top"]))

    return {"left": left, "right": right, "bottom": bottom, "top": top}


def _grid_shape(bounds: dict[str, float], grid_size: float) -> tuple[int, int]:
    cols = int(math.ceil((bounds["right"] - bounds["left"]) / grid_size))
    rows = int(math.ceil((bounds["top"] - bounds["bottom"]) / grid_size))
    return max(cols, 1), max(rows, 1)


def _bfs_path(
    start: GridCell,
    goal: GridCell,
    bounds: dict[str, float],
    cols: int,
    rows: int,
    config: ValidatorConfig,
    blockers: tuple[Blocker, ...],
) -> tuple[tuple[GridCell, ...], int, Counter[str]]:
    frontier: deque[GridCell] = deque([start])
    came_from: dict[GridCell, GridCell | None] = {start: None}
    blocker_counts: Counter[str] = Counter()

    while frontier:
        current = frontier.popleft()
        if current == goal:
            return _reconstruct_path(came_from, goal), len(came_from), blocker_counts

        for neighbor in _neighbors(current):
            col, row = neighbor
            if col < 0 or row < 0 or col >= cols or row >= rows:
                continue
            if neighbor in came_from:
                continue
            blocker = _cell_blocker(neighbor, bounds, config, blockers)
            if blocker is not None:
                blocker_counts[blocker.object_name] += 1
                continue
            came_from[neighbor] = current
            frontier.append(neighbor)

    return (), len(came_from), blocker_counts


def _reconstruct_path(
    came_from: dict[GridCell, GridCell | None],
    goal: GridCell,
) -> tuple[GridCell, ...]:
    path: list[GridCell] = []
    current: GridCell | None = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return tuple(path)


def _neighbors(cell: GridCell) -> Iterable[GridCell]:
    col, row = cell
    yield col + 1, row
    yield col - 1, row
    yield col, row + 1
    yield col, row - 1
    yield col + 1, row + 1
    yield col + 1, row - 1
    yield col - 1, row + 1
    yield col - 1, row - 1


def _cell_blocker(
    cell: GridCell,
    bounds: dict[str, float],
    config: ValidatorConfig,
    blockers: tuple[Blocker, ...],
) -> Blocker | None:
    point = _cell_to_point(cell, bounds, config.grid_size)
    for blocker in blockers:
        if _point_hits_blocker(point, blocker, config.agent_radius):
            return blocker
    return None


def _point_hits_blocker(point: Point, blocker: Blocker, radius: float) -> bool:
    if not _point_near_bb(point, blocker.bb, radius):
        return False

    if blocker.shape_type == "Circle":
        center = _as_point(blocker.data.get("world_center") or blocker.data.get("offset"))
        obstacle_radius = float(blocker.data.get("radius") or 0.0)
        return _distance(point, center) <= obstacle_radius + radius

    if blocker.shape_type == "Segment":
        a = _as_point(blocker.data["world_a"])
        b = _as_point(blocker.data["world_b"])
        segment_radius = float(blocker.data.get("radius") or 0.0)
        return _distance_point_to_segment(point, a, b) <= segment_radius + radius

    if blocker.shape_type == "Poly":
        vertices = [_as_point(vertex) for vertex in blocker.data.get("world_vertices", [])]
        if len(vertices) < 3:
            return False
        if _point_in_polygon(point, vertices):
            return True
        return _distance_to_polygon(point, vertices) <= radius

    return (
        blocker.bb["left"] - radius
        <= point[0]
        <= blocker.bb["right"] + radius
        and blocker.bb["bottom"] - radius
        <= point[1]
        <= blocker.bb["top"] + radius
    )


def _point_near_bb(point: Point, bb: dict[str, float], radius: float) -> bool:
    return (
        bb["left"] - radius <= point[0] <= bb["right"] + radius
        and bb["bottom"] - radius <= point[1] <= bb["top"] + radius
    )


def _point_to_cell(point: Point, bounds: dict[str, float], grid_size: float) -> GridCell:
    return (
        int(math.floor((point[0] - bounds["left"]) / grid_size)),
        int(math.floor((point[1] - bounds["bottom"]) / grid_size)),
    )


def _cell_to_point(cell: GridCell, bounds: dict[str, float], grid_size: float) -> Point:
    return (
        bounds["left"] + (cell[0] + 0.5) * grid_size,
        bounds["bottom"] + (cell[1] + 0.5) * grid_size,
    )


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _distance_point_to_segment(point: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    px, py = point
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return _distance(point, a)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    projection = (ax + t * dx, ay + t * dy)
    return _distance(point, projection)


def _point_in_polygon(point: Point, vertices: list[Point]) -> bool:
    x, y = point
    inside = False
    previous = vertices[-1]
    for current in vertices:
        xi, yi = current
        xj, yj = previous
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi
        if intersects:
            inside = not inside
        previous = current
    return inside


def _distance_to_polygon(point: Point, vertices: list[Point]) -> float:
    distances = [
        _distance_point_to_segment(point, vertices[index], vertices[(index + 1) % len(vertices)])
        for index in range(len(vertices))
    ]
    return min(distances)


def _top_blocker_name(
    blocker_counts: Counter[str],
    blockers: tuple[Blocker, ...],
) -> str | None:
    if not blocker_counts:
        return None
    roles_by_name = {blocker.object_name: blocker.role for blocker in blockers}
    priority_names = [
        name
        for name, _ in blocker_counts.most_common()
        if roles_by_name.get(name) in {"obstacle", "hazard"}
    ]
    if priority_names:
        return priority_names[0]
    return blocker_counts.most_common(1)[0][0]


def _resolve_env_path(env_path: str | Path) -> Path:
    path = Path(env_path)
    if path.suffix != ".py":
        path = GENERATED_ENVS_DIR / f"{path}.py"
    if not path.exists():
        raise FileNotFoundError(f"generated environment not found: {path}")
    return path


def _load_module_from_path(path: Path) -> ModuleType:
    resolved = path.resolve()
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    module_name = f"_harness_generated_{path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import generated environment: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _has_valid_bb(bb: dict[str, Any]) -> bool:
    return all(bb.get(key) is not None for key in ("left", "right", "bottom", "top"))


def _as_point(value: Any) -> Point:
    x, y = value
    return float(x), float(y)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Harness Alpha headless validator")
    parser.add_argument("env", help="generated env file path or module stem in generated_envs/")
    parser.add_argument("--class-name", help="explicit BaseEnv subclass name")
    parser.add_argument("--grid-size", type=float, help="override grid size")
    parser.add_argument("--agent-radius", type=float, help="override agent radius")
    parser.add_argument("--simulation-steps", type=int, default=1)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--include-dynamic-blockers", action="store_true")
    parser.add_argument("--no-kinetic", action="store_true", help="skip physical interaction validation")
    parser.add_argument("--json", action="store_true", help="emit full JSON result")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    defaults = ValidatorConfig()
    config = ValidatorConfig(
        grid_size=args.grid_size or defaults.grid_size,
        agent_radius=args.agent_radius or defaults.agent_radius,
        simulation_steps=args.simulation_steps,
        substeps=args.substeps,
        include_dynamic_blockers=args.include_dynamic_blockers,
        kinetic_validation=not args.no_kinetic,
    )
    result = validate_generated_env(args.env, class_name=args.class_name, config=config)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"SOLVABLE: {result.solvable}")
        print(f"ACCEPTED: {result.accepted}")
        print(f"ACHIEVED_TIER: {result.achieved_tier}")
        print(f"TIER_NAME: {result.tier_name}")
        print(f"OBJECTIVE_TIER: {result.objective_tier}")
        print(f"OPERATIONAL_ACCEPTANCE_TIER: {result.operational_acceptance_tier}")
        print(f"MINIMUM_ACCEPTANCE_TIER: {result.minimum_acceptance_tier}")
        print(f"VERIFICATION_GAP: {result.verification_gap}")
        print(f"REASON: {result.reason}")
        print(f"ENV: {result.env_class}")
        print(f"CONTRACT_VALID: {result.contract_valid}")
        print(f"STRUCTURALLY_VALID: {result.structurally_valid}")
        print(f"OBJECTIVE_VALID: {result.objective_valid}")
        print(f"KINETIC_PROGRESS: {result.kinetic_progress}")
        print(f"KINETIC_SOLVED: {result.kinetic_solved}")
        print(f"VISITED_CELLS: {result.visited_cells}")
        if result.blocking_object:
            print(f"BLOCKING_OBJECT: {result.blocking_object}")

    raise SystemExit(0 if result.solvable else 1)


if __name__ == "__main__":
    main()
