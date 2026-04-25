"""Headless solvability oracle for Harness Alpha generated environments."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass, field
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Iterable

from base_env import BaseEnv


GENERATED_ENVS_DIR = Path("generated_envs")
SOLID_ROLES = {"terrain", "obstacle", "hazard"}
PASS_THROUGH_ROLES = {"agent", "goal"}


Point = tuple[float, float]
GridCell = tuple[int, int]


@dataclass(frozen=True)
class ValidatorConfig:
    """Configuration for headless rollout and grid search."""

    grid_size: float = 24.0
    agent_radius: float = 12.0
    simulation_steps: int = 1
    substeps: int = 1
    include_dynamic_blockers: bool = False
    max_cells: int = 250_000


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "solvable": self.solvable,
            "reason": self.reason,
            "env_class": self.env_class,
            "path": [list(point) for point in self.path],
            "visited_cells": self.visited_cells,
            "blocking_object": self.blocking_object,
            "details": self.details,
        }


def validate_generated_env(
    env_path: str | Path,
    *,
    class_name: str | None = None,
    config: ValidatorConfig | None = None,
) -> ValidationResult:
    """Load a generated environment, run headless physics, and check reachability."""

    validator_config = config or ValidatorConfig()
    env_class = load_env_class(env_path, class_name=class_name)
    env = env_class()

    for _ in range(validator_config.simulation_steps):
        env.step(substeps=validator_config.substeps)

    ground_truth = env.get_ground_truth()
    return validate_ground_truth(ground_truth, config=validator_config)


def validate_ground_truth(
    ground_truth: dict[str, Any],
    *,
    config: ValidatorConfig | None = None,
) -> ValidationResult:
    """Run grid BFS using only BaseEnv ground-truth telemetry."""

    validator_config = _resolve_config_from_ground_truth(ground_truth, config)
    env_class = str(ground_truth.get("env", "UnknownEnv"))

    try:
        start = _resolve_start(ground_truth)
        goal = _resolve_goal(ground_truth)
    except ValueError as exc:
        return ValidationResult(False, str(exc), env_class)

    bounds = _resolve_bounds(ground_truth, start, goal, validator_config)
    cols, rows = _grid_shape(bounds, validator_config.grid_size)
    cell_count = cols * rows
    if cell_count > validator_config.max_cells:
        return ValidationResult(
            False,
            f"Grid too large for validation: {cell_count} cells",
            env_class,
            details={"bounds": bounds, "grid_size": validator_config.grid_size},
        )

    blockers = _extract_blockers(ground_truth, validator_config)
    start_cell = _point_to_cell(start, bounds, validator_config.grid_size)
    goal_cell = _point_to_cell(goal, bounds, validator_config.grid_size)

    start_blocker = _cell_blocker(start_cell, bounds, validator_config, blockers)
    if start_blocker is not None:
        return ValidationResult(
            False,
            f"Start blocked by {start_blocker.object_name}",
            env_class,
            blocking_object=start_blocker.object_name,
            details={"start": start, "start_cell": start_cell},
        )

    goal_blocker = _cell_blocker(goal_cell, bounds, validator_config, blockers)
    if goal_blocker is not None:
        return ValidationResult(
            False,
            f"Goal blocked by {goal_blocker.object_name}",
            env_class,
            blocking_object=goal_blocker.object_name,
            details={"goal": goal, "goal_cell": goal_cell},
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
                "grid_size": validator_config.grid_size,
                "agent_radius": validator_config.agent_radius,
                "path_cells": len(path_cells),
            },
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
                "grid_size": validator_config.grid_size,
                "agent_radius": validator_config.agent_radius,
                "visited_cells": visited_count,
            },
        )

    return ValidationResult(
        False,
        "No path found; search exhausted without a dominant blocking object",
        env_class,
        visited_cells=visited_count,
        details={
            "grid_size": validator_config.grid_size,
            "agent_radius": validator_config.agent_radius,
            "visited_cells": visited_count,
        },
    )


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
    module_name = f"_harness_generated_{path.stem}"
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
    )
    result = validate_generated_env(args.env, class_name=args.class_name, config=config)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"SOLVABLE: {result.solvable}")
        print(f"REASON: {result.reason}")
        print(f"ENV: {result.env_class}")
        print(f"VISITED_CELLS: {result.visited_cells}")
        if result.blocking_object:
            print(f"BLOCKING_OBJECT: {result.blocking_object}")

    raise SystemExit(0 if result.solvable else 1)


if __name__ == "__main__":
    main()
