"""World Architect prompt and code staging utilities for Harness Alpha.

This module does not call an LLM directly. It prepares the strict prompt that
Codex or another provider should answer, then validates and saves the returned
Python environment code under ``generated_envs/``.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import re
import textwrap


GENERATED_ENVS_DIR = Path("generated_envs")

BASE_ENV_CONTRACT = """
Available BaseEnv interface from base_env.py:
- Subclass BaseEnv and implement build_world(self) -> None.
- Do not override reset(), step(), or get_ground_truth().
- Use self.create_dynamic_circle(...) for dynamic circular bodies.
- Use self.create_static_segment(...) for line terrain/walls.
- Use self.create_static_box(...) for rectangular static obstacles/goals.
- If a custom Pymunk shape is unavoidable, add it with self.register_object(...).
- Every relevant body/shape must be registered through one of those helpers.
- Always call self.set_solvability_hint(start=..., goal=..., agent_radius=...).
- Ground truth and success logic must come from Pymunk state, never pixels.
""".strip()

WORLD_ARCHITECT_PROMPT_TEMPLATE = """
You are the World Architect for Harness Alpha, an AI world-models research
harness that generates deterministic 2D Pymunk environments from text.

Your task:
Generate one complete Python module containing one subclass of BaseEnv that
implements the requested world.

Natural language world request:
{world_request}

{base_env_contract}

Hard requirements:
1. Output only Python code inside one fenced ```python code block.
2. Import BaseEnv and EnvConfig from base_env.
3. You may import pymunk, math, and typing utilities if needed.
4. Define exactly one environment class. The class name must be:
   {class_name}
5. The class must inherit directly from BaseEnv.
6. Implement build_world(self) -> None.
7. Generated code must not call self.space.add(...) directly for environment
   objects. Use BaseEnv helper methods or self.register_object(...) so telemetry
   and get_ground_truth() remain complete.
8. Register every physical object with stable, descriptive names.
9. Use role metadata consistently:
   - role="agent" for the controllable or primary body
   - role="goal" for the target
   - role="terrain" for floors, walls, ramps, and boundaries
   - role="obstacle" for blocking geometry
   - role="hazard" for dangerous or failure objects
10. Every object should include useful metadata such as purpose, dimensions,
    prompt phrase, or validator relevance.
11. Include a solvability hint by calling self.set_solvability_hint(...).
    The hint start must match the agent spawn area and the hint goal must match
    the goal center or target region.
12. Keep the world solvable in headless physics. Avoid impossible gaps, sealed
    rooms, unreachable goals, or immediate spawn collisions.
13. Use deterministic values. Do not use unseeded randomness, file IO, network
    calls, image assets, sound assets, pygame, or pixel inspection.
14. Do not include explanatory prose, markdown outside the single code block,
    tests, or command-line behavior.
15. Add these module constants at the bottom:
    GENERATED_ENV_CLASS = {class_name!r}
    SOURCE_PROMPT = {world_request!r}

Quality target:
Make the environment visually interpretable from geometric primitives and
useful for downstream autonomous-agent training. Favor clear coordinates,
simple stable physics, and explicit metadata over clever construction.

Self-check before answering:
- The module imports from base_env.
- The class subclasses BaseEnv.
- build_world creates terrain, an agent, a goal, and any requested objects.
- All objects are registered through BaseEnv helpers or register_object.
- set_solvability_hint is present.
- The code is syntactically valid Python 3.12.
""".strip()

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
CLASS_NAME_RE = re.compile(r"[^0-9a-zA-Z_]+")


@dataclass(frozen=True)
class ArchitectRequest:
    """Normalized request used to render a world-generation prompt."""

    world_request: str
    class_name: str


@dataclass(frozen=True)
class VerificationResult:
    """Static verification result for generated environment code."""

    ok: bool
    errors: tuple[str, ...] = ()
    class_name: str | None = None


def build_architect_request(world_request: str, class_name: str | None = None) -> ArchitectRequest:
    """Create a normalized architect request from natural language input."""

    clean_request = " ".join(world_request.strip().split())
    if not clean_request:
        raise ValueError("world_request cannot be empty")

    resolved_class_name = class_name or _class_name_from_request(clean_request)
    return ArchitectRequest(world_request=clean_request, class_name=resolved_class_name)


def render_prompt(world_request: str, class_name: str | None = None) -> str:
    """Render the strict LLM prompt for a generated BaseEnv subclass."""

    request = build_architect_request(world_request, class_name)
    return WORLD_ARCHITECT_PROMPT_TEMPLATE.format(
        world_request=request.world_request,
        class_name=request.class_name,
        base_env_contract=BASE_ENV_CONTRACT,
    )


def extract_python_code(llm_response: str) -> str:
    """Extract Python code from a fenced LLM response or raw Python text."""

    match = CODE_BLOCK_RE.search(llm_response)
    if match:
        code = match.group(1)
    else:
        code = llm_response
    return textwrap.dedent(code).strip() + "\n"


def verify_generated_code(code: str, *, expected_class_name: str | None = None) -> VerificationResult:
    """Statically verify that generated code follows the BaseEnv contract."""

    errors: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return VerificationResult(ok=False, errors=(f"syntax error: {exc}",))

    class_defs = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    env_classes = [node for node in class_defs if _inherits_from_base_env(node)]

    if len(env_classes) != 1:
        errors.append("generated module must define exactly one BaseEnv subclass")
        class_name = None
    else:
        env_class = env_classes[0]
        class_name = env_class.name
        if expected_class_name and env_class.name != expected_class_name:
            errors.append(
                f"expected class {expected_class_name!r}, found {env_class.name!r}"
            )
        if not _class_has_method(env_class, "build_world"):
            errors.append("BaseEnv subclass must implement build_world(self)")
        for forbidden_method in ("reset", "step", "get_ground_truth"):
            if _class_has_method(env_class, forbidden_method):
                errors.append(f"generated class must not override {forbidden_method}()")

    if not _imports_base_env(tree):
        errors.append("generated module must import BaseEnv from base_env")

    if _has_space_add_call(tree):
        errors.append("generated code must not call self.space.add(...) directly")

    helper_calls = _count_helper_calls(tree)
    if helper_calls == 0:
        errors.append("generated code must use BaseEnv telemetry helpers/register_object")

    if not _has_method_call(tree, "set_solvability_hint"):
        errors.append("generated code must call self.set_solvability_hint(...)")

    if not _has_constant_assignment(tree, "GENERATED_ENV_CLASS"):
        errors.append("generated code must define GENERATED_ENV_CLASS")

    if not _has_constant_assignment(tree, "SOURCE_PROMPT"):
        errors.append("generated code must define SOURCE_PROMPT")

    if not _has_keyword_string_value(tree, "role", "agent"):
        errors.append('generated code must register an object with role="agent"')

    if not _has_keyword_string_value(tree, "role", "goal"):
        errors.append('generated code must register an object with role="goal"')

    if _imports_forbidden_modules(tree):
        errors.append("generated code must not import pygame, assets, networking, or IO modules")

    return VerificationResult(ok=not errors, errors=tuple(errors), class_name=class_name)


def save_generated_code(
    world_request: str,
    llm_response: str,
    *,
    class_name: str | None = None,
    output_dir: Path = GENERATED_ENVS_DIR,
) -> Path:
    """Extract, verify, and save generated environment code."""

    request = build_architect_request(world_request, class_name)
    code = extract_python_code(llm_response)
    result = verify_generated_code(code, expected_class_name=request.class_name)
    if not result.ok:
        error_text = "\n".join(f"- {error}" for error in result.errors)
        raise ValueError(f"generated environment failed verification:\n{error_text}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_module_name_from_class(request.class_name)}.py"
    output_path.write_text(code, encoding="utf-8")
    return output_path


def _class_name_from_request(world_request: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", world_request)
    if not words:
        return "GeneratedEnv"
    candidate = "".join(word.capitalize() for word in words[:6]) + "Env"
    if candidate[0].isdigit():
        candidate = f"Generated{candidate}"
    return candidate


def _module_name_from_class(class_name: str) -> str:
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
    snake = CLASS_NAME_RE.sub("_", snake).strip("_")
    return snake or "generated_env"


def _imports_base_env(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "base_env":
            imported_names = {alias.name for alias in node.names}
            if "BaseEnv" in imported_names:
                return True
    return False


def _imports_forbidden_modules(tree: ast.Module) -> bool:
    forbidden = {"pygame", "cv2", "PIL", "requests", "urllib", "socket", "pathlib", "os", "sys"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in forbidden:
                    return True
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in forbidden:
                return True
    return False


def _inherits_from_base_env(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "BaseEnv":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "BaseEnv":
            return True
    return False


def _class_has_method(node: ast.ClassDef, method_name: str) -> bool:
    return any(
        isinstance(item, ast.FunctionDef) and item.name == method_name
        for item in node.body
    )


def _has_method_call(tree: ast.Module, method_name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == method_name:
                return True
    return False


def _has_constant_assignment(tree: ast.Module, constant_name: str) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == constant_name:
                return True
    return False


def _has_keyword_string_value(tree: ast.Module, keyword_name: str, value: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != keyword_name:
                continue
            if isinstance(keyword.value, ast.Constant) and keyword.value.value == value:
                return True
    return False


def _has_space_add_call(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add":
            continue
        value = node.func.value
        if (
            isinstance(value, ast.Attribute)
            and value.attr == "space"
            and isinstance(value.value, ast.Name)
            and value.value.id == "self"
        ):
            return True
    return False


def _count_helper_calls(tree: ast.Module) -> int:
    helper_names = {
        "create_dynamic_circle",
        "create_static_segment",
        "create_static_box",
        "register_object",
    }
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in helper_names:
                count += 1
    return count


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Harness Alpha World Architect")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prompt_parser = subparsers.add_parser("prompt", help="render the LLM prompt")
    prompt_parser.add_argument("world_request", help="natural language world request")
    prompt_parser.add_argument("--class-name", help="override generated class name")

    verify_parser = subparsers.add_parser("verify", help="verify generated Python code")
    verify_parser.add_argument("code_file", type=Path, help="path to generated Python code")
    verify_parser.add_argument("--class-name", help="expected environment class name")

    save_parser = subparsers.add_parser("save", help="verify and save an LLM response")
    save_parser.add_argument("world_request", help="natural language world request")
    save_parser.add_argument("response_file", type=Path, help="file containing LLM response")
    save_parser.add_argument("--class-name", help="override generated class name")
    save_parser.add_argument("--output-dir", type=Path, default=GENERATED_ENVS_DIR)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "prompt":
        print(render_prompt(args.world_request, class_name=args.class_name))
        return

    if args.command == "verify":
        code = args.code_file.read_text(encoding="utf-8")
        result = verify_generated_code(code, expected_class_name=args.class_name)
        if not result.ok:
            parser.exit(1, "\n".join(result.errors) + "\n")
        print(f"verified: {result.class_name}")
        return

    if args.command == "save":
        response = args.response_file.read_text(encoding="utf-8")
        output_path = save_generated_code(
            args.world_request,
            response,
            class_name=args.class_name,
            output_dir=args.output_dir,
        )
        print(output_path)
        return

    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
