"""Launch the Godot runtime against the latest Harness Alpha world export."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_DIR = Path("godot_client")
DEFAULT_SCHEMA = Path("exports") / "latest_world" / "world_schema.json"


def find_godot_executable() -> Path | None:
    """Return a usable Godot executable path, including winget installs."""

    for name in ("godot", "godot4", "godot_console"):
        found = shutil.which(name)
        if found:
            return Path(found)

    local_app_data = Path.home() / "AppData" / "Local"
    winget_root = local_app_data / "Microsoft" / "WinGet" / "Packages"
    if winget_root.exists():
        candidates = sorted(
            winget_root.glob("GodotEngine.GodotEngine_*/*console*.exe"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(
            sorted(
                winget_root.glob("GodotEngine.GodotEngine_*/Godot*.exe"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def build_godot_command(
    *,
    schema_path: Path = DEFAULT_SCHEMA,
    project_dir: Path = PROJECT_DIR,
    headless: bool = False,
    quit_after: float | None = None,
) -> list[str]:
    godot = find_godot_executable()
    if godot is None:
        raise FileNotFoundError(
            "Godot executable not found. Install Godot 4.x or run "
            "`winget install --id GodotEngine.GodotEngine -e`."
        )
    command = [str(godot)]
    if headless:
        command.append("--headless")
    command.extend(["--path", str(project_dir)])
    if quit_after is not None:
        command.extend(["--quit-after", str(quit_after)])
    command.extend(["--", f"--world-schema={schema_path.resolve()}"])
    return command


def launch_godot_runtime(
    *,
    schema_path: Path = DEFAULT_SCHEMA,
    project_dir: Path = PROJECT_DIR,
    headless: bool = False,
    quit_after: float | None = None,
    wait: bool = True,
) -> int:
    if not schema_path.exists():
        raise FileNotFoundError(f"World schema not found: {schema_path}")
    if not project_dir.exists():
        raise FileNotFoundError(f"Godot project not found: {project_dir}")

    command = build_godot_command(
        schema_path=schema_path,
        project_dir=project_dir,
        headless=headless,
        quit_after=quit_after,
    )
    if wait:
        return subprocess.run(command, check=False).returncode
    subprocess.Popen(command)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch Harness Alpha's Godot runtime.")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--quit-after", type=float)
    parser.add_argument("--no-wait", action="store_true", help="start Godot and return immediately")
    parser.add_argument("--print-command", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    command = build_godot_command(
        schema_path=args.schema,
        project_dir=args.project_dir,
        headless=args.headless,
        quit_after=args.quit_after,
    )
    if args.print_command:
        print(" ".join(f'"{part}"' if " " in part else part for part in command))
        return
    raise SystemExit(
        launch_godot_runtime(
            schema_path=args.schema,
            project_dir=args.project_dir,
            headless=args.headless,
            quit_after=args.quit_after,
            wait=not args.no_wait,
        )
    )


if __name__ == "__main__":
    main()
