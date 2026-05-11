# Harness Alpha Godot Runtime

Godot 4.x visual runtime for Harness Alpha exported worlds.

The Python harness remains the source of truth. It generates, validates, repairs,
and exports `exports/latest_world/world_schema.json`. This Godot client reads
that schema and renders a richer 2D playable preview.

## Run

From the repo root:

```powershell
godot --path godot_client -- --world-schema="$PWD\exports\latest_world\world_schema.json"
```

If your executable is named `godot4`, use that in place of `godot`.

Or use the Python launcher, which can find winget-installed Godot:

```powershell
python godot_bridge.py
```

Headless smoke test:

```powershell
python godot_bridge.py --headless --quit-after 2
```

Or set:

```powershell
$env:HARNESS_WORLD_SCHEMA="$PWD\exports\latest_world\world_schema.json"
godot --path godot_client
```

If no argument is provided, the project tries to load:

```text
../exports/latest_world/world_schema.json
```

relative to `godot_client/`.

## Controls

- `WASD` / Arrow keys: move the rendered agent preview.
- `R`: reset the current exported world.

## Current Phase

Phase 6 MVP:

- loads `world_schema.json`
- reconstructs exported objects as Godot runtime bodies:
  `CharacterBody2D`, `RigidBody2D`, `StaticBody2D`, and `Area2D`
- renders circles, polygons, and segments
- renders simple human, arcade disc, ship, and robot agent avatars
- gives the human avatar explicit idle/run/jump/fall/float/push/kick/throw poses
- infers the current action from input, motion, and nearby objects
- syncs rendered objects to live Godot body positions
- applies push/kick/throw forces to dynamic Godot bodies for playable demo feel
- displays live runtime objective feedback, goal touches, hazard proximity, and reset support
- interprets the renderer-only `visual_program` DSL exported by the Visual Director:
  background particles, ribbons, contours, procedural noise, portals, flame
  effects, field rings, trails, and material accents
- loads renderer-only `semantic_assets` from local Kenney CC0 packs for prompt
  props such as spaceships, meteors, trees, grass, fire particles, and sports
  equipment
- has a Python launch bridge via `python godot_bridge.py`
- supports headless smoke testing with installed Godot 4.x
- applies visual mood/background hints from `visual_brief`
- adds procedural backdrops for space, lava/fire, retro arcade, maze, and field/lab worlds
- applies object accents for goals, hazards, balls, walls, twinkles, and edge glows
- displays verification tier status

It does not replace Python/Pymunk validation.
