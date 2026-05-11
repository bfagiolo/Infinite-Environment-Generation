# General Intuition World Generation Harness

Harness Alpha is a text-to-world factory for verified 2D physics environments. It accepts natural-language commands, generates Pymunk environment code, verifies code-level objectives, repairs failures, exports playable worlds, and presents them through a polished Pygame control room plus a Godot runtime.

The intended evaluator experience is simple: run the dashboard, type or select a prompt, and use the UI from there.

## Fastest Way To Run

```powershell
git clone https://github.com/bfagiolo/Infinite-Environment-Generation.git
cd Infinite-Environment-Generation

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env
```

Open `.env` and add your own OpenAI API key if you want fresh prompt generation:

```env
OPENAI_API_KEY=sk-your-key-here
```

Launch the app:

```powershell
python dashboard.py
```

Once the dashboard opens, the rest is inside the UI:

- Type a prompt or select a saved showcase example.
- Click **Generate** for fresh generation.
- Click **Play in Godot** to control the exported world yourself.
- Click **Watch AI Solve** when the world reached the solved tier.
- Click **Fork 3 Fast Variants** to show quick multiverse-style visual variants.
- Press `F11` or `Alt+Enter` for fullscreen recording mode.

## Recommended First Evaluation Path

Use the **Instant Saved Showcase** examples first. They are prebuilt, load quickly, and show the system working without waiting for live LLM generation.

The saved prompts demonstrate:

- maze navigation
- zero-gravity collection
- seesaw launch physics
- basketball/ballistic scoring
- forest chase dynamics
- water/swim traversal
- spaceship projectile survival
- lava/fireball avoidance

Fresh prompts are supported too, but they call the OpenAI API and may take longer because the harness generates, validates, repairs, and exports a new world.

## What Needs An API Key?

- **Saved showcase examples:** no API key required.
- **Godot playback of saved/generated worlds:** no API key required.
- **Fresh prompt generation:** requires `OPENAI_API_KEY`.

If no API key is set, the saved showcase still works, but live OpenAI-backed generation will not run.

## Godot Runtime

Python/Pymunk is the research harness: generation, physics, validation, repair, and code-level truth live there.

Godot is the presentation layer: it renders exported `world_schema.json` files with richer visuals, avatar animation, replay overlays, variants, and playable controls.

Install Godot 4.x:

```powershell
winget install --id GodotEngine.GodotEngine -e
```

The dashboard launches Godot automatically when available. You can also launch an exported world directly:

```powershell
python godot_bridge.py --schema exports/latest_world/world_schema.json
```

## Optional Visual Assets

The repo can run without downloaded asset packs, but the Godot presentation looks better with local CC0 Kenney packs:

```powershell
python download_assets.py
python asset_resolver.py build
```

Downloaded binary assets live under `assets/library/kenney/` and are intentionally ignored by Git. The semantic asset index is tracked.

## CLI Generation

Fresh OpenAI-backed generation:

```powershell
python harness.py "A tiny maze where the agent must reach a glowing exit." --execution-mode fast
```

Deterministic local smoke backend:

```powershell
python harness.py "A tiny maze where the agent must reach a glowing exit." --backend local
```

## Architecture And Approach

The core idea is verification-first procedural world generation. The LLM is allowed to be creative, but every world must pass structured contracts before it becomes a demo artifact.

### 1. Dashboard Control Room

`dashboard.py` provides the main user interface. It supports typed prompts, saved showcase shortcuts, live generation traces, fast/normal generation modes, Godot launch, AI solve playback, fullscreen recording, quick reset, and fast visual variants.

### 2. Simulation Brief

`simulation_brief.py` turns the raw prompt into a physical interpretation: gravity model, perspective, agent form, core entities, semantic must-happen events, objective type, and validation expectations.

This layer prevents obvious context mistakes, such as treating spaceship shots like soccer shots or treating an agent-fired bullet as an incoming hazard.

### 3. Gameplay Architect

`gameplay_architect.py` adds game-design assumptions: cadence, fairness, readability, safe windows, chaser behavior, recurring hazards, water behavior, projectile timing, and player-facing feel.

This is where prompts become playable worlds rather than static physics diagrams.

### 4. Physics Relation Graph

`physics_relations.py` decomposes tasks into reusable physical relations instead of brittle task templates. Examples include:

- `contact_push`
- `impulse_transfer`
- `projectile_impact_transfer`
- `ballistic_arc_to_region`
- `support_boundary_exit`
- `freefall_after_support_exit`
- `hazard_motion`
- `field_force_transfer`
- `agent_reaches_region`

The validator can then ask relation-level questions: did contact happen, did impulse transfer, did the object enter the region, did the hazard actually threaten the route, did the object fall after leaving support?

### 5. Route-Aware Layout Planner

`layout_planner.py` gives the generator a spatial skeleton before code is written. It protects routes, stages start/goal/object positions, prevents important objects from being sealed behind blockers, and makes maze/platform worlds validator-friendly.

### 6. Environment Spec And API Guardrails

`environment_spec.json` is the source-of-truth contract. It defines required methods, allowed helper APIs, forbidden arguments, object model expectations, mandatory registry variables, objective profiles, capability profiles, visual tags, and validator tiers.

The Architect must generate `BaseEnv` subclasses with:

- `__init__`
- `add_objects`
- `build_world`
- `check_objective`
- `get_ground_truth`

Every objective is code-level, not pixel-guessed.

### 7. BaseEnv / Pymunk Physics Layer

`base_env.py` owns the deterministic Pymunk substrate: bodies, shapes, forces, damping, substepping, constraints, sensors, force zones, recurring hazards, water behavior, support-exit helpers, ballistic helpers, and object registration.

The generated worlds inherit from this stable base rather than inventing raw physics plumbing every time.

### 8. Validator And Probe Library

`validator.py` runs structural checks, semantic checks, relation probes, affordance checks, anti-cheat checks, and headless rollouts.

Validation is tiered:

- lower tiers prove structure and plausible progress
- Tier 4 proves meaningful progress
- Tier 5 proves the code-level objective was solved

Hard/open-ended prompts may be accepted at lower declared tiers, while straightforward finite tasks aim for Tier 5.

### 9. Repair System

`auto_repair.py` handles measured numeric/layout fixes when safe: target placement, sensor sizing, lane alignment, blocked paths, object distance, friction/mass tuning, and route blockers.

When deterministic repair is not enough, the harness gives narrow repair instructions back to the Architect instead of vague failure text.

### 10. Memory, Skills, And Capability Gaps

The harness keeps reusable knowledge in:

- `affordance_blocks/`
- `skills/`
- `policy_memory/`
- `capability_gaps/`
- `.harness_cache/` locally

The goal is not to blindly copy old templates. The goal is to retrieve conceptual priors, recognize related physical relations, and improve future generation/repair behavior.

### 11. Visual Director And Asset Layer

`visual_director.py`, `visual_grammar.py`, `asset_resolver.py`, and `world_exporter.py` keep visuals separate from physics. Worlds can look like lava, forest, space, water, sports, lab, or cave scenes without changing objective truth.

The renderer can add props, backgrounds, particles, avatar styles, trails, countdowns, replay overlays, and variants while leaving validation grounded in Pymunk state.

### 12. Godot Presentation Runtime

`godot_client/` renders exported worlds in Godot 4. It supports playable mode, AI solve playback, deterministic replay, success overlays, replay/quit controls, richer backgrounds, semantic props, and human-like avatar animation.

## Design Principles

- Code-level objectives beat visual guessing.
- Physics relations beat task-name templates.
- Deterministic probes beat vague "looks solvable" claims.
- Repair feedback should be numeric, local, and actionable.
- LLM creativity is useful, but only inside strict contracts.
- Visual richness should never corrupt physics truth.
- Saved demos should prove the system quickly; fresh prompts should show the open-ended harness.

## Inspired By

This project was inspired by several research directions we discussed while building:

- **Eureka: Human-Level Reward Design via Coding Large Language Models** — LLM-generated reward/objective code with iterative feedback.
- **Code as Policies: Language Model Programs for Embodied Control** — grounding language models in constrained API primitives.
- **Voyager: An Open-Ended Embodied Agent with Large Language Models** — reusable skill libraries and improvement through accumulated experience.
- **Word2World** — text-conditioned world generation and the challenge of turning language into interactive environments.
- **POET: Paired Open-Ended Trailblazer** — open-ended environment generation, curriculum growth, and co-evolving challenges/solutions.

## Submission Notes

Commit source code, `saved_demos/`, `godot_client/`, `requirements.txt`, `.env.example`, and this README.

Do **not** commit `.env`, `venv/`, logs, generated one-off environments, or downloaded binary asset packs.
