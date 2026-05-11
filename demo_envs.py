"""Curated saved demo worlds for the Harness Alpha dashboard.

These are intentionally compact, deterministic BaseEnv worlds used for the
instant showcase strip. They do not replace the live Architect pipeline; they
give reviewers a no-wait way to see the range of mechanics and rendering.
"""

from __future__ import annotations

from math import hypot

import pymunk

from base_env import BaseEnv, EnvConfig


class DemoShowcaseEnv(BaseEnv):
    """Small helper base for deterministic saved demos."""

    prompt = "Harness Alpha saved demo"
    objective_type = "navigation_goal"
    objective_description = "Reach the target."
    minimum_tier = 5
    world_context = "top_down_flat"
    movement = "ground_force"
    gravity_label = "zero_g"
    controls = ["apply_force_x", "apply_force_y", "brake"]
    subgoals: list[dict[str, object]] = []

    def __init__(self, config: EnvConfig | None = None):
        super().__init__(config=config or self.default_config(), auto_reset=False)
        self.agent_radius = 15
        self.touch_threshold = 48
        self.agent_strength = 2800
        self.objective_targets: list[str] = []
        self.semantic_requirements: list[dict[str, object]] = []
        self.layout: dict[str, object] = {}
        self.layout_plan = {
            "layout_type": "saved_showcase_demo",
            "route_first": True,
            "source": "curated_demo_library",
        }
        self.gameplay_profile = {
            "source": "curated_demo_library",
            "source_prompt": self.prompt,
            "gameplay_loop": self.objective_type,
            "world_context": {
                "gravity_model": self.world_context,
                "movement_model": self.movement,
                "route_assumption": "stage objectives on reachable routes",
            },
        }
        self.physics_relations = {
            "source": "curated_demo_library",
            "relations": list(self.subgoals),
            "repair_knobs": [{"knob": "layout", "scope": "demo"}],
        }
        self.objective_profile = {
            "objective_type": self.objective_type,
            "objective_description": self.objective_description,
            "success_predicate": "code-level deterministic state satisfies the saved demo objective",
            "targets": self.objective_targets,
            "required_capabilities": ["ground_force", "touch_contact"],
            "progress_metrics": ["distance_to_target"],
            "subgoals": list(self.subgoals),
            "validator_skills": ["saved_demo_oracle"],
            "failure_modes": ["blocked_path", "insufficient_progress"],
            "minimum_acceptance_tier": self.minimum_tier,
        }
        self.capability_profile = {
            "movement": self.movement,
            "interaction": ["touch_contact"],
            "gravity": self.gravity_label,
            "allowed_controls": list(self.controls),
            "forbidden_controls": [
                "teleport",
                "direct_object_move",
                "direct_object_rotation",
                "direct_goal_state_write",
            ],
            "notes": "Saved demo uses deterministic BaseEnv controls.",
        }
        self.reset()

    def default_config(self) -> EnvConfig:
        return EnvConfig(width=960, height=640, gravity=(0, 0), damping=0.97)

    def build_world(self) -> None:
        self.add_objects()

    def reset_objective_state(self) -> None:
        self.targets_touched: set[str] = set()
        self.survival_steps = 0
        self.was_hit = False

    def add_objects(self) -> None:
        raise NotImplementedError

    def check_objective(self) -> bool:
        return False

    def _dist(self, a: str, b: str) -> float:
        pa = self._objects[a].body.position
        pb = self._objects[b].body.position
        return hypot(float(pa.x - pb.x), float(pa.y - pb.y))

    def _near(self, a: str, b: str, threshold: float | None = None) -> bool:
        return self._dist(a, b) <= float(threshold or self.touch_threshold)

    def _bounds(self) -> None:
        self.create_static_segment("floor", a=(40, 40), b=(920, 40), radius=5, role="terrain", metadata={"material": "boundary"})
        self.create_static_segment("ceiling", a=(40, 600), b=(920, 600), radius=5, role="terrain", metadata={"material": "boundary"})
        self.create_static_segment("left_wall", a=(40, 40), b=(40, 600), radius=5, role="terrain", metadata={"material": "boundary"})
        self.create_static_segment("right_wall", a=(920, 40), b=(920, 600), radius=5, role="terrain", metadata={"material": "boundary"})


class DemoTinyMazeEnv(DemoShowcaseEnv):
    prompt = "A tiny neon maze where the agent must reach a glowing exit."
    objective_description = "Navigate a tiny maze to the glowing exit."
    subgoals = [
        {"kind": "agent_reach_region", "target": "maze_waypoint_1", "threshold": 64},
        {"kind": "agent_reach_region", "target": "maze_waypoint_2", "threshold": 64},
        {"kind": "agent_reach_region", "target": "exit_zone", "threshold": 68},
    ]

    def add_objects(self) -> None:
        self.objective_targets.append("exit_zone")
        self._bounds()
        for name, center, size in [
            ("maze_wall_north", (260, 210), (36, 250)),
            ("maze_wall_south", (475, 390), (36, 250)),
            ("maze_wall_gate", (685, 265), (32, 160)),
            ("maze_corner_guard", (690, 505), (250, 34)),
        ]:
            self.create_static_box(name, center=center, size=size, role="obstacle", metadata={"material": "maze_wall", "neon": True})
        self.create_dynamic_circle("agent", pos=(105, 112), radius=self.agent_radius, mass=1.0, role="agent", friction=0.7, metadata={"avatar_hint": "runner"})
        self.create_static_box("maze_waypoint_1", center=(370, 96), size=(74, 74), role="goal", sensor=True, metadata={"waypoint": True, "visual_only": True})
        self.create_static_box("maze_waypoint_2", center=(620, 430), size=(74, 74), role="goal", sensor=True, metadata={"waypoint": True, "visual_only": True})
        self.create_static_box("exit_zone", center=(835, 515), size=(80, 80), role="goal", sensor=True, metadata={"glow": "emerald", "exit": True})
        self.set_solvability_hint(start=(105, 112), goal=(835, 515), agent_radius=self.agent_radius)

    def check_objective(self) -> bool:
        return self._near("agent", "exit_zone", 70)


class DemoZeroGravityCrystalsEnv(DemoShowcaseEnv):
    prompt = "A zero-gravity salvage field where the agent must touch four drifting crystals."
    objective_type = "multi_target_touch"
    objective_description = "Touch four crystals in a zero-gravity salvage field."
    world_context = "zero_g"
    movement = "thrust_2d"
    gravity_label = "zero_g"
    subgoals = [
        {"kind": "agent_touch_object", "object": "crystal_1", "threshold": 62},
        {"kind": "agent_touch_object", "object": "crystal_2", "threshold": 62},
        {"kind": "agent_touch_object", "object": "crystal_3", "threshold": 62},
        {"kind": "agent_touch_object", "object": "crystal_4", "threshold": 62},
    ]

    def add_objects(self) -> None:
        self._bounds()
        self.create_dynamic_circle("agent", pos=(140, 320), radius=16, mass=1.0, role="agent", friction=0.03, metadata={"avatar_hint": "spacesuit"})
        for index, (x, y, vx, vy) in enumerate([(300, 470, 8, -5), (520, 215, -7, 4), (695, 460, -5, -4), (815, 260, 4, 5)], start=1):
            name = f"crystal_{index}"
            self.objective_targets.append(name)
            record = self.create_dynamic_circle(name, pos=(x, y), radius=24, mass=0.25, role="goal", sensor=True, friction=0.0, metadata={"material": "crystal", "glow": "cyan"})
            record.body.velocity = (vx, vy)
        self.set_solvability_hint(start=(140, 320), goal=(815, 260), agent_radius=16)

    def check_objective(self) -> bool:
        for target in list(self.objective_targets):
            if self._near("agent", target, 68):
                self.targets_touched.add(target)
        return len(self.targets_touched) >= len(self.objective_targets)


class DemoSeesawLaunchEnv(DemoShowcaseEnv):
    prompt = "A weighted seesaw puzzle where a heavy ball must launch the agent to a high goal."
    objective_description = "Use a seesaw-like launch scene to reach the high goal."
    world_context = "side_view_gravity"
    gravity_label = "normal"
    subgoals = [
        {
            "kind": "lever_launch",
            "plank": "seesaw_plank",
            "weight": "heavy_ball",
            "impact_region": "launch_pad",
            "target": "high_goal",
            "threshold": 360,
            "min_angle_delta": 0.02,
            "min_agent_lift": -20,
        },
        {"kind": "agent_reach_region", "target": "high_goal", "threshold": 115},
    ]

    def default_config(self) -> EnvConfig:
        return EnvConfig(width=960, height=640, gravity=(0, -650), damping=0.992)

    def add_objects(self) -> None:
        self.objective_targets.append("high_goal")
        self.create_static_box(
            "ground",
            center=(480, 38),
            size=(900, 48),
            role="terrain",
            friction=0.9,
            metadata={"material": "padded_demo_floor", "seesaw_stage": True},
        )
        self.create_static_box(
            "left_load_label",
            center=(370, 104),
            size=(136, 18),
            role="region",
            sensor=True,
            metadata={"visual_only": True, "material": "load_zone_marker", "label": "weight_side"},
        )
        self.create_static_box(
            "right_launch_label",
            center=(640, 104),
            size=(126, 18),
            role="region",
            sensor=True,
            metadata={"visual_only": True, "material": "launch_zone_marker", "label": "launch_side"},
        )
        plank = self.create_dynamic_box(
            "seesaw_plank",
            center=(470, 166),
            size=(390, 28),
            mass=3.8,
            role="mechanism",
            friction=0.42,
            metadata={"material": "seesaw_plank", "seesaw": True, "wood": True, "pivoted_lever": True},
        )
        pivot = self.create_static_box(
            "pivot",
            center=(470, 112),
            size=(54, 92),
            role="support",
            metadata={"material": "seesaw_pivot", "triangle_pivot": True},
        )
        self.register_constraint("seesaw_pivot", type="pivot", body_a=plank.body, body_b=pivot.body, anchor_a=(0, 0), anchor_b=(0, 50))
        self.create_static_box(
            "launch_pad",
            center=(370, 214),
            size=(124, 78),
            role="trigger",
            sensor=True,
            metadata={"launch_pad": True, "material": "impact_pad", "glow": "yellow", "weight_landing_zone": True},
        )
        self.create_dynamic_circle(
            "heavy_ball",
            pos=(372, 264),
            radius=34,
            mass=5.7,
            role="object",
            friction=0.24,
            elasticity=0.02,
            metadata={"heavy": True, "launcher_weight": True, "material": "heavy_iron_ball"},
        )
        self.create_dynamic_circle(
            "agent",
            pos=(642, 235),
            radius=16,
            mass=0.9,
            role="agent",
            friction=0.48,
            metadata={"avatar_hint": "stick", "launch_rider": True, "seesaw_rider": True},
        )
        self.create_static_box(
            "high_goal",
            center=(805, 405),
            size=(142, 142),
            role="goal",
            sensor=True,
            metadata={"glow": "emerald", "high_goal": True, "material": "floating_target_ring"},
        )
        self.set_solvability_hint(start=(625, 245), goal=(805, 445), agent_radius=16)

    def reset_objective_state(self) -> None:
        super().reset_objective_state()
        self.launch_triggered = False

    def after_step(self) -> None:
        super().after_step()
        if self.launch_triggered:
            return
        if not {"agent", "heavy_ball", "launch_pad", "seesaw_plank"}.issubset(self._objects):
            return
        ball = self._objects["heavy_ball"].body
        pad = self._objects["launch_pad"].body
        agent = self._objects["agent"].body
        plank = self._objects["seesaw_plank"].body
        loaded = ball.position.get_distance(pad.position) <= 122 or ball.position.x <= 430
        rotated = abs(float(plank.angle)) >= 0.018
        if loaded and (rotated or self.step_count >= 24):
            agent.apply_impulse_at_world_point(pymunk.Vec2d(390.0, 930.0), agent.position)
            ball.apply_impulse_at_world_point(pymunk.Vec2d(-80.0, -120.0), ball.position)
            plank.angular_velocity -= 1.9
            self.launch_triggered = True

    def check_objective(self) -> bool:
        return self._near("agent", "high_goal", 120)


class DemoBasketballHoopEnv(DemoShowcaseEnv):
    prompt = "A neon basketball court where the agent throws a basketball into a glowing hoop."
    objective_type = "custom_physics"
    objective_description = "Throw the basketball into the hoop sensor."
    world_context = "side_view_gravity"
    gravity_label = "normal"
    subgoals = [{"kind": "ballistic_object_to_region", "object": "basketball", "region": "hoop", "threshold": 64}]

    def default_config(self) -> EnvConfig:
        return EnvConfig(width=960, height=640, gravity=(0, -700), damping=0.992)

    def add_objects(self) -> None:
        self.objective_targets.append("hoop")
        self.create_static_box(
            "neon_court_paint",
            center=(360, 92),
            size=(520, 10),
            role="support",
            sensor=True,
            metadata={"visual_only": True, "court_line": True, "basketball_court": True},
        )
        self.create_static_box(
            "free_throw_lane_visual",
            center=(510, 155),
            size=(140, 92),
            role="region",
            sensor=True,
            metadata={"visual_only": True, "court_paint": True, "basketball_court": True},
        )
        self.create_static_box(
            "scoreboard_visual",
            center=(700, 420),
            size=(120, 44),
            role="region",
            sensor=True,
            metadata={"visual_only": True, "scoreboard": True, "basketball_court": True},
        )
        self.create_ballistic_hoop_challenge(
            "basketball_demo",
            agent_name="agent",
            object_name="basketball",
            target_name="hoop",
            agent_pos=(190, 125),
            object_pos=(250, 125),
            target_center=(535, 260),
            target_size=(120, 105),
            metadata={"court": "basketball", "glow": "orange", "sport": "basketball"},
        )

    def check_objective(self) -> bool:
        return self._near("basketball", "hoop", 78)


class DemoForestBearEscapeEnv(DemoShowcaseEnv):
    prompt = "A forest clearing where the agent must escape to a cabin while a bear chases from the trees."
    objective_description = "Reach the cabin exit while a bear pursues from the forest."
    subgoals = [{"kind": "agent_reach_region", "target": "cabin_exit", "threshold": 68}]

    def add_objects(self) -> None:
        self.objective_targets.append("cabin_exit")
        self._bounds()
        self.create_static_box(
            "mossy_escape_path",
            center=(470, 335),
            size=(770, 86),
            role="region",
            sensor=True,
            metadata={"visual_only": True, "forest_path": True, "material": "moss_path"},
        )
        tree_positions = [
            (205, 170, 38, 82),
            (245, 480, 44, 98),
            (355, 245, 34, 76),
            (440, 500, 46, 104),
            (585, 190, 40, 92),
            (665, 490, 48, 106),
            (760, 280, 40, 92),
            (835, 430, 52, 112),
        ]
        for index, (x, y, width, height) in enumerate(tree_positions, start=1):
            self.create_static_box(
                f"pine_tree_{index}",
                center=(x, y),
                size=(width, height),
                role="obstacle",
                metadata={"material": "evergreen_tree", "forest": True, "blocks_path": True},
            )
        self.create_dynamic_circle(
            "agent",
            pos=(105, 105),
            radius=16,
            mass=1.0,
            role="agent",
            friction=0.72,
            metadata={"avatar_hint": "runner", "runner": True, "forest_escape": True},
        )
        self.create_dynamic_circle(
            "bear_chaser",
            pos=(760, 150),
            radius=25,
            mass=2.1,
            role="hazard",
            friction=0.54,
            metadata={"chaser": True, "animal": "bear", "material": "bear_avatar", "pursuer": True},
        )
        self.create_static_box(
            "cabin_exit",
            center=(840, 520),
            size=(112, 88),
            role="goal",
            sensor=True,
            metadata={"cabin": True, "material": "wood_cabin", "glow": "warm", "safe_house": True},
        )
        self.semantic_requirements.append({"kind": "chasing", "actor": "bear_chaser", "target": "agent"})
        self.set_solvability_hint(start=(105, 105), goal=(840, 520), agent_radius=16)

    def after_step(self) -> None:
        super().after_step()
        if "bear_chaser" in self._objects and "agent" in self._objects:
            bear = self._objects["bear_chaser"].body
            agent = self._objects["agent"].body
            axis = agent.position - bear.position
            if axis.length > 1.0:
                desired = axis.normalized()
                bear.apply_force_at_world_point(desired * 360.0, bear.position)
                bear.velocity = bear.velocity * 0.995

    def check_objective(self) -> bool:
        return self._near("agent", "cabin_exit", 78)


class DemoWaterSwimEnv(DemoShowcaseEnv):
    prompt = "A tropical water channel where the agent jumps in, swims across, and climbs out to reach a beacon."
    objective_description = "Cross the water channel and reach the beacon."
    world_context = "side_view_gravity"
    gravity_label = "normal"
    subgoals = [
        {"kind": "agent_reach_region", "target": "water_pool", "threshold": 95},
        {"kind": "agent_reach_region", "target": "beacon", "threshold": 112},
    ]

    def default_config(self) -> EnvConfig:
        return EnvConfig(width=960, height=640, gravity=(0, -520), damping=0.993)

    def add_objects(self) -> None:
        self.objective_targets.append("beacon")
        self.water_surface_y = 220.0
        self.water_left_x = 318.0
        self.water_right_x = 646.0
        self.create_static_box(
            "left_bank",
            center=(170, 235),
            size=(300, 70),
            role="terrain",
            friction=0.9,
            metadata={"material": "sand_bank", "tropical": True},
        )
        self.create_static_box(
            "right_bank",
            center=(790, 235),
            size=(300, 70),
            role="terrain",
            friction=0.9,
            metadata={"material": "sand_bank", "tropical": True},
        )
        self.create_static_box(
            "left_jump_lip",
            center=(318, 204),
            size=(24, 126),
            role="region",
            sensor=True,
            friction=0.72,
            metadata={"material": "stone_lip", "jump_in_edge": True, "visual_only": True},
        )
        self.create_static_box(
            "right_climb_lip",
            center=(646, 204),
            size=(28, 126),
            role="region",
            sensor=True,
            friction=0.74,
            metadata={"material": "stone_lip", "climb_out_edge": True, "visual_only": True},
        )
        self.create_static_box(
            "water_pool",
            center=(482, 112),
            size=(328, 218),
            role="water",
            sensor=True,
            friction=0.0,
            metadata={
                "water": True,
                "swim_zone": True,
                "material": "tropical_water",
                "surface_y": self.water_surface_y,
                "slows_agent": True,
                "buoyancy": True,
            },
        )
        self.register_force_zone(
            "buoyancy_current",
            center=(482, 122),
            size=(334, 228),
            force=(190, 310),
            affected_names=["agent"],
            metadata={"water": True, "current": True, "swim_current": True},
        )
        self.create_dynamic_circle(
            "agent",
            pos=(112, 300),
            radius=16,
            mass=1.0,
            role="agent",
            friction=0.50,
            metadata={"avatar_hint": "swimmer", "swimmer": True, "can_swim": True},
        )
        self.create_static_box(
            "beacon",
            center=(820, 312),
            size=(124, 118),
            role="goal",
            sensor=True,
            metadata={"glow": "aqua", "beacon": True, "material": "tropical_beacon"},
        )
        self.set_solvability_hint(start=(112, 300), goal=(820, 312), agent_radius=16)

    def reset_objective_state(self) -> None:
        super().reset_objective_state()
        self.entered_water = False
        self.surfaced_from_water = False
        self.climbed_out = False

    def after_step(self) -> None:
        super().after_step()
        if "agent" not in self._objects:
            return
        agent = self._objects["agent"].body
        in_water_x = self.water_left_x <= float(agent.position.x) <= self.water_right_x
        in_water_y = 5.0 <= float(agent.position.y) <= self.water_surface_y + 24.0
        in_water = in_water_x and in_water_y
        if in_water:
            self.entered_water = True
            depth = max(0.0, self.water_surface_y - float(agent.position.y))
            agent.velocity = pymunk.Vec2d(agent.velocity.x * 0.92, agent.velocity.y * 0.74)
            upward = min(520.0, 190.0 + depth * 5.6)
            forward = 230.0 if float(agent.position.x) > 565.0 else 175.0
            agent.apply_force_at_world_point(pymunk.Vec2d(forward, upward), agent.position)
            if abs(float(agent.velocity.x)) < 145.0:
                agent.apply_impulse_at_world_point(pymunk.Vec2d(14.0, 0.0), agent.position)
            if float(agent.position.y) >= self.water_surface_y - 20.0:
                self.surfaced_from_water = True
        elif self.entered_water and float(agent.position.x) > self.water_right_x - 24.0:
            self.climbed_out = True
            if float(agent.velocity.y) < 260.0 and float(agent.position.y) < 292.0:
                agent.apply_impulse_at_world_point(pymunk.Vec2d(50.0, 150.0), agent.position)

    def check_objective(self) -> bool:
        return self._near("agent", "beacon", 126)


class DemoSpaceshipSurvivalEnv(DemoShowcaseEnv):
    prompt = "A space battle where enemy ships fire glowing shots while the agent survives for 8 seconds."
    objective_type = "survival"
    objective_description = "Survive eight seconds while enemy ships fire visible projectiles."
    world_context = "zero_g"
    movement = "thrust_2d"
    gravity_label = "zero_g"
    subgoals = [{"kind": "survive_duration", "duration_seconds": 8.0, "duration_steps": 480}]

    def default_config(self) -> EnvConfig:
        return EnvConfig(width=960, height=640, gravity=(0, 0), damping=0.992)

    def add_objects(self) -> None:
        self._bounds()
        self.create_static_box(
            "distant_capital_ship_shadow",
            center=(735, 330),
            size=(300, 82),
            role="region",
            sensor=True,
            metadata={"material": "capital_ship_shadow", "space_prop": True, "visual_only": True},
        )
        self.create_dynamic_circle(
            "agent",
            pos=(165, 320),
            radius=18,
            mass=1.0,
            role="agent",
            friction=0.02,
            metadata={"avatar_hint": "spaceship", "material": "hero_interceptor", "engine_trail": True},
        )
        ship_positions = [(770, 535), (850, 430), (790, 320), (850, 220), (770, 105)]
        for index, (x, y) in enumerate(ship_positions, start=1):
            self.create_static_box(
                f"enemy_ship_{index}",
                center=(x, y),
                size=(74, 34),
                role="hazard",
                sensor=True,
                metadata={"spaceship": True, "turret": True, "material": "enemy_fighter"},
            )
            shot = self.create_dynamic_circle(
                f"enemy_shot_{index}",
                pos=(x - 76, y + ((index % 2) * 10 - 5)),
                radius=8,
                mass=0.1,
                role="hazard",
                sensor=True,
                friction=0.0,
                metadata={"projectile": True, "laser": True, "material": "laser_bolt", "emitter": f"enemy_ship_{index}"},
            )
            shot.body.velocity = self._shot_velocity(index)
        self.semantic_requirements.append({"kind": "projectile_motion", "role": "hazard", "min_distance": 120})
        self.set_solvability_hint(start=(165, 320), goal=(165, 320), agent_radius=18, notes="Survive for 8 seconds.")

    def after_step(self) -> None:
        super().after_step()
        self.survival_steps += 1
        for name, record in list(self._objects.items()):
            if not name.startswith("enemy_shot_"):
                continue
            if self._dist("agent", name) < 16:
                self.was_hit = True
            if record.body.position.x < 35 or record.body.position.y < 38 or record.body.position.y > self.height - 38:
                source = self._objects.get(name.replace("enemy_shot", "enemy_ship"))
                if source is not None:
                    index = int(name.rsplit("_", 1)[-1])
                    record.body.position = source.body.position + pymunk.Vec2d(-76, ((index % 2) * 10 - 5))
                    record.body.velocity = self._shot_velocity(index)

    def check_objective(self) -> bool:
        return self.survival_steps >= 480 and not self.was_hit

    def _shot_velocity(self, index: int) -> tuple[float, float]:
        lane_drift = [-12.0, 8.0, 0.0, -6.0, 12.0]
        return (-138.0 - index * 12.0, lane_drift[(index - 1) % len(lane_drift)])


class DemoLavaFallingFireEnv(DemoShowcaseEnv):
    prompt = "A lava world where the agent reaches the obsidian exit while staggered fireballs rain from above."
    objective_description = "Reach the exit while fireballs fall in staggered lanes."
    world_context = "side_view_gravity"
    gravity_label = "normal"
    subgoals = [{"kind": "agent_reach_region", "target": "obsidian_exit", "threshold": 80}]

    def default_config(self) -> EnvConfig:
        return EnvConfig(width=960, height=640, gravity=(0, -620), damping=0.992)

    def add_objects(self) -> None:
        self.objective_targets.append("obsidian_exit")
        self.create_static_box("basalt_floor", center=(480, 45), size=(900, 56), role="terrain", friction=0.88, metadata={"material": "basalt", "lava": True})
        self.create_static_box("molten_lava_river", center=(470, 70), size=(370, 42), role="hazard", sensor=True, metadata={"material": "lava_wave", "lava": True, "glow": "orange"})
        self.create_static_box("safe_obsidian_bridge", center=(470, 116), size=(150, 18), role="terrain", friction=0.8, metadata={"material": "obsidian_bridge"})
        self.create_static_box("ceiling_fire_vents", center=(520, 586), size=(650, 24), role="region", sensor=True, metadata={"material": "ember_vent", "lava": True, "visual_only": True})
        self.create_static_box("left_volcanic_spire", center=(62, 226), size=(38, 265), role="region", sensor=True, metadata={"material": "basalt_spire", "lava": True, "visual_only": True})
        self.create_static_box("right_volcanic_spire", center=(908, 226), size=(42, 265), role="region", sensor=True, metadata={"material": "basalt_spire", "lava": True, "visual_only": True})
        self.create_dynamic_circle("agent", pos=(95, 110), radius=12, mass=1.0, role="agent", friction=0.7, metadata={"avatar_hint": "runner", "visual_scale": 0.78})
        self.create_static_box("obsidian_exit", center=(850, 132), size=(86, 95), role="goal", sensor=True, metadata={"material": "obsidian_exit_gate", "glow": "red", "exit": True})
        self.create_recurring_falling_hazards(
            "fire_rain",
            count=5,
            lane_xs=[260, 390, 520, 650, 780],
            spawn_y=585,
            bottom_y=158,
            radius=11,
            speed_y=-245,
            phase_gap_steps=28,
            role="hazard",
            metadata={"material": "falling_fireball", "fireball": True, "lava": True, "projectile": True},
        )
        self.semantic_requirements.append({"kind": "falling_hazards", "role": "hazard", "min_visible_drop": 90})
        self.set_solvability_hint(start=(95, 110), goal=(850, 132), agent_radius=12)

    def after_step(self) -> None:
        super().after_step()
        agent = self._objects.get("agent")
        if agent is None:
            return
        for name, record in self._objects.items():
            if "fireball" not in name:
                continue
            if agent.body.position.get_distance(record.body.position) < 24.0:
                self.was_hit = True

    def check_objective(self) -> bool:
        return self._near("agent", "obsidian_exit", 88) and not self.was_hit
