extends Node2D
class_name WorldLoader

const GRID_MINOR = Color(0.08, 0.42, 0.48, 0.38)
const GRID_MAJOR = Color(0.18, 0.78, 0.86, 0.58)
const TEXT_GOOD = Color(0.72, 1.0, 0.86)
const TEXT_WARN = Color(1.0, 0.45, 0.36)
const ACTION_ORANGE = Color(1.0, 0.42, 0.12, 0.92)
const SENSOR_GREEN = Color(0.15, 1.0, 0.58, 0.82)

var schema: Dictionary = {}
var visual_brief: Dictionary = {}
var visual_program: Dictionary = {}
var semantic_assets: Array = []
var asset_textures: Dictionary = {}
var objects: Array = []
var object_skins: Dictionary = {}
var world_size = Vector2(1024.0, 768.0)
var gravity = Vector2(0.0, -981.0)
var world_scale = 1.0
var world_offset = Vector2.ZERO
var agent_preview_position = Vector2.ZERO
var agent_source_position = Vector2.ZERO
var agent_preview_velocity = Vector2.ZERO
var agent_facing = 1.0
var agent_action = "idle"
var previous_agent_position = Vector2.ZERO
var has_agent = false
var load_error = ""
var elapsed = 0.0
var action_flash_time = 0.0
var action_flash_position = Vector2.ZERO
var action_flash_kind = ""
var object_preview_positions: Dictionary = {}
var object_preview_velocities: Dictionary = {}
var physics_root: Node2D = null
var runtime_bodies: Dictionary = {}
var agent_runtime_body: CharacterBody2D = null
var runtime_physics_ready = false
var source_schema_path = ""
var runtime_goal_contacts: Dictionary = {}
var runtime_objective_satisfied = false
var runtime_hazard_alert = false
var runtime_feedback_message = "Objective active"
var runtime_feedback_time = 0.0
var runtime_survival_elapsed = 0.0
var agent_hit_flash_time = 0.0
var last_goal_distance = INF


func load_world(path: String) -> void:
	source_schema_path = path
	schema = {}
	visual_brief = {}
	visual_program = {}
	semantic_assets = []
	asset_textures = {}
	objects = []
	object_preview_positions = {}
	object_preview_velocities = {}
	runtime_bodies = {}
	agent_runtime_body = null
	runtime_physics_ready = false
	runtime_goal_contacts = {}
	runtime_objective_satisfied = false
	runtime_hazard_alert = false
	runtime_feedback_message = "Objective active"
	runtime_feedback_time = 0.0
	runtime_survival_elapsed = 0.0
	agent_hit_flash_time = 0.0
	last_goal_distance = INF
	elapsed = 0.0
	has_agent = false
	load_error = ""
	_clear_runtime_physics()

	if not FileAccess.file_exists(path):
		load_error = "world_schema.json not found: %s" % path
		queue_redraw()
		return

	var text = FileAccess.get_file_as_string(path)
	var parsed = JSON.parse_string(text)
	if typeof(parsed) != TYPE_DICTIONARY:
		load_error = "world_schema.json is not valid JSON object: %s" % path
		queue_redraw()
		return

	schema = parsed
	visual_brief = schema.get("visual_brief", {})
	semantic_assets = visual_brief.get("semantic_assets", [])
	_load_semantic_asset_textures()
	var raw_recipe = visual_brief.get("recipe", {})
	var recipe: Dictionary = raw_recipe if typeof(raw_recipe) == TYPE_DICTIONARY else {}
	object_skins = recipe.get("object_skins", {})
	var raw_visual_program = visual_brief.get("visual_program", {})
	visual_program = raw_visual_program if typeof(raw_visual_program) == TYPE_DICTIONARY else {}
	if visual_program.is_empty() and recipe.has("visual_program") and typeof(recipe.get("visual_program")) == TYPE_DICTIONARY:
		visual_program = recipe.get("visual_program")
	objects = schema.get("objects", [])

	var world: Dictionary = schema.get("world", {})
	world_size = Vector2(float(world.get("width", 1024.0)), float(world.get("height", 768.0)))
	var gravity_raw: Array = world.get("gravity", [0.0, -981.0])
	if gravity_raw.size() >= 2:
		gravity = Vector2(float(gravity_raw[0]), float(gravity_raw[1]))

	_find_agent()
	_initialize_preview_objects()
	call_deferred("_build_runtime_physics")
	queue_redraw()


func _process(delta: float) -> void:
	elapsed += delta
	if not has_agent:
		queue_redraw()
		return
	if runtime_physics_ready:
		_update_runtime_objective_feedback(delta)
		if action_flash_time > 0.0:
			action_flash_time = max(0.0, action_flash_time - delta)
		if runtime_feedback_time > 0.0:
			runtime_feedback_time = max(0.0, runtime_feedback_time - delta)
		if agent_hit_flash_time > 0.0:
			agent_hit_flash_time = max(0.0, agent_hit_flash_time - delta)
		queue_redraw()
		return
	previous_agent_position = agent_preview_position
	var input = _read_movement_input()
	if input.length() > 0.0:
		agent_preview_position += input * 260.0 * delta
		agent_preview_position.x = clamp(agent_preview_position.x, 0.0, world_size.x)
		agent_preview_position.y = clamp(agent_preview_position.y, 0.0, world_size.y)
	agent_preview_velocity = (agent_preview_position - previous_agent_position) / max(delta, 0.0001)
	if abs(agent_preview_velocity.x) > 8.0:
		agent_facing = 1.0 if agent_preview_velocity.x >= 0.0 else -1.0
	_update_preview_object_motion(delta)
	_update_agent_action(input)
	_apply_agent_action_to_preview_objects(delta)
	_update_runtime_objective_feedback(delta)
	if action_flash_time > 0.0:
		action_flash_time = max(0.0, action_flash_time - delta)
	if runtime_feedback_time > 0.0:
		runtime_feedback_time = max(0.0, runtime_feedback_time - delta)
	if agent_hit_flash_time > 0.0:
		agent_hit_flash_time = max(0.0, agent_hit_flash_time - delta)
	queue_redraw()


func _physics_process(delta: float) -> void:
	if not runtime_physics_ready or agent_runtime_body == null or not has_agent:
		return
	previous_agent_position = agent_preview_position
	var input = _read_movement_input()
	if _agent_uses_freeflight_controls():
		agent_runtime_body.velocity = Vector2(input.x, -input.y) * 330.0
	else:
		var velocity = agent_runtime_body.velocity
		if _agent_is_in_water():
			velocity = _water_runtime_velocity(velocity, input, delta)
		else:
			velocity.x = input.x * 330.0
		if not _agent_is_in_water() and agent_runtime_body.is_on_floor() and input.y > 0.35:
			velocity.y = -430.0
		elif not _agent_is_in_water():
			velocity.y += _screen_gravity_strength() * delta
			velocity.y = min(velocity.y, 780.0)
		if not _agent_is_in_water() and input.y < -0.35 and agent_runtime_body.is_on_floor():
			velocity.x *= 0.78
		agent_runtime_body.velocity = velocity
	agent_runtime_body.move_and_slide()
	_recycle_runtime_projectiles()
	_update_runtime_chasers(delta)
	_update_runtime_lateral_hazards(delta)
	_sync_preview_from_runtime()
	agent_preview_velocity = (agent_preview_position - previous_agent_position) / max(delta, 0.0001)
	if abs(agent_preview_velocity.x) > 8.0:
		agent_facing = 1.0 if agent_preview_velocity.x >= 0.0 else -1.0
	_update_agent_action(input)
	_apply_agent_action_to_preview_objects(delta)
	_update_runtime_objective_feedback(delta)


func _recycle_runtime_projectiles() -> void:
	for name in runtime_bodies.keys():
		var object = _object_by_name(str(name))
		if object == null:
			continue
		if not _object_is_projectile(object):
			continue
		var node = runtime_bodies[name]
		if not (node is RigidBody2D):
			continue
		var rigid = node as RigidBody2D
		var world_pos = _screen_to_world(rigid.global_position)
		var should_recycle = world_pos.x < 35.0 or world_pos.y < 32.0 or world_pos.y > world_size.y - 32.0
		if _object_is_recurring_falling_hazard(object):
			var metadata: Dictionary = object.get("metadata", {})
			should_recycle = world_pos.y <= float(metadata.get("bottom_y", 35.0))
		if not should_recycle:
			continue
		var suffix = _object_numeric_suffix(str(name))
		var spawn_world = _runtime_projectile_spawn_world(object, str(name), suffix)
		rigid.global_position = _world_to_screen(spawn_world)
		rigid.linear_velocity = _projectile_screen_velocity(object, suffix)
		object_preview_positions[str(name)] = spawn_world
		object_preview_velocities[str(name)] = Vector2(rigid.linear_velocity.x, -rigid.linear_velocity.y) / max(world_scale, 0.0001)


func _update_runtime_lateral_hazards(_delta: float) -> void:
	for name in runtime_bodies.keys():
		var object = _object_by_name(str(name))
		if object == null:
			continue
		if not _object_is_recurring_lateral_hazard(object):
			continue
		var node = runtime_bodies[name]
		if not (node is Node2D):
			continue
		var metadata: Dictionary = object.get("metadata", {})
		var spawn_lane = metadata.get("spawn_lane", null)
		var start_world = _object_source_center_world(object)
		if typeof(spawn_lane) == TYPE_ARRAY and spawn_lane.size() >= 2:
			start_world = Vector2(float(spawn_lane[0]), float(spawn_lane[1]))
		var speed_x = float(metadata.get("speed_x", -220.0))
		if abs(speed_x) <= 0.01:
			speed_x = -220.0
		var exit_x = float(metadata.get("exit_x", -80.0 if speed_x < 0.0 else world_size.x + 80.0))
		var phase_index = int(metadata.get("phase_index", _object_numeric_suffix(str(name)) - 1))
		var phase_gap_steps = max(1, int(metadata.get("phase_gap_steps", 45)))
		var time_step = _world_time_step()
		var release_delay = float(max(0, phase_index)) * float(phase_gap_steps) * time_step
		var effective_time = elapsed - release_delay
		var world_pos = start_world
		var world_velocity = Vector2.ZERO
		if effective_time >= 0.0:
			var travel_distance = abs(start_world.x - exit_x)
			var travel_time = max(0.2, travel_distance / abs(speed_x))
			var rest_time = max(0.18, float(phase_gap_steps) * time_step * 0.28)
			var cycle_time = travel_time + rest_time
			var cycle = fmod(effective_time, cycle_time)
			if cycle <= travel_time:
				world_pos.x = start_world.x + speed_x * cycle
				world_velocity = Vector2(speed_x, 0.0)
			else:
				world_pos.x = start_world.x
				world_velocity = Vector2.ZERO
		world_pos.y = start_world.y
		(node as Node2D).global_position = _world_to_screen(world_pos)
		if node is RigidBody2D:
			var rigid = node as RigidBody2D
			rigid.linear_velocity = Vector2(world_velocity.x, -world_velocity.y) * world_scale
			rigid.angular_velocity = 0.0
		object_preview_positions[str(name)] = world_pos
		object_preview_velocities[str(name)] = world_velocity


func _runtime_projectile_spawn_world(object: Dictionary, name: String, suffix: int) -> Vector2:
	if _object_is_recurring_falling_hazard(object):
		var metadata: Dictionary = object.get("metadata", {})
		var lane = metadata.get("spawn_lane", null)
		if typeof(lane) == TYPE_ARRAY and lane.size() >= 2:
			return Vector2(float(lane[0]), float(lane[1]))
	var source_name = name.replace("enemy_shot", "enemy_ship")
	var source = _object_by_name(source_name)
	var spawn_world = _object_source_center_world(source) if source != null else _object_source_center_world(object)
	spawn_world += Vector2(-76.0, float((suffix % 2) * 10 - 5))
	return spawn_world


func _projectile_screen_velocity(object: Dictionary, suffix: int) -> Vector2:
	var body: Dictionary = object.get("body", {})
	var raw_velocity = _vec(body.get("velocity", [0.0, 0.0]))
	var metadata: Dictionary = object.get("metadata", {})
	if _object_is_recurring_falling_hazard(object):
		raw_velocity = Vector2(0.0, float(metadata.get("speed_y", -220.0)))
	if raw_velocity.length() <= 0.01:
		var drift = [-12.0, 8.0, 0.0, -6.0, 12.0]
		raw_velocity = Vector2(-138.0 - float(max(1, suffix)) * 12.0, drift[(max(1, suffix) - 1) % drift.size()])
	return Vector2(raw_velocity.x, -raw_velocity.y) * world_scale


func _runtime_falling_hazard_start(object: Dictionary, fallback: Vector2) -> Vector2:
	var metadata: Dictionary = object.get("metadata", {})
	var lane = metadata.get("spawn_lane", null)
	if typeof(lane) != TYPE_ARRAY or lane.size() < 2:
		return fallback
	var spawn = Vector2(float(lane[0]), float(lane[1]))
	var bottom_y = float(metadata.get("bottom_y", 35.0))
	var phase_index = int(metadata.get("phase_index", 0))
	var lane_span = max(80.0, spawn.y - bottom_y - 30.0)
	var offset = fmod(float(phase_index) * 86.0, lane_span)
	return Vector2(spawn.x, spawn.y - offset)


func _object_numeric_suffix(name: String) -> int:
	var parts = name.split("_")
	if parts.size() <= 0:
		return 1
	return max(1, int(parts[parts.size() - 1]))


func _update_runtime_chasers(delta: float) -> void:
	if agent_runtime_body == null:
		return
	for name in runtime_bodies.keys():
		var object = _object_by_name(str(name))
		if object == null or not _object_is_chaser(object):
			continue
		var node = runtime_bodies[name]
		if not (node is RigidBody2D):
			continue
		var rigid = node as RigidBody2D
		var delta_to_agent = agent_runtime_body.global_position - rigid.global_position
		var distance = delta_to_agent.length()
		if distance <= 2.0:
			continue
		var speed = _chaser_speed(object, distance)
		var desired = delta_to_agent.normalized() * speed
		var blend = clamp(delta * 2.9, 0.0, 1.0)
		rigid.linear_velocity = rigid.linear_velocity.lerp(desired, blend)
		rigid.angular_velocity *= 0.90
		if distance <= _hazard_hit_radius(object) * world_scale + 8.0:
			agent_hit_flash_time = 1.0
			runtime_hazard_alert = true
			if runtime_feedback_time <= 0.0:
				_set_runtime_feedback("CHASE CONTACT - MOVE!", 0.9)


func _chaser_speed(object: Dictionary, distance: float) -> float:
	var metadata: Dictionary = object.get("metadata", {})
	var explicit = metadata.get("chase_speed", null)
	if typeof(explicit) == TYPE_INT or typeof(explicit) == TYPE_FLOAT:
		return max(70.0, float(explicit) * world_scale)
	var base = 170.0 * world_scale
	if distance > 260.0:
		base = 215.0 * world_scale
	elif distance < 80.0:
		base = 128.0 * world_scale
	return base


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_R and source_schema_path != "":
			load_world(source_schema_path)


func _draw() -> void:
	_update_transform()
	_draw_background()
	_draw_semantic_background_assets()
	_draw_grid()
	_draw_world_frame()
	if load_error != "":
		_draw_error(load_error)
		return
	_draw_objects()
	_draw_semantic_foreground_assets()
	_draw_foreground_fx()
	_draw_hud()


func _find_agent() -> void:
	for object in objects:
		if typeof(object) != TYPE_DICTIONARY:
			continue
		if str(object.get("role", "")) != "agent":
			continue
		var body: Dictionary = object.get("body", {})
		agent_source_position = _vec(body.get("position", [0.0, 0.0]))
		agent_preview_position = agent_source_position
		previous_agent_position = agent_source_position
		has_agent = true
		return


func _initialize_preview_objects() -> void:
	for object in objects:
		if typeof(object) != TYPE_DICTIONARY:
			continue
		var name = str(object.get("name", ""))
		if name == "":
			continue
		var body: Dictionary = object.get("body", {})
		var body_type = str(body.get("type", ""))
		var position = _vec(body.get("position", []))
		if position == Vector2.ZERO:
			position = _object_center_world(object)
		object_preview_positions[name] = position
		var velocity = _vec(body.get("velocity", [0.0, 0.0]))
		object_preview_velocities[name] = velocity
		if body_type == "static":
			object_preview_velocities[name] = Vector2.ZERO


func _read_movement_input() -> Vector2:
	var input = Vector2.ZERO
	if Input.is_key_pressed(KEY_A) or Input.is_key_pressed(KEY_LEFT):
		input.x -= 1.0
	if Input.is_key_pressed(KEY_D) or Input.is_key_pressed(KEY_RIGHT):
		input.x += 1.0
	if Input.is_key_pressed(KEY_W) or Input.is_key_pressed(KEY_UP):
		input.y += 1.0
	if Input.is_key_pressed(KEY_S) or Input.is_key_pressed(KEY_DOWN):
		input.y -= 1.0
	if input.length() > 1.0:
		input = input.normalized()
	return input


func _clear_runtime_physics() -> void:
	if physics_root != null and is_instance_valid(physics_root):
		physics_root.queue_free()
	physics_root = null
	runtime_bodies = {}
	agent_runtime_body = null
	runtime_physics_ready = false


func _build_runtime_physics() -> void:
	_clear_runtime_physics()
	if load_error != "" or objects.is_empty():
		return
	_update_transform()
	physics_root = Node2D.new()
	physics_root.name = "RuntimePhysics"
	add_child(physics_root)
	for object in objects:
		if typeof(object) != TYPE_DICTIONARY:
			continue
		var name = str(object.get("name", ""))
		if name == "":
			continue
		var role = str(object.get("role", "")).to_lower()
		var body: Dictionary = object.get("body", {})
		var body_type = str(body.get("type", "")).to_lower()
		var source_world = _object_source_center_world(object)
		if _object_is_recurring_falling_hazard(object):
			source_world = _runtime_falling_hazard_start(object, source_world)
		var source_screen = _world_to_screen(source_world)
		var node: CollisionObject2D = null
		if role == "agent":
			var character = CharacterBody2D.new()
			character.name = "agent_runtime"
			character.position = source_screen
			character.collision_layer = 1
			character.collision_mask = 1
			node = character
			agent_runtime_body = character
		elif _object_uses_area(object):
			var area = Area2D.new()
			area.name = name
			area.position = source_screen
			area.monitoring = true
			area.monitorable = true
			area.collision_layer = 2
			area.collision_mask = 1
			area.body_entered.connect(_on_runtime_area_body_entered.bind(name))
			area.body_exited.connect(_on_runtime_area_body_exited.bind(name))
			node = area
		elif body_type == "dynamic":
			var rigid = RigidBody2D.new()
			rigid.name = name
			rigid.position = source_screen
			rigid.collision_layer = 1
			rigid.collision_mask = 1
			if _object_is_projectile(object):
				rigid.collision_layer = 2
				rigid.collision_mask = 0
			rigid.gravity_scale = 0.0 if (_object_is_floating(object) or _object_is_projectile(object)) else 1.0
			rigid.linear_damp = 0.0 if _object_is_projectile(object) else 0.9
			rigid.angular_damp = 0.0 if _object_is_projectile(object) else 0.9
			var raw_velocity = _vec(body.get("velocity", [0.0, 0.0]))
			rigid.linear_velocity = _projectile_screen_velocity(object, _object_numeric_suffix(name)) if _object_is_projectile(object) else Vector2(raw_velocity.x, -raw_velocity.y) * world_scale
			var raw_mass = body.get("mass", null)
			if typeof(raw_mass) == TYPE_FLOAT or typeof(raw_mass) == TYPE_INT:
				rigid.mass = max(0.05, float(raw_mass))
			node = rigid
		else:
			var static_body = StaticBody2D.new()
			static_body.name = name
			static_body.position = source_screen
			static_body.collision_layer = 1
			static_body.collision_mask = 1
			node = static_body
		if node == null:
			continue
		physics_root.add_child(node)
		runtime_bodies[name] = node
		_add_collision_shapes(node, object, source_screen)
	runtime_physics_ready = agent_runtime_body != null
	if runtime_physics_ready:
		_sync_preview_from_runtime()


func _add_collision_shapes(node: CollisionObject2D, object: Dictionary, body_screen: Vector2) -> void:
	var shapes: Array = object.get("shapes", [])
	for shape in shapes:
		if typeof(shape) != TYPE_DICTIONARY:
			continue
		var collision_shape = _collision_shape_for_exported_shape(shape, body_screen)
		if collision_shape == null:
			continue
		node.add_child(collision_shape)


func _collision_shape_for_exported_shape(shape: Dictionary, body_screen: Vector2) -> CollisionShape2D:
	var shape_type = str(shape.get("type", "")).to_lower()
	var collision = CollisionShape2D.new()
	if shape_type == "circle":
		var circle = CircleShape2D.new()
		circle.radius = max(2.0, float(shape.get("radius", 10.0)) * world_scale)
		collision.shape = circle
		collision.position = _world_to_screen(_vec(shape.get("center", [0.0, 0.0]))) - body_screen
		return collision
	if shape_type == "segment":
		var segment = SegmentShape2D.new()
		segment.a = _world_to_screen(_vec(shape.get("a", [0.0, 0.0]))) - body_screen
		segment.b = _world_to_screen(_vec(shape.get("b", [0.0, 0.0]))) - body_screen
		collision.shape = segment
		return collision
	if shape_type == "polygon":
		var raw_vertices: Array = shape.get("vertices", [])
		if raw_vertices.size() < 3:
			return null
		var polygon = ConvexPolygonShape2D.new()
		var points = PackedVector2Array()
		for raw in raw_vertices:
			points.append(_world_to_screen(_vec(raw)) - body_screen)
		polygon.points = points
		collision.shape = polygon
		return collision
	return null


func _sync_preview_from_runtime() -> void:
	if agent_runtime_body != null:
		agent_preview_position = _screen_to_world(agent_runtime_body.global_position)
	for name in runtime_bodies.keys():
		var node = runtime_bodies[name]
		if node is RigidBody2D:
			object_preview_positions[str(name)] = _screen_to_world(node.global_position)
			var rigid = node as RigidBody2D
			object_preview_velocities[str(name)] = Vector2(rigid.linear_velocity.x, -rigid.linear_velocity.y) / max(world_scale, 0.0001)


func _on_runtime_area_body_entered(body: Node2D, area_name: String) -> void:
	if agent_runtime_body == null or body != agent_runtime_body:
		return
	runtime_goal_contacts[area_name] = true
	var object = _object_by_name(area_name)
	var text = _object_text(object) if object != null else area_name.to_lower()
	if text.find("hazard") >= 0 or text.find("danger") >= 0 or text.find("fire") >= 0 or text.find("lava") >= 0:
		runtime_hazard_alert = true
		_set_runtime_feedback("HAZARD CONTACT: %s" % area_name, 1.8)
	elif _area_counts_as_objective(area_name):
		runtime_objective_satisfied = true
		_set_runtime_feedback("OBJECTIVE TOUCH: %s" % area_name, 2.2)
	else:
		_set_runtime_feedback("CHECKPOINT: %s" % area_name, 0.8)


func _on_runtime_area_body_exited(body: Node2D, area_name: String) -> void:
	if agent_runtime_body == null or body != agent_runtime_body:
		return
	if runtime_goal_contacts.has(area_name):
		runtime_goal_contacts.erase(area_name)


func _update_runtime_objective_feedback(delta: float) -> void:
	runtime_hazard_alert = false
	var nearest_goal_distance = INF
	var touched_named_target = false
	var objective_targets = _objective_targets()
	for object in objects:
		if typeof(object) != TYPE_DICTIONARY:
			continue
		var role = str(object.get("role", "")).to_lower()
		var name = str(object.get("name", "")).to_lower()
		if name == "" or role == "agent":
			continue
		var distance = agent_preview_position.distance_to(_object_center_world(object))
		if _area_counts_as_objective(name):
			nearest_goal_distance = min(nearest_goal_distance, distance)
			if distance <= _objective_touch_radius(object):
				touched_named_target = true
		if role in ["hazard", "danger"] or _shape_is_fire(object):
			if distance <= _hazard_warning_radius(object):
				runtime_hazard_alert = true
			if distance <= _hazard_hit_radius(object):
				agent_hit_flash_time = 1.0
				runtime_hazard_alert = true
				if runtime_feedback_time <= 0.0:
					_set_runtime_feedback("HIT - SHIELDS FLASHING", 0.9)
	if touched_named_target and not runtime_objective_satisfied:
		runtime_objective_satisfied = true
		_set_runtime_feedback("OBJECTIVE COMPLETE IN GODOT RUNTIME", 2.4)
	elif _survival_duration_seconds() > 0.0 and not runtime_objective_satisfied:
		runtime_survival_elapsed += delta
		var remaining = max(0.0, _survival_duration_seconds() - runtime_survival_elapsed)
		if remaining <= 0.0:
			runtime_objective_satisfied = true
			_set_runtime_feedback("SURVIVED - OBJECTIVE COMPLETE", 2.4)
		elif runtime_feedback_time <= 0.0:
			_set_runtime_feedback("SURVIVE: %.1fs" % remaining, 0.35)
	elif nearest_goal_distance < INF and abs(nearest_goal_distance - last_goal_distance) > 12.0 and runtime_feedback_time <= 0.0:
		if nearest_goal_distance < last_goal_distance:
			_set_runtime_feedback("Goal distance %.0f px" % nearest_goal_distance, 0.75)
	last_goal_distance = nearest_goal_distance
	if runtime_hazard_alert and runtime_feedback_time <= 0.0:
		_set_runtime_feedback("Hazard proximity warning", 0.9)


func _set_runtime_feedback(message: String, duration: float) -> void:
	runtime_feedback_message = message
	runtime_feedback_time = duration


func _area_counts_as_objective(area_name: String) -> bool:
	var normalized = area_name.to_lower()
	var targets = _objective_targets()
	if normalized in targets:
		return true
	var object = _object_by_name(area_name)
	if object == null:
		return normalized.find("goal") >= 0 or normalized.find("exit") >= 0
	var text = _object_text(object)
	if text.find("waypoint") >= 0 or text.find("visual_only") >= 0:
		return false
	var role = str(object.get("role", "")).to_lower()
	if role in ["goal", "trigger", "region"]:
		return true
	return normalized.find("goal") >= 0 or normalized.find("exit") >= 0



func _update_transform() -> void:
	var rect = get_viewport_rect()
	var margin = 56.0
	var sx = max(0.1, (rect.size.x - margin * 2.0) / max(world_size.x, 1.0))
	var sy = max(0.1, (rect.size.y - margin * 2.0) / max(world_size.y, 1.0))
	world_scale = min(sx, sy)
	world_offset = Vector2(
		(rect.size.x - world_size.x * world_scale) * 0.5,
		(rect.size.y - world_size.y * world_scale) * 0.5
	)


func _draw_background() -> void:
	var bg = _palette_color("background_color", Color(0.04, 0.07, 0.09))
	draw_rect(get_viewport_rect(), bg, true)
	var mood = str(visual_brief.get("mood", "research_lab"))
	var background = str(visual_brief.get("background", ""))
	if not visual_program.is_empty() and _draw_visual_program_background():
		return
	if mood.find("space") >= 0 or background.find("star") >= 0 or _has_keyword("space") or _has_keyword("asteroid"):
		_draw_starfield()
	elif _has_keyword("lava") or _has_keyword("fire"):
		_draw_lava_wash()
	elif mood.find("retro") >= 0 or background.find("retro") >= 0 or background.find("crt") >= 0:
		_draw_retro_backdrop()
	elif mood.find("maze") >= 0 or _has_keyword("maze"):
		_draw_organic_maze_wash()
	elif mood.find("magnetic") >= 0 or _has_keyword("magnetic") or _has_keyword("field"):
		_draw_field_backdrop()
	else:
		_draw_lab_wash()


func _load_semantic_asset_textures() -> void:
	asset_textures = {}
	if typeof(semantic_assets) != TYPE_ARRAY:
		return
	for asset in semantic_assets:
		if typeof(asset) != TYPE_DICTIONARY:
			continue
		var path = str(asset.get("path", ""))
		if path == "" or asset_textures.has(path):
			continue
		var normalized = path.replace("\\", "/")
		var global_path = ProjectSettings.globalize_path("res://../" + normalized)
		var image = Image.new()
		var err = image.load(global_path)
		if err != OK:
			continue
		asset_textures[path] = ImageTexture.create_from_image(image)


func _draw_semantic_background_assets() -> void:
	if typeof(semantic_assets) != TYPE_ARRAY or semantic_assets.is_empty():
		return
	var rect = get_viewport_rect()
	var drawn = 0
	for index in range(semantic_assets.size()):
		var asset = semantic_assets[index]
		if typeof(asset) != TYPE_DICTIONARY:
			continue
		var role = str(asset.get("role", "background_prop"))
		var placement = str(asset.get("placement", "background"))
		if role == "foreground_prop" and placement == "objective":
			continue
		if _draw_semantic_asset(asset, index, rect, 0.34, true):
			drawn += 1
		if drawn >= 9:
			return


func _draw_semantic_foreground_assets() -> void:
	if typeof(semantic_assets) != TYPE_ARRAY or semantic_assets.is_empty():
		return
	var rect = get_viewport_rect()
	var drawn = 0
	for index in range(semantic_assets.size()):
		var asset = semantic_assets[index]
		if typeof(asset) != TYPE_DICTIONARY:
			continue
		var role = str(asset.get("role", ""))
		var placement = str(asset.get("placement", ""))
		if role != "foreground_prop" and placement != "objective":
			continue
		if _draw_semantic_asset(asset, index, rect, 0.58, false):
			drawn += 1
		if drawn >= 4:
			return


func _draw_semantic_asset(asset: Dictionary, index: int, rect: Rect2, alpha: float, background_pass: bool) -> bool:
	var path = str(asset.get("path", ""))
	if path == "" or not asset_textures.has(path):
		return false
	var texture = asset_textures[path]
	if texture == null:
		return false
	var natural: Vector2 = texture.get_size()
	if natural.x <= 0.0 or natural.y <= 0.0:
		return false
	var semantic = str(asset.get("semantic", "")).to_lower()
	var placement = str(asset.get("placement", "background")).to_lower()
	var max_size = 132.0 if background_pass else 74.0
	if semantic.find("tree") >= 0 or semantic.find("building") >= 0 or semantic.find("castle") >= 0:
		max_size = 180.0
	elif semantic.find("ship") >= 0 or semantic.find("asteroid") >= 0 or semantic.find("meteor") >= 0:
		max_size = 92.0
	elif semantic.find("fire") >= 0 or semantic.find("spark") >= 0:
		max_size = 72.0
	elif not background_pass:
		max_size = 54.0
	var scale = max_size / max(natural.x, natural.y)
	var size = natural * scale
	var pos = _semantic_asset_position(index, rect, placement, size, background_pass)
	var tint = Color(1.0, 1.0, 1.0, alpha)
	if semantic.find("fire") >= 0 or semantic.find("lava") >= 0:
		tint = Color(1.0, 0.62 + 0.16 * sin(elapsed * 6.0 + float(index)), 0.28, min(0.82, alpha + 0.20))
	elif semantic.find("space") >= 0 or semantic.find("ship") >= 0 or semantic.find("asteroid") >= 0:
		tint = Color(0.82, 0.92, 1.0, alpha + 0.10)
	elif semantic.find("tree") >= 0 or semantic.find("grass") >= 0:
		tint = Color(0.78, 1.0, 0.72, alpha + 0.08)
	draw_texture_rect(texture, Rect2(pos, size), false, tint)
	return true


func _semantic_asset_position(index: int, rect: Rect2, placement: String, size: Vector2, background_pass: bool) -> Vector2:
	var seed = float((_program_seed() + index * 271) % 10000)
	var x = fmod(seed * 0.137 + float(index) * 91.0, max(rect.size.x - size.x, 1.0))
	var y = fmod(seed * 0.071 + float(index) * 59.0, max(rect.size.y - size.y, 1.0))
	if placement in ["sky", "horizon"]:
		y = 24.0 + fmod(seed * 0.041 + float(index) * 37.0, max(rect.size.y * 0.32, 1.0))
	elif placement in ["edges", "floor", "platforms"]:
		y = rect.size.y - size.y - 28.0 - fmod(seed * 0.021 + float(index) * 17.0, 76.0)
		x = fmod(seed * 0.119 + float(index) * 121.0, max(rect.size.x - size.x, 1.0))
	elif placement in ["hazard", "background"]:
		y = fmod(seed * 0.089 + float(index) * 73.0 + elapsed * 9.0, max(rect.size.y - size.y, 1.0))
	elif placement == "objective":
		x = rect.size.x - size.x - 42.0 - fmod(seed * 0.013, 60.0)
		y = world_offset.y + world_size.y * world_scale * 0.48 - size.y * 0.5 + fmod(seed * 0.017, 46.0)
	if not background_pass:
		y = clamp(y, world_offset.y + 20.0, world_offset.y + world_size.y * world_scale - size.y - 20.0)
	return Vector2(clamp(x, 8.0, rect.size.x - size.x - 8.0), clamp(y, 8.0, rect.size.y - size.y - 8.0))


func _draw_visual_program_background() -> bool:
	var layers: Array = visual_program.get("background_layers", [])
	if layers.is_empty():
		return false
	for layer in layers:
		if typeof(layer) != TYPE_DICTIONARY:
			continue
		var primitive = str(layer.get("primitive", layer.get("type", "")))
		if primitive == "particle_field" or primitive == "parallax_dots":
			_draw_program_particle_field(layer)
		elif primitive == "ribbon_flow":
			_draw_program_ribbon_flow(layer)
		elif primitive == "contour_lines":
			_draw_program_contour_lines(layer)
		elif primitive == "grid_overlay":
			_draw_program_grid_overlay(layer)
		elif primitive == "facet_field":
			_draw_program_facet_field(layer)
		elif primitive == "texture_noise" or primitive == "silhouette_motes":
			_draw_program_texture_noise(layer)
		elif primitive == "heat_shimmer":
			_draw_program_heat_shimmer(layer)
		elif primitive == "scanline_roll":
			_draw_program_scanline_roll(layer)
		elif primitive == "radial_glow":
			_draw_program_radial_glow(layer)
	return true


func _draw_program_particle_field(layer: Dictionary) -> void:
	var rect = get_viewport_rect()
	var count = int(clamp(float(layer.get("count", 90)), 8.0, 260.0))
	var color = _program_color(layer, _palette_color("primary", Color(0.30, 0.92, 1.0)))
	var motion = str(layer.get("motion", "drift"))
	var speed = float(layer.get("speed", 0.20))
	for index in range(count):
		var base_x = fmod(float(index * 137 + _program_seed()), max(rect.size.x, 1.0))
		var base_y = fmod(float(index * 59 + _program_seed() / 3), max(rect.size.y, 1.0))
		var dx = elapsed * speed * (18.0 + float(index % 7) * 4.0)
		var dy = elapsed * speed * (22.0 + float(index % 5) * 5.0)
		if motion == "updraft":
			base_y = fmod(base_y - dy + rect.size.y, rect.size.y)
			base_x += sin(elapsed * 0.9 + float(index)) * 9.0
		elif motion == "orbit":
			base_x += sin(elapsed * speed + float(index)) * 34.0
			base_y += cos(elapsed * speed * 0.8 + float(index)) * 16.0
		else:
			base_x = fmod(base_x + dx, rect.size.x)
			base_y = fmod(base_y + dy * 0.25, rect.size.y)
		var alpha = 0.12 + 0.24 * (0.5 + 0.5 * sin(elapsed * 2.0 + float(index)))
		draw_circle(Vector2(base_x, base_y), 1.0 + float(index % 3) * 0.45, Color(color.r, color.g, color.b, alpha))


func _draw_program_ribbon_flow(layer: Dictionary) -> void:
	var rect = get_viewport_rect()
	var count = int(clamp(float(layer.get("count", 8)), 2.0, 32.0))
	var speed = float(layer.get("speed", 0.18))
	var color = _program_color(layer, _palette_color("hot", ACTION_ORANGE))
	for index in range(count):
		var x = fmod(float(index * 97 + _program_seed() / 5), max(rect.size.x, 1.0))
		var sway = sin(elapsed * (1.1 + speed) + float(index)) * 24.0
		var alpha = 0.045 + 0.08 * (float(index % 5) / 5.0)
		var start = Vector2(x + sway, 0.0)
		var end = Vector2(x - sway + 45.0, rect.size.y)
		draw_line(start, end, Color(color.r, color.g, color.b, alpha), 2.0 + float(index % 3))


func _draw_program_contour_lines(layer: Dictionary) -> void:
	var rect = get_viewport_rect()
	var count = int(clamp(float(layer.get("count", 10)), 3.0, 28.0))
	var color = _program_color(layer, _palette_color("secondary", Color(0.7, 0.45, 1.0)))
	for index in range(count):
		var y = rect.size.y * (float(index) + 0.5) / float(count)
		var points = PackedVector2Array()
		for step in range(0, int(rect.size.x) + 18, 18):
			var x = float(step)
			var wave = sin(x * 0.018 + elapsed * 1.6 + float(index)) * (9.0 + float(index % 4) * 3.0)
			points.append(Vector2(x, y + wave))
		draw_polyline(points, Color(color.r, color.g, color.b, 0.075), 1.5)


func _draw_program_grid_overlay(layer: Dictionary) -> void:
	var rect = get_viewport_rect()
	var cell = max(12.0, float(layer.get("cell", 32.0)))
	var color = _program_color(layer, _palette_color("primary", Color(0.25, 0.9, 1.0)))
	var x = fmod(float(_program_seed() % 19), cell)
	while x < rect.size.x:
		draw_line(Vector2(x, 0), Vector2(x, rect.size.y), Color(color.r, color.g, color.b, 0.065), 1.0)
		x += cell
	var y = fmod(float(_program_seed() % 23), cell)
	while y < rect.size.y:
		draw_line(Vector2(0, y), Vector2(rect.size.x, y), Color(color.r, color.g, color.b, 0.065), 1.0)
		y += cell


func _draw_program_facet_field(layer: Dictionary) -> void:
	var rect = get_viewport_rect()
	var count = int(clamp(float(layer.get("count", 32)), 6.0, 80.0))
	var color = _program_color(layer, _palette_color("secondary", Color(0.55, 0.85, 1.0)))
	for index in range(count):
		var x = fmod(float(index * 73 + _program_seed()), max(rect.size.x, 1.0))
		var y = fmod(float(index * 47 + _program_seed() / 2), max(rect.size.y, 1.0))
		var size = 12.0 + float(index % 5) * 5.0
		var points = PackedVector2Array([
			Vector2(x, y - size),
			Vector2(x + size * 0.8, y),
			Vector2(x, y + size),
			Vector2(x - size * 0.8, y),
		])
		draw_colored_polygon(points, Color(color.r, color.g, color.b, 0.026))


func _draw_program_texture_noise(layer: Dictionary) -> void:
	var rect = get_viewport_rect()
	var count = int(clamp(float(layer.get("count", 70)), 10.0, 170.0))
	var color = _program_color(layer, _palette_color("secondary", Color(0.55, 0.85, 0.55)))
	for index in range(count):
		var x = fmod(float(index * 89 + _program_seed()), max(rect.size.x, 1.0))
		var y = fmod(float(index * 53 + _program_seed() / 7), max(rect.size.y, 1.0))
		var length = 8.0 + float(index % 6) * 3.0
		var angle = elapsed * 0.2 + float(index) * 0.7
		var p0 = Vector2(x, y)
		var p1 = p0 + Vector2(cos(angle), sin(angle)) * length
		draw_line(p0, p1, Color(color.r, color.g, color.b, 0.045), 1.0)


func _draw_program_heat_shimmer(layer: Dictionary) -> void:
	var rect = get_viewport_rect()
	var color = _program_color(layer, _palette_color("hot", ACTION_ORANGE))
	for index in range(9):
		var y = rect.size.y - float(index + 1) * rect.size.y / 10.0
		var offset = sin(elapsed * 2.0 + float(index)) * 20.0
		draw_line(Vector2(offset, y), Vector2(rect.size.x + offset, y - 18.0), Color(color.r, color.g, color.b, 0.035), 4.0)


func _draw_program_scanline_roll(layer: Dictionary) -> void:
	var rect = get_viewport_rect()
	var color = _program_color(layer, _palette_color("secondary", Color(1.0, 0.3, 0.9)))
	for y in range(int(fmod(elapsed * 28.0, 8.0)), int(rect.size.y), 8):
		draw_line(Vector2(0, y), Vector2(rect.size.x, y), Color(color.r, color.g, color.b, 0.055), 1.0)


func _draw_program_radial_glow(layer: Dictionary) -> void:
	var rect = get_viewport_rect()
	var count = int(clamp(float(layer.get("count", 3)), 1.0, 8.0))
	var color = _program_color(layer, _palette_color("primary", Color(0.30, 0.92, 1.0)))
	for index in range(count):
		var center = Vector2(
			rect.size.x * (0.18 + 0.64 * fmod(float(index * 37 + _program_seed() % 100), 100.0) / 100.0),
			rect.size.y * (0.20 + 0.58 * fmod(float(index * 53 + _program_seed() % 100), 100.0) / 100.0)
		)
		draw_circle(center, 140.0 + float(index) * 30.0, Color(color.r, color.g, color.b, 0.032))


func _draw_world_frame() -> void:
	var top_left = _world_to_screen(Vector2(0.0, world_size.y))
	var bottom_right = _world_to_screen(Vector2(world_size.x, 0.0))
	var rect = Rect2(top_left, bottom_right - top_left).abs()
	var primary = _palette_color("primary", Color(0.25, 0.9, 1.0))
	draw_rect(rect.grow(8.0), Color(primary.r, primary.g, primary.b, 0.10), false, 12.0)
	draw_rect(rect, Color(primary.r, primary.g, primary.b, 0.62), false, 2.0)


func _draw_grid() -> void:
	var step = 32.0 * world_scale
	if step < 8.0:
		return
	var rect = get_viewport_rect()
	var x = fmod(world_offset.x, step)
	var index = 0
	while x < rect.size.x:
		draw_line(Vector2(x, 0), Vector2(x, rect.size.y), GRID_MAJOR if index % 4 == 0 else GRID_MINOR, 1.0)
		x += step
		index += 1
	var y = fmod(world_offset.y, step)
	index = 0
	while y < rect.size.y:
		draw_line(Vector2(0, y), Vector2(rect.size.x, y), GRID_MAJOR if index % 4 == 0 else GRID_MINOR, 1.0)
		y += step
		index += 1


func _draw_objects() -> void:
	for object in objects:
		if typeof(object) != TYPE_DICTIONARY:
			continue
		var role = str(object.get("role", ""))
		var color = _object_color(object)
		_draw_object_pre_fx(object, color)
		var shapes: Array = object.get("shapes", [])
		for shape in shapes:
			if typeof(shape) != TYPE_DICTIONARY:
				continue
			if role == "agent":
				_draw_agent(shape, object)
			else:
				_draw_shape(shape, color, object)
		_draw_object_post_fx(object, color)


func _draw_shape(shape: Dictionary, color: Color, object: Dictionary) -> void:
	var type = str(shape.get("type", ""))
	var skin = _skin_for_object(object)
	var material = str(skin.get("material", ""))
	var offset = _object_preview_offset(object)
	if type == "circle":
		var center = _world_to_screen(_vec(shape.get("center", [0.0, 0.0])) + offset)
		var radius = float(shape.get("radius", 10.0)) * world_scale
		if material.find("laser_bolt") >= 0:
			_draw_laser_bolt_object(center, radius, color, skin, object)
			return
		if material.find("bear_avatar") >= 0:
			_draw_bear_object(center, radius, color, skin, object)
			return
		if material.find("heavy_iron_ball") >= 0:
			_draw_heavy_weight_object(center, radius, color, skin, object)
			return
		if _shape_is_fire(object) or material.find("ember") >= 0:
			_draw_fire_circle(center, radius, color)
			return
		draw_circle(center, radius + 7.0, Color(color.r, color.g, color.b, 0.14))
		draw_circle(center, radius, color)
		draw_arc(center, radius, 0.0, TAU, 36, color.lightened(0.35), 2.0)
		if _shape_is_ball(object):
			_draw_ball_marks(center, radius, color)
	elif type == "segment":
		var a = _world_to_screen(_vec(shape.get("a", [0.0, 0.0])) + offset)
		var b = _world_to_screen(_vec(shape.get("b", [0.0, 0.0])) + offset)
		var width = max(2.0, float(shape.get("radius", 2.0)) * 2.0 * world_scale)
		draw_line(a, b, Color(color.r, color.g, color.b, 0.25), width + 7.0)
		draw_line(a, b, color, width)
	elif type == "polygon":
		var raw_vertices: Array = shape.get("vertices", [])
		var vertices = PackedVector2Array()
		for raw in raw_vertices:
			vertices.append(_world_to_screen(_vec(raw) + offset))
		if vertices.size() >= 3:
			if material.find("tropical_water") >= 0:
				_draw_water_pool_object(vertices, skin)
				return
			if material.find("water_current") >= 0:
				_draw_water_current_object(vertices, skin)
				return
			if material.find("tropical_beacon") >= 0:
				_draw_tropical_beacon_object(vertices, skin)
				return
			if material.find("seesaw_plank") >= 0:
				_draw_seesaw_plank_object(vertices, object, skin)
				return
			if material.find("seesaw_pivot") >= 0:
				_draw_seesaw_pivot_object(vertices, skin)
				return
			if material.find("evergreen_tree") >= 0:
				_draw_tree_object(vertices, skin)
				return
			if material.find("wood_cabin") >= 0:
				_draw_cabin_object(vertices, skin)
				return
			if material.find("moss_path") >= 0:
				draw_colored_polygon(vertices, Color(color.r, color.g, color.b, 0.28))
				var path_outline = PackedVector2Array(vertices)
				path_outline.append(vertices[0])
				draw_polyline(path_outline, color.lightened(0.42), 2.0)
				return
			if material.find("enemy_fighter") >= 0:
				_draw_enemy_fighter_object(vertices, color, skin)
				return
			if material.find("capital_ship_shadow") >= 0:
				_draw_capital_ship_shadow_object(vertices, color, skin)
				return
			if material.find("lava_wave") >= 0 or material.find("molten_surface") >= 0:
				_draw_lava_pool_object(vertices, color, skin)
				return
			if material.find("obsidian_exit_gate") >= 0:
				_draw_obsidian_exit_object(vertices, color, skin)
				return
			draw_colored_polygon(vertices, color)
			var outline = PackedVector2Array(vertices)
			outline.append(vertices[0])
			draw_polyline(outline, color.lightened(0.38), 2.0)
			if _skin_has_fx(skin, "edge_bloom") or _skin_has_fx(skin, "edge_light") or str(object.get("name", "")).to_lower().find("wall") >= 0:
				draw_polyline(outline, Color(color.r, color.g, color.b, 0.34), 7.0)


func _draw_laser_bolt_object(center: Vector2, radius: float, color: Color, skin: Dictionary, object: Dictionary) -> void:
	var name = str(object.get("name", ""))
	var velocity: Vector2 = object_preview_velocities.get(name, Vector2(-1.0, 0.0))
	var direction = Vector2(velocity.x, -velocity.y)
	if direction.length() <= 0.01:
		direction = Vector2.LEFT
	direction = direction.normalized()
	var fill = _skin_color_value(skin, "fill", color)
	var outline = _skin_color_value(skin, "outline", Color(1.0, 0.92, 0.98, 1.0))
	var glow = _skin_color_value(skin, "glow", fill)
	var start = center - direction * max(18.0, radius * 3.2)
	var end = center + direction * max(28.0, radius * 5.6)
	var pulse = 0.72 + 0.28 * sin(elapsed * 18.0 + center.x * 0.03)
	draw_line(start, end, Color(glow.r, glow.g, glow.b, 0.32 * pulse), max(10.0, radius * 4.0))
	draw_line(start, end, fill, max(4.0, radius * 1.8))
	draw_line(start, end, outline, max(1.4, radius * 0.55))
	draw_circle(end, max(3.0, radius * 0.85), Color(outline.r, outline.g, outline.b, 0.82))


func _draw_enemy_fighter_object(vertices: PackedVector2Array, color: Color, skin: Dictionary) -> void:
	var rect = _bounds_for_vertices(vertices)
	var center = rect.get_center()
	var unit = max(14.0, max(rect.size.x, rect.size.y) * 0.31)
	var fill = _skin_color_value(skin, "fill", Color(0.13, 0.08, 0.20, 0.96))
	var outline = _skin_color_value(skin, "outline", color)
	var glow = _skin_color_value(skin, "glow", outline)
	var hull = PackedVector2Array([
		center + Vector2(-unit * 1.35, 0.0),
		center + Vector2(-unit * 0.10, -unit * 0.95),
		center + Vector2(unit * 0.95, -unit * 0.55),
		center + Vector2(unit * 0.36, 0.0),
		center + Vector2(unit * 0.95, unit * 0.55),
		center + Vector2(-unit * 0.10, unit * 0.95),
	])
	var pulse = 0.58 + 0.42 * sin(elapsed * 5.2 + center.y * 0.03)
	draw_polyline(PackedVector2Array([hull[0], hull[1], hull[2], hull[3], hull[4], hull[5], hull[0]]), Color(glow.r, glow.g, glow.b, 0.28 * pulse), 8.0)
	draw_colored_polygon(hull, fill)
	var outline_points = PackedVector2Array(hull)
	outline_points.append(hull[0])
	draw_polyline(outline_points, outline, 2.6)
	draw_circle(center + Vector2(-unit * 0.42, 0.0), max(2.4, unit * 0.13), outline.lightened(0.45))
	draw_circle(center + Vector2(-unit * 1.05, -unit * 0.22), max(2.2, unit * 0.10), Color(glow.r, glow.g, glow.b, 0.72 * pulse))
	draw_circle(center + Vector2(-unit * 1.05, unit * 0.22), max(2.2, unit * 0.10), Color(glow.r, glow.g, glow.b, 0.72 * pulse))


func _draw_capital_ship_shadow_object(vertices: PackedVector2Array, color: Color, skin: Dictionary) -> void:
	var fill = _skin_color_value(skin, "fill", Color(0.07, 0.08, 0.15, 0.56))
	var outline = _skin_color_value(skin, "outline", color)
	draw_colored_polygon(vertices, Color(fill.r, fill.g, fill.b, 0.42))
	var outline_points = PackedVector2Array(vertices)
	outline_points.append(vertices[0])
	draw_polyline(outline_points, Color(outline.r, outline.g, outline.b, 0.38), 2.0)
	var rect = _bounds_for_vertices(vertices)
	for index in range(6):
		var y = rect.position.y + rect.size.y * (float(index) + 1.0) / 7.0
		draw_line(Vector2(rect.position.x + 16.0, y), Vector2(rect.position.x + rect.size.x - 18.0, y - 7.0), Color(outline.r, outline.g, outline.b, 0.16), 1.0)


func _draw_lava_pool_object(vertices: PackedVector2Array, color: Color, skin: Dictionary) -> void:
	var rect = _bounds_for_vertices(vertices)
	var fill = _skin_color_value(skin, "fill", Color(1.0, 0.22, 0.04, 0.92))
	var outline = _skin_color_value(skin, "outline", Color(1.0, 0.72, 0.18, 0.96))
	var glow = _skin_color_value(skin, "glow", color)
	draw_colored_polygon(vertices, fill)
	var outline_points = PackedVector2Array(vertices)
	outline_points.append(vertices[0])
	draw_polyline(outline_points, Color(glow.r, glow.g, glow.b, 0.42), 8.0)
	draw_polyline(outline_points, outline, 2.0)
	for index in range(5):
		var y = rect.position.y + rect.size.y * (0.16 + float(index) * 0.17)
		var wave = PackedVector2Array()
		for step in range(0, int(rect.size.x) + 10, 10):
			var x = rect.position.x + float(step)
			wave.append(Vector2(x, y + sin(x * 0.045 + elapsed * 3.2 + float(index) * 1.4) * (3.0 + float(index))))
		draw_polyline(wave, outline if index == 0 else Color(outline.r, outline.g, outline.b, 0.52), 2.2 if index == 0 else 1.2)
	for index in range(10):
		var bx = rect.position.x + fmod(float(index * 43) + elapsed * 28.0, max(rect.size.x, 1.0))
		var by = rect.position.y + rect.size.y - fmod(float(index * 19) + elapsed * 35.0, max(rect.size.y, 1.0))
		draw_circle(Vector2(bx, by), 1.8 + float(index % 3), Color(1.0, 0.84, 0.25, 0.38))


func _draw_obsidian_exit_object(vertices: PackedVector2Array, color: Color, skin: Dictionary) -> void:
	var rect = _bounds_for_vertices(vertices)
	var fill = _skin_color_value(skin, "fill", Color(0.12, 0.07, 0.12, 0.94))
	var outline = _skin_color_value(skin, "outline", Color(1.0, 0.82, 0.28, 0.96))
	var glow = _skin_color_value(skin, "glow", color)
	var pulse = 0.55 + 0.45 * sin(elapsed * 4.4)
	draw_colored_polygon(vertices, fill)
	var outline_points = PackedVector2Array(vertices)
	outline_points.append(vertices[0])
	draw_polyline(outline_points, Color(glow.r, glow.g, glow.b, 0.34 + pulse * 0.18), 9.0)
	draw_polyline(outline_points, outline, 2.8)
	draw_arc(rect.get_center(), max(rect.size.x, rect.size.y) * (0.54 + pulse * 0.08), 0.0, TAU, 54, Color(glow.r, glow.g, glow.b, 0.66), 3.0)
	draw_circle(rect.get_center(), max(5.0, min(rect.size.x, rect.size.y) * 0.11), outline.lightened(0.24))


func _draw_agent(shape: Dictionary, object: Dictionary) -> void:
	var avatar = str(visual_brief.get("agent_avatar", "human"))
	if avatar == "arcade_disc":
		_draw_arcade_agent(shape)
		return
	if avatar == "ship":
		_draw_ship_agent(shape)
		return
	if avatar == "robot":
		_draw_robot_agent(shape)
		return
	var center = _world_to_screen(agent_preview_position)
	var primary = _contrast_color(_palette_color("primary", Color(0.30, 0.92, 1.0)).lightened(0.16), 0.42)
	var secondary = _contrast_color(_palette_color("secondary", Color(0.93, 1.0, 0.98)).lightened(0.10), 0.36)
	var hot = _contrast_color(_palette_color("hot", ACTION_ORANGE), 0.40)
	var radius = max(13.0, float(shape.get("radius", 15.0)) * world_scale * 1.55)
	var pose = _human_pose_points(center, radius, agent_facing, agent_action)
	var head: Vector2 = pose["head"]
	var neck: Vector2 = pose["neck"]
	var hip: Vector2 = pose["hip"]
	var hand_a: Vector2 = pose["hand_a"]
	var hand_b: Vector2 = pose["hand_b"]
	var knee_a: Vector2 = pose["knee_a"]
	var knee_b: Vector2 = pose["knee_b"]
	var foot_a: Vector2 = pose["foot_a"]
	var foot_b: Vector2 = pose["foot_b"]
	var width = max(2.8, radius * 0.16)
	_draw_agent_shadow(center, radius, primary)
	_draw_agent_motion_streaks(center, radius, primary, hot)
	for pair in [[neck, hip], [neck, hand_a], [neck, hand_b], [hip, knee_a], [knee_a, foot_a], [hip, knee_b], [knee_b, foot_b]]:
		_draw_limb(pair[0], pair[1], width + 5.0, Color(primary.r, primary.g, primary.b, 0.14))
	for pair in [[neck, hip], [neck, hand_a], [neck, hand_b], [hip, knee_a], [knee_a, foot_a], [hip, knee_b], [knee_b, foot_b]]:
		_draw_limb(pair[0], pair[1], width, primary)
	var head_radius = max(5.0, radius * 0.38)
	draw_circle(head, head_radius + 4.0, Color(primary.r, primary.g, primary.b, 0.16))
	draw_circle(head, head_radius, primary)
	draw_circle(head + Vector2(agent_facing * head_radius * 0.28, -head_radius * 0.22), max(1.4, head_radius * 0.18), secondary)
	if agent_action in ["kick", "throw", "push"]:
		_draw_agent_action_cue(pose, radius)


func _draw_arcade_agent(shape: Dictionary) -> void:
	var center = _world_to_screen(agent_preview_position)
	var radius = max(12.0, float(shape.get("radius", 15.0)) * world_scale * 1.35)
	var primary = _palette_color("primary", Color(1.0, 0.91, 0.25))
	var secondary = _palette_color("secondary", Color(1.0, 0.2, 0.92))
	draw_circle(center, radius + 8.0, Color(secondary.r, secondary.g, secondary.b, 0.18))
	draw_circle(center, radius, primary)
	var mouth = PackedVector2Array([
		center,
		center + Vector2(radius * 1.08, -radius * (0.24 + 0.10 * sin(elapsed * 12.0))),
		center + Vector2(radius * 1.08, radius * (0.24 + 0.10 * sin(elapsed * 12.0))),
	])
	draw_colored_polygon(mouth, _palette_color("background_color", Color(0.04, 0.04, 0.08)))
	draw_arc(center, radius, 0.0, TAU, 32, secondary, 2.5)


func _draw_ship_agent(shape: Dictionary) -> void:
	var center = _world_to_screen(agent_preview_position)
	var unit = max(15.0, float(shape.get("radius", 15.0)) * world_scale * 1.65)
	var primary = _palette_color("primary", Color(0.35, 0.9, 1.0))
	var hot = _palette_color("hot", ACTION_ORANGE)
	var hit_mix = clamp(agent_hit_flash_time, 0.0, 1.0)
	var hull_color = primary.lerp(Color(1.0, 0.08, 0.08, 1.0), hit_mix)
	var outline_color = primary.lightened(0.35).lerp(Color(1.0, 0.72, 0.58, 1.0), hit_mix)
	var points = PackedVector2Array([
		center + Vector2(unit * 1.25, 0.0),
		center + Vector2(-unit * 0.85, -unit * 0.72),
		center + Vector2(-unit * 0.48, 0.0),
		center + Vector2(-unit * 0.85, unit * 0.72),
	])
	if agent_hit_flash_time > 0.0:
		draw_circle(center, unit * (1.45 + 0.18 * sin(elapsed * 28.0)), Color(1.0, 0.08, 0.04, 0.28 * hit_mix))
	draw_colored_polygon(points, Color(hull_color.r, hull_color.g, hull_color.b, 0.86))
	var outline = PackedVector2Array(points)
	outline.append(points[0])
	draw_polyline(outline, outline_color, 2.0 + hit_mix * 2.0)
	var flame = PackedVector2Array([
		center + Vector2(-unit * 0.74, -unit * 0.38),
		center + Vector2(-unit * (1.35 + 0.18 * sin(elapsed * 18.0)), 0.0),
		center + Vector2(-unit * 0.74, unit * 0.38),
	])
	draw_colored_polygon(flame, Color(hot.r, hot.g, hot.b, 0.62))


func _draw_robot_agent(shape: Dictionary) -> void:
	var center = _world_to_screen(agent_preview_position)
	var unit = max(13.0, float(shape.get("radius", 15.0)) * world_scale * 1.45)
	var primary = _palette_color("primary", Color(0.36, 0.9, 1.0))
	var secondary = _palette_color("secondary", Color(0.7, 0.45, 1.0))
	var body = Rect2(center + Vector2(-unit * 0.55, -unit * 1.15), Vector2(unit * 1.1, unit * 1.35))
	var head = Rect2(center + Vector2(-unit * 0.48, -unit * 1.90), Vector2(unit * 0.96, unit * 0.62))
	draw_rect(body.grow(5.0), Color(primary.r, primary.g, primary.b, 0.16), true)
	draw_rect(body, Color(0.04, 0.08, 0.10, 0.95), true)
	draw_rect(body, primary, false, 2.5)
	draw_rect(head, Color(0.04, 0.08, 0.10, 0.95), true)
	draw_rect(head, secondary, false, 2.5)
	draw_circle(head.get_center() + Vector2(unit * 0.20, 0.0), max(2.0, unit * 0.11), secondary.lightened(0.35))


func _human_pose_points(center: Vector2, radius: float, facing: float, action: String) -> Dictionary:
	var gait = sin(elapsed * 10.5)
	var bob = sin(elapsed * 8.0) * radius * 0.08
	var hip = center + Vector2(-facing * radius * 0.06, -radius * 0.22 + bob)
	var neck = center + Vector2(facing * radius * 0.08, -radius * 1.30 + bob)
	var head = center + Vector2(facing * radius * 0.16, -radius * 2.06 + bob)
	var hand_a: Vector2
	var hand_b: Vector2
	var knee_a: Vector2
	var knee_b: Vector2
	var foot_a: Vector2
	var foot_b: Vector2

	if action == "kick":
		neck += Vector2(facing * radius * 0.28, radius * 0.04)
		head += Vector2(facing * radius * 0.28, radius * 0.04)
		hand_a = neck + Vector2(-facing * radius * 0.24, radius * 0.58)
		hand_b = neck + Vector2(-facing * radius * 0.82, radius * 0.44)
		knee_a = hip + Vector2(facing * radius * 0.72, radius * 0.46)
		foot_a = hip + Vector2(facing * radius * 1.72, radius * 0.56)
		knee_b = hip + Vector2(-facing * radius * 0.44, radius * 0.78)
		foot_b = hip + Vector2(-facing * radius * 1.02, radius * 1.34)
	elif action == "throw":
		neck += Vector2(facing * radius * 0.20, -radius * 0.08)
		head += Vector2(facing * radius * 0.20, -radius * 0.08)
		hand_a = neck + Vector2(facing * radius * 1.30, -radius * 0.56)
		hand_b = neck + Vector2(-facing * radius * 0.70, radius * 0.52)
		knee_a = hip + Vector2(facing * radius * 0.46, radius * 0.72)
		foot_a = hip + Vector2(facing * radius * 0.88, radius * 1.32)
		knee_b = hip + Vector2(-facing * radius * 0.46, radius * 0.72)
		foot_b = hip + Vector2(-facing * radius * 0.84, radius * 1.32)
	elif action == "swim":
		hip = center + Vector2(-facing * radius * 0.12, radius * 0.14 + bob)
		neck = center + Vector2(facing * radius * 0.24, -radius * 0.34 + bob)
		head = center + Vector2(facing * radius * 0.62, -radius * 0.56 + bob)
		hand_a = neck + Vector2(facing * radius * (1.04 + 0.18 * gait), radius * (0.08 - 0.18 * abs(gait)))
		hand_b = neck + Vector2(-facing * radius * (0.78 - 0.12 * gait), radius * (0.20 + 0.15 * abs(gait)))
		knee_a = hip + Vector2(facing * radius * 0.44, radius * (0.24 + 0.14 * gait))
		foot_a = hip + Vector2(facing * radius * 0.98, radius * (0.30 - 0.16 * gait))
		knee_b = hip + Vector2(-facing * radius * 0.38, radius * (0.22 - 0.12 * gait))
		foot_b = hip + Vector2(-facing * radius * 0.90, radius * (0.32 + 0.14 * gait))
	elif action == "push":
		neck += Vector2(facing * radius * 0.38, radius * 0.18)
		head += Vector2(facing * radius * 0.38, radius * 0.18)
		hip += Vector2(-facing * radius * 0.20, radius * 0.12)
		hand_a = neck + Vector2(facing * radius * 1.28, radius * 0.20)
		hand_b = neck + Vector2(facing * radius * 1.08, radius * 0.46)
		knee_a = hip + Vector2(facing * radius * 0.28, radius * 0.78)
		foot_a = hip + Vector2(facing * radius * 0.64, radius * 1.38)
		knee_b = hip + Vector2(-facing * radius * 0.64, radius * 0.76)
		foot_b = hip + Vector2(-facing * radius * 1.16, radius * 1.22)
	elif action == "jump":
		hand_a = neck + Vector2(facing * radius * 0.64, -radius * 0.42)
		hand_b = neck + Vector2(-facing * radius * 0.58, -radius * 0.32)
		knee_a = hip + Vector2(facing * radius * 0.46, radius * 0.50)
		foot_a = hip + Vector2(facing * radius * 0.92, radius * 0.88)
		knee_b = hip + Vector2(-facing * radius * 0.42, radius * 0.50)
		foot_b = hip + Vector2(-facing * radius * 0.88, radius * 0.82)
	elif action == "fall":
		hand_a = neck + Vector2(facing * radius * 0.72, -radius * 0.34)
		hand_b = neck + Vector2(-facing * radius * 0.66, -radius * 0.20)
		knee_a = hip + Vector2(facing * radius * 0.38, radius * 0.50)
		foot_a = hip + Vector2(facing * radius * 0.48, radius * 1.08)
		knee_b = hip + Vector2(-facing * radius * 0.42, radius * 0.44)
		foot_b = hip + Vector2(-facing * radius * 0.78, radius * 0.88)
	elif action == "float":
		hand_a = neck + Vector2(facing * radius * (0.62 + 0.12 * gait), -radius * (0.12 + 0.06 * gait))
		hand_b = neck + Vector2(-facing * radius * (0.56 - 0.08 * gait), radius * (0.36 + 0.08 * gait))
		knee_a = hip + Vector2(facing * radius * 0.36, radius * (0.58 + 0.06 * gait))
		foot_a = hip + Vector2(facing * radius * 0.84, radius * (0.82 - 0.08 * gait))
		knee_b = hip + Vector2(-facing * radius * 0.34, radius * (0.58 - 0.06 * gait))
		foot_b = hip + Vector2(-facing * radius * 0.78, radius * (0.92 + 0.06 * gait))
	elif action == "run":
		hand_a = neck + Vector2(-facing * radius * 0.56 * gait, radius * 0.48)
		hand_b = neck + Vector2(facing * radius * 0.56 * gait, radius * 0.42)
		knee_a = hip + Vector2(facing * radius * 0.50 * gait, radius * 0.72)
		foot_a = hip + Vector2(facing * radius * (0.96 * gait + 0.10), radius * 1.32)
		knee_b = hip + Vector2(-facing * radius * 0.50 * gait, radius * 0.72)
		foot_b = hip + Vector2(-facing * radius * (0.96 * gait - 0.10), radius * 1.32)
	else:
		hand_a = neck + Vector2(facing * radius * 0.48, radius * 0.62)
		hand_b = neck + Vector2(-facing * radius * 0.40, radius * 0.66)
		knee_a = hip + Vector2(facing * radius * 0.30, radius * 0.78)
		foot_a = hip + Vector2(facing * radius * 0.46, radius * 1.34)
		knee_b = hip + Vector2(-facing * radius * 0.30, radius * 0.78)
		foot_b = hip + Vector2(-facing * radius * 0.46, radius * 1.34)

	return {
		"head": head,
		"neck": neck,
		"hip": hip,
		"hand_a": hand_a,
		"hand_b": hand_b,
		"knee_a": knee_a,
		"knee_b": knee_b,
		"foot_a": foot_a,
		"foot_b": foot_b,
	}


func _draw_agent_shadow(center: Vector2, radius: float, color: Color) -> void:
	_draw_filled_ellipse(
		center + Vector2(0.0, radius * 1.18),
		Vector2(radius * 1.10, radius * 0.14),
		Color(color.r, color.g, color.b, 0.08)
	)


func _draw_agent_motion_streaks(center: Vector2, radius: float, color: Color, hot: Color) -> void:
	if agent_action not in ["run", "kick", "throw", "push", "jump", "fall", "swim"]:
		return
	var direction = -agent_facing
	var count = 3 if agent_action in ["run", "jump", "fall"] else 5
	for index in range(count):
		var offset = Vector2(direction * radius * (0.7 + float(index) * 0.28), radius * (0.15 + 0.18 * float(index % 2)))
		var p0 = center + offset
		var p1 = p0 + Vector2(direction * radius * (0.45 + float(index) * 0.08), -radius * 0.08)
		var streak_color = _palette_color("secondary", color) if agent_action == "swim" else (hot if agent_action in ["kick", "throw"] else color)
		draw_line(p0, p1, Color(streak_color.r, streak_color.g, streak_color.b, 0.22 - float(index) * 0.03), max(1.4, radius * 0.055))


func _draw_agent_action_cue(pose: Dictionary, radius: float) -> void:
	var source: Vector2 = pose["foot_a"] if agent_action == "kick" else pose["hand_a"]
	var target = source + Vector2(agent_facing * radius * (1.35 if agent_action == "kick" else 1.05), -radius * (0.02 if agent_action == "kick" else 0.32))
	if action_flash_time > 0.0:
		target = _world_to_screen(action_flash_position)
	draw_line(source, target, Color(1.0, 0.80, 0.32, 0.82), max(2.0, radius * 0.075))
	draw_circle(target, max(3.0, radius * 0.16), Color(ACTION_ORANGE.r, ACTION_ORANGE.g, ACTION_ORANGE.b, 0.70))
	for index in range(4):
		var angle = (-0.35 + float(index) * 0.24) if agent_action == "kick" else (-0.65 + float(index) * 0.30)
		var burst = target + Vector2(agent_facing * cos(angle), sin(angle)) * radius * (0.48 + float(index) * 0.10)
		draw_line(target, burst, Color(ACTION_ORANGE.r, ACTION_ORANGE.g, ACTION_ORANGE.b, 0.64), max(1.2, radius * 0.045))


func _draw_limb(a: Vector2, b: Vector2, width: float, color: Color) -> void:
	draw_line(a, b, color, width)
	draw_circle(a, width * 0.5, color)
	draw_circle(b, width * 0.5, color)


func _draw_filled_ellipse(center: Vector2, radii: Vector2, color: Color) -> void:
	var points = PackedVector2Array()
	for index in range(28):
		var angle = TAU * float(index) / 28.0
		points.append(center + Vector2(cos(angle) * radii.x, sin(angle) * radii.y))
	draw_colored_polygon(points, color)


func _draw_object_pre_fx(object: Dictionary, color: Color) -> void:
	var center = _object_center_screen(object)
	var role = str(object.get("role", "")).to_lower()
	var name = str(object.get("name", "")).to_lower()
	var skin = _skin_for_object(object)
	var pulse = 0.5 + 0.5 * sin(elapsed * 3.2)
	_draw_visual_program_object_effects(object, color, true)
	if role in ["goal", "trigger", "region"] or _skin_has_fx(skin, "breathing_ring") or name.find("goal") >= 0:
		draw_circle(center, 22.0 + pulse * 8.0, Color(SENSOR_GREEN.r, SENSOR_GREEN.g, SENSOR_GREEN.b, 0.10 + 0.12 * pulse))
		draw_arc(center, 28.0 + pulse * 10.0, 0.0, TAU, 48, Color(SENSOR_GREEN.r, SENSOR_GREEN.g, SENSOR_GREEN.b, 0.56), 2.0)
	if role in ["hazard", "danger"] or _shape_is_fire(object):
		draw_circle(center, 26.0 + pulse * 12.0, Color(1.0, 0.18, 0.04, 0.12 + 0.12 * pulse))


func _draw_visual_program_object_effects(object: Dictionary, fallback_color: Color, before_shape: bool) -> void:
	if visual_program.is_empty():
		return
	var effects: Array = visual_program.get("object_effects", [])
	if effects.is_empty():
		return
	for effect in effects:
		if typeof(effect) != TYPE_DICTIONARY:
			continue
		if not _program_effect_matches_object(effect, object):
			continue
		var primitive = str(effect.get("primitive", effect.get("type", "")))
		var color = _program_color(effect, fallback_color)
		var center = _object_center_screen(object)
		var pulse = 0.5 + 0.5 * sin(elapsed * (3.0 + float(_program_seed() % 3)) + center.x * 0.01)
		if primitive == "portal_ring" and before_shape:
			draw_circle(center, 24.0 + pulse * 14.0, Color(color.r, color.g, color.b, 0.10 + pulse * 0.08))
			draw_arc(center, 32.0 + pulse * 13.0, 0.0, TAU, 52, Color(color.r, color.g, color.b, 0.66), 2.4)
		elif primitive == "danger_pulse" and before_shape:
			draw_circle(center, 30.0 + pulse * 18.0, Color(color.r, color.g, color.b, 0.10 + pulse * 0.16))
		elif primitive == "impact_shadow" and before_shape:
			_draw_filled_ellipse(center + Vector2(0, 20), Vector2(28 + pulse * 8, 8 + pulse * 2), Color(color.r, color.g, color.b, 0.10))
		elif primitive == "motion_trail" and before_shape:
			var name = str(object.get("name", ""))
			var velocity: Vector2 = object_preview_velocities.get(name, Vector2.ZERO)
			if velocity.length() > 10.0:
				var trail_end = _world_to_screen(_object_center_world(object) - velocity.normalized() * min(70.0, velocity.length() * 0.18))
				draw_line(trail_end, center, Color(color.r, color.g, color.b, 0.26), 5.0)
		elif primitive == "flame_orb" and not before_shape:
			draw_circle(center, 22.0 + pulse * 9.0, Color(1.0, 0.16, 0.03, 0.20))
			for index in range(5):
				var angle = elapsed * 4.0 + float(index) * TAU / 5.0
				var ember = center + Vector2(cos(angle), sin(angle)) * (18.0 + pulse * 8.0)
				draw_circle(ember, 2.0 + pulse * 2.0, Color(color.r, color.g, color.b, 0.72))
		elif primitive == "neon_outline" and not before_shape:
			draw_circle(center, 25.0 + pulse * 4.0, Color(color.r, color.g, color.b, 0.18))
			draw_arc(center, 28.0 + pulse * 4.0, 0.0, TAU, 40, Color(color.r, color.g, color.b, 0.68), 2.0)
		elif primitive == "rim_glow" and before_shape:
			draw_circle(center, 22.0 + pulse * 7.0, Color(color.r, color.g, color.b, 0.11))
		elif primitive == "field_ring" and before_shape:
			var rings = int(clamp(float(effect.get("rings", 3)), 1.0, 7.0))
			for ring in range(rings):
				draw_arc(center, 24.0 + float(ring) * 13.0 + pulse * 8.0, 0.0, TAU, 50, Color(color.r, color.g, color.b, 0.23), 1.4)
		elif primitive == "spark_burst" and not before_shape:
			for index in range(6):
				var angle = float(index) * TAU / 6.0 + elapsed * 1.8
				var p0 = center + Vector2(cos(angle), sin(angle)) * 12.0
				var p1 = center + Vector2(cos(angle), sin(angle)) * (22.0 + pulse * 7.0)
				draw_line(p0, p1, Color(color.r, color.g, color.b, 0.72), 1.4)
		elif primitive == "material_stripes" and not before_shape:
			for index in range(4):
				var y = center.y - 20.0 + float(index) * 12.0
				draw_line(Vector2(center.x - 28.0, y), Vector2(center.x + 28.0, y - 10.0), Color(color.r, color.g, color.b, 0.22), 2.0)
		elif primitive == "cracked_surface" and not before_shape:
			for index in range(5):
				var angle = float(index) * 1.23 + elapsed * 0.04
				var p0 = center + Vector2(cos(angle), sin(angle)) * 8.0
				var p1 = center + Vector2(cos(angle + 0.35), sin(angle + 0.35)) * 30.0
				draw_line(p0, p1, Color(color.r, color.g, color.b, 0.20), 1.6)


func _program_effect_matches_object(effect: Dictionary, object: Dictionary) -> bool:
	var selector = str(effect.get("selector", "*")).to_lower()
	if selector == "*" or selector == "":
		return true
	var text = _object_text(object)
	for token in selector.split("|"):
		var normalized = token.strip_edges()
		if normalized.begins_with("*"):
			normalized = normalized.substr(1)
		if normalized.ends_with("*"):
			normalized = normalized.substr(0, normalized.length() - 1)
		if normalized != "" and text.find(normalized) >= 0:
			return true
	return false


func _draw_object_post_fx(object: Dictionary, color: Color) -> void:
	var center = _object_center_screen(object)
	var skin = _skin_for_object(object)
	_draw_visual_program_object_effects(object, color, false)
	if _skin_has_fx(skin, "twinkle") or _skin_has_fx(skin, "edge_bloom"):
		for index in range(3):
			var angle = elapsed * (1.3 + float(index) * 0.25) + float(index) * 2.1
			var p = center + Vector2(cos(angle), sin(angle)) * (18.0 + float(index) * 5.0)
			draw_circle(p, 1.8 + float(index) * 0.5, color.lightened(0.45))


func _draw_foreground_fx() -> void:
	if _has_keyword("fire") or _has_keyword("lava"):
		for index in range(36):
			var x = fmod(float(index * 97) + elapsed * (20.0 + float(index % 7) * 5.0), get_viewport_rect().size.x)
			var y = get_viewport_rect().size.y - fmod(float(index * 53) + elapsed * 38.0, 220.0)
			var alpha = 0.10 + 0.10 * sin(elapsed * 4.0 + float(index))
			draw_circle(Vector2(x, y), 1.4 + float(index % 3), Color(1.0, 0.35, 0.08, alpha))
	elif _has_keyword("space") or _has_keyword("asteroid"):
		for index in range(18):
			var x = fmod(float(index * 151) - elapsed * (12.0 + float(index % 5)), get_viewport_rect().size.x)
			var y = fmod(float(index * 71), get_viewport_rect().size.y)
			draw_line(Vector2(x, y), Vector2(x - 18.0, y + 4.0), Color(0.65, 0.9, 1.0, 0.16), 1.0)


func _draw_fire_circle(center: Vector2, radius: float, color: Color) -> void:
	var pulse = 0.5 + 0.5 * sin(elapsed * 8.0 + center.x * 0.01)
	var display_radius = radius * 0.76
	draw_circle(center, display_radius + 8.0 + pulse * 4.0, Color(1.0, 0.10, 0.02, 0.13))
	draw_circle(center, display_radius + 3.0, Color(1.0, 0.34, 0.06, 0.72))
	draw_circle(center + Vector2(0, -display_radius * 0.18), display_radius * 0.62, Color(1.0, 0.86, 0.18, 0.92))
	draw_arc(center, display_radius + 5.0, 0.0, TAU, 36, Color(1.0, 0.72, 0.2, 0.78), 2.0)


func _draw_bear_object(center: Vector2, radius: float, color: Color, skin: Dictionary, object: Dictionary) -> void:
	var fill = _skin_color_value(skin, "fill", Color(0.44, 0.27, 0.14, 0.96))
	var outline = _skin_color_value(skin, "outline", Color(0.98, 0.77, 0.48, 0.95))
	var glow = _skin_color_value(skin, "glow", color)
	var velocity = Vector2.ZERO
	var name = str(object.get("name", ""))
	if object_preview_positions.has(name):
		var source = _object_source_center_world(object)
		velocity = object_preview_positions[name] - source
	var facing = 1.0 if velocity.x >= -0.2 else -1.0
	var stride = sin(elapsed * 12.0 + center.x * 0.02)
	var body_rect = Rect2(center - Vector2(radius * 1.25, radius * 0.48), Vector2(radius * 2.45, radius * 1.36))
	body_rect.position.x -= facing * radius * 0.18
	draw_colored_polygon(_ellipse_points(body_rect.grow(radius * 0.28), 28), Color(glow.r, glow.g, glow.b, 0.16))
	draw_colored_polygon(_ellipse_points(body_rect, 28), fill)
	draw_arc(body_rect.get_center(), max(radius * 1.02, 8.0), 0.0, TAU, 36, outline, max(2.0, radius * 0.13))
	var head = center + Vector2(facing * radius * 0.98, -radius * 0.14)
	draw_circle(head, radius * 0.68, fill)
	draw_arc(head, radius * 0.68, 0.0, TAU, 28, outline, max(2.0, radius * 0.12))
	draw_circle(head + Vector2(-facing * radius * 0.18, -radius * 0.50), radius * 0.22, fill.darkened(0.2))
	draw_circle(head + Vector2(facing * radius * 0.18, -radius * 0.48), radius * 0.20, fill.darkened(0.2))
	draw_circle(head + Vector2(facing * radius * 0.23, -radius * 0.12), max(1.6, radius * 0.10), Color(0.02, 0.02, 0.01, 1.0))
	draw_circle(head + Vector2(facing * radius * 0.56, radius * 0.12), max(2.0, radius * 0.12), Color(0.02, 0.018, 0.012, 1.0))
	for index in range(4):
		var xmul = [-0.66, -0.22, 0.28, 0.68][index]
		var phase = stride if index % 2 == 0 else -stride
		var hip = center + Vector2(radius * xmul, radius * 0.72)
		var paw = hip + Vector2(facing * phase * radius * 0.18, radius * 0.58)
		draw_line(hip, paw, fill.darkened(0.25), max(3.0, radius * 0.20))
		draw_circle(paw, max(2.0, radius * 0.13), outline)


func _draw_heavy_weight_object(center: Vector2, radius: float, color: Color, skin: Dictionary, object: Dictionary) -> void:
	var fill = _skin_color_value(skin, "fill", Color(0.35, 0.37, 0.40, 0.96))
	var outline = _skin_color_value(skin, "outline", Color(0.88, 0.92, 0.91, 0.95))
	var glow = _skin_color_value(skin, "glow", color)
	draw_circle(center, radius * 1.45, Color(glow.r, glow.g, glow.b, 0.12))
	draw_circle(center, radius, fill)
	draw_arc(center, radius, 0.0, TAU, 36, outline, max(2.0, radius * 0.12))
	draw_circle(center + Vector2(-radius * 0.26, -radius * 0.25), radius * 0.24, fill.lightened(0.34))
	draw_circle(center + Vector2(radius * 0.25, radius * 0.22), radius * 0.32, fill.darkened(0.30))


func _draw_water_pool_object(vertices: PackedVector2Array, skin: Dictionary) -> void:
	var rect = _bounds_for_vertices(vertices)
	var fill = _skin_color_value(skin, "fill", Color(0.10, 0.56, 0.77, 0.56))
	var outline = _skin_color_value(skin, "outline", Color(0.40, 0.97, 1.0, 0.86))
	var foam = _skin_color_value(skin, "foam", Color(0.88, 1.0, 0.96, 0.88))
	draw_colored_polygon(vertices, Color(fill.r, fill.g, fill.b, 0.48))
	var outline_points = PackedVector2Array(vertices)
	outline_points.append(vertices[0])
	draw_polyline(outline_points, Color(outline.r, outline.g, outline.b, 0.62), 2.0)
	for line_index in range(5):
		var y = rect.position.y + rect.size.y * (0.12 + float(line_index) * 0.17)
		var wave = PackedVector2Array()
		for step in range(0, int(rect.size.x) + 10, 10):
			var x = rect.position.x + float(step)
			wave.append(Vector2(x, y + sin(x * 0.035 + elapsed * 2.0 + float(line_index)) * (3.0 + float(line_index) * 0.8)))
		draw_polyline(wave, foam if line_index == 0 else Color(outline.r, outline.g, outline.b, 0.38), 1.6 if line_index == 0 else 1.0)
	for index in range(12):
		var bx = rect.position.x + fmod(float(index * 37) + elapsed * 16.0, max(rect.size.x, 1.0))
		var by = rect.position.y + rect.size.y - fmod(float(index * 29) + elapsed * 24.0, max(rect.size.y, 1.0))
		draw_circle(Vector2(bx, by), 1.4 + float(index % 3), Color(foam.r, foam.g, foam.b, 0.28))


func _draw_water_current_object(vertices: PackedVector2Array, skin: Dictionary) -> void:
	var rect = _bounds_for_vertices(vertices)
	var outline = _skin_color_value(skin, "outline", Color(0.72, 1.0, 0.95, 0.50))
	for index in range(4):
		var y = rect.position.y + rect.size.y * (0.25 + float(index) * 0.16)
		var phase = fmod(elapsed * 58.0 + float(index) * 43.0, max(rect.size.x, 1.0))
		var start = Vector2(rect.position.x + phase, y)
		var end = start + Vector2(34.0, -5.0)
		draw_line(start, end, Color(outline.r, outline.g, outline.b, 0.32), 2.0)
		draw_circle(end, 2.2, Color(outline.r, outline.g, outline.b, 0.38))


func _draw_tropical_beacon_object(vertices: PackedVector2Array, skin: Dictionary) -> void:
	var rect = _bounds_for_vertices(vertices)
	var fill = _skin_color_value(skin, "fill", Color(0.22, 0.91, 1.0, 0.92))
	var outline = _skin_color_value(skin, "outline", Color(0.95, 1.0, 0.98, 0.96))
	var glow = _skin_color_value(skin, "glow", fill)
	var pulse = 0.62 + 0.38 * sin(elapsed * 5.5)
	draw_circle(rect.get_center(), max(rect.size.x, rect.size.y) * (0.72 + pulse * 0.14), Color(glow.r, glow.g, glow.b, 0.12))
	draw_rect(rect, fill.darkened(0.36), true)
	draw_rect(rect, outline, false, 2.0)
	draw_circle(rect.position + Vector2(rect.size.x * 0.5, rect.size.y * 0.34), max(5.0, rect.size.x * 0.23), fill)
	draw_line(rect.get_center(), rect.get_center() + Vector2(-rect.size.x * 0.9, -rect.size.y * 0.65), Color(glow.r, glow.g, glow.b, 0.28), 2.0)
	draw_line(rect.get_center(), rect.get_center() + Vector2(rect.size.x * 0.9, -rect.size.y * 0.65), Color(glow.r, glow.g, glow.b, 0.28), 2.0)


func _draw_seesaw_plank_object(vertices: PackedVector2Array, object: Dictionary, skin: Dictionary) -> void:
	var fill = _skin_color_value(skin, "fill", Color(0.64, 0.38, 0.16, 0.96))
	var outline = _skin_color_value(skin, "outline", Color(1.0, 0.84, 0.45, 0.96))
	var stripe = _skin_color_value(skin, "stripe", Color(0.36, 0.20, 0.10, 0.95))
	var glow = _skin_color_value(skin, "glow", outline)
	draw_colored_polygon(vertices, Color(glow.r, glow.g, glow.b, 0.10))
	draw_colored_polygon(vertices, fill)
	var polyline = PackedVector2Array(vertices)
	polyline.append(vertices[0])
	draw_polyline(polyline, outline, 2.5)
	var rect = _bounds_for_vertices(vertices)
	var center = rect.get_center()
	var angle = 0.0
	var body: Dictionary = object.get("body", {})
	if body.has("angle"):
		angle = float(body.get("angle"))
	var axis = Vector2(cos(angle), -sin(angle))
	var normal = Vector2(-axis.y, axis.x)
	var half_len = max(rect.size.x, rect.size.y) * 0.48
	for offset in [-0.23, 0.23]:
		draw_line(center - axis * half_len + normal * offset * 17.0, center + axis * half_len + normal * offset * 17.0, stripe, 1.5)
	draw_circle(center - axis * half_len, 5.0, _palette_color("hot", Color(1.0, 0.45, 0.25)))
	draw_circle(center + axis * half_len, 5.0, _palette_color("secondary", Color(0.20, 1.0, 0.65)))


func _draw_seesaw_pivot_object(vertices: PackedVector2Array, skin: Dictionary) -> void:
	var rect = _bounds_for_vertices(vertices)
	var fill = _skin_color_value(skin, "fill", Color(0.36, 0.31, 0.26, 0.96))
	var outline = _skin_color_value(skin, "outline", Color(1.0, 0.86, 0.54, 0.96))
	var glow = _skin_color_value(skin, "glow", outline)
	var triangle = PackedVector2Array([
		Vector2(rect.position.x + rect.size.x * 0.5, rect.position.y - 5.0),
		Vector2(rect.position.x - 10.0, rect.position.y + rect.size.y),
		Vector2(rect.position.x + rect.size.x + 10.0, rect.position.y + rect.size.y),
	])
	draw_colored_polygon(triangle, Color(glow.r, glow.g, glow.b, 0.10))
	draw_colored_polygon(triangle, fill)
	triangle.append(triangle[0])
	draw_polyline(triangle, outline, 2.6)
	draw_circle(Vector2(rect.position.x + rect.size.x * 0.5, rect.position.y + rect.size.y * 0.18), max(4.0, rect.size.x * 0.14), outline)


func _draw_tree_object(vertices: PackedVector2Array, skin: Dictionary) -> void:
	var rect = _bounds_for_vertices(vertices)
	var fill = _skin_color_value(skin, "fill", Color(0.12, 0.48, 0.24, 0.96))
	var outline = _skin_color_value(skin, "outline", Color(0.51, 0.90, 0.48, 0.92))
	var trunk = _skin_color_value(skin, "trunk", Color(0.36, 0.22, 0.12, 0.96))
	var cx = rect.position.x + rect.size.x * 0.5
	var bottom = rect.position.y + rect.size.y
	var trunk_rect = Rect2(Vector2(cx - rect.size.x * 0.10, bottom - rect.size.y * 0.34), Vector2(rect.size.x * 0.20, rect.size.y * 0.34))
	draw_rect(trunk_rect, trunk)
	for layer in range(3):
		var y = bottom - rect.size.y * (0.22 + float(layer) * 0.25)
		var half = rect.size.x * (0.78 - float(layer) * 0.13)
		var points = PackedVector2Array([
			Vector2(cx, y - rect.size.y * 0.35),
			Vector2(cx - half, y + rect.size.y * 0.16),
			Vector2(cx + half, y + rect.size.y * 0.16),
		])
		draw_colored_polygon(points, fill.lightened(0.08 * float(layer)))
		var outline_points = PackedVector2Array(points)
		outline_points.append(points[0])
		draw_polyline(outline_points, outline, 1.5)


func _draw_cabin_object(vertices: PackedVector2Array, skin: Dictionary) -> void:
	var rect = _bounds_for_vertices(vertices)
	var fill = _skin_color_value(skin, "fill", Color(0.47, 0.28, 0.16, 0.96))
	var outline = _skin_color_value(skin, "outline", Color(1.0, 0.86, 0.59, 0.95))
	var roof = _skin_color_value(skin, "roof", Color(0.27, 0.14, 0.10, 0.98))
	var glow = _skin_color_value(skin, "glow", Color(1.0, 0.78, 0.32, 0.95))
	draw_rect(rect.grow(8.0), Color(glow.r, glow.g, glow.b, 0.10), false, 4.0)
	draw_rect(rect, fill)
	draw_rect(rect, outline, false, 2.0)
	var roof_points = PackedVector2Array([
		Vector2(rect.position.x - 8.0, rect.position.y + rect.size.y * 0.35),
		Vector2(rect.position.x + rect.size.x * 0.5, rect.position.y - rect.size.y * 0.28),
		Vector2(rect.position.x + rect.size.x + 8.0, rect.position.y + rect.size.y * 0.35),
	])
	draw_colored_polygon(roof_points, roof)
	roof_points.append(roof_points[0])
	draw_polyline(roof_points, outline, 2.0)
	var window = Rect2(rect.position + Vector2(rect.size.x * 0.58, rect.size.y * 0.42), Vector2(rect.size.x * 0.22, rect.size.y * 0.22))
	draw_rect(window.grow(7.0), Color(glow.r, glow.g, glow.b, 0.17), false, 3.0)
	draw_rect(window, glow)
	var door = Rect2(rect.position + Vector2(rect.size.x * 0.20, rect.size.y * 0.52), Vector2(rect.size.x * 0.20, rect.size.y * 0.48))
	draw_rect(door, fill.darkened(0.26))


func _bounds_for_vertices(vertices: PackedVector2Array) -> Rect2:
	if vertices.is_empty():
		return Rect2()
	var min_pos = vertices[0]
	var max_pos = vertices[0]
	for point in vertices:
		min_pos.x = min(min_pos.x, point.x)
		min_pos.y = min(min_pos.y, point.y)
		max_pos.x = max(max_pos.x, point.x)
		max_pos.y = max(max_pos.y, point.y)
	return Rect2(min_pos, max_pos - min_pos)


func _ellipse_points(rect: Rect2, segments: int = 24) -> PackedVector2Array:
	var points = PackedVector2Array()
	var center = rect.get_center()
	var rx = rect.size.x * 0.5
	var ry = rect.size.y * 0.5
	for index in range(max(8, segments)):
		var angle = TAU * float(index) / float(max(8, segments))
		points.append(center + Vector2(cos(angle) * rx, sin(angle) * ry))
	return points


func _skin_color_value(skin: Dictionary, key: String, fallback: Color) -> Color:
	var raw = skin.get(key, null)
	if typeof(raw) == TYPE_ARRAY and raw.size() >= 3:
		return Color(float(raw[0]) / 255.0, float(raw[1]) / 255.0, float(raw[2]) / 255.0, fallback.a)
	if typeof(raw) == TYPE_STRING:
		var text = str(raw)
		if text.begins_with("#") and text.length() == 7:
			return Color.html(text)
	return fallback


func _draw_ball_marks(center: Vector2, radius: float, color: Color) -> void:
	draw_arc(center, radius * 0.62, -0.9, 0.9, 16, Color(0.05, 0.10, 0.14, 0.42), 2.0)
	draw_arc(center, radius * 0.62, PI - 0.9, PI + 0.9, 16, Color(0.05, 0.10, 0.14, 0.42), 2.0)
	draw_line(center + Vector2(0, -radius * 0.72), center + Vector2(0, radius * 0.72), Color(0.05, 0.10, 0.14, 0.34), 1.6)


func _update_agent_action(input: Vector2) -> void:
	var nearest = _nearest_interactable_object()
	var zero_g = _agent_uses_freeflight_controls()
	if nearest != null:
		var target_pos = _object_center_world(nearest)
		if target_pos != Vector2.ZERO and abs(target_pos.x - agent_preview_position.x) > 1.0:
			agent_facing = 1.0 if target_pos.x >= agent_preview_position.x else -1.0
		if _object_is_throwable(nearest):
			agent_action = "throw"
			return
		if _object_is_kickable(nearest):
			agent_action = "kick"
			return
		if _object_is_dynamic(nearest):
			agent_action = "push"
			return
	if _agent_is_in_water():
		agent_action = "swim"
		return
	if zero_g:
		agent_action = "float"
	elif agent_preview_velocity.y > 80.0:
		agent_action = "jump"
	elif agent_preview_velocity.y < -80.0:
		agent_action = "fall"
	elif abs(agent_preview_velocity.x) > 35.0 or input.length() > 0.2:
		agent_action = "run"
	else:
		agent_action = "idle"


func _apply_agent_action_to_preview_objects(delta: float) -> void:
	var target = _nearest_interactable_object()
	if target == null:
		return
	if not (agent_action in ["kick", "push", "throw"]):
		return
	var name = str(target.get("name", ""))
	if name == "":
		return
	var target_pos = _object_center_world(target)
	var offset = target_pos - agent_preview_position
	if offset.length() <= 1.0 or offset.length() > _interaction_radius(target):
		return
	var axis = offset.normalized()
	if runtime_physics_ready and runtime_bodies.has(name) and runtime_bodies[name] is RigidBody2D:
		var rigid = runtime_bodies[name] as RigidBody2D
		var screen_axis = _world_axis_to_screen(axis)
		if agent_action == "push":
			rigid.apply_central_force(screen_axis * 13500.0)
		elif action_flash_time <= 0.0:
			var impulse = 230.0 if agent_action == "kick" else 190.0
			rigid.apply_central_impulse(screen_axis * impulse)
			action_flash_time = 0.22
			action_flash_kind = agent_action
			action_flash_position = target_pos
		return
	var velocity: Vector2 = object_preview_velocities.get(name, Vector2.ZERO)
	if agent_action == "push":
		velocity += axis * 190.0 * delta
	elif action_flash_time <= 0.0:
		if agent_action == "kick":
			velocity += axis * 245.0
		else:
			velocity += (axis + Vector2(0.0, 0.45)).normalized() * 215.0
		action_flash_time = 0.22
		action_flash_kind = agent_action
		action_flash_position = target_pos
	object_preview_velocities[name] = velocity.limit_length(420.0)


func _update_preview_object_motion(delta: float) -> void:
	for object in objects:
		if typeof(object) != TYPE_DICTIONARY:
			continue
		var name = str(object.get("name", ""))
		if name == "" or not object_preview_positions.has(name):
			continue
		if not _object_is_dynamic(object):
			continue
		var pos: Vector2 = object_preview_positions.get(name, Vector2.ZERO)
		var vel: Vector2 = object_preview_velocities.get(name, Vector2.ZERO)
		if gravity.length() > 30.0 and not _object_is_floating(object):
			vel.y += gravity.y * delta * 0.10
		pos += vel * delta
		pos.x = clamp(pos.x, 0.0, world_size.x)
		pos.y = clamp(pos.y, 0.0, world_size.y)
		vel *= pow(0.86, delta * 8.0)
		object_preview_positions[name] = pos
		object_preview_velocities[name] = vel


func _nearest_interactable_object():
	var best = null
	var best_distance = INF
	for object in objects:
		if typeof(object) != TYPE_DICTIONARY:
			continue
		if str(object.get("role", "")) == "agent":
			continue
		if not _object_is_dynamic(object):
			continue
		var distance = agent_preview_position.distance_to(_object_center_world(object))
		if distance < best_distance:
			best_distance = distance
			best = object
	if best != null and best_distance <= _interaction_radius(best):
		return best
	return null


func _interaction_radius(object: Dictionary) -> float:
	var radius = 95.0
	for shape in object.get("shapes", []):
		if typeof(shape) == TYPE_DICTIONARY and shape.has("radius"):
			radius = max(radius, float(shape.get("radius", 15.0)) * 4.0)
	return radius


func _object_is_dynamic(object: Dictionary) -> bool:
	var body: Dictionary = object.get("body", {})
	return str(body.get("type", "")) == "dynamic"


func _object_is_projectile(object: Dictionary) -> bool:
	var text = _object_text(object)
	return text.find("projectile") >= 0 or text.find("laser") >= 0 or text.find("shot") >= 0 or text.find("bolt") >= 0


func _object_is_recurring_falling_hazard(object: Dictionary) -> bool:
	var metadata: Dictionary = object.get("metadata", {})
	return bool(metadata.get("falling", false)) and str(metadata.get("kind", "")).to_lower().find("recurring_falling_hazard") >= 0


func _object_is_recurring_lateral_hazard(object: Dictionary) -> bool:
	var metadata: Dictionary = object.get("metadata", {})
	var text = _object_text(object)
	return bool(metadata.get("lateral", false)) and (
		str(metadata.get("kind", "")).to_lower().find("recurring_lateral_hazard") >= 0
		or str(metadata.get("recurring_lateral_hazard", "")) != ""
		or text.find("car_hazard_stream") >= 0
	)


func _world_time_step() -> float:
	var world: Dictionary = schema.get("world", {})
	return max(0.001, float(world.get("time_step", 1.0 / 60.0)))


func _object_is_chaser(object: Dictionary) -> bool:
	if not _object_is_dynamic(object) or _object_is_projectile(object):
		return false
	var role = str(object.get("role", "")).to_lower()
	if role == "agent":
		return false
	var text = _object_text(object) + " " + JSON.stringify(schema.get("objective", {})).to_lower()
	for token in [
		"chaser",
		"chase",
		"chasing",
		"pursuer",
		"pursue",
		"pursuing",
		"hunter",
		"hunt",
		"hunting",
		"stalker",
		"stalk",
		"stalking",
		"follower",
		"follow",
		"following",
		"runs_after",
		"runs after",
		"bear",
		"enemy_agent",
		"enemy agent",
		"angry_agent",
		"angry agent"
	]:
		if text.find(token) >= 0:
			return true
	return false


func _object_uses_area(object: Dictionary) -> bool:
	var role = str(object.get("role", "")).to_lower()
	if _object_is_dynamic(object) and _object_is_projectile(object):
		return false
	if role in ["goal", "trigger", "region", "sensor"]:
		return true
	var name = str(object.get("name", "")).to_lower()
	if name.find("goal") >= 0 or name.find("plate") >= 0 or name.find("sensor") >= 0 or name.find("exit") >= 0:
		return true
	var shapes: Array = object.get("shapes", [])
	for shape in shapes:
		if typeof(shape) == TYPE_DICTIONARY and bool(shape.get("sensor", false)):
			return true
	return false


func _object_is_floating(object: Dictionary) -> bool:
	return gravity.length() < 30.0 or _object_text(object).find("float") >= 0


func _agent_uses_freeflight_controls() -> bool:
	if gravity.length() < 30.0:
		return true
	var objective: Dictionary = schema.get("objective", {})
	var capability: Dictionary = objective.get("capability_profile", {})
	var text = (
		str(capability.get("movement", "")) + " " +
		str(capability.get("gravity", "")) + " " +
		str(capability.get("notes", ""))
	).to_lower()
	for token in ["thrust", "freeflight", "free_flight", "zero_g", "zero-g", "flying", "flight", "spaceship", "hover"]:
		if text.find(token) >= 0:
			return true
	return false


func _screen_gravity_strength() -> float:
	if gravity.length() < 30.0:
		return 0.0
	return max(120.0, abs(gravity.y))


func _water_runtime_velocity(current_velocity: Vector2, input: Vector2, delta: float) -> Vector2:
	var info = _water_info_at_agent()
	var surface_y = float(info.get("surface_y", agent_preview_position.y))
	var right_edge = float(info.get("right", world_size.x))
	var depth = max(0.0, surface_y - agent_preview_position.y)
	var waterline_y = surface_y - 10.0
	var near_surface = abs(agent_preview_position.y - waterline_y) <= 24.0
	var near_exit = agent_preview_position.x >= right_edge - 32.0
	var target_x = input.x * 185.0
	if input.x > 0.2:
		target_x += 62.0
	elif input.x == 0.0:
		target_x += 38.0
	var velocity = current_velocity
	velocity.x = lerp(velocity.x, target_x, clamp(delta * 4.8, 0.0, 1.0))
	velocity.y *= pow(0.18, delta)
	if agent_preview_position.y < waterline_y - 8.0:
		var buoyancy = 130.0 + min(300.0, depth * 4.7)
		velocity.y -= buoyancy * delta
	else:
		var surface_error = agent_preview_position.y - waterline_y
		velocity.y = lerp(velocity.y, surface_error * 2.8, clamp(delta * 5.0, 0.0, 1.0))
	if input.y > 0.25:
		velocity.y -= 95.0 * delta
	elif input.y < -0.25:
		velocity.y += 130.0 * delta
	if near_surface:
		velocity.y = clamp(velocity.y, -70.0, 80.0)
	if near_exit and (near_surface or input.y > 0.25):
		velocity.x = max(velocity.x, 235.0)
		velocity.y = min(velocity.y, -285.0)
	return velocity.limit_length(520.0)


func _water_info_at_agent() -> Dictionary:
	for object in objects:
		if typeof(object) != TYPE_DICTIONARY:
			continue
		var role = str(object.get("role", "")).to_lower()
		var text = _object_text(object)
		if role != "water" and text.find("swim_zone") < 0 and text.find("tropical_water") < 0:
			continue
		var surface_y = INF
		var left = INF
		var right = -INF
		var bottom = INF
		var top = -INF
		for shape in object.get("shapes", []):
			if typeof(shape) != TYPE_DICTIONARY:
				continue
			var bb = _shape_bounds_world(shape)
			if bb.size == Vector2.ZERO:
				continue
			left = min(left, bb.position.x)
			right = max(right, bb.position.x + bb.size.x)
			bottom = min(bottom, bb.position.y)
			top = max(top, bb.position.y + bb.size.y)
		var metadata: Dictionary = object.get("metadata", {})
		if metadata.has("surface_y"):
			surface_y = float(metadata.get("surface_y"))
		elif top > -INF:
			surface_y = top
		if left < INF and right > -INF:
			return {
				"surface_y": surface_y,
				"left": left,
				"right": right,
				"bottom": bottom,
				"top": top,
			}
	return {"surface_y": agent_preview_position.y, "left": 0.0, "right": world_size.x}


func _object_is_kickable(object: Dictionary) -> bool:
	var text = _object_text(object)
	return text.find("ball") >= 0 or text.find("puck") >= 0 or text.find("soccer") >= 0 or text.find("kick") >= 0 or text.find("shot") >= 0


func _object_is_throwable(object: Dictionary) -> bool:
	var text = _object_text(object)
	if text.find("ball") >= 0 and (_has_keyword("throw") or _has_keyword("basket")):
		return true
	return text.find("throw") >= 0 or text.find("projectile") >= 0 or text.find("toss") >= 0 or text.find("basket") >= 0


func _objective_targets() -> Array:
	var objective: Dictionary = schema.get("objective", {})
	var raw_targets: Array = objective.get("objective_targets", [])
	var targets: Array = []
	for target in raw_targets:
		targets.append(str(target).to_lower())
	var objective_profile: Dictionary = objective.get("objective_profile", {})
	var profile_targets: Array = objective_profile.get("targets", [])
	for target in profile_targets:
		var normalized = str(target).to_lower()
		if not (normalized in targets):
			targets.append(normalized)
	return targets


func _objective_touch_radius(object: Dictionary) -> float:
	var radius = 44.0
	for shape in object.get("shapes", []):
		if typeof(shape) == TYPE_DICTIONARY:
			if shape.has("radius"):
				radius = max(radius, float(shape.get("radius", 15.0)) + 28.0)
			elif shape.has("bb"):
				var bb: Dictionary = shape.get("bb", {})
				var width = abs(float(bb.get("right", 0.0)) - float(bb.get("left", 0.0)))
				var height = abs(float(bb.get("top", 0.0)) - float(bb.get("bottom", 0.0)))
				radius = max(radius, min(90.0, max(width, height) * 0.55 + 24.0))
	return radius


func _hazard_warning_radius(object: Dictionary) -> float:
	var radius = 42.0
	for shape in object.get("shapes", []):
		if typeof(shape) == TYPE_DICTIONARY and shape.has("radius"):
			radius = max(radius, float(shape.get("radius", 15.0)) + 34.0)
	return radius


func _hazard_hit_radius(object: Dictionary) -> float:
	var radius = 21.0
	for shape in object.get("shapes", []):
		if typeof(shape) == TYPE_DICTIONARY and shape.has("radius"):
			radius = max(radius, float(shape.get("radius", 8.0)) * 2.25)
	if _object_is_projectile(object):
		radius = max(radius, 26.0)
	return radius


func _draw_hud() -> void:
	var verification: Dictionary = schema.get("verification", {})
	var accepted = bool(verification.get("accepted", false))
	var text = "TIER %s %s" % [str(verification.get("achieved_tier", "?")), str(verification.get("tier_name", "unknown"))]
	var pos = Vector2(24, get_viewport_rect().size.y - 42)
	draw_string(ThemeDB.fallback_font, pos, text, HORIZONTAL_ALIGNMENT_LEFT, -1, 16, TEXT_GOOD if accepted else TEXT_WARN)
	var physics_text = "GODOT PHYSICS: ON" if runtime_physics_ready else "GODOT PHYSICS: PREVIEW"
	draw_string(ThemeDB.fallback_font, pos + Vector2(0, -22), physics_text, HORIZONTAL_ALIGNMENT_LEFT, -1, 14, TEXT_GOOD if runtime_physics_ready else TEXT_WARN)
	var objective_color = TEXT_GOOD if runtime_objective_satisfied else (TEXT_WARN if runtime_hazard_alert else Color(0.80, 0.94, 1.0))
	var objective_text = "OBJECTIVE: COMPLETE" if runtime_objective_satisfied else "OBJECTIVE: ACTIVE"
	if runtime_hazard_alert and not runtime_objective_satisfied:
		objective_text = "OBJECTIVE: ACTIVE / HAZARD NEAR"
	draw_string(ThemeDB.fallback_font, Vector2(24, 88), objective_text, HORIZONTAL_ALIGNMENT_LEFT, -1, 18, objective_color)
	var duration = _survival_duration_seconds()
	if duration > 0.0:
		var remaining = max(0.0, duration - runtime_survival_elapsed)
		var timer_color = TEXT_GOOD if runtime_objective_satisfied else _palette_color("hot", ACTION_ORANGE)
		var timer_panel = Rect2(Vector2(get_viewport_rect().size.x - 258.0, 24.0), Vector2(218.0, 62.0))
		draw_rect(timer_panel, Color(0.02, 0.05, 0.08, 0.72), true)
		draw_rect(timer_panel, Color(timer_color.r, timer_color.g, timer_color.b, 0.65), false, 2.0)
		draw_string(ThemeDB.fallback_font, timer_panel.position + Vector2(20.0, 39.0), "SURVIVE", HORIZONTAL_ALIGNMENT_LEFT, -1, 13, _palette_color("primary", Color(0.55, 0.92, 1.0)))
		draw_string(ThemeDB.fallback_font, timer_panel.position + Vector2(20.0, 64.0), "%04.1fs" % remaining, HORIZONTAL_ALIGNMENT_LEFT, -1, 25, timer_color)
	if runtime_feedback_message != "":
		var alpha = 1.0 if runtime_feedback_time > 0.0 else 0.58
		var feedback_color = Color(objective_color.r, objective_color.g, objective_color.b, alpha)
		draw_string(ThemeDB.fallback_font, Vector2(24, 113), runtime_feedback_message, HORIZONTAL_ALIGNMENT_LEFT, -1, 14, feedback_color)
	_draw_controls_overlay()


func _draw_controls_overlay() -> void:
	var lines = _control_lines()
	if lines.is_empty():
		return
	var rect = get_viewport_rect()
	var width = 390.0
	var row_height = 19.0
	var height = 48.0 + row_height * float(lines.size())
	var panel = Rect2(Vector2(rect.size.x - width - 24.0, 82.0), Vector2(width, height))
	var primary = _palette_color("primary", Color(0.30, 0.92, 1.0))
	draw_rect(panel, Color(0.02, 0.05, 0.06, 0.72), true)
	draw_rect(panel, Color(primary.r, primary.g, primary.b, 0.56), false, 1.4)
	draw_string(ThemeDB.fallback_font, panel.position + Vector2(16, 25), "PLAYER CONTROLS", HORIZONTAL_ALIGNMENT_LEFT, -1, 15, primary.lightened(0.18))
	var y = panel.position.y + 50.0
	for line in lines:
		draw_string(ThemeDB.fallback_font, Vector2(panel.position.x + 16.0, y), str(line), HORIZONTAL_ALIGNMENT_LEFT, -1, 13, Color(0.82, 0.95, 0.97, 0.92))
		y += row_height


func _control_lines() -> Array:
	var lines: Array = []
	var zero_g = _agent_uses_freeflight_controls()
	if zero_g:
		lines.append("Move: WASD / Arrow Keys steer in zero gravity")
	else:
		lines.append("Move: A/D or Arrow Left/Right; W/Up jumps only when grounded")
	if _has_water_context():
		lines.append("Water: fall in, A/D swim, W/Up floats and climbs out")
	lines.append("Reset: R")
	if _has_human_interaction_context():
		lines.append("Action: move into objects to push / kick / throw")
	if _has_hazard_context():
		lines.append("Hazards: dodge red/fire objects; warning appears nearby")
	if _has_space_battle_context():
		lines.append("Space battle: thrust between glowing bolts until the timer clears")
	lines.append("Current pose: %s" % agent_action.to_upper())
	var objective = _objective_summary()
	if objective != "":
		lines.append("Goal: %s" % objective)
	return lines


func _has_water_context() -> bool:
	if _has_keyword("water") or _has_keyword("swim"):
		return true
	for object in objects:
		if typeof(object) != TYPE_DICTIONARY:
			continue
		var role = str(object.get("role", "")).to_lower()
		var text = _object_text(object)
		if role == "water" or text.find("swim_zone") >= 0 or text.find("tropical_water") >= 0:
			return true
	return false


func _has_human_interaction_context() -> bool:
	var objective_text = JSON.stringify(schema.get("objective", {})).to_lower()
	if objective_text.find("push") >= 0 or objective_text.find("kick") >= 0 or objective_text.find("throw") >= 0 or objective_text.find("move_object") >= 0:
		return true
	for object in objects:
		if typeof(object) != TYPE_DICTIONARY:
			continue
		if _object_is_dynamic(object) and str(object.get("role", "")).to_lower() != "agent":
			return true
	return false


func _has_hazard_context() -> bool:
	if _has_keyword("fire") or _has_keyword("lava") or _has_keyword("hazard"):
		return true
	for object in objects:
		if typeof(object) != TYPE_DICTIONARY:
			continue
		var role = str(object.get("role", "")).to_lower()
		if role in ["hazard", "danger"] or _shape_is_fire(object):
			return true
	return false


func _has_space_battle_context() -> bool:
	if _has_keyword("space") or _has_keyword("spaceship") or _has_keyword("laser"):
		return true
	var text = (JSON.stringify(schema.get("objective", {})) + " " + JSON.stringify(visual_brief)).to_lower()
	return text.find("space") >= 0 or text.find("ship") >= 0 or text.find("laser") >= 0 or text.find("projectile") >= 0


func _survival_duration_seconds() -> float:
	var objective: Dictionary = schema.get("objective", {})
	var profile: Dictionary = objective.get("objective_profile", {})
	var subgoal_sources: Array = []
	if objective.has("subgoals"):
		subgoal_sources.append_array(objective.get("subgoals", []))
	if profile.has("subgoals"):
		subgoal_sources.append_array(profile.get("subgoals", []))
	for item in subgoal_sources:
		if typeof(item) != TYPE_DICTIONARY:
			continue
		var kind = str(item.get("kind", "")).to_lower()
		if kind.find("survive") < 0 and kind.find("duration") < 0:
			continue
		if item.has("duration_seconds"):
			return max(0.0, float(item.get("duration_seconds")))
		if item.has("duration_steps"):
			return max(0.0, float(item.get("duration_steps")) / 60.0)
	var text = JSON.stringify(objective).to_lower()
	if text.find("survive") >= 0:
		return 8.0
	return 0.0


func _objective_summary() -> String:
	var objective: Dictionary = schema.get("objective", {})
	var profile: Dictionary = objective.get("objective_profile", {})
	var description = str(profile.get("objective_description", ""))
	if description == "":
		description = str(objective.get("objective_type", ""))
	return _shorten_text(description, 58)


func _shorten_text(text: String, limit: int) -> String:
	if text.length() <= limit:
		return text
	return text.substr(0, max(0, limit - 3)) + "..."


func _draw_error(message: String) -> void:
	draw_string(ThemeDB.fallback_font, Vector2(42, 92), message, HORIZONTAL_ALIGNMENT_LEFT, -1, 18, TEXT_WARN)


func _draw_starfield() -> void:
	for index in range(90):
		var x = fmod(float(index * 137), get_viewport_rect().size.x)
		var y = fmod(float(index * 59), get_viewport_rect().size.y)
		var alpha = 0.35 + 0.35 * sin(Time.get_ticks_msec() * 0.001 + index)
		draw_circle(Vector2(x, y), 1.0 + float(index % 3) * 0.35, Color(0.7, 0.9, 1.0, alpha))
	for index in range(4):
		var center = Vector2(
			fmod(float(index * 251) + elapsed * (6.0 + index), get_viewport_rect().size.x),
			fmod(float(index * 137) + 80.0, get_viewport_rect().size.y)
		)
		draw_circle(center, 80.0 + float(index) * 22.0, Color(0.35, 0.14, 0.9, 0.035))


func _draw_lava_wash() -> void:
	for index in range(10):
		var y = get_viewport_rect().size.y - float(index) * 34.0
		var alpha = 0.035 + float(index) * 0.004
		draw_rect(Rect2(Vector2(0, y), Vector2(get_viewport_rect().size.x, 26)), Color(1.0, 0.22, 0.05, alpha), true)
	for index in range(7):
		var y = get_viewport_rect().size.y - 60.0 - float(index) * 26.0
		var sway = sin(elapsed * 1.7 + float(index)) * 18.0
		draw_line(Vector2(sway, y), Vector2(get_viewport_rect().size.x + sway, y - 22.0), Color(1.0, 0.38, 0.08, 0.045), 5.0)


func _draw_retro_backdrop() -> void:
	var rect = get_viewport_rect()
	for y in range(0, int(rect.size.y), 6):
		draw_line(Vector2(0, y), Vector2(rect.size.x, y), Color(1.0, 0.3, 0.95, 0.055), 1.0)
	for index in range(80):
		var x = fmod(float(index * 67), rect.size.x)
		var y = fmod(float(index * 41) + elapsed * 15.0, rect.size.y)
		var spark = _palette_color("secondary", Color(1.0, 0.3, 0.9))
		draw_rect(Rect2(Vector2(x, y), Vector2(2, 2)), Color(spark.r, spark.g, spark.b, 0.22), true)


func _draw_organic_maze_wash() -> void:
	var rect = get_viewport_rect()
	var color = _palette_color("secondary", Color(0.34, 0.85, 0.45))
	for index in range(18):
		var x = fmod(float(index * 83) + sin(elapsed + index) * 12.0, rect.size.x)
		draw_line(Vector2(x, 0), Vector2(x - 34.0, rect.size.y), Color(color.r, color.g, color.b, 0.055), 3.0)


func _draw_field_backdrop() -> void:
	var rect = get_viewport_rect()
	var primary = _palette_color("primary", Color(0.32, 0.9, 1.0))
	for index in range(12):
		var x = float(index) * rect.size.x / 11.0
		var offset = sin(elapsed * 1.5 + float(index)) * 28.0
		draw_line(Vector2(x + offset, 0), Vector2(x - offset, rect.size.y), Color(primary.r, primary.g, primary.b, 0.06), 2.0)


func _draw_lab_wash() -> void:
	var rect = get_viewport_rect()
	draw_circle(rect.size * 0.22, 240.0, Color(0.1, 0.8, 1.0, 0.035))
	draw_circle(rect.size * 0.78, 260.0, Color(0.5, 0.2, 1.0, 0.028))


func _object_color(object: Dictionary) -> Color:
	var role = str(object.get("role", "")).to_lower()
	var name = str(object.get("name", "")).to_lower()
	var metadata = JSON.stringify(object.get("metadata", {})).to_lower()
	var skin = _skin_for_object(object)
	var material = str(skin.get("material", "")).to_lower()
	if role == "agent":
		return _contrast_color(_palette_color("primary", Color(0.30, 0.92, 1.0)), 0.42)
	if skin.has("fill"):
		return _skin_color_value(skin, "fill", _contrast_color(_palette_color("secondary", Color(0.55, 0.72, 0.78)), 0.28))
	if role in ["goal", "trigger", "region"] or name.find("goal") >= 0:
		return _contrast_color(Color(0.18, 1.0, 0.58, 0.84), 0.36)
	if role in ["hazard", "danger"] or name.find("fire") >= 0 or metadata.find("lava") >= 0:
		return _contrast_color(Color(1.0, 0.22, 0.08, 0.92), 0.40)
	if role in ["object"] or name.find("ball") >= 0:
		if material.find("stone") >= 0 or name.find("rock") >= 0:
			return Color(0.62, 0.68, 0.72, 0.96)
		return Color(0.88, 0.94, 1.0, 0.94)
	if role in ["support", "terrain"] or name.find("wall") >= 0:
		if material.find("neon") >= 0:
			return _palette_color("secondary", Color(0.85, 0.28, 1.0))
		if _has_keyword("lava") or material.find("basalt") >= 0:
			return Color(0.20, 0.11, 0.12, 0.96)
		return _palette_color("secondary", Color(0.55, 0.72, 0.78)).darkened(0.10)
	return _contrast_color(_palette_color("hot", Color(1.0, 0.36, 0.48)), 0.32)


func _skin_for_object(object: Dictionary) -> Dictionary:
	var name = str(object.get("name", "")).to_lower()
	var role = str(object.get("role", "")).to_lower()
	for pattern in object_skins.keys():
		var key = str(pattern).to_lower()
		var skin = object_skins[pattern]
		if typeof(skin) != TYPE_DICTIONARY:
			continue
		if key == name or key == role:
			return skin
		if key.begins_with("*") and key.ends_with("*"):
			var needle = key.substr(1, key.length() - 2)
			if name.find(needle) >= 0 or role.find(needle) >= 0:
				return skin
		elif key.begins_with("*"):
			var suffix = key.substr(1)
			if name.ends_with(suffix):
				return skin
		elif key.ends_with("*"):
			var prefix = key.substr(0, key.length() - 1)
			if name.begins_with(prefix):
				return skin
	return {}


func _skin_has_fx(skin: Dictionary, fx_name: String) -> bool:
	var fx: Array = skin.get("fx", [])
	return fx_name in fx


func _shape_is_fire(object: Dictionary) -> bool:
	var text = _object_text(object)
	return text.find("fire") >= 0 or text.find("lava") >= 0 or text.find("ember") >= 0


func _shape_is_ball(object: Dictionary) -> bool:
	var text = _object_text(object)
	return text.find("ball") >= 0 or text.find("puck") >= 0 or text.find("marble") >= 0


func _object_center_screen(object: Dictionary) -> Vector2:
	return _world_to_screen(_object_center_world(object))


func _object_center_world(object: Dictionary) -> Vector2:
	var name = str(object.get("name", ""))
	if name != "" and object_preview_positions.has(name):
		return object_preview_positions[name]
	var body: Dictionary = object.get("body", {})
	var pos = _vec(body.get("position", []))
	if pos != Vector2.ZERO:
		return pos
	var shapes: Array = object.get("shapes", [])
	for shape in shapes:
		if typeof(shape) != TYPE_DICTIONARY:
			continue
		if shape.has("center"):
			return _vec(shape.get("center", [0.0, 0.0]))
		if shape.has("bb"):
			var bb: Dictionary = shape.get("bb", {})
			if bb.has("left") and bb.has("right") and bb.has("top") and bb.has("bottom"):
				return Vector2(
					(float(bb.get("left")) + float(bb.get("right"))) * 0.5,
					(float(bb.get("bottom")) + float(bb.get("top"))) * 0.5
				)
	return Vector2.ZERO


func _agent_is_in_water() -> bool:
	for object in objects:
		if typeof(object) != TYPE_DICTIONARY:
			continue
		var role = str(object.get("role", "")).to_lower()
		var text = _object_text(object)
		if role != "water" and text.find("swim_zone") < 0 and text.find("water") < 0:
			continue
		var shapes: Array = object.get("shapes", [])
		for shape in shapes:
			if typeof(shape) != TYPE_DICTIONARY:
				continue
			var bb = _shape_bounds_world(shape)
			if bb.size == Vector2.ZERO:
				continue
			if agent_preview_position.x >= bb.position.x and agent_preview_position.x <= bb.position.x + bb.size.x and agent_preview_position.y >= bb.position.y and agent_preview_position.y <= bb.position.y + bb.size.y + 54.0:
				return true
	return false


func _shape_bounds_world(shape: Dictionary) -> Rect2:
	if shape.has("bb"):
		var bb: Dictionary = shape.get("bb", {})
		if bb.has("left") and bb.has("right") and bb.has("top") and bb.has("bottom"):
			var left = float(bb.get("left"))
			var right = float(bb.get("right"))
			var bottom = float(bb.get("bottom"))
			var top = float(bb.get("top"))
			return Rect2(Vector2(min(left, right), min(bottom, top)), Vector2(abs(right - left), abs(top - bottom)))
	if shape.has("center") and shape.has("radius"):
		var c = _vec(shape.get("center", [0.0, 0.0]))
		var r = float(shape.get("radius", 0.0))
		return Rect2(c - Vector2(r, r), Vector2(r * 2.0, r * 2.0))
	if shape.has("vertices"):
		var vertices: Array = shape.get("vertices", [])
		if vertices.is_empty():
			return Rect2()
		var min_pos = _vec(vertices[0])
		var max_pos = min_pos
		for raw in vertices:
			var point = _vec(raw)
			min_pos.x = min(min_pos.x, point.x)
			min_pos.y = min(min_pos.y, point.y)
			max_pos.x = max(max_pos.x, point.x)
			max_pos.y = max(max_pos.y, point.y)
		return Rect2(min_pos, max_pos - min_pos)
	return Rect2()


func _object_preview_offset(object: Dictionary) -> Vector2:
	var name = str(object.get("name", ""))
	if name == "" or not object_preview_positions.has(name):
		return Vector2.ZERO
	var body: Dictionary = object.get("body", {})
	var source = _vec(body.get("position", []))
	if source == Vector2.ZERO:
		source = _object_source_center_world(object)
	return object_preview_positions[name] - source


func _object_source_center_world(object: Dictionary) -> Vector2:
	var body: Dictionary = object.get("body", {})
	var pos = _vec(body.get("position", []))
	if pos != Vector2.ZERO:
		return pos
	var shapes: Array = object.get("shapes", [])
	for shape in shapes:
		if typeof(shape) != TYPE_DICTIONARY:
			continue
		if shape.has("center"):
			return _vec(shape.get("center", [0.0, 0.0]))
		if shape.has("bb"):
			var bb: Dictionary = shape.get("bb", {})
			if bb.has("left") and bb.has("right") and bb.has("top") and bb.has("bottom"):
				return Vector2(
					(float(bb.get("left")) + float(bb.get("right"))) * 0.5,
					(float(bb.get("bottom")) + float(bb.get("top"))) * 0.5
				)
	return Vector2.ZERO


func _object_text(object: Dictionary) -> String:
	return ("%s %s %s" % [
		str(object.get("name", "")),
		str(object.get("role", "")),
		JSON.stringify(object.get("metadata", {})),
	]).to_lower()


func _object_by_name(name: String):
	var normalized = name.to_lower()
	for object in objects:
		if typeof(object) != TYPE_DICTIONARY:
			continue
		if str(object.get("name", "")).to_lower() == normalized:
			return object
	return null


func _palette_color(name: String, fallback: Color) -> Color:
	var palette: Dictionary = visual_brief.get("palette", {})
	var raw = palette.get(name, null)
	if typeof(raw) == TYPE_ARRAY and raw.size() >= 3:
		return Color(float(raw[0]) / 255.0, float(raw[1]) / 255.0, float(raw[2]) / 255.0, 1.0)
	return fallback


func _contrast_color(color: Color, minimum: float) -> Color:
	var background = _palette_color("background_color", Color(0.04, 0.07, 0.09))
	if _luminance_delta(color, background) >= minimum:
		return color
	var bright = color.lerp(Color(1.0, 1.0, 1.0, color.a), 0.72)
	if _luminance_delta(bright, background) >= minimum:
		return bright
	return color.lerp(Color(0.0, 0.0, 0.0, color.a), 0.72)


func _luminance_delta(a: Color, b: Color) -> float:
	return abs(_relative_luminance(a) - _relative_luminance(b))


func _relative_luminance(color: Color) -> float:
	return 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b


func _program_color(item: Dictionary, fallback: Color) -> Color:
	var role = str(item.get("color_role", ""))
	if role == "background":
		return _palette_color("background_color", fallback)
	if role == "primary":
		return _palette_color("primary", fallback)
	if role == "secondary":
		return _palette_color("secondary", fallback)
	if role == "hazard" or role == "hot":
		return _palette_color("hot", fallback)
	if role == "shadow":
		var recipe: Dictionary = visual_brief.get("recipe", {})
		var palette: Dictionary = recipe.get("palette", {})
		var shadow = palette.get("shadow", null)
		if typeof(shadow) == TYPE_STRING:
			return _hex_to_color(str(shadow), fallback)
	return fallback


func _hex_to_color(value: String, fallback: Color) -> Color:
	var text = value.strip_edges()
	if text.begins_with("#"):
		text = text.substr(1)
	if text.length() != 6:
		return fallback
	var r = text.substr(0, 2).hex_to_int()
	var g = text.substr(2, 2).hex_to_int()
	var b = text.substr(4, 2).hex_to_int()
	return Color(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, 1.0)


func _program_seed() -> int:
	var seed = visual_program.get("seed", null)
	if typeof(seed) == TYPE_INT:
		return int(seed)
	var recipe: Dictionary = visual_brief.get("recipe", {})
	seed = recipe.get("seed", 1701)
	if typeof(seed) == TYPE_INT:
		return int(seed)
	if typeof(seed) == TYPE_FLOAT:
		return int(seed)
	return 1701


func _has_keyword(keyword: String) -> bool:
	var context: Dictionary = visual_brief.get("semantic_context", {})
	var keywords: Array = context.get("keywords", [])
	return keyword in keywords


func _world_to_screen(point: Vector2) -> Vector2:
	return Vector2(
		world_offset.x + point.x * world_scale,
		world_offset.y + (world_size.y - point.y) * world_scale
	)


func _screen_to_world(point: Vector2) -> Vector2:
	return Vector2(
		(point.x - world_offset.x) / max(world_scale, 0.0001),
		world_size.y - ((point.y - world_offset.y) / max(world_scale, 0.0001))
	)


func _world_axis_to_screen(axis: Vector2) -> Vector2:
	var screen_axis = Vector2(axis.x, -axis.y)
	if screen_axis.length() <= 0.0001:
		return Vector2.RIGHT
	return screen_axis.normalized()


func _vec(value) -> Vector2:
	if typeof(value) == TYPE_ARRAY and value.size() >= 2:
		return Vector2(float(value[0]), float(value[1]))
	return Vector2.ZERO
