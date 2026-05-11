extends Node2D

const WorldLoaderScript = preload("res://scripts/WorldLoader.gd")

var loader: Node
var status_label: Label
var schema_path = ""


func _ready() -> void:
	schema_path = _schema_path_from_args()
	_build_ui()
	loader = WorldLoaderScript.new()
	add_child(loader)
	loader.load_world(schema_path)
	_update_status()


func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_SIZE_CHANGED and loader:
		loader.queue_redraw()


func _build_ui() -> void:
	var canvas = CanvasLayer.new()
	add_child(canvas)

	status_label = Label.new()
	status_label.position = Vector2(24, 18)
	status_label.add_theme_font_size_override("font_size", 16)
	status_label.add_theme_color_override("font_color", Color(0.83, 0.97, 0.98))
	status_label.add_theme_color_override("font_shadow_color", Color(0.0, 0.0, 0.0, 0.8))
	status_label.add_theme_constant_override("shadow_offset_x", 2)
	status_label.add_theme_constant_override("shadow_offset_y", 2)
	canvas.add_child(status_label)


func _update_status() -> void:
	var tier = "unverified"
	var accepted = false
	var mood = "unknown"
	if loader.schema.has("verification"):
		var verification: Dictionary = loader.schema["verification"]
		tier = "%s / tier %s" % [str(verification.get("tier_name", "unknown")), str(verification.get("achieved_tier", "?"))]
		accepted = bool(verification.get("accepted", false))
	if loader.visual_brief:
		mood = str(loader.visual_brief.get("mood", "unknown"))
	status_label.text = "HARNESS ALPHA GODOT RUNTIME\n%s\naccepted=%s | mood=%s\nWASD / arrows control the runtime agent | R resets" % [
		schema_path,
		str(accepted),
		mood,
	]


func _schema_path_from_args() -> String:
	var args = OS.get_cmdline_user_args()
	for arg in args:
		if arg.begins_with("--world-schema="):
			return arg.trim_prefix("--world-schema=")
		if arg.ends_with(".json"):
			return arg

	var env_path = OS.get_environment("HARNESS_WORLD_SCHEMA")
	if env_path != "":
		return env_path

	return ProjectSettings.globalize_path("res://../exports/latest_world/world_schema.json")
