"""Blender animation capture for NEO VIDEO GEN.

Example:
blender --background scene.blend --python scripts/blender_capture.py -- --output ./neo_data/scene_001 --start 1 --end 9 --profile FULL
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

PASS_FLAGS = {
    "depth": ("use_pass_z", "view"),
    "normal": ("use_pass_normal", "view"),
    "position": ("use_pass_position", "view"),
    "vector": ("use_pass_vector", "view"),
    "uv": ("use_pass_uv", "view"),
    "object_index": ("use_pass_object_index", "view"),
    "material_index": ("use_pass_material_index", "view"),
    "diffuse_direct": ("use_pass_diffuse_direct", "view"),
    "diffuse_indirect": ("use_pass_diffuse_indirect", "view"),
    "diffuse_color": ("use_pass_diffuse_color", "view"),
    "glossy_direct": ("use_pass_glossy_direct", "view"),
    "glossy_indirect": ("use_pass_glossy_indirect", "view"),
    "glossy_color": ("use_pass_glossy_color", "view"),
    "transmission_direct": ("use_pass_transmission_direct", "view"),
    "transmission_indirect": ("use_pass_transmission_indirect", "view"),
    "transmission_color": ("use_pass_transmission_color", "view"),
    "emit": ("use_pass_emit", "view"),
    "environment": ("use_pass_environment", "view"),
    "shadow": ("use_pass_shadow", "view"),
    "ambient_occlusion": ("use_pass_ambient_occlusion", "view"),
    "cryptomatte_object": ("use_pass_cryptomatte_object", "cycles"),
    "cryptomatte_material": ("use_pass_cryptomatte_material", "cycles"),
    "cryptomatte_asset": ("use_pass_cryptomatte_asset", "cycles"),
}


def as_list(value):
    if hasattr(value, "to_list"):
        return value.to_list()
    try:
        return [as_list(v) for v in value]
    except TypeError:
        return value


def matrix_rows(matrix):
    return [[float(value) for value in row] for row in matrix]


def capture_object(obj):
    item = {
        "name": obj.name,
        "type": obj.type,
        "data_name": getattr(obj.data, "name", None),
        "world_matrix": matrix_rows(obj.matrix_world),
        "location": as_list(obj.matrix_world.translation),
        "rotation": as_list(obj.matrix_world.to_euler()),
        "scale": as_list(obj.matrix_world.to_scale()),
        "dimensions": as_list(obj.dimensions),
        "parent": obj.parent.name if obj.parent else None,
        "materials": [slot.material.name if slot.material else None for slot in obj.material_slots],
    }
    if obj.type == "MESH":
        item["mesh"] = {
            "vertices": len(obj.data.vertices),
            "edges": len(obj.data.edges),
            "polygons": len(obj.data.polygons),
            "bounds_world": [as_list(obj.matrix_world @ Vector(corner)) for corner in obj.bound_box],
        }
    if obj.type == "ARMATURE":
        item["bones"] = [
            {
                "name": bone.name,
                "parent": bone.parent.name if bone.parent else None,
                "head_world": as_list(obj.matrix_world @ bone.head_local),
                "tail_world": as_list(obj.matrix_world @ bone.tail_local),
            }
            for bone in obj.data.bones
        ]
        item["pose_bones"] = [
            {
                "name": bone.name,
                "matrix_world": matrix_rows(obj.matrix_world @ bone.matrix),
                "rotation_quaternion": as_list(bone.rotation_quaternion),
            }
            for bone in obj.pose.bones
        ]
    if obj.type == "CAMERA":
        cam = obj.data
        item["camera"] = {
            "lens_mm": cam.lens,
            "sensor_width_mm": cam.sensor_width,
            "sensor_height_mm": cam.sensor_height,
            "sensor_fit": cam.sensor_fit,
            "clip_start": cam.clip_start,
            "clip_end": cam.clip_end,
            "dof_enabled": cam.dof.use_dof,
            "focus_distance": cam.dof.focus_distance,
            "f_stop": cam.dof.aperture_fstop,
            "projection_matrix": matrix_rows(cam.calc_matrix_camera(
                bpy.context.evaluated_depsgraph_get(),
                x=bpy.context.scene.render.resolution_x,
                y=bpy.context.scene.render.resolution_y,
                scale_x=bpy.context.scene.render.pixel_aspect_x,
                scale_y=bpy.context.scene.render.pixel_aspect_y,
            )),
        }
    return item


def enable_passes(layer):
    enabled, unavailable = [], []
    for name, (attribute, owner) in PASS_FLAGS.items():
        target = layer.cycles if owner == "cycles" and hasattr(layer, "cycles") else layer
        if not hasattr(target, attribute):
            unavailable.append(name)
            continue
        try:
            setattr(target, attribute, True)
            enabled.append(name)
        except (TypeError, RuntimeError):
            unavailable.append(name)
    return enabled, unavailable


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--profile", choices=["FULL", "CORE"], default="FULL")
    args = parser.parse_args(argv)

    scene = bpy.context.scene
    start = args.start if args.start is not None else scene.frame_start
    end = args.end if args.end is not None else scene.frame_end
    output = Path(args.output).expanduser().resolve()
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    enabled, unavailable = enable_passes(bpy.context.view_layer)

    scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec = "ZIP"
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True

    fps = scene.render.fps / scene.render.fps_base
    manifest = {
        "scene_id": output.name,
        "engine": "blender",
        "engine_version": bpy.app.version_string,
        "schema_version": "0.1.0",
        "capture_profile": args.profile,
        "coordinate_system": "Blender world coordinates; right-handed; Z-up",
        "units": scene.unit_settings.system,
        "scene_metadata": {
            "render_engine": scene.render.engine,
            "resolution": [scene.render.resolution_x, scene.render.resolution_y],
            "fps": fps,
            "requested_channels": list(PASS_FLAGS),
        },
        "frames": [],
        "assets": [],
        "available_channels": enabled,
        "unavailable_channels": unavailable,
        "notes": ["Pass availability depends on Blender version and render engine."],
    }

    for frame in range(start, end + 1):
        scene.frame_set(frame)
        stem = f"frame_{frame:06d}"
        scene.render.filepath = str(frames_dir / stem)
        bpy.ops.render.render(write_still=True)
        timestamp = (frame - scene.frame_start) / fps
        state = {
            "frame_index": frame,
            "timestamp_seconds": timestamp,
            "camera": scene.camera.name if scene.camera else None,
            "objects": [capture_object(obj) for obj in scene.objects],
            "world": {"name": scene.world.name if scene.world else None} if scene.world else None,
        }
        state_file = frames_dir / f"{stem}.json"
        state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
        manifest["frames"].append({
            "frame_index": frame,
            "timestamp_seconds": timestamp,
            "passes": {"multilayer_exr": f"frames/{stem}.exr"},
            "state_file": f"frames/{stem}.json",
            "available_channels": enabled,
            "unavailable_channels": unavailable,
        })

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"NEO capture finished: {output}")


if __name__ == "__main__":
    main()
