"""Capture Blender render passes, evaluated geometry and per-frame scene state.

Run:
blender --background scene.blend --python scripts/blender_capture.py -- --output ./neo_data/scene_001 --start 1 --end 9 --profile FULL
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any

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


def plain(value: Any) -> Any:
    if hasattr(value, "to_list"):
        return value.to_list()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return [plain(item) for item in value]
    except TypeError:
        return str(value)


def matrix_rows(matrix) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:100] or "object"


def capture_material(mat) -> dict[str, Any]:
    result = {
        "name": mat.name,
        "diffuse_color": plain(mat.diffuse_color),
        "roughness": getattr(mat, "roughness", None),
        "metallic": getattr(mat, "metallic", None),
        "blend_method": getattr(mat, "surface_render_method", None),
    }
    if mat.use_nodes and mat.node_tree:
        result["nodes"] = [
            {"name": node.name, "type": node.bl_idname}
            for node in mat.node_tree.nodes
        ]
        principled = next(
            (node for node in mat.node_tree.nodes if node.bl_idname == "ShaderNodeBsdfPrincipled"),
            None,
        )
        if principled:
            for socket_name in ("Base Color", "Metallic", "Roughness", "Alpha", "IOR"):
                socket = principled.inputs.get(socket_name)
                if socket and socket.is_linked is False:
                    result[socket_name.lower().replace(" ", "_")] = plain(socket.default_value)
    return result


def export_evaluated_mesh(obj, geometry_dir: Path, frame_stem: str) -> dict[str, Any]:
    """Export evaluated mesh geometry using only Blender's standard Python runtime."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        vertices_local = [list(v.co) for v in mesh.vertices]
        vertices_world = [list(evaluated.matrix_world @ v.co) for v in mesh.vertices]
        normal_matrix = evaluated.matrix_world.to_3x3().inverted().transposed()
        normals_world = []
        for vertex in mesh.vertices:
            normal = normal_matrix @ vertex.normal
            if normal.length:
                normal.normalize()
            normals_world.append(list(normal))
        face_offsets = [0]
        face_indices = []
        face_materials = []
        face_normals = []
        for polygon in mesh.polygons:
            face_indices.extend(int(index) for index in polygon.vertices)
            face_offsets.append(len(face_indices))
            face_materials.append(int(polygon.material_index))
            face_normals.append(list(polygon.normal))
        edge_indices = [list(edge.vertices) for edge in mesh.edges]
        uv_layer = mesh.uv_layers.active
        uv_values = [list(loop.uv) for loop in uv_layer.data] if uv_layer else []
        payload = {
            "vertices_local": vertices_local,
            "vertices_world": vertices_world,
            "vertex_normals_world": normals_world,
            "edges": edge_indices,
            "face_offsets": face_offsets,
            "face_vertex_indices": face_indices,
            "face_material_indices": face_materials,
            "face_normals_local": face_normals,
            "loop_uvs": uv_values,
            "world_matrix": matrix_rows(evaluated.matrix_world),
        }
        target = geometry_dir / f"{frame_stem}_{safe_name(obj.name)}.json.gz"
        with gzip.open(target, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        return {
            "file": str(target),
            "format": "json.gz",
            "vertices": len(mesh.vertices),
            "edges": len(mesh.edges),
            "polygons": len(mesh.polygons),
            "has_uv": bool(uv_layer),
        }
    finally:
        evaluated.to_mesh_clear()


def capture_object(obj, geometry_dir: Path, frame_stem: str) -> dict[str, Any]:
    item: dict[str, Any] = {
        "name": obj.name,
        "type": obj.type,
        "data_name": getattr(obj.data, "name", None),
        "world_matrix": matrix_rows(obj.matrix_world),
        "location": plain(obj.matrix_world.translation),
        "rotation": plain(obj.matrix_world.to_euler()),
        "scale": plain(obj.matrix_world.to_scale()),
        "dimensions": plain(obj.dimensions),
        "parent": obj.parent.name if obj.parent else None,
        "parent_bone": getattr(obj, "parent_bone", None),
        "materials": [
            capture_material(slot.material) if slot.material else None
            for slot in obj.material_slots
        ],
        "modifiers": [
            {
                "name": modifier.name,
                "type": modifier.type,
                "show_viewport": modifier.show_viewport,
                "show_render": modifier.show_render,
            }
            for modifier in obj.modifiers
        ],
    }
    if obj.type == "MESH":
        item["evaluated_geometry"] = export_evaluated_mesh(obj, geometry_dir, frame_stem)
        item["mesh"] = {
            "vertices": len(obj.data.vertices),
            "edges": len(obj.data.edges),
            "polygons": len(obj.data.polygons),
            "bounds_world": [
                plain(obj.matrix_world @ Vector(corner)) for corner in obj.bound_box
            ],
        }
    if obj.type == "ARMATURE":
        item["bones"] = [
            {
                "name": bone.name,
                "parent": bone.parent.name if bone.parent else None,
                "head_world": plain(obj.matrix_world @ bone.head_local),
                "tail_world": plain(obj.matrix_world @ bone.tail_local),
            }
            for bone in obj.data.bones
        ]
        item["pose_bones"] = [
            {
                "name": bone.name,
                "matrix_world": matrix_rows(obj.matrix_world @ bone.matrix),
                "head_world": plain(obj.matrix_world @ bone.head),
                "tail_world": plain(obj.matrix_world @ bone.tail),
                "rotation_quaternion": plain(bone.rotation_quaternion),
                "rotation_euler": plain(bone.rotation_euler),
                "location": plain(bone.location),
                "scale": plain(bone.scale),
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
    if obj.type == "LIGHT":
        item["light"] = {
            "type": obj.data.type,
            "energy": obj.data.energy,
            "color": plain(obj.data.color),
            "shape": getattr(obj.data, "shape", None),
            "size": getattr(obj.data, "size", None),
            "size_y": getattr(obj.data, "size_y", None),
            "use_nodes": getattr(obj.data, "use_nodes", None),
        }
    if obj.particle_systems:
        item["particle_systems"] = [
            {
                "name": system.name,
                "type": system.settings.type,
                "count": system.settings.count,
                "frame_start": system.settings.frame_start,
                "frame_end": system.settings.frame_end,
            }
            for system in obj.particle_systems
        ]
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
    if end < start:
        raise ValueError("end frame must be greater than or equal to start frame")
    output = Path(args.output).expanduser().resolve()
    frames_dir = output / "frames"
    geometry_dir = output / "geometry"
    frames_dir.mkdir(parents=True, exist_ok=True)
    geometry_dir.mkdir(parents=True, exist_ok=True)
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
        "available_channels": enabled + ["evaluated_mesh_geometry", "scene_state_json"],
        "unavailable_channels": unavailable,
        "notes": [
            "Multilayer EXR contains enabled passes supported by this Blender version and render engine.",
            "Evaluated mesh arrays are saved per object and frame in geometry/.",
            "Unsupported channels are explicitly listed; custom AOVs require scene-specific shader setup.",
        ],
    }

    for frame in range(start, end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        stem = f"frame_{frame:06d}"
        scene.render.filepath = str(frames_dir / stem)
        bpy.ops.render.render(write_still=True)

        objects = []
        geometry_failures = []
        for obj in scene.objects:
            try:
                objects.append(capture_object(obj, geometry_dir, stem))
            except Exception as exc:
                geometry_failures.append({"object": obj.name, "error": str(exc)})
                objects.append({
                    "name": obj.name, "type": obj.type,
                    "world_matrix": matrix_rows(obj.matrix_world),
                    "geometry_export_error": str(exc),
                })

        timestamp = (frame - scene.frame_start) / fps
        state = {
            "frame_index": frame,
            "timestamp_seconds": timestamp,
            "camera": scene.camera.name if scene.camera else None,
            "objects": objects,
            "world": {
                "name": scene.world.name if scene.world else None,
                "color": plain(scene.world.color) if scene.world else None,
            } if scene.world else None,
            "render": {
                "engine": scene.render.engine,
                "resolution": [scene.render.resolution_x, scene.render.resolution_y],
                "fps": fps,
            },
            "geometry_export_failures": geometry_failures,
        }
        state_file = frames_dir / f"{stem}.json"
        state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
        manifest["frames"].append({
            "frame_index": frame,
            "timestamp_seconds": timestamp,
            "passes": {
                "multilayer_exr": f"frames/{stem}.exr",
                "evaluated_geometry_dir": "geometry/",
            },
            "state_file": f"frames/{stem}.json",
            "available_channels": manifest["available_channels"],
            "unavailable_channels": unavailable,
            "geometry_export_failures": geometry_failures,
        })

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"NEO FULL capture finished: {output}")
    print(f"Requested render passes available: {len(enabled)}; unavailable: {len(unavailable)}")


if __name__ == "__main__":
    main()
