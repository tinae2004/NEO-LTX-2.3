"""Export Unreal Editor actor/component state for NEO VIDEO GEN.

Execute inside Unreal Editor's Python environment. This exports scene metadata;
Movie Render Queue render passes must be configured separately.
"""
from __future__ import annotations

import json
import os
import unreal


def vec(value):
    return {"x": float(value.x), "y": float(value.y), "z": float(value.z)}


def rot(value):
    return {"pitch": float(value.pitch), "yaw": float(value.yaw), "roll": float(value.roll)}


def transform(value):
    return {
        "location": vec(value.translation),
        "rotation": rot(value.rotation.rotator()),
        "scale": vec(value.scale3d),
    }


def export_level(path):
    actors = unreal.EditorLevelLibrary.get_all_level_actors()
    records = []
    for actor in actors:
        record = {
            "name": actor.get_name(),
            "label": actor.get_actor_label(),
            "class": actor.get_class().get_name(),
            "world_transform": transform(actor.get_actor_transform()),
            "parent": actor.get_attach_parent_actor().get_name()
                if actor.get_attach_parent_actor() else None,
            "components": [],
        }
        for component in actor.get_components_by_class(unreal.ActorComponent):
            component_record = {
                "name": component.get_name(),
                "class": component.get_class().get_name(),
                "relative_transform": transform(component.get_relative_transform()),
            }
            if hasattr(component, "get_bone_names"):
                try:
                    component_record["bone_names"] = [str(x) for x in component.get_bone_names()]
                except Exception as exc:
                    component_record["bone_names_unavailable"] = str(exc)
            for prop in ("field_of_view", "current_focal_length", "current_aperture",
                         "intensity", "light_color", "temperature"):
                try:
                    component_record[prop] = str(component.get_editor_property(prop))
                except Exception:
                    pass
            record["components"].append(component_record)
        records.append(record)

    world = unreal.EditorLevelLibrary.get_editor_world()
    payload = {
        "scene_id": world.get_name() if world else "unreal_level",
        "engine": "unreal",
        "engine_version": unreal.SystemLibrary.get_engine_version(),
        "schema_version": "0.1.0",
        "capture_profile": "SCENE_METADATA",
        "coordinate_system": "Unreal world coordinates; X-forward, Y-right, Z-up",
        "units": "centimeters",
        "scene_metadata": {
            "level": world.get_path_name() if world else None,
            "actor_count": len(records),
            "render_passes_note": "Configure Movie Render Queue passes separately.",
        },
        "actors": records,
        "frames": [],
        "assets": [],
    }
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    unreal.log("NEO VIDEO GEN scene export: " + path)


export_level(os.environ.get("NEO_UNREAL_EXPORT", "Saved/NEO/scene_state.json"))
