"""Create a compact animated mannequin scene for NEO capture smoke tests.

Run:
blender --background --python scripts/create_blender_demo_scene.py -- --output ./demo_scene.blend
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.armatures, bpy.data.materials):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def make_material(name, color, roughness=0.6):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Roughness"].default_value = roughness
    return mat


def create_rig():
    data = bpy.data.armatures.new("NEO_Demo_Armature")
    rig = bpy.data.objects.new("NEO_Demo_Rig", data)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    specs = [
        ("Root", (0, 0, 0.0), (0, 0, 0.25), None),
        ("Pelvis", (0, 0, 0.85), (0, 0, 1.05), "Root"),
        ("Spine", (0, 0, 1.0), (0, 0, 1.35), "Pelvis"),
        ("Neck", (0, 0, 1.35), (0, 0, 1.52), "Spine"),
        ("Head", (0, 0, 1.52), (0, 0, 1.82), "Neck"),
        ("Shoulder_L", (-0.18, 0, 1.34), (-0.38, 0, 1.34), "Spine"),
        ("Elbow_L", (-0.38, 0, 1.34), (-0.55, 0, 1.10), "Shoulder_L"),
        ("Wrist_L", (-0.55, 0, 1.10), (-0.62, 0, 0.94), "Elbow_L"),
        ("Shoulder_R", (0.18, 0, 1.34), (0.38, 0, 1.34), "Spine"),
        ("Elbow_R", (0.38, 0, 1.34), (0.55, 0, 1.10), "Shoulder_R"),
        ("Wrist_R", (0.55, 0, 1.10), (0.62, 0, 0.94), "Elbow_R"),
        ("Hip_L", (-0.12, 0, 0.87), (-0.13, 0, 0.52), "Pelvis"),
        ("Knee_L", (-0.13, 0, 0.52), (-0.13, 0, 0.12), "Hip_L"),
        ("Ankle_L", (-0.13, 0, 0.12), (-0.13, -0.12, 0.08), "Knee_L"),
        ("Hip_R", (0.12, 0, 0.87), (0.13, 0, 0.52), "Pelvis"),
        ("Knee_R", (0.13, 0, 0.52), (0.13, 0, 0.12), "Hip_R"),
        ("Ankle_R", (0.13, 0, 0.12), (0.13, -0.12, 0.08), "Knee_R"),
        ("Nose", (0, -0.13, 1.67), (0, -0.24, 1.67), "Head"),
    ]
    bones = {}
    for name, head, tail, parent_name in specs:
        bone = data.edit_bones.new(name)
        bone.head, bone.tail = head, tail
        bones[name] = bone
        if parent_name:
            bone.parent = bones[parent_name]
            bone.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    rig.show_in_front = True
    return rig


def attach_sphere(rig, bone_name, name, location, scale, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(mat)
    world_matrix = obj.matrix_world.copy()
    obj.parent = rig
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    obj.matrix_world = world_matrix
    return obj


def create_character(rig):
    skin = make_material("NEO_Skin", (0.36, 0.19, 0.11))
    shirt = make_material("NEO_Shirt", (0.035, 0.12, 0.32))
    trousers = make_material("NEO_Trousers", (0.07, 0.075, 0.09))
    shoes = make_material("NEO_Shoes", (0.015, 0.015, 0.018), 0.4)
    parts = [
        ("Pelvis", "Body_Pelvis", (0, 0, 0.92), (0.23, 0.14, 0.17), trousers),
        ("Spine", "Body_Torso", (0, 0, 1.19), (0.28, 0.16, 0.30), shirt),
        ("Head", "Body_Head", (0, 0, 1.68), (0.16, 0.15, 0.21), skin),
        ("Shoulder_L", "UpperArm_L", (-0.30, 0, 1.31), (0.12, 0.12, 0.22), shirt),
        ("Elbow_L", "LowerArm_L", (-0.49, 0, 1.18), (0.085, 0.09, 0.19), skin),
        ("Wrist_L", "Hand_L", (-0.61, 0, 1.00), (0.07, 0.055, 0.10), skin),
        ("Shoulder_R", "UpperArm_R", (0.30, 0, 1.31), (0.12, 0.12, 0.22), shirt),
        ("Elbow_R", "LowerArm_R", (0.49, 0, 1.18), (0.085, 0.09, 0.19), skin),
        ("Wrist_R", "Hand_R", (0.61, 0, 1.00), (0.07, 0.055, 0.10), skin),
        ("Hip_L", "UpperLeg_L", (-0.13, 0, 0.70), (0.12, 0.13, 0.25), trousers),
        ("Knee_L", "LowerLeg_L", (-0.13, 0, 0.32), (0.09, 0.10, 0.22), trousers),
        ("Ankle_L", "Foot_L", (-0.13, -0.10, 0.09), (0.11, 0.20, 0.07), shoes),
        ("Hip_R", "UpperLeg_R", (0.13, 0, 0.70), (0.12, 0.13, 0.25), trousers),
        ("Knee_R", "LowerLeg_R", (0.13, 0, 0.32), (0.09, 0.10, 0.22), trousers),
        ("Ankle_R", "Foot_R", (0.13, -0.10, 0.09), (0.11, 0.20, 0.07), shoes),
    ]
    for bone, name, location, scale, mat in parts:
        attach_sphere(rig, bone, name, location, scale, mat)


def create_environment():
    floor_mat = make_material("NEO_Floor", (0.12, 0.13, 0.15), 0.9)
    bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, -0.03))
    bpy.context.object.name = "Environment_Floor"
    bpy.context.object.data.materials.append(floor_mat)

    bpy.ops.mesh.primitive_cube_add(size=1, location=(1.2, 0, 0.45))
    table = bpy.context.object
    table.name = "Prop_Table"
    table.dimensions = (1.2, 0.7, 0.9)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.12, location=(1.1, -0.05, 1.0))
    bpy.context.object.name = "Prop_Cup"

    bpy.ops.object.camera_add(location=(3.3, -6.5, 2.0))
    camera = bpy.context.object
    camera.name = "NEO_Camera"
    direction = Vector((0, 0, 0.95)) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 50
    camera.data.dof.use_dof = True
    camera.data.dof.focus_distance = 7.0
    camera.data.dof.aperture_fstop = 5.6
    bpy.context.scene.camera = camera

    bpy.ops.object.light_add(type="AREA", location=(2, -3, 4))
    key = bpy.context.object
    key.name = "NEO_Key_Light"
    key.data.energy = 800
    key.data.shape = "DISK"
    key.data.size = 4
    key.rotation_euler = (0.25, 0, 0.5)

    bpy.ops.object.light_add(type="AREA", location=(-3, -1, 2.5))
    fill = bpy.context.object
    fill.name = "NEO_Fill_Light"
    fill.data.energy = 300
    fill.data.size = 3
    fill.rotation_euler = (0.4, 0, -0.8)

    world = bpy.data.worlds.new("NEO_World") if not bpy.data.worlds else bpy.data.worlds[0]
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.08, 0.09, 0.12, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35


def animate(rig):
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 27
    pose = rig.pose.bones
    for frame in (1, 9, 10, 18, 19, 27):
        scene.frame_set(frame)
        for bone in pose:
            bone.rotation_mode = "XYZ"
        if frame in (1,):
            pose["Head"].rotation_euler[2] = 0
            pose["Shoulder_L"].rotation_euler[1] = 0
            pose["Elbow_L"].rotation_euler[1] = 0
            pose["Hip_L"].rotation_euler[1] = 0
            pose["Hip_R"].rotation_euler[1] = 0
        elif frame in (9, 10):
            pose["Head"].rotation_euler[2] = 0.75
        elif frame in (18, 19):
            pose["Head"].rotation_euler[2] = 0.75
            pose["Shoulder_L"].rotation_euler[1] = -1.0
            pose["Elbow_L"].rotation_euler[1] = -0.35
        elif frame == 27:
            pose["Head"].rotation_euler[2] = 0.75
            pose["Shoulder_L"].rotation_euler[1] = -1.0
            pose["Elbow_L"].rotation_euler[1] = -0.35
            pose["Hip_L"].rotation_euler[1] = 0.35
            pose["Hip_R"].rotation_euler[1] = -0.35
        for name in ("Head", "Shoulder_L", "Elbow_L", "Hip_L", "Hip_R"):
            pose[name].keyframe_insert(data_path="rotation_euler", frame=frame, group=name)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./neo_demo_scene.blend")
    args = parser.parse_args(argv)

    clear_scene()
    rig = create_rig()
    create_character(rig)
    animate(rig)
    create_environment()
    scene = bpy.context.scene
    engines = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    if args.renderer == "CYCLES":
        if "CYCLES" not in engines:
            raise RuntimeError("Cycles renderer is not available in this Blender build")
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 12
    else:
        scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.fps = 24
    scene.view_settings.view_transform = "AgX"
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    print(f"Saved NEO demo scene: {output}")


if __name__ == "__main__":
    main()
