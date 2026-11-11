# NEO Data Capture Specification

## Capture contract

Every capture records engine/version, renderer/settings, coordinate convention, units, requested channels, available channels, unavailable channels, frame index, timestamp and asset provenance.

Never mark a requested channel as captured unless the exporter confirms it.

## Blender

Run:

    blender --background scene.blend --python scripts/blender_capture.py -- --output ./neo_data/scene_001 --start 1 --end 9 --profile FULL

The script attempts depth/Z, normals, position, vector, UV, object/material index, diffuse, glossy, transmission, emission, environment, shadow, ambient-occlusion and Cryptomatte passes when the active Blender version and render engine expose them. It writes multilayer OpenEXR frames and per-frame JSON with object transforms, mesh bounds/counts, materials, parent links, armature bones/pose matrices and camera parameters.

Custom AOVs, cloth/particle internals and Geometry Nodes-specific attributes need scene-specific shader or extraction logic.

## Unreal Engine

Run scripts/unreal_export_scene.py in the Unreal Editor Python environment. It exports actor/component transforms, attachment hierarchy, available skeletal bone names and common camera/light properties.

Configure Movie Render Queue separately for beauty, depth, motion vectors, object IDs/stencil, EXR layers and custom post-process outputs. Support varies by Unreal version and renderer. The exporter is not claiming to capture every Unreal subsystem yet; extend it for Sequencer, skeletal transforms, physics, Niagara, materials and render-pass configuration as those requirements are tested.

## Unified training sample

The first translator prototype expects normalized NPZ samples:

| Key | Shape | Meaning |
|---|---|---|
| joints_3d | [52, 3] | normalized 3D joint positions |
| camera | [16] | normalized camera/control features |
| action_id | scalar integer | motion/action label |
| target_flow | [2, 64, 64] | target 2D displacement |
| target_visibility | [1, 64, 64] | visibility/reconstruction target |
| target_depth_delta | [1, 64, 64] | depth-change target |

A conversion stage must derive these from engine captures. The raw exporters do not yet create training NPZ files.

## Coordinate normalization

Blender and Unreal use different axis conventions and units. Convert both into a documented NEO convention before combining datasets. Store original engine coordinates as well as normalized coordinates; never silently discard the source transform.

## Asset provenance

Record asset source, licence, attribution and permitted use. Use only assets with rights suitable for the intended model training and commercial deployment.
