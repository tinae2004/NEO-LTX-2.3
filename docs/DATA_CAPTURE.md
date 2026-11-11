# NEO Data Capture Specification

## Capture contract

Every capture records engine/version, renderer/settings, coordinate convention, units, requested channels, available channels, unavailable channels, frame index, timestamp and asset provenance. Never mark a channel as captured unless the exporter confirms it.

## Blender capture

Create the included demo scene:

    blender --background --python scripts/create_blender_demo_scene.py -- --output ./demo_scene.blend

Capture it:

    blender --background ./demo_scene.blend --python scripts/blender_capture.py -- --output ./neo_data/scene_001 --start 1 --end 9 --profile FULL

The exporter attempts depth/Z, normals, position, vector, UV, object/material index, diffuse, glossy, transmission, emission, environment, shadow, ambient-occlusion and Cryptomatte passes when exposed by the active Blender version and render engine. It writes multilayer OpenEXR frames, per-frame JSON scene state and evaluated mesh arrays containing vertices, normals, topology, UVs, face material indices and world transforms.

Custom AOVs, cloth/particle internals and Geometry Nodes-specific attributes need scene-specific shader or extraction logic. The manifest records unavailable passes and geometry-export failures.

## Unreal Engine capture

Run scripts/unreal_export_scene.py in the Unreal Editor Python environment. It exports actor/component transforms, attachment hierarchy, available skeletal bone names and common camera/light properties.

Configure Movie Render Queue separately for beauty, depth, motion vectors, object IDs/stencil, EXR layers and custom post-process outputs. Support varies by Unreal version and renderer. The current exporter is a level-state snapshot; complete Sequencer time-scrubbing and per-frame render automation are still future work.

## Unified training sample

The initial translator uses these arrays:

| Key | Shape | Meaning |
|---|---|---|
| joints_3d | [2, 17, 3] | source and target 3D joint positions in camera coordinates |
| camera | [2, 4, 4] | source and target camera projection matrices |
| action_id | scalar integer | motion/action label |
| target_flow | [2, 64, 64] | target 2D joint-displacement control field |
| target_visibility | [1, 64, 64] | target joint-visibility heatmap |
| target_depth_delta | [1, 64, 64] | depth change near joints |

The Blender converter builds adjacent-frame samples from the standard NEO 17-joint rig. It is skeleton-level supervision, not dense surface flow.

## Coordinate normalization

Blender and Unreal use different axis conventions and units. Convert both into a documented NEO convention before combining datasets. Store original engine coordinates as well as normalized coordinates; never silently discard the source transform.

## Asset provenance

Record asset source, licence, attribution and permitted use. Use only assets with rights suitable for intended model training and commercial deployment.
