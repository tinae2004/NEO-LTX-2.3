# Blender Data Factory quick start

This is the first real-engine path for NEO Motion Translator. Run the Blender capture stage on a machine with Blender installed; use ordinary Python with NumPy for dataset conversion and training.

## 1. Create the demo scene

From the repository root:

    blender --background --python scripts/create_blender_demo_scene.py -- --output ./demo_scene.blend

The procedural scene contains a simple 17-joint mannequin, room, table, cup, camera, lights and short head/arm/leg motion segments. It is a capture smoke-test rig, not a production-quality human character.

## 2. Capture one action segment

Head turn, frames 1–9:

    blender --background ./demo_scene.blend --python scripts/blender_capture.py -- --output ./neo_data/head_turn --start 1 --end 9 --profile FULL

Left arm raise, frames 10–18:

    blender --background ./demo_scene.blend --python scripts/blender_capture.py -- --output ./neo_data/arm_raise --start 10 --end 18 --profile FULL

Walk-step proxy, frames 19–27:

    blender --background ./demo_scene.blend --python scripts/blender_capture.py -- --output ./neo_data/walk_step --start 19 --end 27 --profile FULL

The FULL profile writes multilayer EXR frames, per-frame scene metadata and evaluated mesh geometry as compressed JSON. The exporter uses Blender's bundled Python standard library and does not require NumPy inside Blender. Exact render-pass availability depends on Blender version and render engine; check each manifest's available and unavailable channel lists.

## 3. Convert captured animation pairs into training arrays

Use one action label per capture:

    python scripts/build_dataset_from_blender.py --capture ./neo_data/head_turn --output ./datasets/head_turn --action-id 0
    python scripts/build_dataset_from_blender.py --capture ./neo_data/arm_raise --output ./datasets/arm_raise --action-id 1
    python scripts/build_dataset_from_blender.py --capture ./neo_data/walk_step --output ./datasets/walk_step --action-id 4

The converter runs in ordinary Python and requires NumPy. It uses adjacent per-frame scene-state JSON records and the demo rig's standard 17 bones. Adapt BONE_FOR_JOINT for other rigs. The target control maps are sparse skeleton-joint flow/visibility/depth fields, not dense clothing or surface flow.

## 4. Merge captures and train

    python scripts/merge_datasets.py --inputs ./datasets/head_turn ./datasets/arm_raise ./datasets/walk_step --output ./datasets/blender_mix
    pip install -e ".[train]"
    python training/train_translator.py --data ./datasets/blender_mix --epochs 10 --batch-size 16 --output ./runs/neo_motion_translator.pt

Use more varied captures, rigs, camera positions and held-out scenes before treating the model as useful. Three short demo clips are only a pipeline proof, not a meaningful production training corpus.

## 5. Current limitations

- The Blender capture path is exercised by a headless Blender smoke test; validate additional render passes on the exact Blender version and renderer used for production.
- The Unreal exporter currently captures current-level scene metadata; a full Sequencer/Movie Render Queue temporal capture adapter remains future work.
- No FLUX control adapter exists yet. The trained prototype outputs skeleton-motion control maps only.
- Projection is calculated analytically; the learned model is a baseline for sparse control-field generation.
