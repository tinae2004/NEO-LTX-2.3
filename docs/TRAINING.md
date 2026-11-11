# NEO Motion Translator training protocol

## What this prototype learns

The first model receives two 3D skeleton states, source and target 4x4 camera projection matrices, and a discrete action ID. It learns sparse 2D control fields around projected joints:

- 2D displacement / flow;
- target visibility heatmap;
- depth change.

It does not yet understand full body surfaces, texture, clothing, complex object occlusion, or FLUX's internal conditioning. It is not a finished 2D animator.

## Deterministic projection versus learned control

Known 3D points and camera matrices can be projected into 2D with exact camera geometry. That operation should be calculated, not learned. The neural component is an initial baseline for turning paired skeleton states into spatial control fields. Later work must extend it to dense surface correspondence, visibility, object interactions and appearance-aware reconstruction.

## Synthetic smoke training

Install:

    pip install -e ".[train]"

Generate a reproducible toy dataset:

    python scripts/generate_synthetic_dataset.py --samples 1024 --size 64 --output ./datasets/synthetic_v0

Train:

    python training/train_translator.py --data ./datasets/synthetic_v0 --epochs 10 --batch-size 16

The synthetic data validates the training loop and tensor contracts. It is not a substitute for engine-generated training samples.

## Blender-generated training data

Follow docs/BLENDER_QUICKSTART.md to build a demo scene, capture motion, convert adjacent frame pairs to NPZ arrays, and merge captures.

The initial Blender converter requires a compatible 17-joint rig. It projects 3D joint locations into each frame camera and creates sparse Gaussian control labels. This is skeleton-level supervision, not dense body-surface flow.

## Promotion criteria before FLUX integration

1. Validation flow, visibility and depth metrics improve over an untrained baseline.
2. Outputs remain correct for held-out action variations and camera parameters.
3. Compare analytic projected joints against ground truth.
4. Add dense geometry/surface correspondence and render-derived visibility labels.
5. Validate on different rigs, body proportions, clothing and scenes.
6. Implement and evaluate a FLUX-compatible spatial-control adapter.

Never label a checkpoint as production-ready solely because synthetic training loss decreases.
