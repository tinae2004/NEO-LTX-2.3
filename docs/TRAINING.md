# NEO Motion Translator: training protocol

## What this prototype learns

The first model is deliberately narrow. It receives two 3D skeleton states, a 4x4 camera projection matrix and a discrete action ID. It learns sparse 2D control fields around projected joints:

- 2D displacement / flow;
- target visibility heatmap;
- depth change.

It does not yet understand full body surfaces, texture, clothing, object geometry, complex occlusion, or FLUX's internal conditioning. It is not a finished 2D animator.

## Why projection is deterministic

A known 3D point and camera matrix can be projected into 2D with standard camera geometry. We should calculate that exactly. The learnable part is the image-space control representation and, in later versions, visibility, surface correspondence and appearance-aware reconstruction.

## Reproducible smoke training

Generate a small synthetic skeleton dataset:

    pip install -e ".[train]"
    python scripts/generate_synthetic_dataset.py --samples 1024 --size 64 --output ./datasets/synthetic_v0

Train:

    python training/train_translator.py --data ./datasets/synthetic_v0 --epochs 10 --batch-size 16

For CPU-only smoke checks:

    python training/train_translator.py --data ./datasets/synthetic_v0 --epochs 2 --batch-size 4 --device cpu

The synthetic dataset is a pipeline test and a first skeleton-control baseline. It is not a replacement for Blender/Unreal captures.

## Promotion criteria before FLUX integration

1. Validation flow/visibility/depth metrics improve over an untrained baseline.
2. Outputs remain correct for held-out action variations and camera parameters.
3. Compare analytic projected joints against ground truth.
4. Add dense geometry/surface correspondence and render-derived visibility labels.
5. Validate on different rigs, body proportions, clothing and scenes.
6. Only then train and evaluate a FLUX-compatible control adapter.

Never label a checkpoint as production-ready solely because synthetic training loss decreases.
