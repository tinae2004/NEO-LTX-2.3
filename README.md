---
title: NEO VIDEO GEN
emoji: 🎬
colorFrom: purple
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# NEO VIDEO GEN

NEO's experimental 3D-to-2D motion-control research platform.

Blender and Unreal Engine generate controlled 3D scenes and ground-truth data. NEO maintains deterministic world state, translates 3D movement into 2D spatial-control signals, and will eventually use a compatible FLUX adapter to render photorealistic frames.

## Current capabilities

- Blender FULL capture prototype: multilayer EXR, render passes, evaluated mesh geometry and per-frame scene state.
- Blender demo scene generator with a simple 17-joint rig and head/arm/leg motion segments.
- Blender capture-to-NPZ converter for adjacent frame pairs.
- Unreal level/actor scene metadata exporter.
- Unified dataset schema and validator.
- Synthetic skeleton-motion dataset generator.
- PyTorch motion-translator training baseline.
- Gradio project dashboard and dataset inspector.

## Important status

This is an experimental research foundation, not a finished video generator. The translator is a narrow skeleton-control baseline. The Unreal tool currently exports a level-state snapshot; full Sequencer/Movie Render Queue temporal capture still needs implementation and engine-side testing. No FLUX control adapter or production video renderer is included yet.

## Quick start: dashboard

    pip install -r requirements.txt
    python app.py

## Blender quick start

Follow docs/BLENDER_QUICKSTART.md. In brief:

    blender --background --python scripts/create_blender_demo_scene.py -- --output ./demo_scene.blend
    blender --background ./demo_scene.blend --python scripts/blender_capture.py -- --output ./neo_data/head_turn --start 1 --end 9 --profile FULL
    python scripts/build_dataset_from_blender.py --capture ./neo_data/head_turn --output ./datasets/head_turn --action-id 0
    python training/train_translator.py --data ./datasets/head_turn --epochs 10 --batch-size 16

For multiple actions, convert separate captures and merge them with scripts/merge_datasets.py. Read docs/TRAINING.md before interpreting the resulting checkpoint.

## Synthetic smoke training

    pip install -e ".[train]"
    python scripts/generate_synthetic_dataset.py --samples 1024 --output ./datasets/synthetic_v0
    python training/train_translator.py --data ./datasets/synthetic_v0 --epochs 10

Synthetic data verifies the training pipeline; it is not a production-quality motion corpus.

## Data rights

Only use characters, animation, textures, footage and model checkpoints whose licences permit the intended use. Record source, licence, attribution and permitted use for every dataset asset. Download availability does not imply commercial model-training permission.
