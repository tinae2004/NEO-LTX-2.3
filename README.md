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

## Architecture

Blender and Unreal Engine generate controlled 3D scenes and ground-truth data. NEO maintains deterministic world state, translates 3D movement into 2D spatial-control signals, and will eventually use a compatible FLUX adapter to render photorealistic frames.

The system is divided into:
1. Blender Data Factory.
2. Unreal scene and cinematic capture.
3. Unified scene/frame dataset.
4. NEO Motion Translator.
5. FLUX visual-control bridge.
6. Continuity validation and video assembly.

## Important status

This is a clean project foundation and prototype, not a finished video generator. The repository does not yet contain a trained translator, a working FLUX control adapter, or a production video renderer. The Gradio interface is a project dashboard and dataset inspector.

## Quick start

    pip install -r requirements.txt
    python app.py

## Blender capture

Open an animated Blender scene and run:

    blender --background scene.blend --python scripts/blender_capture.py -- --output ./neo_data/scene_001 --start 1 --end 9 --profile FULL

The exporter attempts all supported configured render passes, writes multilayer OpenEXR frames and per-frame scene-state JSON. Unsupported channels are recorded.

## Unreal capture

Run scripts/unreal_export_scene.py inside the Unreal Editor Python environment. It exports actor, component, camera and light metadata. Configure Movie Render Queue passes separately using the guide in docs/DATA_CAPTURE.md.

## Training

Install optional training dependencies:

    pip install -e ".[train]"

The initial PyTorch prototype expects normalized NPZ samples. Their required keys and shapes are documented in docs/DATA_CAPTURE.md. Raw engine captures need a conversion/normalization stage before training.

## Data rights

Only use characters, animation, textures, footage and model checkpoints whose licences permit the intended use. Record source, licence, attribution and permitted use for every asset. Download availability does not imply commercial model-training permission.
