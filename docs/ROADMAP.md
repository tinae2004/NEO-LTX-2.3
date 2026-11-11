# NEO VIDEO GEN Roadmap

## Phase 0 — foundation
- [x] Clean NEO VIDEO GEN project branch.
- [x] Shared manifest and dataset validator.
- [x] Blender capture prototype.
- [x] Unreal scene metadata exporter prototype.
- [x] Initial PyTorch motion-translator scaffold.
- [ ] Run scripts in real Blender and Unreal installations.
- [ ] Add CI and integration tests.

## Phase 1 — Blender Data Factory
- Create a controlled rigged-character scene.
- Capture head turn, walk, hand raise, sit and object pickup.
- Validate EXR channels and frame state.
- Add asset provenance.
- Build raw capture to NPZ converter.

## Phase 2 — Unreal Data Factory
- Create a controlled cinematic level.
- Configure Movie Render Queue EXR passes.
- Export Sequencer, camera, lights, skeletal state and actors.
- Normalize axes and units.
- Compare equivalent scenes across engines.

## Phase 3 — Motion Translator
- Train the 3D-state-to-2D-control baseline.
- Evaluate flow, visibility and depth-control accuracy.
- Test unseen rigs, actions and cameras.
- Separate scene-state correctness from visual quality.

## Phase 4 — FLUX Bridge
- Select and verify the exact FLUX checkpoint and licence.
- Implement reference-image conditioning.
- Train a compatible spatial-control adapter.
- Evaluate identity, geometry, prompt adherence and temporal consistency.

## Phase 5 — NEO VIDEO GEN
- Add shot planning and scene execution.
- Add frame generation, continuity validation and video assembly.
- Benchmark quality, latency, VRAM and cost.
- Claim only measured performance.

## Hardware
Start translator experiments on an RTX 3090 24GB using small crops and short sequences. Generate engine data in separate jobs. Rent larger GPUs only when measured memory or throughput requires them.
