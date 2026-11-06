# NEO-LTX 2.3 implementation plan

## Target
Audio/video inference on one RTX 3090 or RTX 4090 with 24 GB VRAM. This is a research target; no 24 GB fit or speed claim is made before measurement.

## Architecture
1. DeepSeek Director: screenplay, motion, camera, dialogue and sound direction; emits validated structured plans.
2. Conditioning bridge: initially uses the official LTX text/image conditioning interfaces unchanged.
3. LTX audio/video pipeline: preserve official transformer, VAEs, vocoder, scheduler and decoder.
4. Reasoning observer: inspect previews at explicit supported checkpoints; do not mutate latents arbitrarily.
5. Temporal state: persist characters, objects, scene progress, references and completed outputs.
6. Quantization lab: compare upstream FP8 paths and compatible 4-bit backends on Ampere/Ada.
7. Recovery: resume only at validated generation boundaries.

## Gemma replacement
Gemma's embeddings and LTX connectors are trained together. DeepSeek text/hidden states are not interchangeable. Initially DeepSeek directs while the original Gemma encoder supplies LTX-compatible conditioning. A Gemma-free route requires training and evaluating a DeepSeek-to-LTX adapter against the original path before removing Gemma.

## Phases
### 0. Baseline
Pin LTX-2.3 checkpoint and dependencies. Run unchanged audio/video inference. Record GPU, driver, CUDA, PyTorch, settings, peak VRAM, host RAM, latency, A/V sync and output quality.
### 1. Director
Add schema-validated DeepSeek plans for visual prompt, motion, camera, dialogue, music, ambience and reference timestamps.
### 2. Temporal memory
Persist scene and character state; add observer hooks only at supported pipeline/sampler boundaries.
### 3. Quantization
Compare BF16 baseline where possible, upstream FP8 scaled-MM/cast paths, and a separately validated 4-bit backend. Do not assume Blackwell-only NVFP4 kernels work on RTX 3090/4090. Measure total runtime memory including activations, attention, VAEs and workspaces.
### 4. Conditioning adapter
Train and evaluate a compatible adapter. Remove Gemma only if video/audio quality and prompt adherence meet measured acceptance criteria.
### 5. Multi-reference and long-form
Test 2, 5, 10 and up to 50 timestamped image guides. Explore bounded latent windows, temporal state and continuous user-facing jobs. A continuous job does not imply one unbounded GPU tensor or one infinite diffusion pass.

## Benchmark matrix
RTX 3090 24 GB and RTX 4090 24 GB: FP8 scaled-MM, FP8 cast, compatible 4-bit backend; audio/video on. Record peak allocated/reserved VRAM, host RAM, throughput, load time, output duration, A/V sync and human quality review.

## Reproducibility
Keep checkpoints outside Git. Preserve upstream licenses. Record exact checkpoint hashes and environment. Never label a mode supported until end-to-end tests pass.