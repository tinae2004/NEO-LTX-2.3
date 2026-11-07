# 24 GB GPU quantization experiments

Target hardware: NVIDIA RTX 3090 (Ampere, 24 GB) and RTX 4090 (Ada, 24 GB).

## Safe starting point

- Start with the upstream `fp8-cast` policy. It downcasts transformer linear weights during loading and does not require the optional NVFP4 CUDA extension.
- Use CPU or disk offload for stage transitions and keep allocator settings documented per run.
- Keep Gemma/text encoder and VAEs at supported precision until measured otherwise; transformer quantization alone does not guarantee the complete pipeline fits.
- Prefer short, low-resolution, low-frame-count smoke tests before increasing output size.
- Record peak allocated and reserved VRAM separately for transformer, video VAE, audio VAE, text encoder and any Shona TTS stage.

## Important hardware restriction

The upstream LTX optimization guide states that `nvfp4-cast` and `nvfp4-prequant` require Blackwell (SM >= 10) and the optional `ltx-kernels` extension. RTX 3090 (SM 8.6) and RTX 4090 (SM 8.9) do not meet that requirement. Do not select NVFP4 on these cards or describe it as their 4-bit solution.

The repository currently has no verified, supported 4-bit LTX transformer backend for RTX 3090/4090. Any third-party 4-bit implementation must be added behind a capability-checked backend and tested against output quality and numerical stability before it is enabled.

## Required benchmark record

Each run must record:

- exact model revision and SHA-256 checkpoint hashes;
- GPU name, compute capability, driver, CUDA and PyTorch versions;
- quantization backend and offload mode;
- resolution, frame count, fps, audio generation settings and chunk length;
- peak `torch.cuda.max_memory_allocated()` and `max_memory_reserved()`;
- host RAM peak, generation latency and OOM/retry count;
- audio/video synchronization and visual/audio quality review.

## Shona dialogue on a 24 GB card

Run LTX video/audio generation and Shona TTS sequentially by default. Unload the LTX transformer and VAE before loading the Shona TTS model, then mix the synthesized dialogue with the retained LTX soundtrack on CPU/FFmpeg. Do not assume both models can remain resident together.

## Acceptance criteria

A 24 GB claim is valid only after a complete end-to-end run on an actual RTX 3090 or RTX 4090, including model loading, prompt encoding, denoising, audio/video decoding, Shona dialogue synthesis, muxing and output verification. Record peak allocated/reserved VRAM and run repeated jobs to catch fragmentation. Until then, 24 GB is a target, not a verified guarantee.
