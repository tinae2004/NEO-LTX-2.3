# Quantization experiments

Compare the official LTX FP8 implementations with a compatible 4-bit backend on RTX 3090 (Ampere) and RTX 4090 (Ada).

Do not assume NVFP4 kernels that require Blackwell are supported on these GPUs. Check backend capability guards before use.

Each run must record checkpoint hash, software versions, GPU, peak allocated/reserved VRAM, host RAM, latency, resolution, frames, audio settings, A/V sync and quality review. Keep VAEs and sensitive components in higher precision until validated.