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

NEO's generation console supports LTX-2.3 joint audio/video workflows and a separate HunyuanVideo 1.5 distilled 480p visual workflow.

## HunyuanVideo 1.5 — 480p fast mode

1. Install the official HunyuanVideo 1.5 ComfyUI model files and compatible nodes.
2. Prepare a ComfyUI API-format workflow using the **480p I2V step-distilled** checkpoint.
3. Use CFG 1, flow shift 7 and 8 or 12 inference steps. The NEO UI can override the sampler step count; checkpoint, CFG, flow shift and resolution must already be correct in the workflow.
4. Export the workflow in **API format**.
5. In the Hunyuan 480p Fast tab, enter the endpoint and workflow, provide the prompt/image/sampler node IDs, upload a CO-PLANNER character start image, and generate.

Node IDs are specific to each workflow. The tab requires an image because this fast distilled profile is I2V. See [docs/HUNYUAN_480P.md](docs/HUNYUAN_480P.md) for setup and benchmark instructions.

**Important:** HunyuanVideo 1.5 produces the visual clip, not Shona speech. The Shona TTS, lip-sync and final audio-mixing stages are not yet wired into this tab. LTX-2.3 remains available for joint audio/video workflows.

## LTX-2.3 audio/video mode

1. Run/update ComfyUI with native LTX-2.3 support.
2. Install the LTX-2.3 checkpoint, compatible text encoder and required VAEs.
3. Load an LTX-2.3 Text-to-Video or Image-to-Video workflow and export **API format** JSON.
4. In the LTX tab, enter the ComfyUI URL, upload that JSON, enter the positive-prompt node ID and prompt. For image-to-video, set the image-loader node ID and upload a start image.
5. Click **Generate LTX video with sound**. NEO submits once, polls the job and downloads the output. It will not blindly retry an uncertain paid submission.

## RTX 3090 24GB test policy

Start with short, low-resolution output on the 24GB RTX 3090. Use Ampere-compatible quantization/workflows; do not assume every FP8 kernel is supported on RTX 30-series. Log peak VRAM, generation wall time, output duration and audio/video synchronization. Offloading may be required and may reduce speed. Do not assume 720p or long clips will fit until measured.

## Honest status

The app submits generation jobs to an already-running ComfyUI endpoint. It does **not** rent Vast.ai GPUs automatically. GPU rental is a separate, manually controlled action. Hunyuan 480p is a visual-only mode; Shona speech and lip-sync integration remain future work.
