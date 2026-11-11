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

NEO's generation console integrating LTX-2.3 for joint audio/video output.

## Generate a video
1. Run/update ComfyUI with native LTX-2.3 support.
2. Install the LTX-2.3 distilled checkpoint, compatible text encoder and required VAEs.
3. Load an LTX-2.3 Text-to-Video or Image-to-Video workflow and export **API format** JSON.
4. In the NEO Generate tab, enter the ComfyUI URL, upload that JSON, enter the positive-prompt node ID and prompt. For image-to-video, set the image-loader node ID and upload a start image.
5. Click **Generate video with sound**. NEO submits once, polls the job and downloads the output. It will not blindly retry an uncertain paid submission.

Use the node IDs in your exported workflow, not the example IDs in the UI. The workflow must save a video (for example with SaveVideo).

## RTX 3090 24GB test
Start with short, low-resolution output on the 24GB RTX 3090 and the LTX-2.3 distilled checkpoint. Use an Ampere-compatible quantization/workflow; do not assume every FP8 kernel is supported on RTX 30-series. Log peak VRAM, generation wall time, output duration and audio/video synchronization. Offloading may be required and may reduce speed. Do not assume 720p or long clips will fit until measured.

## Honest status
The previous Blender-to-skeleton translator was a research baseline, not a video generator. Blender remains for future paired audiovisual data creation. This app submits generation jobs to an already-running ComfyUI endpoint; it does **not** rent Vast.ai GPUs automatically. GPU rental is a separate, manually controlled action under the approved budget.
