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

NEO's generation console. This branch adds the first **HunyuanVideo 1.5 text-to-video worker** for 16GB-class RTX testing while retaining the existing LTX-2.3 ComfyUI prototype.

## First generation target: HunyuanVideo 1.5 T2V

- Target GPU: RTX 5060 Ti 16GB or another NVIDIA GPU with at least 14 GiB VRAM.
- Initial mode: 480p text-to-video, no image input.
- Memory strategy: model/group CPU offloading.
- Initial smoke test: 145 frames (~6 seconds at 24 fps), 50 inference steps.
- Worker launcher: `scripts/run_hunyuan_t2v_16gb.sh`
- Setup and limitations: `docs/HUNYUAN_T2V_16GB.md`

Run on a rented Vast GPU after attaching/mounting persistent storage:

```bash
export NEO_DATA_DIR=/workspace/neo-data
bash scripts/run_hunyuan_t2v_16gb.sh "A cinematic wide shot of a Zimbabwean village at sunrise"
```

Tencent documents a 14GB minimum with model offloading enabled. A 16GB card is a tight-fit target, not a guarantee. Prefer a host with 64 GiB system RAM for the first test; the script warns below 48 GiB. Validate actual generation, peak VRAM, RAM, disk use and speed before using it for customer jobs.

## Existing LTX-2.3 prototype

The Gradio app still contains the earlier LTX-2.3 ComfyUI client. It requires an already-running ComfyUI endpoint and an API-format workflow JSON. It does not automatically rent Vast GPUs.

## Honest implementation status

The Hunyuan script is a worker bootstrap and smoke-test path. It does **not yet** rent GPUs, create or attach Vast volumes, expose a customer-facing generation API, queue jobs, handle payments, or automatically destroy instances. Those are subsequent integration steps. Keep model weights and outputs under `NEO_DATA_DIR` on persistent storage to avoid repeated large downloads.
