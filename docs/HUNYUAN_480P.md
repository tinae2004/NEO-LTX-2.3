# HunyuanVideo 1.5 — 480p fast visual mode

## Purpose

NEO now exposes a separate HunyuanVideo 1.5 480p Image-to-Video tab. It is a visual-generation path, not a native Shona audio/video generator. Keep LTX-2.3 available for workflows that produce audio and video together.

## Official fast profile

Use the official **HunyuanVideo 1.5 480P I2V step-distilled** weights and a matching ComfyUI workflow.

| Setting | Value |
|---|---:|
| Resolution | 480p |
| Mode | Image-to-Video |
| Distillation | Step-distilled |
| CFG | 1 |
| Flow shift | 7 |
| Inference steps | 8 or 12 |
| Start frame | Required |

Tencent's ComfyUI guide recommends 8 or 12 steps for the 480p I2V step-distilled model. Four steps are possible but reduce quality. Do not confuse this with CFG-distilled checkpoints: those use 50 steps. See the official [ComfyUI guide](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/ComfyUI/README.md) and [model repository](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5).

## ComfyUI setup

1. Install/update ComfyUI and install the HunyuanVideo 1.5 compatible model files and nodes using Tencent/Comfy-Org instructions.
2. Load a workflow specifically configured for 480p I2V **step-distilled** weights. Do not use a 720p workflow or the 480p CFG-distilled checkpoint by mistake.
3. Confirm the workflow uses CFG 1 and flow shift 7, and that the sampler exposes a numeric `steps` input.
4. Export the workflow using ComfyUI's **API format**, not the normal UI workflow format.
5. In NEO's Hunyuan 480p Fast tab, supply:
   - ComfyUI endpoint URL
   - API-format workflow JSON
   - prompt text node ID
   - image-loader node ID
   - sampler node ID
   - CO-PLANNER start image
   - 8 or 12 steps

The node IDs are workflow-specific. Read them from the exported JSON; do not copy IDs from another workflow. The NEO client changes only the selected prompt, uploaded start image and sampler step count. It does not silently rewrite CFG, flow shift, checkpoint names, resolution or custom-node settings.

## Shona audiovisual plan

HunyuanVideo 1.5 generates the visual clip only. It does not produce a Shona speech track. The planned NEO pipeline is:

1. CO-PLANNER writes and reviews the Shona dialogue.
2. A separately trained/licensed Shona TTS voice produces speech audio.
3. The visual clip is generated with Hunyuan 480p I2V.
4. A lip-sync stage aligns the character's mouth to the speech.
5. NEO assembles speech, ambience and sound effects on a shared timeline.

The current Hunyuan tab implements step 3 only. Steps 1, 2, 4 and 5 are not represented as completed functionality. Do not advertise the Hunyuan tab as generating a Shona-speaking video until those stages are integrated and tested.

## Hardware and benchmark policy

Do not promise RTX 3090 performance from Tencent's RTX 4090 speedup claim. First run short 480p tests on the actual rented GPU and record:
- GPU model and VRAM
- peak VRAM use
- model load time
- generation wall time
- clip duration and frame rate
- failed/OOM attempts
- output validity and visual quality

Start with 8 steps for speed and compare 12 steps for quality. Test one short clip before enabling paid queue workloads. CPU offloading may reduce memory pressure but can increase wall time. Do not rent a 48GB GPU solely on an assumption that it will be needed; measure first.

## Official references

- [Tencent HunyuanVideo 1.5 repository](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)
- [Tencent ComfyUI usage guide](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/ComfyUI/README.md)
- [Comfy-Org repackaged checkpoint](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged)
