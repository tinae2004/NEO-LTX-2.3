# NEO HunyuanVideo 1.5 — 16GB RTX T2V MVP

## Scope

This branch adds the first HunyuanVideo 1.5 **text-to-video** worker path. It is separate from the existing LTX-2.3 ComfyUI client.

- Target: one NVIDIA RTX GPU with at least 14 GiB reported VRAM (the initial marketplace target is RTX 5060 Ti 16GB).
- Mode: text-to-video only; no input image.
- Initial output: 480p, 145 frames (~6 seconds at 24 fps), 50 inference steps.
- CPU/group offloading enabled; super-resolution and remote prompt rewriting disabled.
- Model/code/data paths are under `NEO_DATA_DIR`, which should point to a mounted persistent volume when one is available.

## Important hardware caveat

Tencent documents a 14GB minimum **with model offloading enabled**. A 16GB card is therefore a narrow-fit target, not a guarantee. The worker checks VRAM and warns when system RAM is below 48 GiB; the actual minimum host RAM is not specified by Tencent. A marketplace offer with 30–40 GiB system RAM may still fail during CPU offload. Prefer a host with 64 GiB or more for the first test.

## Run on a Vast GPU

After renting a compatible GPU and attaching/mounting storage:

```bash
export NEO_DATA_DIR=/workspace/neo-data
bash scripts/run_hunyuan_t2v_16gb.sh "A cinematic wide shot of a Zimbabwean village at sunrise, gentle camera movement"
```

The script:
1. Checks NVIDIA driver, GPU VRAM and host RAM.
2. Clones Tencent's official HunyuanVideo-1.5 repository if absent.
3. Installs the official repository requirements.
4. Downloads the base model, Qwen2.5-VL text encoder, ByT5 and Glyph-SDXL assets if missing. Hugging Face downloads can resume when rerun.
5. Runs the official `generate.py` in 480p T2V mode.
6. Saves the MP4 under `$NEO_DATA_DIR/outputs/hunyuan`.

Environment overrides:
- `NEO_DATA_DIR`: persistent storage root (default `/workspace/neo-data`)
- `VIDEO_LENGTH`: frames (default 145; use 121 for roughly 5 seconds)
- `INFERENCE_STEPS`: inference steps (default 50)
- `SEED`: random seed (default 123)

## Current limitations

- This is a worker bootstrap and smoke-test script, **not yet an automatic Vast.ai renter or production customer API**.
- It does not create or attach Vast volumes. The volume must be mounted before running.
- It does not yet include queueing, authentication, billing, upload callbacks, or automatic GPU destruction.
- Model downloads are large. Persist `NEO_DATA_DIR` across rentals to avoid downloading the weights every time.
- Validate one successful generation and measure wall time, peak VRAM, system RAM, disk use and output quality before accepting customer jobs.
