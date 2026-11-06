# NEO checkpoint integration with LTX

## Live sampler callback path

The built-in deterministic Euler sampler and ancestral Euler sampler accept an
optional callback with this signature:

```python
Callable[[int, int, LatentState | None, LatentState | None], None]
```

The callback receives `(completed_step, total_steps, video_state, audio_state)`.
It fires only after the updates for both modalities have completed. If a modality
is absent, its state is `None`. The terminal denoised prediction is also reported.

The chunk denoiser forwards this callback through `DiffusionStage`. The distilled
pipeline exposes a higher-level callback on both `stream_chunks` and `__call__`:

```python
def on_checkpoint(stage, chunk, step, total, video_state, audio_state):
    # Record metadata or enqueue a small event. Do not mutate sampler states.
    print(stage, chunk, step, total)

pipeline.stream_chunks(
    prompt="A continuous cinematic shot",
    seed=42,
    height=512,
    width=768,
    frame_rate=24,
    images=[],
    num_frames=121,
    step_callback=on_checkpoint,
)
```

The pipeline callback adds stage and 1-based chunk identity. Stage names are
`stage_1_base` and `stage_2_refine`. Because `stream_chunks` is lazy, callbacks
fire as downstream code consumes the returned decoded-chunk iterator.

## Safety and runtime behavior

The callback is observational. It receives live latent-state objects and must
not mutate or retain them: retaining references in a long-lived list keeps the
corresponding GPU allocations alive. It runs synchronously on the sampling thread;
expensive network calls, image decoding, or blocking work in the callback will
stall GPU sampling. Capture compact metadata or deliberately copy only the data
needed for a bounded CPU-side preview.
Callback exceptions propagate and stop generation, so callers that treat
observation as best-effort must catch and record their own errors.

DeepSeek may return a revised director plan at a checkpoint, but that plan must
be translated into supported conditioning objects and applied at a defined
stage/chunk boundary. Arbitrary in-place latent edits are not supported.
The existing `ReasoningObserver.inspect` API expects a decoded preview path;
this sampler hook supplies latents only and does not yet decode RGB/audio previews
or automatically call DeepSeek.

## Tests and remaining boundary

CPU tests live at
`packages/ltx-pipelines/tests/ltx_pipelines/test_sampler_callbacks.py`.
They cover synchronized two-modality callbacks, both Euler variants, and
video-only/audio-only callback behavior.

The following still require separate implementation and validation:

- A sparse preview decoder operating on detached copies, with interval and memory limits.
- An adapter that supplies decoded previews and production state to DeepSeek.
- Translation of approved director revisions into valid LTX conditioning at safe
  chunk/stage boundaries.
- Full package test/lint execution and an end-to-end run on RTX 3090 and RTX 4090.

No claim is made that DeepSeek can inspect every GPU step in real time or modify
model latents safely.
