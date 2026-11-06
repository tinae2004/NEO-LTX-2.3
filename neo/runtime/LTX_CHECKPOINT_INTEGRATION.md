# NEO checkpoint integration with LTX

## What is now wired

The upstream-compatible sampler function
`ltx_pipelines.utils.samplers.euler_denoising_loop` accepts an optional
`step_callback`:

```python
Callable[[int, int, LatentState | None, LatentState | None], None]
```

It is called after each Euler update, after both audio and video states have
advanced. The callback receives (completed_step, total_steps, video_state,
audio_state). This is the first real NEO hook inside the LTX audio/video
denoising loop.

## Attach NEO to a DiffusionStage

```python
from functools import partial
from ltx_pipelines.utils.samplers import euler_denoising_loop

def on_checkpoint(step, total, video_state, audio_state):
    # Keep this callback lightweight. Record step/shape/device metadata.
    # If a preview is required, decode a COPY in a separate preview path;
    # never mutate the sampler's states here.
    neo_runtime.record_latent_checkpoint(
        step=step,
        total_steps=total,
        video_state=video_state,
        audio_state=audio_state,
    )

result = stage(
    denoiser=denoiser,
    sigmas=sigmas,
    noiser=noiser,
    modalities=modalities,
    loop=partial(euler_denoising_loop, step_callback=on_checkpoint),
)
```

The callback is intentionally observational. DeepSeek may return a revised
director plan at a checkpoint, but that plan must be translated into supported
conditioning objects and applied at a defined stage/chunk boundary. Arbitrary
in-place latent edits are not supported.

## Current scope and limitations

- The callback is wired to the standard Euler denoising loop.
- Other samplers (ancestral Euler, CFG++, gradient-estimating Euler, Res2S)
  do not yet expose this callback.
- The callback receives latent states, not decoded RGB frames or audio samples.
- A decoded preview path still needs to be implemented and benchmarked.
- The high-level distilled chunk pipeline currently calls its chunk denoiser
  without exposing this callback. The next integration step is to thread an
  observer option through chunk planning/denoising without breaking streaming,
  carry-frame continuity, or audio synchronization.
- No claim is made that DeepSeek can inspect every GPU step in real time or
  modify model latents safely.

## Recommended next implementation

1. Add the same post-update callback contract to the sampler variants actually
   selected by the pinned LTX-2.3 checkpoint.
2. Thread an optional NEO observer from the pipeline entrypoint through
   `denoise_chunks` and `DiffusionStage`.
3. Add an opt-in preview decoder that works on detached copies at sparse
   checkpoints, with a configurable interval.
4. Send compact preview + scene state to DeepSeek; validate its response against
   the DirectorPlan schema.
5. Apply approved changes only as new conditioning at a chunk/stage boundary.
6. Add tests for synchronized callback order, absent modalities, callback errors,
   resume behavior, and unchanged output when the callback is omitted.
