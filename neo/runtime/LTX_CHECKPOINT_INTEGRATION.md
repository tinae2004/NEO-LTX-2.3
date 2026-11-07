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

## Bounded latent checkpoint capture

`neo.runtime.checkpoints.LatentCheckpointStore` can be passed directly as the
pipeline callback. It captures every configured Nth step plus the final step,
copies only the video/audio latent tensors to CPU, writes atomically, and prunes
older snapshots to a configured maximum. It does not retain GPU tensor references.

Example:

```python
from neo.runtime.checkpoints import LatentCheckpointStore

checkpoint_store = LatentCheckpointStore(
    "neo_vault/checkpoints",
    every_n_steps=4,
    max_snapshots=8,
)
pipeline.stream_chunks(..., step_callback=checkpoint_store)
```

Important: these saved tensors are still patchified sampler tokens, not RGB
frames and not directly VAE-decodable. The checkpoint consumer must reconstruct
the correct chunk target shape, clear extra conditioning tokens, and unpatchify
with the matching `VideoLatentTools` before passing video latents to the VAE.
Audio latents likewise require their matching audio decode path. The capture
writes synchronously and can briefly stall sampling while copying/saving; use a
sparse interval and fast local storage.

The following still require separate implementation and validation:

- A shape-aware unpatchify + sparse preview decoder operating on detached copies, with interval and memory limits.
- An adapter that supplies decoded previews and production state to DeepSeek.
- Translation of approved director revisions into valid LTX conditioning at safe
  chunk/stage boundaries.
- Full package test/lint execution and an end-to-end run on RTX 3090 and RTX 4090.

No claim is made that DeepSeek can inspect every GPU step in real time or modify
model latents safely.

## Decoded stage previews and DeepSeek vision

Sampler callbacks still receive patchified live tokens. For visual inspection,
use the separate distilled-pipeline preview callback. It runs once per selected
stage-2 chunk, after the diffusion stage has returned, the transformer context
has exited, and the video latent has been unpatchified. The VAE is therefore
not built while the denoiser is resident.

```python
from neo.director.client import DeepSeekDirector
from neo.runtime.observer import ReasoningObserver, ReasoningPreviewAdapter
from neo.runtime.state import ProductionState

production = ProductionState("neo_vault/production.json")
director = DeepSeekDirector()  # default model: deepseek-flash
observer = ReasoningObserver(director)
reasoning = ReasoningPreviewAdapter(
    observer,
    brief="Continue the scene while preserving character and prop continuity.",
    production_state=production,
    every_n_chunks=2,
)

pipeline.stream_chunks(
    prompt="A continuous cinematic shot",
    seed=42,
    height=512,
    width=768,
    frame_rate=24,
    images=[],
    num_frames=241,
    preview_callback=reasoning,
    preview_directory="neo_vault/previews",
    preview_every_n_chunks=2,
)
```

The writer saves the first and last decoded RGB frames side by side in one PNG
contact sheet and keeps at most eight preview files per writer instance. The pipeline defaults to one preview per stage-2
chunk; raise preview_every_n_chunks to reduce VAE decode and DeepSeek request
overhead. The DeepSeek adapter can be configured with the same every_n_chunks
interval. Preview generation and DeepSeek calls are synchronous between chunks,
not inside denoising steps, so they pause progression to the next chunk but do
not block an active sampler loop.

The director client attaches the PNG bytes as an image input to DeepSeek's
vision-capable chat-completions request (default model deepseek-flash). It also
sends the production-state snapshot, current stage/chunk and scene brief. A
missing preview file is an error rather than silently claiming visual review.

The returned DirectorPlan is persisted as data in ProductionState. It is not
applied to a running LTX pipeline: changing visual prompts, camera direction,
references or audio requires a separate adapter that translates approved plan
fields into supported LTX conditionings at a later chunk boundary. The preview
callback is also not a live per-denoising-step vision loop.

