# Runtime research

Implemented runtime pieces:
- crash-safe production-state persistence
- bounded CPU latent checkpoint snapshots
- sampler-step observation for synchronized audio/video states
- unpatchified, post-stage RGB preview decoding at a safe VAE boundary
- optional DeepSeek visual review with production-state persistence

The distilled pipeline's optional `preview_callback` runs after each selected stage-2 chunk has completed and its transformer context has exited. `preview_every_n_chunks` controls preview frequency. A first/last-frame contact sheet is written to a bounded preview directory and passed to the callback.

`ReasoningPreviewAdapter` can be passed directly as that callback. It sends the preview and production state to DeepSeek, then stores the resulting DirectorPlan. DeepSeek calls are synchronous between chunks, so they pause progression to the next chunk.

The returned plan is data only. Any intervention must be translated into an explicit supported conditioning mechanism at a later chunk boundary and tested. An LLM must not arbitrarily edit diffusion latents. Long-form generation, automatic plan application, and GPU end-to-end validation remain open work.
