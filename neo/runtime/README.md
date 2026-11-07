# Runtime research

Implemented runtime pieces:
- crash-safe production-state persistence
- bounded CPU latent checkpoint snapshots
- sampler-step observation for synchronized audio/video states
- unpatchified, post-stage RGB preview decoding at a safe VAE boundary
- optional DeepSeek visual review with production-state persistence

The distilled pipeline's optional `preview_callback` runs after each selected stage-2 chunk has completed and its transformer context has exited. `preview_every_n_chunks` controls preview frequency. A first/last-frame contact sheet is written to a bounded preview directory and passed to the callback.

`ReasoningPreviewAdapter` can be passed directly as that callback. It sends the preview and production state to DeepSeek, then stores the resulting DirectorPlan. DeepSeek calls are synchronous between chunks, so they pause progression to the next chunk.

With `apply_director_plans=True`, the pipeline converts the plan's visual, motion, camera and audio text into a new prompt, runs the existing Gemma PromptEncoder, and supplies its video/audio embeddings to subsequent stage-2 chunks. Image-reference application, persistent context restoration on resume, and full GPU validation remain open work. Plans never directly edit diffusion latents.
