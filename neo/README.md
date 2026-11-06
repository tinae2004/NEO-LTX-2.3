# NEO-LTX 2.3

Experimental audio/video research layer for LTX-2.3, targeting RTX 3090 and RTX 4090 (24 GB).

## Principles
- Preserve the upstream LTX model and conditioning path until replacements are trained and validated.
- DeepSeek is the director/reasoner, not a drop-in replacement for Gemma embeddings.
- Keep audio and video enabled in the initial target.
- Benchmark FP8 and compatible 4-bit methods independently.
- Treat 50-frame guidance, denoising feedback and long-form generation as research goals, not guaranteed capabilities.

See `docs/NEO_LTX23_IMPLEMENTATION_PLAN.md`.