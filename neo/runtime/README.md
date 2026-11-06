# Runtime research

Planned responsibilities:
- job lifecycle and durable production state
- DeepSeek pre-generation planning
- observer callbacks at supported sampler boundaries
- safe pause/resume at validated boundaries
- benchmark and output recording

Live denoising feedback is experimental. Any intervention must use an explicit conditioning mechanism and controlled tests; an LLM must not arbitrarily edit diffusion latents.