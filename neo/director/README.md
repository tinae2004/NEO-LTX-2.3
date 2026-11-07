# DeepSeek Director

`client.py` calls the DeepSeek OpenAI-compatible chat-completions API and validates its JSON response as a `DirectorPlan`. Configure `DEEPSEEK_API_KEY`; optionally set `DEEPSEEK_MODEL` and `DEEPSEEK_BASE_URL`. The default model is `deepseek-flash`, which accepts image inputs.

When the production state includes a decoded `preview_path`, the client reads the PNG/JPEG from disk and sends its actual bytes as an image input (low detail) alongside the scene brief and production-state JSON. A missing or unsupported preview is an error, not a text-only fallback.

The director emits a structured plan. It does not directly produce LTX embeddings. The upstream Gemma encoder remains responsible for converting text into the conditioning representation expected by the pretrained LTX model until a learned replacement adapter is trained and evaluated.

The plan schema accepts up to 50 timestamped reference guides. When `apply_director_plans=True`, visual, motion, camera and audio directions are re-encoded by the existing Gemma PromptEncoder and become the text context for subsequent stage-2 chunks. Reference-image guides are retained in the plan but are not yet applied as image conditionings.
