# DeepSeek Director

`client.py` calls the DeepSeek OpenAI-compatible chat-completions API and validates its JSON response as a `DirectorPlan`. Configure `DEEPSEEK_API_KEY`; optionally set `DEEPSEEK_MODEL` and `DEEPSEEK_BASE_URL`. The default model is `deepseek-flash`, which accepts image inputs.

When the production state includes a decoded `preview_path`, the client reads the PNG/JPEG from disk and sends its actual bytes as an image input (low detail) alongside the scene brief and production-state JSON. A missing or unsupported preview is an error, not a text-only fallback.

The director emits a structured plan. It does not directly produce LTX embeddings. The upstream Gemma encoder remains responsible for converting text into the conditioning representation expected by the pretrained LTX model until a learned replacement adapter is trained and evaluated.

The plan schema accepts up to 50 timestamped reference guides. The current preview adapter persists returned plans for review; it does not yet translate plan changes into LTX conditioning or automatically apply references to later chunks.
