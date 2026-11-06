# DeepSeek Director

`client.py` calls the DeepSeek chat-completions API and validates its JSON response as a `DirectorPlan`. Configure `DEEPSEEK_API_KEY`; optionally set `DEEPSEEK_MODEL` and `DEEPSEEK_BASE_URL`.

The director emits a structured plan. It does not directly produce LTX embeddings. The upstream Gemma encoder remains responsible for converting text into the conditioning representation expected by the pretrained LTX model until a learned replacement adapter is trained and evaluated.

The plan schema accepts up to 50 timestamped reference guides, but the current LTX pipeline adapter has not yet been wired to consume this list.