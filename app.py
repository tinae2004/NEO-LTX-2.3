"""NEO VIDEO GEN dashboard; not a renderer."""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from neo_video_gen import __version__
from neo_video_gen.config import PROJECT_ROOT
from neo_video_gen.dataset.validate import validate_manifest

README = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
ROADMAP = (PROJECT_ROOT / "docs" / "ROADMAP.md").read_text(encoding="utf-8")
CAPTURE_GUIDE = (PROJECT_ROOT / "docs" / "DATA_CAPTURE.md").read_text(encoding="utf-8")


def inspect_dataset(path_text: str) -> tuple[str, str]:
    path = Path(path_text).expanduser()
    if not path.exists():
        return "Path does not exist.", ""
    manifests = [path] if path.is_file() else sorted(path.rglob("manifest.json"))
    if not manifests:
        return "No manifest.json found.", ""
    reports = []
    for manifest in manifests[:100]:
        try:
            reports.append(validate_manifest(manifest))
        except Exception as exc:
            reports.append({"manifest": str(manifest), "valid": False, "errors": [str(exc)]})
    return f"Inspected {len(reports)} manifest(s).", json.dumps(reports, indent=2)


with gr.Blocks(title="NEO VIDEO GEN", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"# 🎬 NEO VIDEO GEN\n"
        f"**3D world understanding → 2D motion control → photorealistic image generation**\n\n"
        f"Research foundation · version {__version__}"
    )
    with gr.Row():
        gr.Markdown("### Blender Data Factory\nRig, animate and capture 3D scenes.")
        gr.Markdown("### Unreal Data Factory\nCapture cinematic scene state and render passes.")
        gr.Markdown("### Motion Translator\nLearn 3D-to-2D spatial control.")
    with gr.Tab("Project"):
        gr.Markdown(README)
    with gr.Tab("Roadmap"):
        gr.Markdown(ROADMAP)
    with gr.Tab("Capture specification"):
        gr.Markdown(CAPTURE_GUIDE)
    with gr.Tab("Dataset inspector"):
        path_box = gr.Textbox(label="Dataset folder or manifest", value="./neo_data")
        inspect_btn = gr.Button("Inspect manifests", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
        output = gr.Code(label="Validation report", language="json")
        inspect_btn.click(inspect_dataset, inputs=path_box, outputs=[status, output])
    gr.Markdown("---\nThis dashboard does not claim that model training or video generation is operational yet.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
