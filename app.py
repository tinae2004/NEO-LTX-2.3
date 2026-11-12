"""NEO VIDEO GEN — LTX-2.3 generation console."""
from __future__ import annotations
import json
from pathlib import Path
import gradio as gr
from neo_video_gen import __version__
from neo_video_gen.config import PROJECT_ROOT
from neo_video_gen.comfy_client import generate_video
from neo_video_gen.dataset.validate import validate_manifest

README=(PROJECT_ROOT/"README.md").read_text(encoding="utf-8")
ROADMAP=(PROJECT_ROOT/"docs"/"ROADMAP.md").read_text(encoding="utf-8")
CAPTURE=(PROJECT_ROOT/"docs"/"DATA_CAPTURE.md").read_text(encoding="utf-8")
BLENDER=(PROJECT_ROOT/"docs"/"BLENDER_QUICKSTART.md").read_text(encoding="utf-8")
TRAINING=(PROJECT_ROOT/"docs"/"TRAINING.md").read_text(encoding="utf-8")
SHONA=(PROJECT_ROOT/"docs"/"SHONA_ENGINE.md").read_text(encoding="utf-8")

def inspect_dataset(path_text):
    p=Path(path_text).expanduser()
    if not p.exists(): return "Path does not exist.",""
    manifests=[p] if p.is_file() else sorted(p.rglob("manifest.json"))
    if not manifests: return "No manifest.json found.",""
    reports=[]
    for m in manifests[:100]:
        try: reports.append(validate_manifest(m))
        except Exception as e: reports.append({"manifest":str(m),"valid":False,"errors":[str(e)]})
    return f"Inspected {len(reports)} manifest(s).",json.dumps(reports,indent=2)

def run_ltx(endpoint,wf,prompt,prompt_node,start_image,image_node,timeout):
    if wf is None: raise gr.Error("Upload a ComfyUI API-format workflow JSON.")
    try:
        out=generate_video(endpoint,Path(wf).read_text(encoding="utf-8"),prompt,str(prompt_node).strip(),str(start_image) if start_image else None,str(image_node).strip() if image_node else None,timeout_seconds=int(timeout))
    except Exception as e: raise gr.Error(f"LTX generation failed: {e}") from e
    return out,f"Generation complete: {out}"

with gr.Blocks(title="NEO VIDEO GEN",theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 🎬 NEO VIDEO GEN\n**LTX-2.3 · joint audio/video · RTX 3090 test console**\n\nVersion {__version__}")
    with gr.Tab("Generate video"):
        gr.Markdown("Connect to a running ComfyUI server with LTX-2.3 models installed. Export your workflow in **API format**.")
        endpoint=gr.Textbox(label="ComfyUI server URL",placeholder="https://your-comfyui-endpoint")
        wf=gr.File(label="ComfyUI API workflow JSON",file_types=[".json"],type="filepath")
        prompt=gr.Textbox(label="Video + sound prompt",lines=5,placeholder="Describe the scene, camera movement, dialogue and sound...")
        with gr.Row():
            prompt_node=gr.Textbox(label="Prompt node ID",value="267:266")
            image_node=gr.Textbox(label="Image-loader node ID (optional)",value="269")
        start_image=gr.Image(label="Start image (optional)",type="filepath")
        timeout=gr.Slider(label="Timeout seconds",minimum=60,maximum=3600,value=1800,step=60)
        btn=gr.Button("Generate video with sound",variant="primary")
        video=gr.Video(label="Generated NEO video")
        status=gr.Textbox(label="Status",interactive=False)
        btn.click(run_ltx,[endpoint,wf,prompt,prompt_node,start_image,image_node,timeout],[video,status])
    with gr.Tab("Project"): gr.Markdown(README)
    with gr.Tab("Blender quick start"): gr.Markdown(BLENDER)
    with gr.Tab("Capture specification"): gr.Markdown(CAPTURE)
    with gr.Tab("Training guide"): gr.Markdown(TRAINING)
    with gr.Tab("Shona Engine"): gr.Markdown(SHONA)
    with gr.Tab("Roadmap"): gr.Markdown(ROADMAP)
    with gr.Tab("Dataset inspector"):
        path=gr.Textbox(label="Dataset folder or manifest",value="./neo_data")
        inspect=gr.Button("Inspect manifests",variant="primary")
        state=gr.Textbox(label="Status",interactive=False); report=gr.Code(label="Validation report",language="json")
        inspect.click(inspect_dataset,path,[state,report])
    gr.Markdown("---\nThis console submits real generation jobs to your configured ComfyUI server; GPU provisioning is separate.")
if __name__=="__main__": demo.launch(server_name="0.0.0.0",server_port=7860)
