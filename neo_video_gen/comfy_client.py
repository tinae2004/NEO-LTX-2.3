"""Submit a ComfyUI API workflow and collect its generated video."""
from __future__ import annotations
import json, mimetypes, time
from pathlib import Path
from urllib.parse import urljoin
import requests

def generate_video(base_url, workflow_json, prompt, prompt_node_id, image_path=None, image_node_id=None, image_input_name="image", timeout_seconds=1800):
    if not base_url.strip().startswith(("http://", "https://")):
        raise ValueError("ComfyUI URL must start with http:// or https://")
    base = base_url.strip().rstrip("/") + "/"
    workflow = json.loads(workflow_json)
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("Workflow must be a non-empty API-format JSON object")
    if prompt_node_id not in workflow:
        raise ValueError(f"Prompt node {prompt_node_id!r} is not in the workflow")
    inputs = workflow[prompt_node_id].get("inputs", {})
    fields = [k for k,v in inputs.items() if isinstance(v,str) and k.lower() in {"text","prompt"}]
    if not fields:
        raise ValueError("Selected prompt node has no text or prompt input")
    inputs[fields[0]] = prompt
    session = requests.Session()
    if image_path:
        if not image_node_id or image_node_id not in workflow:
            raise ValueError("Select a valid image loader node when uploading a start image")
        p = Path(image_path)
        with p.open("rb") as fh:
            r=session.post(urljoin(base,"upload/image"),files={"image":(p.name,fh,mimetypes.guess_type(p.name)[0] or "application/octet-stream")},data={"type":"input","overwrite":"true"},timeout=120)
        r.raise_for_status()
        name=r.json().get("name")
        if not name: raise RuntimeError("ComfyUI image upload returned no filename")
        image_inputs=workflow[image_node_id].setdefault("inputs",{})
        if image_input_name not in image_inputs: raise ValueError(f"Image node lacks {image_input_name!r} input")
        image_inputs[image_input_name]=name
    r=session.post(urljoin(base,"prompt"),json={"prompt":workflow},timeout=120); r.raise_for_status()
    prompt_id=r.json().get("prompt_id")
    if not prompt_id: raise RuntimeError(f"ComfyUI returned no prompt_id: {r.text[:500]}")
    deadline=time.monotonic()+timeout_seconds
    while time.monotonic()<deadline:
        r=session.get(urljoin(base,f"history/{prompt_id}"),timeout=60); r.raise_for_status()
        rec=r.json().get(prompt_id)
        if rec:
            status=rec.get("status",{})
            if status.get("status_str")=="error": raise RuntimeError(f"ComfyUI failed: {status.get('messages',[])}")
            for node in rec.get("outputs",{}).values():
                for key in ("videos","gifs","images"):
                    for item in node.get(key,[]):
                        q={"filename":item["filename"],"subfolder":item.get("subfolder",""),"type":item.get("type","output")}
                        f=session.get(urljoin(base,"view"),params=q,timeout=180); f.raise_for_status()
                        ext=Path(item["filename"]).suffix or (".mp4" if key=="videos" else ".png")
                        dest=Path("outputs")/f"neo_{prompt_id[:8]}{ext}"; dest.parent.mkdir(exist_ok=True)
                        dest.write_bytes(f.content); return str(dest.resolve())
            if status.get("completed"): raise RuntimeError("Job completed without a saved video. Ensure workflow ends with SaveVideo.")
        time.sleep(3)
    raise TimeoutError(f"ComfyUI job {prompt_id} exceeded {timeout_seconds}s")
