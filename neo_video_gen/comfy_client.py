"""Submit a ComfyUI API workflow and collect its generated video."""
from __future__ import annotations

import json
import mimetypes
import time
from pathlib import Path
from urllib.parse import urljoin

import requests


def generate_video(
    base_url,
    workflow_json,
    prompt,
    prompt_node_id,
    image_path=None,
    image_node_id=None,
    image_input_name="image",
    timeout_seconds=1800,
    sampler_node_id=None,
    sampler_steps=None,
):
    if not base_url.strip().startswith(("http://", "https://")):
        raise ValueError("ComfyUI URL must start with http:// or https://")
    base = base_url.strip().rstrip("/") + "/"
    workflow = json.loads(workflow_json)
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("Workflow must be a non-empty API-format JSON object")
    if prompt_node_id not in workflow:
        raise ValueError(f"Prompt node {prompt_node_id!r} is not in the workflow")
    inputs = workflow[prompt_node_id].get("inputs", {})
    fields = [
        key for key, value in inputs.items()
        if isinstance(value, str) and key.lower() in {"text", "prompt"}
    ]
    if not fields:
        raise ValueError("Selected prompt node has no text or prompt input")
    inputs[fields[0]] = prompt

    if sampler_steps is not None:
        if not sampler_node_id or sampler_node_id not in workflow:
            raise ValueError("Select a valid sampler node when overriding inference steps")
        sampler_inputs = workflow[sampler_node_id].get("inputs", {})
        if "steps" not in sampler_inputs:
            raise ValueError("Selected sampler node has no 'steps' input")
        steps = int(sampler_steps)
        if steps < 1 or steps > 100:
            raise ValueError("Sampler steps must be between 1 and 100")
        sampler_inputs["steps"] = steps

    session = requests.Session()
    if image_path:
        if not image_node_id or image_node_id not in workflow:
            raise ValueError("Select a valid image loader node when uploading a start image")
        path = Path(image_path)
        with path.open("rb") as handle:
            response = session.post(
                urljoin(base, "upload/image"),
                files={
                    "image": (
                        path.name,
                        handle,
                        mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                    )
                },
                data={"type": "input", "overwrite": "true"},
                timeout=120,
            )
        response.raise_for_status()
        name = response.json().get("name")
        if not name:
            raise RuntimeError("ComfyUI image upload returned no filename")
        image_inputs = workflow[image_node_id].setdefault("inputs", {})
        if image_input_name not in image_inputs:
            raise ValueError(f"Image node lacks {image_input_name!r} input")
        image_inputs[image_input_name] = name

    response = session.post(
        urljoin(base, "prompt"), json={"prompt": workflow}, timeout=120
    )
    response.raise_for_status()
    prompt_id = response.json().get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI returned no prompt_id: {response.text[:500]}")

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = session.get(urljoin(base, f"history/{prompt_id}"), timeout=60)
        response.raise_for_status()
        record = response.json().get(prompt_id)
        if record:
            status = record.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI failed: {status.get('messages', [])}")
            for node in record.get("outputs", {}).values():
                for key in ("videos", "gifs", "images"):
                    for item in node.get(key, []):
                        query = {
                            "filename": item["filename"],
                            "subfolder": item.get("subfolder", ""),
                            "type": item.get("type", "output"),
                        }
                        file_response = session.get(
                            urljoin(base, "view"), params=query, timeout=180
                        )
                        file_response.raise_for_status()
                        extension = Path(item["filename"]).suffix or (
                            ".mp4" if key == "videos" else ".png"
                        )
                        destination = Path("outputs") / f"neo_{prompt_id[:8]}{extension}"
                        destination.parent.mkdir(exist_ok=True)
                        destination.write_bytes(file_response.content)
                        return str(destination.resolve())
            if status.get("completed"):
                raise RuntimeError(
                    "Job completed without a saved video. Ensure workflow ends with SaveVideo."
                )
        time.sleep(3)
    raise TimeoutError(f"ComfyUI job {prompt_id} exceeded {timeout_seconds}s")
