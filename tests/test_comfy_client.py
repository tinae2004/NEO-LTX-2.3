import json
import pytest
from neo_video_gen.comfy_client import generate_video

def test_missing_prompt_node():
    with pytest.raises(ValueError, match="Prompt node"):
        generate_video("http://localhost:8188", json.dumps({"1":{"inputs":{}}}), "test", "6")

def test_missing_prompt_field():
    with pytest.raises(ValueError, match="no text"):
        generate_video("http://localhost:8188", json.dumps({"1":{"inputs":{"seed":2}}}), "test", "1")

def test_url_validation():
    with pytest.raises(ValueError, match="http"):
        generate_video("localhost:8188", json.dumps({"1":{"inputs":{"text":""}}}), "test", "1")
