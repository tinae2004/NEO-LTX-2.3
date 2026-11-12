from scripts.run_vast_gpu_training import remote_training_script


def test_remote_training_script_renders_fstring_and_training_commands():
    script = remote_training_script(epochs=17, samples=2345)

    assert " --samples 2345 " in script
    assert "--epochs 17 " in script
    assert "nvidia-smi" in script
    assert "training/train_translator.py" in script
    assert "print(json.dumps({k:m[k]" in script
    assert "NameError" not in script


def test_remote_training_script_bootstraps_python_and_cuda_torch():
    script = remote_training_script(epochs=2, samples=1000)

    assert "python3 -m venv --system-site-packages /workspace/neo/.venv" in script
    assert "source /workspace/neo/.venv/bin/activate" in script
    assert "python -m pip install --no-deps -e ." in script
    assert "python3-pip" not in script
    assert "apt-get install" not in script
    assert "torch.cuda.is_available()" in script
    assert "torch.cuda.get_device_name(0)" in script
    assert "pip install --index-url https://download.pytorch.org" not in script
    assert "Blender" not in script
    assert "datasets/combined_v1/train.npz" in script
    # The prepared dataset must exist before the rented GPU is used for training.
    assert script.index("python - <<'PY'") > script.index("training/train_translator.py")


def test_deepseek_agent_redacts_api_tokens():
    from scripts.deepseek_agent import redact

    value = redact("Authorization: Bearer abcdef123456 api_key=sk-abcdefghijklmnop")
    assert "abcdef123456" not in value
    assert "sk-abcdefghijklmnop" not in value
    assert "[REDACTED]" in value


def test_rent_with_refresh_researches_after_stale_vast_offer():
    from scripts.run_vast_gpu_training import rent_with_refresh

    class FakeAPI:
        def __init__(self):
            self.search_calls = 0
            self.create_calls = []

        def search(self, gpu_names):
            self.search_calls += 1
            if self.search_calls == 1:
                return [{
                    "id": 101, "gpu_name": "RTX 3090", "dph_total": 0.18,
                    "reliability": 0.995,
                }]
            return [{
                "id": 202, "gpu_name": "RTX 3090", "dph_total": 0.19,
                "reliability": 0.996,
            }]

        def create(self, offer_id, disk_gb, label):
            self.create_calls.append(offer_id)
            if offer_id == 101:
                raise RuntimeError('Vast API PUT failed (400): "no_such_ask"')
            return {"new_contract": 9876}

    api = FakeAPI()
    instance_id, selected = rent_with_refresh(
        api, ["RTX 3090", "RTX 4090"], "RTX 3090,RTX 4090",
        max_hourly=0.30, budget=5.0, max_hours=0.3,
        disk_gb=100, label="test", max_searches=3, offers_per_search=5,
    )

    assert instance_id == 9876
    assert selected["id"] == 202
    assert api.search_calls == 2
    assert api.create_calls == [101, 202]


def test_rent_with_refresh_never_retries_excluded_stale_offer():
    from scripts.run_vast_gpu_training import rent_with_refresh

    class FakeAPI:
        def __init__(self):
            self.search_calls = 0
            self.create_calls = []

        def search(self, gpu_names):
            self.search_calls += 1
            return [{
                "id": 303, "gpu_name": "RTX 3090", "dph_total": 0.18,
                "reliability": 0.995,
            }]

        def create(self, offer_id, disk_gb, label):
            self.create_calls.append(offer_id)
            raise RuntimeError("no_such_ask")

    api = FakeAPI()
    try:
        rent_with_refresh(
            api, ["RTX 3090", "RTX 4090"], "RTX 3090,RTX 4090",
            max_hourly=0.30, budget=5.0, max_hours=0.3,
            disk_gb=100, label="test", max_searches=3, offers_per_search=5,
        )
    except RuntimeError as exc:
        assert "fresh searches" in str(exc)
    else:
        raise AssertionError("Expected bounded rental failure")

    assert api.search_calls == 3
    assert api.create_calls == [303]
