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

    assert "python3-pip" in script
    assert "python-is-python3" in script
    assert "download.pytorch.org/whl/cu128" in script
    assert "torch.cuda.is_available()" in script
    assert script.index("apt-get install") < script.index("python -c")
    # The remaining heredoc is intentional: it validates metrics after training.
    assert script.index("python - <<'PY'") > script.index("training/train_translator.py")


def test_deepseek_agent_redacts_api_tokens():
    from scripts.deepseek_agent import redact

    value = redact("Authorization: Bearer abcdef123456 api_key=sk-abcdefghijklmnop")
    assert "abcdef123456" not in value
    assert "sk-abcdefghijklmnop" not in value
    assert "[REDACTED]" in value
