from scripts.run_vast_gpu_training import remote_training_script


def test_remote_training_script_renders_fstring_and_training_commands():
    script = remote_training_script(epochs=17, samples=2345)

    assert " --samples 2345 " in script
    assert "--epochs 17 " in script
    assert "nvidia-smi" in script
    assert "training/train_translator.py" in script
    assert "print(json.dumps({k:m[k]" in script
    assert "NameError" not in script
