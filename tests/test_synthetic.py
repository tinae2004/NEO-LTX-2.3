import numpy as np

from neo_video_gen.dataset.synthetic import ACTION_NAMES, generate_arrays, save_dataset


def test_synthetic_dataset_shapes():
    data = generate_arrays(sample_count=12, image_size=32, seed=11)
    assert data["joints_3d"].shape == (12, 2, 17, 3)
    assert data["camera"].shape == (12, 2, 4, 4)
    assert data["action_id"].shape == (12,)
    assert data["target_flow"].shape == (12, 2, 32, 32)
    assert data["target_visibility"].shape == (12, 1, 32, 32)
    assert data["target_depth_delta"].shape == (12, 1, 32, 32)
    assert set(data["action_id"].tolist()).issubset(set(range(len(ACTION_NAMES))))


def test_dataset_split_is_reproducible(tmp_path):
    train_a, val_a = save_dataset(tmp_path / "a", sample_count=20, image_size=32, seed=4)
    train_b, val_b = save_dataset(tmp_path / "b", sample_count=20, image_size=32, seed=4)
    with np.load(train_a) as a, np.load(train_b) as b:
        assert np.array_equal(a["joints_3d"], b["joints_3d"])
    with np.load(val_a) as a, np.load(val_b) as b:
        assert np.array_equal(a["target_flow"], b["target_flow"])
