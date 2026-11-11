import pytest
import torch

from neo_video_gen.translator.model import MotionTranslator


def test_translator_forward_shapes():
    model = MotionTranslator()
    output = model(
        torch.randn(2, 2, 17, 3),
        torch.eye(4).reshape(1, 4, 4).repeat(2, 1, 1),
        torch.tensor([0, 3]),
    )
    assert output["flow"].shape == (2, 2, 64, 64)
    assert output["visibility_logits"].shape == (2, 1, 64, 64)
    assert output["depth_delta"].shape == (2, 1, 64, 64)


def test_translator_rejects_wrong_joint_shape():
    model = MotionTranslator()
    with pytest.raises(ValueError):
        model(torch.randn(2, 17, 3), torch.eye(4).repeat(2, 1, 1), torch.tensor([0, 1]))
