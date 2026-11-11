"""Minimal supervised trainer for the NEO Motion Translator prototype."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from neo_video_gen.translator.model import MotionTranslator


class MotionSamples(Dataset):
    def __init__(self, root: Path):
        self.files = sorted(root.rglob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No NPZ training samples found under {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with np.load(self.files[index]) as sample:
            return (
                torch.from_numpy(sample["joints_3d"].astype("float32")),
                torch.from_numpy(sample["camera"].astype("float32")),
                torch.tensor(int(sample["action_id"]), dtype=torch.long),
                torch.from_numpy(sample["target_flow"].astype("float32")),
                torch.from_numpy(sample["target_visibility"].astype("float32")),
                torch.from_numpy(sample["target_depth_delta"].astype("float32")),
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/translator.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = MotionSamples(args.data)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = MotionTranslator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for joints, camera, action, flow, visibility, depth in loader:
            joints, camera, action = joints.to(device), camera.to(device), action.to(device)
            flow, visibility, depth = flow.to(device), visibility.to(device), depth.to(device)
            prediction = model(joints, camera, action)
            loss = torch.nn.functional.smooth_l1_loss(prediction["flow"], flow)
            loss += torch.nn.functional.binary_cross_entropy_with_logits(
                prediction["visibility_logits"], visibility
            )
            loss += torch.nn.functional.smooth_l1_loss(prediction["depth_delta"], depth)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        print(f"epoch={epoch + 1} loss={total / max(1, len(loader)):.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "grid_size": 64}, args.output)
    print(f"Saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()
