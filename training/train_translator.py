"""Train the first NEO skeleton-motion control-field baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from neo_video_gen.translator.model import MotionTranslator


class MotionSamples(Dataset):
    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path) as data:
            self.joints = data["joints_3d"].astype("float32")
            self.camera = data["camera"].astype("float32")
            self.actions = data["action_id"].astype("int64")
            self.flow = data["target_flow"].astype("float32")
            self.visibility = data["target_visibility"].astype("float32")
            self.depth = data["target_depth_delta"].astype("float32")
        count = len(self.joints)
        if not all(len(value) == count for value in (
            self.camera, self.actions, self.flow, self.visibility, self.depth
        )):
            raise ValueError("Dataset arrays have inconsistent sample counts")

    def __len__(self) -> int:
        return len(self.joints)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.joints[index]),
            torch.from_numpy(self.camera[index]),
            torch.tensor(self.actions[index], dtype=torch.long),
            torch.from_numpy(self.flow[index]),
            torch.from_numpy(self.visibility[index]),
            torch.from_numpy(self.depth[index]),
        )


def evaluate(model, loader, device):
    model.eval()
    totals = {"flow_l1": 0.0, "visibility_bce": 0.0, "depth_l1": 0.0}
    batches = 0
    with torch.no_grad():
        for joints, camera, action, flow, visibility, depth in loader:
            prediction = model(joints.to(device), camera.to(device), action.to(device))
            totals["flow_l1"] += torch.nn.functional.l1_loss(
                prediction["flow"], flow.to(device)
            ).item()
            totals["visibility_bce"] += torch.nn.functional.binary_cross_entropy_with_logits(
                prediction["visibility_logits"], visibility.to(device)
            ).item()
            totals["depth_l1"] += torch.nn.functional.l1_loss(
                prediction["depth_delta"], depth.to(device)
            ).item()
            batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Directory containing train.npz and val.npz")
    parser.add_argument("--output", type=Path, default=Path("runs/neo_motion_translator.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA-enabled PyTorch device is available")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    train_data = MotionSamples(args.data / "train.npz")
    val_data = MotionSamples(args.data / "val.npz")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    model = MotionTranslator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for joints, camera, action, flow, visibility, depth in train_loader:
            joints, camera, action = joints.to(device), camera.to(device), action.to(device)
            flow, visibility, depth = flow.to(device), visibility.to(device), depth.to(device)
            prediction = model(joints, camera, action)
            flow_loss = torch.nn.functional.smooth_l1_loss(prediction["flow"], flow)
            visibility_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                prediction["visibility_logits"], visibility
            )
            depth_loss = torch.nn.functional.smooth_l1_loss(prediction["depth_delta"], depth)
            loss = flow_loss + visibility_loss + depth_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
        metrics = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch + 1,
            "train_loss": total / max(1, len(train_loader)),
            **metrics,
        }
        history.append(record)
        print(json.dumps(record))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"joint_count": 17, "camera_dim": 16, "action_count": 6, "output_size": 64},
        "history": history,
        "training_data": str(args.data.resolve()),
        "device": str(device),
        "note": "Synthetic skeleton-motion control baseline; not a FLUX adapter.",
    }, args.output)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({
        "device": str(device), "epochs": args.epochs, "history": history,
    }, indent=2), encoding="utf-8")
    print(f"Checkpoint: {args.output}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
