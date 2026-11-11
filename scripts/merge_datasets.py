"""Merge multiple NEO train.npz / val.npz captures into one action-balanced dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

KEYS = (
    "joints_3d", "camera", "action_id", "target_flow",
    "target_visibility", "target_depth_delta",
)


def merge_splits(inputs: list[Path], output: Path, seed: int = 7):
    loaded = {"train": [], "val": []}
    for root in inputs:
        for split in ("train", "val"):
            path = root / f"{split}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}")
            with np.load(path) as archive:
                missing = [key for key in KEYS if key not in archive]
                if missing:
                    raise ValueError(f"{path} missing arrays: {missing}")
                loaded[split].append({key: archive[key] for key in KEYS})
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    results = {}
    for split, parts in loaded.items():
        merged = {key: np.concatenate([part[key] for part in parts], axis=0) for key in KEYS}
        order = rng.permutation(len(merged["action_id"]))
        merged = {key: value[order] for key, value in merged.items()}
        out_path = output / f"{split}.npz"
        np.savez_compressed(out_path, **merged)
        results[split] = {"path": str(out_path), "samples": len(order)}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(merge_splits(args.inputs, args.output, args.seed))


if __name__ == "__main__":
    main()
