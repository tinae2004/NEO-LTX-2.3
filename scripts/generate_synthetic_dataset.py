"""Generate a labelled synthetic skeleton-motion dataset for smoke training."""
from __future__ import annotations

import argparse

from neo_video_gen.dataset.synthetic import ACTION_NAMES, save_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./datasets/synthetic_v0")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    train_path, val_path = save_dataset(args.output, args.samples, image_size=args.size, seed=args.seed)
    print(f"Actions: {', '.join(ACTION_NAMES)}")
    print(f"Training data: {train_path}")
    print(f"Validation data: {val_path}")


if __name__ == "__main__":
    main()
