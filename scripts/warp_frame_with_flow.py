"""Warp one source RGB frame using a NEO dense-flow pair from NPZ."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from neo_video_gen.temporal.pixel_reuse import warp_rgb_forward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--flow-npz", type=Path, required=True)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holes", type=Path, required=True)
    args = parser.parse_args()
    image = np.asarray(Image.open(args.image).convert("RGB"))
    with np.load(args.flow_npz) as data:
        index = args.pair_index
        count = len(data["flow_forward"])
        if index < 0 or index >= count:
            raise IndexError(f"pair-index must be in [0, {count - 1}]")
        result = warp_rgb_forward(
            image, data["flow_forward"][index], data["source_visibility"][index],
            data["target_visible_from_source"][index], data["source_depth"][index],
            data["depth_delta"][index], data["target_coverage"][index],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.holes.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result["warped_rgb"], mode="RGB").save(args.output)
    Image.fromarray((result["hole_mask"] * 255).astype(np.uint8), mode="L").save(args.holes)
    print(f"Saved warped image: {args.output}")
    print(f"Saved hole mask: {args.holes}")
    print(f"Target coverage: {result['warped_coverage'].mean():.4f}")

if __name__ == "__main__":
    main()
