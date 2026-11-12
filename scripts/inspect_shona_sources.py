#!/usr/bin/env python3
"""Inspect Hugging Face dataset configs/splits before downloading large speech corpora."""
from __future__ import annotations

import argparse
import json

SUGGESTED_SOURCES = [
    "google/WaxalNLP",
    "Teleagents/zimbabwe-all-speech",
    "manassehzw/sna-dataset",
    "Professor/shona-speech-data",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", dest="datasets", help="HF dataset repo ID; repeatable")
    args = parser.parse_args()
    try:
        from datasets import load_dataset_builder
    except ImportError as exc:
        raise SystemExit("Install optional tools: pip install -e '.[shona-data]'") from exc

    for repo_id in args.datasets or SUGGESTED_SOURCES:
        print(f"\n### {repo_id}")
        try:
            builder = load_dataset_builder(repo_id)
            configs = []
            for config in builder.builder_configs:
                try:
                    info = load_dataset_builder(repo_id, name=config.name).info
                    splits = {
                        name: getattr(split, "num_examples", None)
                        for name, split in (info.splits or {}).items()
                    }
                    features = list(info.features.keys()) if info.features else []
                    configs.append({"config": config.name, "splits": splits, "features": features})
                except Exception as exc:
                    configs.append({"config": config.name, "inspection_error": str(exc)})
            print(json.dumps({
                "dataset": repo_id,
                "license": getattr(builder.info, "license", None),
                "configs": configs,
            }, indent=2, ensure_ascii=False))
        except Exception as exc:
            print(json.dumps({"dataset": repo_id, "error": str(exc)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
