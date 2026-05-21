#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/private/tmp/TorchUMM/src/umm/post_training/LatentUMM/eval/gen/wise/final_data.json")
DEFAULT_OUTPUT = ROOT / "benchmarks" / "generation" / "wise" / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Split TorchUMM WISE final_data.json into benchmark data files.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Path to final_data.json")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Output directory for WISE JSON files")
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"WISE source must be a JSON list, got {type(data).__name__}")

    buckets = {
        "cultural_common_sense.json": [],
        "spatio-temporal_reasoning.json": [],
        "natural_science.json": [],
    }
    for item in data:
        pid = int(item["prompt_id"])
        if 1 <= pid <= 400:
            buckets["cultural_common_sense.json"].append(item)
        elif 401 <= pid <= 700:
            buckets["spatio-temporal_reasoning.json"].append(item)
        elif 701 <= pid <= 1000:
            buckets["natural_science.json"].append(item)
        else:
            raise ValueError(f"Unexpected WISE prompt_id: {pid}")

    for name, rows in buckets.items():
        (output_dir / name).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {len(data)} WISE items to {output_dir}")


if __name__ == "__main__":
    main()
