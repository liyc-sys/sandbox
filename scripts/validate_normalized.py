#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "benchmarks.json"
NORM_ROOT = ROOT / "data" / "normalized"


REQUIRED_TOP_LEVEL = [
    "uid",
    "benchmark",
    "split",
    "task_type",
    "prompt",
    "media",
    "answer",
    "metadata",
    "source",
]


def load_benchmarks():
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return data["benchmarks"]


def validate_file(path):
    errors = []
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            count += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"{path}:{line_no}: invalid json: {e}")
                continue
            for key in REQUIRED_TOP_LEVEL:
                if key not in obj:
                    errors.append(f"{path}:{line_no}: missing field {key}")
            if "prompt" in obj and not isinstance(obj["prompt"], dict):
                errors.append(f"{path}:{line_no}: prompt must be object")
            if "media" in obj and not isinstance(obj["media"], list):
                errors.append(f"{path}:{line_no}: media must be list")
            if "answer" in obj and not isinstance(obj["answer"], dict):
                errors.append(f"{path}:{line_no}: answer must be object")
    return count, errors


def main():
    parser = argparse.ArgumentParser(description="Validate normalized benchmark JSONL files.")
    parser.add_argument("--benchmark", help="Validate one benchmark only.")
    parser.add_argument("--all", action="store_true", help="Validate all benchmark folders.")
    args = parser.parse_args()

    benchmarks = load_benchmarks()
    if not args.all and not args.benchmark:
        parser.error("Please pass --benchmark <name> or --all")

    targets = [args.benchmark] if args.benchmark else sorted(benchmarks.keys())
    total_files = 0
    total_rows = 0
    total_errors = 0

    for benchmark in targets:
        bench_dir = NORM_ROOT / benchmark
        if not bench_dir.exists():
            print(f"[WARN] missing normalized dir: {bench_dir}")
            continue
        for path in sorted(bench_dir.glob("*.jsonl")):
            total_files += 1
            rows, errors = validate_file(path)
            total_rows += rows
            if errors:
                total_errors += len(errors)
                print(f"[FAIL] {path} rows={rows} errors={len(errors)}")
                for err in errors[:20]:
                    print("  ", err)
            else:
                print(f"[OK] {path} rows={rows}")

    print(f"validated_files={total_files} validated_rows={total_rows} errors={total_errors}")


if __name__ == "__main__":
    main()
