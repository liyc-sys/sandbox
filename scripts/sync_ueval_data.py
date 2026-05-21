#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarks" / "generation" / "ueval" / "data"
DEFAULT_DATASET = "zlab-princeton/UEval"


def _records_from_json_like(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, dict) or hasattr(item, "items")]
    if isinstance(raw, dict):
        for key in ("data", "items", "entries", "examples", "prompts", "test"):
            value = raw.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, dict) or hasattr(item, "items")]
        return [dict(raw)]
    return []


def _read_record_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _records_from_json_like(json.loads(path.read_text(encoding="utf-8")))
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(dict(json.loads(line)))
        return rows
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("reading parquet requires pyarrow") from exc
        return [dict(row) for row in pq.read_table(path).to_pylist()]
    return []


def _read_local_source(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        rows = _read_record_file(path)
        if rows:
            return rows
        raise FileNotFoundError(f"unsupported UEval source file: {path}")

    if not path.is_dir():
        raise FileNotFoundError(f"UEval source not found: {path}")

    for rel in ("test.json", "test.jsonl", "test.parquet", "data/test.json", "data/test.jsonl", "data/test.parquet"):
        candidate = path / rel
        if candidate.is_file():
            rows = _read_record_file(candidate)
            if rows:
                return rows

    try:
        from datasets import load_from_disk

        loaded = load_from_disk(str(path))
        if isinstance(loaded, dict):
            loaded = loaded["test"] if "test" in loaded else next(iter(loaded.values()))
        return [dict(item) for item in loaded]
    except Exception:
        pass

    try:
        from datasets import load_dataset

        loaded = load_dataset(str(path), split="test")
        return [dict(item) for item in loaded]
    except Exception:
        pass

    rows: list[dict[str, Any]] = []
    for pattern in ("*.json", "*.jsonl", "*.csv", "*.parquet"):
        for candidate in sorted(path.glob(pattern)):
            rows.extend(_read_record_file(candidate))
    if rows:
        return rows
    raise FileNotFoundError(f"UEval source directory is empty or unsupported: {path}")


def _load_hf_dataset(dataset_id: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("loading from Hugging Face requires `datasets`; use the project .venv or install datasets") from exc
    return [dict(item) for item in load_dataset(dataset_id, split=split)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize UEval data into benchmarks/generation/ueval/data/test.json.")
    parser.add_argument(
        "--source",
        default=DEFAULT_DATASET,
        help="Local file/dir or Hugging Face dataset id. Default: zlab-princeton/UEval",
    )
    parser.add_argument("--split", default="test", help="Hugging Face split when --source is a dataset id")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Directory to write test.json")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke fixtures")
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    if source.exists():
        rows = _read_local_source(source)
    else:
        rows = _load_hf_dataset(args.source, args.split)

    if args.limit is not None:
        rows = rows[: args.limit]

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.split}.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} UEval items to {out_path}")


if __name__ == "__main__":
    main()
