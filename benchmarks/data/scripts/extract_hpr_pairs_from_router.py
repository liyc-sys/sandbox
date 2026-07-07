#!/usr/bin/env python3
"""Extract HPR pair candidates from routed feedback instances."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LOCAL_TYPES = {"local_preference", "local_correction"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def stable_id(*parts: Any, prefix: str = "") -> str:
    raw = "\n".join(str(p) for p in parts)
    return prefix + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def router_type(row: dict[str, Any], label_field: str) -> str:
    if label_field == "router":
        return str(row.get("router_feedback_type") or "neutral")
    if label_field == "gold":
        return str(row.get("gold_feedback_type") or "neutral")
    raise ValueError(f"unknown label field: {label_field}")


def confidence(row: dict[str, Any], label_field: str) -> float | None:
    value = row.get(f"{label_field}_confidence")
    if value is None and label_field == "gold":
        value = 1.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_pairs(
    rows: list[dict[str, Any]],
    *,
    label_field: str,
    min_confidence: float,
    splits: set[str],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for row in rows:
        feedback_type = router_type(row, label_field)
        if feedback_type not in LOCAL_TYPES:
            continue
        if splits and str(row.get("split")) not in splits:
            continue
        conf = confidence(row, label_field)
        if conf is not None and conf < min_confidence:
            continue
        action = row.get("action") or {}
        rejected_text = str(action.get("text") or "").strip()
        if not rejected_text:
            continue
        state = row.get("state") or row.get("prompt")
        if not state:
            state = {"state_ref": row.get("state_ref")}
        instance_id = str(row.get("instance_id"))
        pairs.append(
            {
                "pair_id": stable_id(instance_id, label_field, feedback_type, "hpr", prefix="hpr_"),
                "instance_id": instance_id,
                "trajectory_id": row.get("trajectory_id"),
                "split": row.get("split"),
                "benchmark": row.get("benchmark"),
                "domain": row.get("domain"),
                "task_id": row.get("task_id"),
                "prompt": state,
                "chosen": None,
                "rejected": {"text": rejected_text},
                "feedback_text": row.get("feedback_text"),
                "feedback_type": feedback_type,
                "router_confidence": conf,
                "router_model": row.get("router_model"),
                "construction": f"{label_field}_routed_needs_hindsight_regeneration",
            }
        )
    return pairs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feedback-instances", type=Path, required=True)
    p.add_argument("--output-hpr-pairs", type=Path, required=True)
    p.add_argument("--label-field", choices=["router", "gold"], default="router")
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--splits", nargs="+", default=["train_update", "train", "dev"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_jsonl(args.feedback_instances)
    pairs = extract_pairs(
        rows,
        label_field=args.label_field,
        min_confidence=args.min_confidence,
        splits=set(args.splits),
    )
    n = write_jsonl(args.output_hpr_pairs, pairs)
    summary = {
        "feedback_instances": str(args.feedback_instances),
        "output_hpr_pairs": str(args.output_hpr_pairs),
        "label_field": args.label_field,
        "min_confidence": args.min_confidence,
        "splits": args.splits,
        "pairs_written": n,
        "pairs_by_split": dict(sorted(Counter(str(p.get("split")) for p in pairs).items())),
        "pairs_by_feedback_type": dict(sorted(Counter(str(p.get("feedback_type")) for p in pairs).items())),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
