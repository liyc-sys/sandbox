#!/usr/bin/env python3
"""Extract HPR pair candidates from arbitrary routed feedback types.

This is used for the HPR-All stress test, where every non-neutral feedback
instance is deliberately converted into a hindsight-preference pair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


NON_NEUTRAL_TYPES = {
    "local_preference",
    "local_correction",
    "tool_api_outcome",
    "delayed_trajectory_outcome",
}


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


def feedback_type(row: dict[str, Any], label_field: str) -> str:
    if label_field == "router":
        return str(row.get("router_feedback_type") or row.get("gold_feedback_type") or "neutral")
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


def action_text(row: dict[str, Any], mode: str) -> str:
    action = row.get("action")
    if isinstance(action, dict):
        text = str(action.get("text") or "").strip()
        raw = str(action.get("raw_text") or "").strip()
        if mode == "text":
            return text
        if mode == "raw":
            return raw
        if mode == "text_or_raw":
            return text or raw
    return str(action or "").strip()


def extract_pairs(
    rows: list[dict[str, Any]],
    *,
    label_field: str,
    feedback_types: set[str],
    min_confidence: float,
    splits: set[str],
    action_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    pairs: list[dict[str, Any]] = []
    skipped = {"empty_action": 0, "low_confidence": 0, "wrong_split": 0, "wrong_type": 0}
    for row in rows:
        split = str(row.get("split") or "")
        if splits and split not in splits:
            skipped["wrong_split"] += 1
            continue
        typ = feedback_type(row, label_field)
        if typ not in feedback_types:
            skipped["wrong_type"] += 1
            continue
        conf = confidence(row, label_field)
        if conf is not None and conf < min_confidence:
            skipped["low_confidence"] += 1
            continue
        rejected_text = action_text(row, action_mode)
        if not rejected_text:
            skipped["empty_action"] += 1
            continue
        state = row.get("state") or row.get("prompt") or {"state_ref": row.get("state_ref")}
        instance_id = str(row.get("instance_id"))
        pairs.append(
            {
                "pair_id": stable_id(instance_id, label_field, typ, "hpr_all", prefix="hpr_all_"),
                "instance_id": instance_id,
                "trajectory_id": row.get("trajectory_id"),
                "split": split,
                "benchmark": row.get("benchmark"),
                "domain": row.get("domain"),
                "task_id": row.get("task_id"),
                "prompt": state,
                "chosen": None,
                "rejected": {"text": rejected_text},
                "feedback_text": row.get("feedback_text"),
                "feedback_type": typ,
                "router_confidence": conf,
                "router_model": row.get("router_model"),
                "construction": f"{label_field}_{typ}_as_hpr_needs_hindsight_regeneration",
            }
        )
    return pairs, skipped


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feedback-instances", type=Path, required=True)
    p.add_argument("--output-hpr-pairs", type=Path, required=True)
    p.add_argument("--label-field", choices=["router", "gold"], default="gold")
    p.add_argument("--feedback-types", nargs="+", default=sorted(NON_NEUTRAL_TYPES))
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--splits", nargs="+", default=["train_update", "dev"])
    p.add_argument("--action-mode", choices=["text", "raw", "text_or_raw"], default="text_or_raw")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pairs, skipped = extract_pairs(
        read_jsonl(args.feedback_instances),
        label_field=args.label_field,
        feedback_types=set(args.feedback_types),
        min_confidence=args.min_confidence,
        splits=set(args.splits),
        action_mode=args.action_mode,
    )
    n = write_jsonl(args.output_hpr_pairs, pairs)
    summary = {
        "feedback_instances": str(args.feedback_instances),
        "output_hpr_pairs": str(args.output_hpr_pairs),
        "label_field": args.label_field,
        "feedback_types": args.feedback_types,
        "action_mode": args.action_mode,
        "pairs_written": n,
        "pairs_by_split": dict(sorted(Counter(str(p.get("split")) for p in pairs).items())),
        "pairs_by_feedback_type": dict(sorted(Counter(str(p.get("feedback_type")) for p in pairs).items())),
        "skipped": skipped,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
