#!/usr/bin/env python3
"""Prepare oracle-routed tau3 HPR pairs.

The qwenmax HPR run already contains regenerated chosen responses for many
local instances. The oracle variant should use gold local labels, so this
script maps any available regenerated pair by instance_id onto the gold/rule
local-pair file and leaves the missing ones marked for regeneration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold-hpr-pairs", type=Path, required=True)
    p.add_argument("--generated-hpr-pairs", type=Path, required=True)
    p.add_argument("--output-hpr-pairs", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train_update", "dev"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    generated_by_instance: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(args.generated_hpr_pairs):
        if row.get("chosen") and row.get("rejected"):
            generated_by_instance[str(row.get("instance_id"))] = row

    allowed_splits = set(args.splits)
    out: list[dict[str, Any]] = []
    reused = 0
    missing = 0
    for row in read_jsonl(args.gold_hpr_pairs):
        if row.get("split") not in allowed_splits:
            continue
        row = dict(row)
        inst_id = str(row.get("instance_id"))
        generated = generated_by_instance.get(inst_id)
        if generated:
            row["chosen"] = generated.get("chosen")
            row["generator_model"] = generated.get("generator_model")
            row["construction"] = generated.get("construction") or "hindsight_regenerated"
            row.pop("skip_reason", None)
            reused += 1
        else:
            row["chosen"] = None
            row["construction"] = "needs_hindsight_regeneration"
            row["skip_reason"] = "missing_generated_chosen_for_oracle_local"
            missing += 1
        out.append(row)

    written = write_jsonl(args.output_hpr_pairs, out)
    print(
        json.dumps(
            {
                "output_hpr_pairs": str(args.output_hpr_pairs),
                "written": written,
                "reused_generated": reused,
                "needs_regeneration": missing,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
