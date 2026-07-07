#!/usr/bin/env python3
"""Materialize SWE-bench Lite metadata for local experiment planning."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import datasets


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "raw" / "swe_bench_lite"
ANALYSIS_DIR = ROOT / "analysis"


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    ds = datasets.load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    rows = [dict(ex) for ex in ds]
    rows.sort(key=lambda x: str(x.get("instance_id", "")))

    records = []
    repo_counts = Counter()
    created_counts = Counter()
    for idx, ex in enumerate(rows):
        instance_id = str(ex.get("instance_id"))
        repo = str(ex.get("repo"))
        repo_counts[repo] += 1
        created_counts[str(ex.get("created_at", ""))[:7] if ex.get("created_at") else "unknown"] += 1
        records.append({
            "index": idx,
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": ex.get("base_commit"),
            "problem_statement": ex.get("problem_statement"),
            "patch": ex.get("patch"),
            "test_patch": ex.get("test_patch"),
            "hints_text": ex.get("hints_text"),
            "created_at": ex.get("created_at"),
            "version": ex.get("version"),
            "FAIL_TO_PASS": ex.get("FAIL_TO_PASS"),
            "PASS_TO_PASS": ex.get("PASS_TO_PASS"),
        })

    write_jsonl(OUT_DIR / "swe_bench_lite_test.jsonl", records)

    dev = records[:15]
    train = records[15:60]
    eval_rows = records[60:]
    write_jsonl(OUT_DIR / "swe_bench_lite_train.jsonl", train)
    write_jsonl(OUT_DIR / "swe_bench_lite_dev.jsonl", dev)
    write_jsonl(OUT_DIR / "swe_bench_lite_eval.jsonl", eval_rows)

    summary = {
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "hf_split": "test",
        "total": len(records),
        "train": len(train),
        "dev": len(dev),
        "eval": len(eval_rows),
        "split_rule": "sort by instance_id; first 15 dev, next 45 train, remaining 240 reported eval; this is not a leaderboard split",
        "unique_instance_ids": len({r["instance_id"] for r in records}),
        "repo_counts": dict(sorted(repo_counts.items())),
        "created_month_counts": dict(sorted(created_counts.items())),
        "files": {
            "all": str(OUT_DIR / "swe_bench_lite_test.jsonl"),
            "train": str(OUT_DIR / "swe_bench_lite_train.jsonl"),
            "dev": str(OUT_DIR / "swe_bench_lite_dev.jsonl"),
            "eval": str(OUT_DIR / "swe_bench_lite_eval.jsonl"),
        },
        "example_keys": sorted(rows[0].keys()) if rows else [],
        "first_examples": [
            {
                "instance_id": r["instance_id"],
                "repo": r["repo"],
                "problem_statement_head": (r.get("problem_statement") or "")[:240],
            }
            for r in records[:5]
        ],
    }
    with (ANALYSIS_DIR / "swe_bench_lite_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
