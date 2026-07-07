#!/usr/bin/env python3
"""Re-split already materialized SWE-bench Lite JSONL without re-downloading.

The first split used sorted instance ids, which accidentally put all 45 train
issues in Django.  This deterministic replacement stratifies by repository so
train/dev cover the same repository families as reported eval.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "raw" / "swe_bench_lite"
ANALYSIS_DIR = ROOT / "analysis"
SEED = 20260523


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def allocate_counts(group_sizes: dict[str, int], total: int, *, min_one: bool) -> dict[str, int]:
    counts = {k: 0 for k in group_sizes}
    remaining = total
    eligible = {k: v for k, v in group_sizes.items() if v > 0}
    if min_one:
        for key in sorted(eligible):
            counts[key] = 1
            remaining -= 1
        if remaining < 0:
            raise ValueError("total is smaller than number of non-empty groups")

    raw = []
    base_sum = 0
    denom = sum(eligible.values())
    for key, size in eligible.items():
        quota = remaining * size / denom if denom else 0
        base = int(quota)
        cap = size - counts[key]
        base = min(base, cap)
        counts[key] += base
        base_sum += base
        raw.append((quota - base, key))

    leftover = remaining - base_sum
    for _, key in sorted(raw, reverse=True):
        if leftover <= 0:
            break
        cap = group_sizes[key] - counts[key]
        if cap <= 0:
            continue
        counts[key] += 1
        leftover -= 1
    if leftover:
        for key in sorted(group_sizes, key=lambda k: group_sizes[k] - counts[k], reverse=True):
            if leftover <= 0:
                break
            cap = group_sizes[key] - counts[key]
            take = min(cap, leftover)
            counts[key] += take
            leftover -= take
    if sum(counts.values()) != total:
        raise AssertionError((sum(counts.values()), total, counts))
    return counts


def stratified_split(records: list[dict], *, train_n: int, dev_n: int) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(SEED)
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_repo[str(row.get("repo") or "")].append(row)
    for repo_rows in by_repo.values():
        repo_rows.sort(key=lambda x: str(x.get("instance_id", "")))
        rng.shuffle(repo_rows)

    train_counts = allocate_counts({k: len(v) for k, v in by_repo.items()}, train_n, min_one=True)
    train: list[dict] = []
    remaining_by_repo: dict[str, list[dict]] = {}
    for repo in sorted(by_repo):
        rows = by_repo[repo]
        n = train_counts[repo]
        train.extend(rows[:n])
        remaining_by_repo[repo] = rows[n:]

    dev_counts = allocate_counts({k: len(v) for k, v in remaining_by_repo.items()}, dev_n, min_one=False)
    dev: list[dict] = []
    eval_rows: list[dict] = []
    for repo in sorted(remaining_by_repo):
        rows = remaining_by_repo[repo]
        n = dev_counts[repo]
        dev.extend(rows[:n])
        eval_rows.extend(rows[n:])

    for split_rows in [train, dev, eval_rows]:
        split_rows.sort(key=lambda x: str(x.get("instance_id", "")))
    return train, dev, eval_rows


def main() -> int:
    records = list(read_jsonl(DATA_DIR / "swe_bench_lite_test.jsonl"))
    records.sort(key=lambda x: str(x.get("instance_id", "")))
    train, dev, eval_rows = stratified_split(records, train_n=45, dev_n=15)
    write_jsonl(DATA_DIR / "swe_bench_lite_train.jsonl", train)
    write_jsonl(DATA_DIR / "swe_bench_lite_dev.jsonl", dev)
    write_jsonl(DATA_DIR / "swe_bench_lite_eval.jsonl", eval_rows)

    repo_counts = Counter(str(r.get("repo")) for r in records)
    split_repo_counts = {
        "train": dict(sorted(Counter(str(r.get("repo")) for r in train).items())),
        "dev": dict(sorted(Counter(str(r.get("repo")) for r in dev).items())),
        "eval": dict(sorted(Counter(str(r.get("repo")) for r in eval_rows).items())),
    }
    summary = {
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "hf_split": "test",
        "total": len(records),
        "train": len(train),
        "dev": len(dev),
        "eval": len(eval_rows),
        "split_rule": (
            "deterministic repository-stratified split with seed 20260523; "
            "45 train, 15 dev, 240 reported eval; this is not a leaderboard split"
        ),
        "unique_instance_ids": len({r.get("instance_id") for r in records}),
        "repo_counts": dict(sorted(repo_counts.items())),
        "split_repo_counts": split_repo_counts,
        "files": {
            "all": str(DATA_DIR / "swe_bench_lite_test.jsonl"),
            "train": str(DATA_DIR / "swe_bench_lite_train.jsonl"),
            "dev": str(DATA_DIR / "swe_bench_lite_dev.jsonl"),
            "eval": str(DATA_DIR / "swe_bench_lite_eval.jsonl"),
        },
        "first_examples": [
            {
                "instance_id": r.get("instance_id"),
                "repo": r.get("repo"),
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
