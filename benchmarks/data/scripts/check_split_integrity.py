#!/usr/bin/env python3
"""Check local experiment split counts and id disjointness."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ids(rows: list[dict], key: str) -> set[str]:
    values = {str(row.get(key)) for row in rows}
    if None in values:
        raise ValueError(f"missing key {key}")
    return values


def check_count(name: str, rows: list[dict], expected: int) -> None:
    actual = len(rows)
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected}, found {actual}")


def check_disjoint(name: str, left: set[str], right: set[str]) -> None:
    overlap = left & right
    if overlap:
        sample = sorted(overlap)[:5]
        raise AssertionError(f"{name}: overlap size {len(overlap)} sample={sample}")


def check_tau3() -> None:
    base = DATA / "raw" / "tau3_bench"
    clean89_manifest = DATA / "splits" / "tau3_reported_test_clean89_manifest.json"
    expected = {
        "airline": {"train_update": 26, "dev": 4, "reported_test": 20},
        "retail": {"train_update": 63, "dev": 11, "reported_test": 40},
        "telecom": {"train_update": 63, "dev": 11, "reported_test": 40},
    }
    total_train = total_dev = total_eval = 0
    reported_rows = []
    for domain, split_counts in expected.items():
        split_ids = {}
        for split, count in split_counts.items():
            path = base / f"{domain}_{split}_tasks.jsonl"
            rows = read_jsonl(path)
            check_count(f"tau3/{domain}/{split}", rows, count)
            if split == "reported_test":
                reported_rows.extend(dict(row, domain=domain, split=split) for row in rows)
            split_ids[split] = ids(rows, "task_id")
            if len(split_ids[split]) != count:
                raise AssertionError(f"tau3/{domain}/{split}: duplicate task ids")
        check_disjoint(f"tau3/{domain}: train-dev", split_ids["train_update"], split_ids["dev"])
        check_disjoint(
            f"tau3/{domain}: train-eval",
            split_ids["train_update"],
            split_ids["reported_test"],
        )
        check_disjoint(f"tau3/{domain}: dev-eval", split_ids["dev"], split_ids["reported_test"])
        total_train += split_counts["train_update"]
        total_dev += split_counts["dev"]
        total_eval += split_counts["reported_test"]
    if (total_train, total_dev, total_eval) != (152, 26, 100):
        raise AssertionError(f"tau3 totals mismatch: {(total_train, total_dev, total_eval)}")

    with clean89_manifest.open(encoding="utf-8") as f:
        manifest = json.load(f)
    excluded = {
        (str(domain), str(task_id))
        for domain, task_ids in manifest["excluded_task_ids_by_domain"].items()
        for task_id in task_ids
    }
    reported_ids = {(str(row["domain"]), str(row["task_id"])) for row in reported_rows}
    missing = excluded - reported_ids
    if missing:
        raise AssertionError(f"tau3 clean89 excluded ids missing from reported_test: {sorted(missing)[:5]}")
    clean_rows = [
        row
        for row in reported_rows
        if (str(row["domain"]), str(row["task_id"])) not in excluded
    ]
    expected_clean = int(manifest["expected_count"])
    check_count("tau3/reported_test_clean89", clean_rows, expected_clean)
    print(
        f"tau3-bench OK: train={total_train}, dev={total_dev}, "
        f"official_eval={total_eval}, reported_clean_eval={expected_clean}"
    )


def check_swe_lite() -> None:
    base = DATA / "raw" / "swe_bench_lite"
    expected = {"train": 45, "dev": 15, "eval": 240, "test": 300}
    split_ids = {}
    for split, count in expected.items():
        path = base / f"swe_bench_lite_{split}.jsonl"
        rows = read_jsonl(path)
        check_count(f"swe_lite/{split}", rows, count)
        split_ids[split] = ids(rows, "instance_id")
        if len(split_ids[split]) != count:
            raise AssertionError(f"swe_lite/{split}: duplicate instance ids")
    check_disjoint("swe_lite: train-dev", split_ids["train"], split_ids["dev"])
    check_disjoint("swe_lite: train-eval", split_ids["train"], split_ids["eval"])
    check_disjoint("swe_lite: dev-eval", split_ids["dev"], split_ids["eval"])
    covered = split_ids["train"] | split_ids["dev"] | split_ids["eval"]
    if covered != split_ids["test"]:
        raise AssertionError("swe_lite: train/dev/eval union does not match materialized all/test file")
    print("SWE-bench Lite OK: train=45, dev=15, eval=240")


def main() -> int:
    try:
        check_tau3()
        check_swe_lite()
    except Exception as exc:
        print(f"split integrity check failed: {exc}", file=sys.stderr)
        return 1
    print("All checked local splits are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
