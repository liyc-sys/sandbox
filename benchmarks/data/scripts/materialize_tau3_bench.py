#!/usr/bin/env python3
"""Materialize tau2/tau3-bench task metadata for local experiment planning."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
TAU3_ROOT = PROJECT_ROOT / "vendor" / "tau2-bench"
DOMAIN_ROOT = TAU3_ROOT / "data" / "tau2" / "domains"
OUT_DIR = ROOT / "raw" / "tau3_bench"
ANALYSIS_DIR = ROOT / "analysis"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def get_instructions(task: dict[str, Any]) -> str:
    instructions = (task.get("user_scenario") or {}).get("instructions")
    if isinstance(instructions, str):
        return instructions
    if isinstance(instructions, dict):
        parts = []
        for key in ["known_info", "unknown_info", "reason_for_call", "task_instructions"]:
            value = instructions.get(key)
            if value:
                parts.append(str(value))
        return "\n".join(parts)
    return ""


def task_stats(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts = Counter()
    reward_basis_counts = Counter()
    action_nums = []
    nl_nums = []
    env_assertion_nums = []
    instruction_lens = []
    for task in tasks:
        criteria = task.get("evaluation_criteria") or {}
        actions = criteria.get("actions") or []
        nl_assertions = criteria.get("nl_assertions") or []
        env_assertions = criteria.get("env_assertions") or []
        reward_basis = criteria.get("reward_basis") or []
        action_nums.append(len(actions))
        nl_nums.append(len(nl_assertions))
        env_assertion_nums.append(len(env_assertions))
        instruction_lens.append(len(get_instructions(task).split()))
        for action in actions:
            action_counts[str(action.get("name"))] += 1
        if isinstance(reward_basis, list):
            for item in reward_basis:
                reward_basis_counts[str(item)] += 1
        elif reward_basis:
            reward_basis_counts[str(reward_basis)] += 1

    def mean(xs: list[int]) -> float:
        return round(sum(xs) / len(xs), 2) if xs else 0.0

    return {
        "num_tasks": len(tasks),
        "min_actions": min(action_nums) if action_nums else 0,
        "max_actions": max(action_nums) if action_nums else 0,
        "mean_actions": mean(action_nums),
        "mean_nl_assertions": mean(nl_nums),
        "mean_env_assertions": mean(env_assertion_nums),
        "mean_instruction_words": mean(instruction_lens),
        "tasks_with_nl_assertions": sum(1 for x in nl_nums if x),
        "tasks_with_env_assertions": sum(1 for x in env_assertion_nums if x),
        "action_counts": dict(sorted(action_counts.items())),
        "reward_basis_counts": dict(sorted(reward_basis_counts.items())),
    }


def load_domain_tasks(domain: str) -> dict[str, dict[str, Any]]:
    domain_dir = DOMAIN_ROOT / domain
    if domain == "banking_knowledge":
        task_paths = sorted((domain_dir / "tasks").glob("task_*.json"))
        tasks = [read_json(path) for path in task_paths]
    else:
        tasks = read_json(domain_dir / "tasks.json")
    return {str(task["id"]): task for task in tasks}


def split_ids(domain: str, tasks_by_id: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    domain_dir = DOMAIN_ROOT / domain
    split_path = domain_dir / "split_tasks.json"
    if split_path.exists():
        raw = read_json(split_path)
        return {str(k): [str(x) for x in v] for k, v in raw.items()}
    ids = sorted(tasks_by_id)
    if domain == "banking_knowledge":
        train = ids[:60]
        dev = ids[60:75]
        test = ids[75:]
        return {"train": train, "dev": dev, "test": test, "base": ids}
    return {"base": ids}


def materialize_split(domain: str, split: str, ids: list[str], tasks_by_id: dict[str, dict[str, Any]]):
    rows = []
    missing = []
    for index, task_id in enumerate(ids):
        task = tasks_by_id.get(task_id)
        if task is None:
            missing.append(task_id)
            continue
        rows.append({
            "domain": domain,
            "split": split,
            "index": index,
            "task_id": task_id,
            "metadata": task,
        })
    out_path = OUT_DIR / f"{domain}_{split}_tasks.jsonl"
    write_jsonl(out_path, rows)
    tasks = [r["metadata"] for r in rows]
    stats = task_stats(tasks)
    stats.update({
        "file": str(out_path),
        "missing_task_ids": missing,
        "first_examples": [
            {
                "task_id": row["task_id"],
                "instruction_head": get_instructions(row["metadata"])[:260],
                "num_actions": len((row["metadata"].get("evaluation_criteria") or {}).get("actions") or []),
                "num_nl_assertions": len((row["metadata"].get("evaluation_criteria") or {}).get("nl_assertions") or []),
            }
            for row in rows[:3]
        ],
    })
    return stats


def materialize_derived_core_splits(
    domain: str,
    official_splits: dict[str, list[str]],
    tasks_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Reserve a small dev set from official train for threshold/prompt tuning."""
    if "train" not in official_splits or "test" not in official_splits:
        return {}
    train_ids = sorted(official_splits["train"])
    dev_size = max(1, round(len(train_ids) * 0.15))
    dev_ids = train_ids[:dev_size]
    update_train_ids = train_ids[dev_size:]
    return {
        "train_update": materialize_split(domain, "train_update", update_train_ids, tasks_by_id),
        "dev": materialize_split(domain, "dev", dev_ids, tasks_by_id),
        "reported_test": materialize_split(domain, "reported_test", official_splits["test"], tasks_by_id),
        "rule": (
            "For core text domains, reserve the first 15% of official train "
            "task ids after string sort as dev; train updates use the "
            "remaining official train ids; reported_test is the official test split."
        ),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    domains = ["airline", "retail", "telecom", "banking_knowledge"]
    summary: dict[str, Any] = {
        "source_repo": str(TAU3_ROOT),
        "version_label": "tau3-bench via tau2-bench repository",
        "domains": {},
        "recommended_main_experiment": {
            "train": ["retail_train", "airline_train", "telecom_train"],
            "dev": ["retail_test/dev-subset", "airline_test/dev-subset", "telecom_small"],
            "test": ["retail_test", "airline_test", "telecom_base/test-heldout"],
            "optional": ["banking_knowledge as knowledge-heavy out-of-domain diagnostic"],
        },
    }
    for domain in domains:
        tasks_by_id = load_domain_tasks(domain)
        splits = split_ids(domain, tasks_by_id)
        domain_summary = {
            "total_unique_tasks": len(tasks_by_id),
            "available_splits": sorted(splits),
            "splits": {},
            "derived_experiment_splits": {},
        }
        for split, ids in sorted(splits.items()):
            domain_summary["splits"][split] = materialize_split(domain, split, ids, tasks_by_id)
        if domain in {"airline", "retail", "telecom"}:
            domain_summary["derived_experiment_splits"] = materialize_derived_core_splits(
                domain, splits, tasks_by_id
            )
        summary["domains"][domain] = domain_summary

    with (ANALYSIS_DIR / "tau3_bench_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
