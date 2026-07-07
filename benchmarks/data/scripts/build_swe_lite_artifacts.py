#!/usr/bin/env python3
"""Build SWE-bench Lite routed-feedback artifacts.

This builder materializes the delayed-outcome boundary setting used by the
routed-feedback experiments.  SWE-bench Lite provides issue text and a gold
patch, but final correctness is only known after applying the patch and running
tests.  We therefore expose only delayed trajectory outcome feedback for
training/dev.  No local HPR pairs are created by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SYSTEM_PROMPT = (
    "You are a software engineering agent. Given a GitHub issue and repository "
    "metadata, produce a minimal unified diff patch that fixes the issue. "
    "Return only the patch."
)

USER_TEMPLATE = """Repository: {repo}
Base commit: {base_commit}
Instance id: {instance_id}

Issue:
{problem_statement}

Produce a unified diff patch that fixes the issue."""

NO_PATCH = "No code changes are needed."
POS_FEEDBACK = "The submitted patch resolves the issue and passes the target tests."
NEG_FEEDBACK = "The submitted patch does not resolve the issue or fails the target tests."


def stable_id(*parts: Any, prefix: str = "") -> str:
    raw = "\n".join(str(p) for p in parts)
    return prefix + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def state_for(row: dict[str, Any]) -> dict[str, Any]:
    user = USER_TEMPLATE.format(
        repo=row.get("repo") or "",
        base_commit=row.get("base_commit") or "",
        instance_id=row.get("instance_id") or "",
        problem_statement=str(row.get("problem_statement") or "").strip(),
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "latest_user_message": user,
        "prompt_text": "",
    }


def make_action(text: str) -> dict[str, Any]:
    return {"text": text, "raw_text": text, "tool_calls": []}


def delayed_instance(
    *,
    traj_id: str,
    task_id: str,
    split: str,
    state_ref: str,
    response: str,
    feedback_text: str,
    reward: float,
    suffix: str,
    repo: str,
) -> dict[str, Any]:
    return {
        "instance_id": stable_id(traj_id, "delayed", suffix, prefix="fb_"),
        "trajectory_id": traj_id,
        "turn_id": 1 if reward > 0 else 2,
        "benchmark": "swe_bench_lite",
        "domain": repo,
        "split": split,
        "task_id": task_id,
        "state_ref": state_ref,
        "action": make_action(response),
        "next_state": {"role": "evaluator", "content": feedback_text},
        "feedback_text": feedback_text,
        "gold_feedback_type": "delayed_trajectory_outcome",
        "gold_label_source": "swe_bench_lite_gold_patch",
        "router_feedback_type": "delayed_trajectory_outcome",
        "router_model": "oracle-boundary-rule",
        "router_confidence": 1.0,
        "local_label": int(reward),
        "critique": None,
        "hint": None,
        "hpr": None,
        "ppo": {"reward": reward, "reward_source": "swe_bench_lite_gold_patch"},
    }


def load_split(path: Path, split: str) -> list[tuple[str, dict[str, Any]]]:
    return [(split, row) for row in read_jsonl(path)]


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[tuple[str, dict[str, Any]]] = []
    rows.extend(load_split(args.train, "train_update"))
    rows.extend(load_split(args.dev, "dev"))
    if args.eval and args.eval.exists():
        rows.extend(load_split(args.eval, "eval"))

    trajectories: list[dict[str, Any]] = []
    instances: list[dict[str, Any]] = []
    hpr_pairs: list[dict[str, Any]] = []
    hpr_pairs_all: list[dict[str, Any]] = []
    ppo_rewards: list[dict[str, Any]] = []

    for split, row in rows:
        task_id = str(row.get("instance_id") or "")
        if not task_id:
            continue
        repo = str(row.get("repo") or "unknown_repo")
        patch = str(row.get("patch") or "").strip()
        if not patch:
            continue
        traj_id = stable_id("swe-bench-lite", task_id, prefix="traj_")
        state_ref = f"{traj_id}:turn:1:state"
        state = state_for(row)

        pos_inst = delayed_instance(
            traj_id=traj_id,
            task_id=task_id,
            split=split,
            state_ref=state_ref,
            response=patch,
            feedback_text=POS_FEEDBACK,
            reward=1.0,
            suffix="positive",
            repo=repo,
        )
        neg_inst = delayed_instance(
            traj_id=traj_id,
            task_id=task_id,
            split=split,
            state_ref=state_ref,
            response=NO_PATCH,
            feedback_text=NEG_FEEDBACK,
            reward=0.0,
            suffix="negative",
            repo=repo,
        )
        instances.extend([pos_inst, neg_inst])

        delayed_pair = {
            "pair_id": stable_id(pos_inst["instance_id"], neg_inst["instance_id"], "hpr", prefix="hpr_"),
            "instance_id": pos_inst["instance_id"],
            "trajectory_id": traj_id,
            "turn_id": 1,
            "split": split,
            "task_id": task_id,
            "benchmark": "swe_bench_lite",
            "domain": repo,
            "prompt": state,
            "chosen": {"text": patch},
            "rejected": {"text": NO_PATCH},
            "feedback_text": POS_FEEDBACK,
            "feedback_type": "delayed_trajectory_outcome",
            "construction": "swe_gold_patch_vs_no_patch",
        }
        hpr_pairs_all.append(delayed_pair)

        for inst in [pos_inst, neg_inst]:
            ppo_rewards.append(
                {
                    "reward_id": stable_id(inst["instance_id"], "ppo", prefix="ppo_"),
                    "instance_id": inst["instance_id"],
                    "trajectory_id": traj_id,
                    "turn_id": inst["turn_id"],
                    "split": split,
                    "task_id": task_id,
                    "benchmark": "swe_bench_lite",
                    "domain": repo,
                    "reward": inst["ppo"]["reward"],
                    "reward_source": "swe_bench_lite_gold_patch",
                    "feedback_type": "delayed_trajectory_outcome",
                }
            )

        trajectories.append(
            {
                "trajectory_id": traj_id,
                "benchmark": "swe_bench_lite",
                "domain": repo,
                "split": split,
                "task_id": task_id,
                "task_index": row.get("index"),
                "seed": None,
                "policy_checkpoint": args.policy_checkpoint,
                "router_checkpoint": args.router_checkpoint,
                "environment_version": "swe-bench-lite-offline-v1",
                "turns": [
                    {
                        "turn_id": 1,
                        "state": state,
                        "action": make_action(patch),
                        "next_state": {"role": "evaluator", "content": POS_FEEDBACK},
                        "tool_calls": [],
                        "raw_messages": state["messages"],
                    },
                    {
                        "turn_id": 2,
                        "state": state,
                        "action": make_action(NO_PATCH),
                        "next_state": {"role": "evaluator", "content": NEG_FEEDBACK},
                        "tool_calls": [],
                        "raw_messages": state["messages"],
                    },
                ],
                "final_outcome": {
                    "success": True,
                    "score": 1.0,
                    "evaluator": "swe_bench_lite_gold_patch_proxy",
                    "raw": {
                        "instance_id": task_id,
                        "repo": repo,
                        "base_commit": row.get("base_commit"),
                        "fail_to_pass": row.get("FAIL_TO_PASS"),
                    },
                },
            }
        )

    counts = {
        "trajectories": len(trajectories),
        "feedback_instances": len(instances),
        "hpr_pairs": len(hpr_pairs),
        "hpr_pairs_all": len(hpr_pairs_all),
        "ppo_rewards": len(ppo_rewards),
        "trajectories_by_split": dict(Counter(t["split"] for t in trajectories)),
        "instances_by_split": dict(Counter(i["split"] for i in instances)),
        "instances_by_feedback_type": dict(Counter(i["gold_feedback_type"] for i in instances)),
        "repos": dict(Counter(t["domain"] for t in trajectories)),
    }
    return {
        "trajectories": trajectories,
        "feedback_instances": instances,
        "hpr_pairs": hpr_pairs,
        "hpr_pairs_all": hpr_pairs_all,
        "ppo_rewards": ppo_rewards,
        "counts": counts,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--eval", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--policy-checkpoint", default="Qwen3-4B-Thinking-2507")
    p.add_argument("--router-checkpoint", default="oracle-boundary-rule")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    built = build(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "trajectories": args.output_dir / "trajectories.jsonl",
        "feedback_instances": args.output_dir / "feedback_instances.jsonl",
        "hpr_pairs": args.output_dir / "hpr_pairs.jsonl",
        "hpr_pairs_all": args.output_dir / "hpr_pairs_all.jsonl",
        "ppo_rewards": args.output_dir / "ppo_rewards.jsonl",
    }
    for key, path in files.items():
        write_jsonl(path, built[key])
    manifest = {
        "version": "swe-bench-lite-routed-offline-v1",
        "split_policy": "train/dev/eval are disjoint by SWE-bench instance_id; trainer consumes train_update only.",
        "feedback_policy": "SWE-bench Lite is treated as delayed-outcome boundary; local HPR pairs are empty.",
        "counts": built["counts"],
        "files": {k: str(v) for k, v in files.items()},
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
