#!/usr/bin/env python3
"""Build OpenClaw/GSM8K routed-feedback artifacts.

The OpenClaw-RL repository materializes the GSM8K test set as 1,319 personal
homework tasks.  This builder keeps that task order and emits the same common
contract used by the routed-feedback trainers:

  - trajectories.jsonl
  - feedback_instances.jsonl
  - hpr_pairs.jsonl            (local preference/correction only)
  - hpr_pairs_all.jsonl        (local pairs plus delayed-outcome pairs)
  - ppo_rewards.jsonl
  - manifest.json

Splits follow the paper protocol: 0-999 train_update, 1000-1099 dev, and
1100-1318 eval.  Training scripts ignore eval rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SYSTEM_PROMPT = (
    "You are a helpful AI assistant integrated into the OpenClaw desktop app. "
    "The user keeps homework problems in homework/i.txt. Show the full math "
    "work, but write naturally, without heavy markdown formatting."
)

FIRST_MESSAGE = (
    "Hey, I have my homework in the file homework/{index}.txt in your workspace. "
    "Can you read it and help me solve it? Show me the answer first -- don't "
    "write to the file until I tell you to.\n\nProblem:\n{question}"
)

LOCAL_FEEDBACK = (
    "This is too formatted and AI-like. Rewrite it in a natural homework style, "
    "but keep the complete reasoning and the final answer."
)

DELAYED_FEEDBACK_POS = "The homework answer is correct."
DELAYED_FEEDBACK_NEG = "The homework answer is incorrect."


def stable_id(*parts: Any, prefix: str = "") -> str:
    raw = "\n".join(str(p) for p in parts)
    return prefix + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


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


def load_hf_gsm8k_test() -> dict[str, str]:
    """Return question -> full GSM8K answer rationale for the test split."""
    try:
        from datasets import load_dataset
    except Exception:
        return {}
    try:
        ds = load_dataset("gsm8k", "main", split="test", download_mode="reuse_dataset_if_exists")
    except Exception:
        ds = load_dataset("gsm8k", "main", split="test")
    return {str(row["question"]).strip(): str(row["answer"]) for row in ds}


def clean_rationale(answer: str, final_answer: str) -> str:
    text = str(answer or "").strip()
    if not text:
        return f"The answer is {final_answer}."
    text = re.sub(r"<<([^<>]+)>>", "", text)
    text = re.sub(r"####\s*([-+.,/$\d]+)", r"So the answer is \1.", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = " ".join(lines)
    if final_answer and final_answer not in text[-80:]:
        text = f"{text} So the answer is {final_answer}."
    return text


def ai_like_solution(rationale: str, final_answer: str) -> str:
    sentences = split_sentences(rationale)
    body = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences[:6]))
    return (
        "### Step-by-step solution\n\n"
        f"{body}\n\n"
        f"**Final Answer:** \\boxed{{{final_answer}}}"
    )


def conversational_solution(rationale: str, final_answer: str) -> str:
    sentences = split_sentences(rationale)
    if len(sentences) <= 3:
        text = " ".join(sentences)
    else:
        midpoint = (len(sentences) + 1) // 2
        text = " ".join(sentences[:midpoint]) + "\n\n" + " ".join(sentences[midpoint:])
    if final_answer and final_answer not in text[-100:]:
        text = f"{text}\n\nSo the answer is {final_answer}."
    return text.strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def wrong_answer(final_answer: str) -> str:
    raw = str(final_answer or "").replace(",", "").replace("$", "").strip()
    try:
        value = float(raw)
        if value.is_integer():
            return str(int(value + 1))
        return f"{value + 1:.2f}".rstrip("0").rstrip(".")
    except ValueError:
        return raw + "1"


def wrong_solution(rationale: str, final_answer: str) -> str:
    bad = wrong_answer(final_answer)
    sentences = split_sentences(rationale)
    prefix = " ".join(sentences[: max(1, min(3, len(sentences)))])
    return f"{prefix} I made a final arithmetic slip, so the answer is {bad}."


def split_for_index(index: int, train_end: int, dev_end: int, eval_end: int) -> str:
    if index <= train_end:
        return "train_update"
    if index <= dev_end:
        return "dev"
    if index <= eval_end:
        return "eval"
    return "unused"


def state_for(index: int, question: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": FIRST_MESSAGE.format(index=index, question=question)},
        ],
        "latest_user_message": FIRST_MESSAGE.format(index=index, question=question),
        "prompt_text": "",
    }


def make_action(text: str) -> dict[str, Any]:
    return {"text": text, "raw_text": text, "tool_calls": []}


def build(args: argparse.Namespace) -> dict[str, Any]:
    openclaw_rows = read_json(args.dataset)
    if not isinstance(openclaw_rows, list):
        raise SystemExit(f"{args.dataset} must contain a JSON list")
    hf_answers = load_hf_gsm8k_test()

    trajectories: list[dict[str, Any]] = []
    instances: list[dict[str, Any]] = []
    local_hpr_pairs: list[dict[str, Any]] = []
    all_hpr_pairs: list[dict[str, Any]] = []
    ppo_rewards: list[dict[str, Any]] = []
    missing_rationale = 0

    last_index = min(args.eval_end, len(openclaw_rows) - 1)
    for index in range(args.train_start, last_index + 1):
        row = openclaw_rows[index]
        question = str(row.get("question") or "").strip()
        final_answer = str(row.get("ground_truth_answer") or "").strip()
        full_answer = hf_answers.get(question, "")
        if not full_answer:
            missing_rationale += 1
        rationale = clean_rationale(full_answer, final_answer)
        chosen_text = conversational_solution(rationale, final_answer)
        rejected_style_text = ai_like_solution(rationale, final_answer)
        rejected_wrong_text = wrong_solution(rationale, final_answer)
        split = split_for_index(index, args.train_end, args.dev_end, args.eval_end)
        if split == "unused":
            continue

        task_id = f"gsm8k-{index}"
        traj_id = stable_id("openclaw-gsm8k", task_id, prefix="traj_")
        state_ref = f"{traj_id}:turn:1:state"
        state = state_for(index, question)

        turns = [
            {
                "turn_id": 1,
                "state": state,
                "action": make_action(rejected_style_text),
                "next_state": {"role": "user", "content": LOCAL_FEEDBACK},
                "tool_calls": [],
                "raw_messages": state["messages"],
            },
            {
                "turn_id": 2,
                "state": state,
                "action": make_action(chosen_text),
                "next_state": {"role": "evaluator", "content": DELAYED_FEEDBACK_POS},
                "tool_calls": [],
                "raw_messages": state["messages"],
            },
            {
                "turn_id": 3,
                "state": state,
                "action": make_action(rejected_wrong_text),
                "next_state": {"role": "evaluator", "content": DELAYED_FEEDBACK_NEG},
                "tool_calls": [],
                "raw_messages": state["messages"],
            },
        ]

        local_inst = {
            "instance_id": stable_id(traj_id, "local", prefix="fb_"),
            "trajectory_id": traj_id,
            "turn_id": 1,
            "benchmark": "openclaw_personal_gsm8k",
            "domain": "gsm8k_student",
            "split": split,
            "task_id": task_id,
            "state_ref": state_ref,
            "action": make_action(rejected_style_text),
            "next_state": turns[0]["next_state"],
            "feedback_text": LOCAL_FEEDBACK,
            "gold_feedback_type": "local_preference",
            "gold_label_source": "openclaw_style_rule",
            "router_feedback_type": "local_preference",
            "router_model": "rule-router-v0",
            "router_confidence": 1.0,
            "local_label": -1,
            "critique": "The solution uses conspicuous AI-style formatting.",
            "hint": "Rewrite naturally while preserving all reasoning and the answer.",
            "hpr": {
                "accepted": True,
                "chosen": {"text": chosen_text},
                "rejected": {"text": rejected_style_text},
                "judge_model": "openclaw_style_rule",
                "judge_score": 1.0,
            },
            "ppo": None,
        }
        pos_inst = delayed_instance(
            traj_id, task_id, split, state_ref, chosen_text, DELAYED_FEEDBACK_POS, 1.0, "positive"
        )
        neg_inst = delayed_instance(
            traj_id, task_id, split, state_ref, rejected_wrong_text, DELAYED_FEEDBACK_NEG, 0.0, "negative"
        )
        instances.extend([local_inst, pos_inst, neg_inst])

        local_pair = {
            "pair_id": stable_id(local_inst["instance_id"], "hpr", prefix="hpr_"),
            "instance_id": local_inst["instance_id"],
            "trajectory_id": traj_id,
            "turn_id": 1,
            "split": split,
            "task_id": task_id,
            "benchmark": "openclaw_personal_gsm8k",
            "domain": "gsm8k_student",
            "prompt": state,
            "chosen": {"text": chosen_text},
            "rejected": {"text": rejected_style_text},
            "feedback_text": LOCAL_FEEDBACK,
            "feedback_type": "local_preference",
            "construction": "openclaw_style_pair",
        }
        delayed_pair = {
            "pair_id": stable_id(pos_inst["instance_id"], neg_inst["instance_id"], "hpr", prefix="hpr_"),
            "instance_id": pos_inst["instance_id"],
            "trajectory_id": traj_id,
            "turn_id": 2,
            "split": split,
            "task_id": task_id,
            "benchmark": "openclaw_personal_gsm8k",
            "domain": "gsm8k_student",
            "prompt": state,
            "chosen": {"text": chosen_text},
            "rejected": {"text": rejected_wrong_text},
            "feedback_text": "The final answer should match the ground-truth solution.",
            "feedback_type": "delayed_trajectory_outcome",
            "construction": "openclaw_delayed_pair_for_hpr_all",
        }
        local_hpr_pairs.append(local_pair)
        all_hpr_pairs.extend([local_pair, delayed_pair])

        for inst in [pos_inst, neg_inst]:
            ppo_rewards.append(
                {
                    "reward_id": stable_id(inst["instance_id"], "ppo", prefix="ppo_"),
                    "instance_id": inst["instance_id"],
                    "trajectory_id": traj_id,
                    "turn_id": inst["turn_id"],
                    "split": split,
                    "task_id": task_id,
                    "benchmark": "openclaw_personal_gsm8k",
                    "domain": "gsm8k_student",
                    "reward": inst["ppo"]["reward"],
                    "reward_source": "gsm8k_answer_rule",
                    "feedback_type": "delayed_trajectory_outcome",
                }
            )

        trajectories.append(
            {
                "trajectory_id": traj_id,
                "benchmark": "openclaw_personal_gsm8k",
                "domain": "gsm8k_student",
                "split": split,
                "task_id": task_id,
                "task_index": index,
                "seed": None,
                "policy_checkpoint": args.policy_checkpoint,
                "router_checkpoint": args.router_checkpoint,
                "environment_version": "openclaw-gsm8k-offline-v1",
                "turns": turns,
                "final_outcome": {
                    "success": True,
                    "score": 1.0,
                    "evaluator": "gsm8k_ground_truth",
                    "raw": {"ground_truth_answer": final_answer},
                },
            }
        )

    counts = {
        "trajectories": len(trajectories),
        "feedback_instances": len(instances),
        "hpr_pairs": len(local_hpr_pairs),
        "hpr_pairs_all": len(all_hpr_pairs),
        "ppo_rewards": len(ppo_rewards),
        "missing_hf_rationales": missing_rationale,
        "trajectories_by_split": dict(Counter(t["split"] for t in trajectories)),
        "instances_by_split": dict(Counter(i["split"] for i in instances)),
        "instances_by_feedback_type": dict(Counter(i["gold_feedback_type"] for i in instances)),
    }
    return {
        "trajectories": trajectories,
        "instances": instances,
        "hpr_pairs": local_hpr_pairs,
        "hpr_pairs_all": all_hpr_pairs,
        "ppo_rewards": ppo_rewards,
        "counts": counts,
    }


def delayed_instance(
    traj_id: str,
    task_id: str,
    split: str,
    state_ref: str,
    response: str,
    feedback_text: str,
    reward: float,
    suffix: str,
) -> dict[str, Any]:
    return {
        "instance_id": stable_id(traj_id, "delayed", suffix, prefix="fb_"),
        "trajectory_id": traj_id,
        "turn_id": 2 if reward > 0 else 3,
        "benchmark": "openclaw_personal_gsm8k",
        "domain": "gsm8k_student",
        "split": split,
        "task_id": task_id,
        "state_ref": state_ref,
        "action": make_action(response),
        "next_state": {"role": "evaluator", "content": feedback_text},
        "feedback_text": feedback_text,
        "gold_feedback_type": "delayed_trajectory_outcome",
        "gold_label_source": "gsm8k_answer_rule",
        "router_feedback_type": "delayed_trajectory_outcome",
        "router_model": "rule-router-v0",
        "router_confidence": 1.0,
        "local_label": int(reward),
        "critique": None,
        "hint": None,
        "hpr": None,
        "ppo": {"reward": reward, "reward_source": "gsm8k_answer_rule"},
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train-start", type=int, default=0)
    p.add_argument("--train-end", type=int, default=999)
    p.add_argument("--dev-end", type=int, default=1099)
    p.add_argument("--eval-end", type=int, default=1318)
    p.add_argument("--policy-checkpoint", default="Qwen3-4B-Thinking-2507")
    p.add_argument("--router-checkpoint", default="rule-router-v0")
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
    write_jsonl(files["trajectories"], built["trajectories"])
    write_jsonl(files["feedback_instances"], built["instances"])
    write_jsonl(files["hpr_pairs"], built["hpr_pairs"])
    write_jsonl(files["hpr_pairs_all"], built["hpr_pairs_all"])
    write_jsonl(files["ppo_rewards"], built["ppo_rewards"])
    manifest = {
        "version": "openclaw-gsm8k-routed-v1",
        "builder": "data/scripts/build_openclaw_gsm8k_artifacts.py",
        "dataset": str(args.dataset),
        "files": {k: str(v) for k, v in files.items()},
        "split_rule": {
            "train_update": [args.train_start, args.train_end],
            "dev": [args.train_end + 1, args.dev_end],
            "eval": [args.dev_end + 1, args.eval_end],
        },
        "counts": built["counts"],
        "leakage": {
            "task_overlap": [],
            "trajectory_overlap": [],
            "note": "Splits are disjoint by GSM8K task index.",
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(args.output_dir), "counts": built["counts"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
