#!/usr/bin/env python3
"""Convert tau3 real rollout records into routed-feedback data artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


LOCAL_TYPES = {"local_preference", "local_correction"}
TOOL_OUTCOME = "tool_api_outcome"
DELAYED_OUTCOME = "delayed_trajectory_outcome"
NEUTRAL = "neutral"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
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
                yield obj


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


def flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return " ".join(p for p in parts if p)
    return str(content)


def message_text(msg: dict[str, Any] | None) -> str:
    if not msg:
        return ""
    content = flatten_content(msg.get("content"))
    tool_calls = msg.get("tool_calls") or []
    if tool_calls and not content:
        return json.dumps(tool_calls, ensure_ascii=False, sort_keys=True)
    if tool_calls:
        return content + "\n" + json.dumps(tool_calls, ensure_ascii=False, sort_keys=True)
    return content


def compact_text(text: str, limit: int = 2400) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n...[truncated]...\n" + text[-limit // 2 :]


def latest_user_message(messages: list[dict[str, Any]], upto: int) -> str:
    for msg in reversed(messages[:upto]):
        if msg.get("role") == "user":
            return flatten_content(msg.get("content"))
    return ""


def prior_context(messages: list[dict[str, Any]], upto: int) -> dict[str, Any]:
    return {
        "messages": messages[:upto],
        "latest_user_message": latest_user_message(messages, upto),
        "prompt_text": "\n".join(
            f"{m.get('role')}: {compact_text(message_text(m), 800)}"
            for m in messages[max(0, upto - 8) : upto]
        ),
    }


def visible_action(action_msg: dict[str, Any]) -> dict[str, Any]:
    text = message_text(action_msg)
    return {
        "text": flatten_content(action_msg.get("content")),
        "raw_text": text,
        "tool_calls": action_msg.get("tool_calls") or [],
    }


def role_of(msg: dict[str, Any] | None) -> str:
    return str((msg or {}).get("role") or "")


def tool_success(tool_msg: dict[str, Any]) -> bool:
    text = flatten_content(tool_msg.get("content")).lower()
    if any(marker in text for marker in ["error", "exception", "failed", "invalid"]):
        return False
    return True


def user_feedback_type(feedback_text: str) -> str:
    text = feedback_text.lower()
    new_requirement_markers = [
        "actually i now",
        "i just realized",
        "now i want",
        "new preference",
        "also can you",
        "by the way",
    ]
    correction_markers = [
        "wrong",
        "incorrect",
        "not right",
        "fix",
        "change",
        "correct",
        "redo",
        "try again",
        "not that",
        "i meant",
        "you should",
        "should have",
        "instead",
    ]
    preference_markers = [
        "rewrite",
        "more natural",
        "less ai",
        "ai-like",
        "too structured",
        "too polished",
        "casual",
        "tone",
        "style",
        "cheaper",
        "cheap",
        "faster",
        "shorter",
        "simpler",
    ]
    if any(p in text for p in new_requirement_markers):
        return NEUTRAL
    if any(p in text for p in correction_markers):
        return "local_correction"
    if any(p in text for p in preference_markers):
        return "local_preference"
    return NEUTRAL


def is_context_supported_feedback(messages: list[dict[str, Any]], assistant_index: int) -> bool:
    """Whether a user message can critique a prior substantive assistant action."""
    prior_user_seen = any(m.get("role") == "user" for m in messages[:assistant_index])
    prior_tool_seen = any(m.get("role") == "tool" for m in messages[:assistant_index])
    assistant_text = flatten_content(messages[assistant_index].get("content")).strip().lower()
    greeting_only = assistant_text in {
        "hi! how can i help you today?",
        "hello! how can i help you today?",
        "how can i help you today?",
    }
    return prior_user_seen or prior_tool_seen or not greeting_only


def critique_from_feedback(feedback_type: str, feedback_text: str) -> str | None:
    if feedback_type == "local_preference":
        return feedback_text or "The user expressed a local preference about the previous response."
    if feedback_type == "local_correction":
        return feedback_text or "The user corrected the previous response."
    return None


def hint_from_feedback(feedback_type: str) -> str | None:
    if feedback_type == "local_preference":
        return "Revise the response to satisfy the context-supported local preference without changing unrelated behavior."
    if feedback_type == "local_correction":
        return "Revise the response to address the correction while preserving valid prior work."
    return None


def reward_components(reward_info: dict[str, Any] | None) -> dict[str, Any]:
    if not reward_info:
        return {}
    return {
        k: v
        for k, v in reward_info.items()
        if k not in {"reward"} and not k.startswith("_")
    }


def final_reward(simulation: dict[str, Any]) -> float | None:
    reward_info = simulation.get("reward_info") or {}
    reward = reward_info.get("reward")
    if reward is None:
        return None
    try:
        return float(reward)
    except (TypeError, ValueError):
        return None


def make_terminal_next_state(simulation: dict[str, Any]) -> dict[str, Any]:
    reward_info = simulation.get("reward_info") or {}
    return {
        "role": "terminal",
        "content": json.dumps(
            {
                "termination_reason": simulation.get("termination_reason"),
                "reward": reward_info.get("reward"),
                "reward_components": reward_components(reward_info),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def add_terminal_feedback(
    *,
    simulation: dict[str, Any],
    rec: dict[str, Any],
    trajectory_id: str,
    turns: list[dict[str, Any]],
    instances: list[dict[str, Any]],
    ppo_rewards: list[dict[str, Any]],
    router_checkpoint: str,
) -> None:
    messages = simulation.get("messages") or []
    if not turns:
        return
    last_turn = turns[-1]
    turn_id = len(turns)
    reward = final_reward(simulation)
    label = 1 if reward and reward > 0 else 0
    next_state = make_terminal_next_state(simulation)
    state = {
        "messages": messages,
        "latest_user_message": latest_user_message(messages, len(messages)),
        "prompt_text": "\n".join(
            f"{m.get('role')}: {compact_text(message_text(m), 800)}"
            for m in messages[max(0, len(messages) - 8) :]
        ),
    }
    action = last_turn["action"]
    terminal_turn = {
        "turn_id": turn_id,
        "state": state,
        "action": action,
        "next_state": next_state,
        "tool_calls": action.get("tool_calls") or [],
        "raw_messages": messages[max(0, len(messages) - 8) :],
        "terminal_feedback": True,
    }
    turns.append(terminal_turn)

    instance_id = stable_id(trajectory_id, turn_id, DELAYED_OUTCOME, reward, prefix="fb_")
    instance = {
        "instance_id": instance_id,
        "trajectory_id": trajectory_id,
        "turn_id": turn_id,
        "benchmark": "tau3_bench",
        "domain": rec.get("domain"),
        "split": rec.get("split"),
        "task_id": rec.get("task_id"),
        "state_ref": f"{trajectory_id}:turn:{turn_id}:state",
        "action": action,
        "next_state": next_state,
        "feedback_text": next_state.get("content"),
        "gold_feedback_type": DELAYED_OUTCOME,
        "gold_label_source": "tau3_bench_reward",
        "router_feedback_type": DELAYED_OUTCOME,
        "router_model": router_checkpoint,
        "router_confidence": 1.0,
        "local_label": label,
        "critique": None,
        "hint": None,
        "hpr": None,
        "ppo": {"reward": reward, "reward_source": "tau3_bench_reward"},
        "raw_id": simulation.get("id"),
    }
    instances.append(instance)
    ppo_rewards.append(
        {
            "reward_id": stable_id(instance_id, "ppo", prefix="ppo_"),
            "instance_id": instance_id,
            "trajectory_id": trajectory_id,
            "turn_id": turn_id,
            "split": rec.get("split"),
            "benchmark": "tau3_bench",
            "domain": rec.get("domain"),
            "task_id": rec.get("task_id"),
            "reward": reward,
            "reward_source": "tau3_bench_reward",
            "feedback_type": DELAYED_OUTCOME,
        }
    )


def build_from_rollouts(
    records: list[dict[str, Any]],
    *,
    policy_checkpoint: str,
    router_checkpoint: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    trajectories: list[dict[str, Any]] = []
    instances: list[dict[str, Any]] = []
    hpr_pairs: list[dict[str, Any]] = []
    ppo_rewards: list[dict[str, Any]] = []

    for rec in records:
        if rec.get("status") != "ok":
            continue
        simulation = rec.get("simulation") or {}
        messages = simulation.get("messages") or []
        domain = str(rec.get("domain"))
        split = str(rec.get("split"))
        task_id = str(rec.get("task_id"))
        seed = rec.get("seed")
        trajectory_id = stable_id("tau3_bench", domain, split, task_id, seed, simulation.get("id"), prefix="traj_")
        turns: list[dict[str, Any]] = []

        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            action = visible_action(msg)
            if not action["raw_text"].strip():
                continue
            next_msg = messages[i + 1] if i + 1 < len(messages) else None
            next_state = next_msg or make_terminal_next_state(simulation)
            next_role = role_of(next_state)
            feedback_text = flatten_content(next_state.get("content"))
            if next_role == "tool":
                feedback_type = TOOL_OUTCOME
                label = 1 if tool_success(next_state) else -1
            elif next_role == "user":
                feedback_type = user_feedback_type(feedback_text)
                if feedback_type in LOCAL_TYPES and not is_context_supported_feedback(messages, i):
                    feedback_type = NEUTRAL
                label = -1 if feedback_type in LOCAL_TYPES else None
            else:
                feedback_type = NEUTRAL
                label = None

            turn_id = len(turns)
            state = prior_context(messages, i)
            turn = {
                "turn_id": turn_id,
                "state": state,
                "action": action,
                "next_state": next_state,
                "tool_calls": action.get("tool_calls") or [],
                "raw_messages": messages[max(0, i - 8) : i + 2],
            }
            turns.append(turn)

            instance_id = stable_id(trajectory_id, turn_id, feedback_type, feedback_text, prefix="fb_")
            instance = {
                "instance_id": instance_id,
                "trajectory_id": trajectory_id,
                "turn_id": turn_id,
                "benchmark": "tau3_bench",
                "domain": domain,
                "split": split,
                "task_id": task_id,
                "state_ref": f"{trajectory_id}:turn:{turn_id}:state",
                "action": action,
                "next_state": next_state,
                "feedback_text": feedback_text or None,
                "gold_feedback_type": feedback_type,
                "gold_label_source": "tau3_bench_environment_rule",
                "router_feedback_type": feedback_type,
                "router_model": router_checkpoint,
                "router_confidence": 1.0,
                "local_label": label,
                "critique": critique_from_feedback(feedback_type, feedback_text),
                "hint": hint_from_feedback(feedback_type),
                "hpr": None,
                "ppo": None,
                "raw_id": simulation.get("id"),
            }

            if feedback_type in LOCAL_TYPES and action["text"].strip():
                hpr_pairs.append(
                    {
                        "pair_id": stable_id(instance_id, "hpr", prefix="hpr_"),
                        "instance_id": instance_id,
                        "trajectory_id": trajectory_id,
                        "split": split,
                        "benchmark": "tau3_bench",
                        "domain": domain,
                        "task_id": task_id,
                        "prompt": state,
                        "chosen": None,
                        "rejected": {"text": action["text"]},
                        "feedback_text": feedback_text,
                        "feedback_type": feedback_type,
                        "construction": "needs_hindsight_regeneration",
                    }
                )
                instance["hpr"] = {
                    "accepted": False,
                    "chosen": None,
                    "rejected": {"text": action["text"]},
                    "judge_model": None,
                    "judge_score": None,
                    "skip_reason": "needs_hindsight_regeneration",
                }

            reward = None
            reward_source = None
            if feedback_type == TOOL_OUTCOME and label is not None:
                reward = float(label)
                reward_source = "tool_message_rule"
            if reward is not None:
                instance["ppo"] = {"reward": reward, "reward_source": reward_source}
                ppo_rewards.append(
                    {
                        "reward_id": stable_id(instance_id, "ppo", prefix="ppo_"),
                        "instance_id": instance_id,
                        "trajectory_id": trajectory_id,
                        "turn_id": turn_id,
                        "split": split,
                        "benchmark": "tau3_bench",
                        "domain": domain,
                        "task_id": task_id,
                        "reward": reward,
                        "reward_source": reward_source,
                        "feedback_type": feedback_type,
                    }
                )

            instances.append(instance)

        add_terminal_feedback(
            simulation=simulation,
            rec=rec,
            trajectory_id=trajectory_id,
            turns=turns,
            instances=instances,
            ppo_rewards=ppo_rewards,
            router_checkpoint=router_checkpoint,
        )

        reward = final_reward(simulation)
        trajectories.append(
            {
                "trajectory_id": trajectory_id,
                "benchmark": "tau3_bench",
                "domain": domain,
                "split": split,
                "task_id": task_id,
                "task_index": rec.get("task_index"),
                "seed": seed,
                "policy_checkpoint": policy_checkpoint,
                "router_checkpoint": router_checkpoint,
                "environment_version": "tau2-bench",
                "turns": turns,
                "final_outcome": {
                    "success": bool(reward and reward > 0),
                    "score": reward,
                    "evaluator": "tau2_bench",
                    "raw": {
                        "simulation_id": simulation.get("id"),
                        "termination_reason": simulation.get("termination_reason"),
                        "reward_info": simulation.get("reward_info"),
                    },
                },
            }
        )

    return trajectories, instances, hpr_pairs, ppo_rewards


def counts_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(r.get(key)) for r in rows).items()))


def split_overlap_report(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for traj in trajectories:
        by_split[str(traj.get("split"))].add((str(traj.get("domain")), str(traj.get("task_id"))))
    out: dict[str, Any] = {}
    splits = sorted(by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            overlap = sorted(by_split[a] & by_split[b])
            out[f"{a}__{b}"] = [f"{d}:{tid}" for d, tid in overlap]
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollouts", type=Path, required=True, help="raw_simulations.jsonl or its containing rollout directory")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--policy-checkpoint", default="Qwen3-4B")
    p.add_argument("--router-checkpoint", default="rule-router-v0")
    p.add_argument("--allow-empty", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rollout_path = args.rollouts
    if rollout_path.is_dir():
        rollout_path = rollout_path / "raw_simulations.jsonl"
    if not rollout_path.exists():
        raise SystemExit(f"missing rollout file: {rollout_path}")
    records = list(read_jsonl(rollout_path))
    if not records and not args.allow_empty:
        raise SystemExit("no rollout records")

    trajectories, instances, hpr_pairs, ppo_rewards = build_from_rollouts(
        records,
        policy_checkpoint=args.policy_checkpoint,
        router_checkpoint=args.router_checkpoint,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "trajectories": args.output_dir / "trajectories.jsonl",
        "feedback_instances": args.output_dir / "feedback_instances.jsonl",
        "hpr_pairs": args.output_dir / "hpr_pairs.jsonl",
        "ppo_rewards": args.output_dir / "ppo_rewards.jsonl",
        "manifest": args.output_dir / "manifest.json",
    }
    n_traj = write_jsonl(paths["trajectories"], trajectories)
    n_inst = write_jsonl(paths["feedback_instances"], instances)
    n_hpr = write_jsonl(paths["hpr_pairs"], hpr_pairs)
    n_ppo = write_jsonl(paths["ppo_rewards"], ppo_rewards)

    leakage = split_overlap_report(trajectories)
    manifest = {
        "version": "2026-05-20",
        "builder": "data/scripts/build_tau3_routed_data.py",
        "input": str(rollout_path),
        "policy_checkpoint": args.policy_checkpoint,
        "router_checkpoint": args.router_checkpoint,
        "counts": {
            "trajectories": n_traj,
            "feedback_instances": n_inst,
            "hpr_pairs_pending_regeneration": n_hpr,
            "ppo_rewards": n_ppo,
            "trajectories_by_split": counts_by(trajectories, "split"),
            "instances_by_split": counts_by(instances, "split"),
            "instances_by_feedback_type": counts_by(instances, "gold_feedback_type"),
        },
        "leakage": leakage,
        "notes": [
            "HPR pairs created by this converter contain rejected responses and feedback only; chosen responses must be generated by the HPR regeneration stage.",
            "Final tau3 rewards come from tau2-bench reward_info and are kept only for their original split.",
            "No reported_test trajectory should be consumed for updates.",
        ],
    }
    with paths["manifest"].open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    bad_overlap = {k: v for k, v in leakage.items() if v}
    if bad_overlap:
        raise SystemExit(f"train/eval leakage detected: {bad_overlap}")

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "trajectories": n_traj,
        "feedback_instances": n_inst,
        "hpr_pairs_pending_regeneration": n_hpr,
        "ppo_rewards": n_ppo,
        "feedback_types": manifest["counts"]["instances_by_feedback_type"],
        "leakage": leakage,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
