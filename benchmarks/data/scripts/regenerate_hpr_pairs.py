#!/usr/bin/env python3
"""Regenerate HPR chosen responses for feedback-pair candidates."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable


LOCAL_TYPES = {"local_preference", "local_correction"}
NON_NEUTRAL_TYPES = {
    "local_preference",
    "local_correction",
    "tool_api_outcome",
    "delayed_trajectory_outcome",
}


HPR_SYSTEM_PROMPT = """\
You regenerate a corrected assistant response for hindsight preference training.

Given the prior conversation context, the original assistant response, and the
user's local feedback, write the response the assistant should have given at
that same turn. The new response must:
- satisfy the user's context-supported feedback;
- preserve all valid task-relevant content from the original response;
- avoid adding facts that were not available at that point;
- output only the revised assistant response, with no explanation.
"""


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


def compact_text(text: str, limit: int = 5000) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n...[truncated]...\n" + text[-limit // 2 :]


def flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(content)


def build_state_index(trajectories_path: Path | None) -> dict[str, dict[str, Any]]:
    if trajectories_path is None:
        return {}
    states: dict[str, dict[str, Any]] = {}
    for traj in read_jsonl(trajectories_path):
        traj_id = str(traj.get("trajectory_id"))
        for turn in traj.get("turns") or []:
            turn_id = turn.get("turn_id")
            state = turn.get("state") or {}
            if isinstance(state, dict):
                states[f"{traj_id}:turn:{turn_id}:state"] = state
    return states


def resolve_prompt(pair: dict[str, Any], states: dict[str, dict[str, Any]]) -> dict[str, Any]:
    prompt = pair.get("prompt") or {}
    if isinstance(prompt, dict) and prompt.get("messages"):
        return prompt
    state_ref = prompt.get("state_ref") if isinstance(prompt, dict) else None
    if state_ref and state_ref in states:
        return states[state_ref]
    traj_id = pair.get("trajectory_id")
    turn_id = pair.get("turn_id")
    if traj_id is not None and turn_id is not None:
        key = f"{traj_id}:turn:{turn_id}:state"
        if key in states:
            return states[key]
    return prompt if isinstance(prompt, dict) else {}


def prompt_for_pair(pair: dict[str, Any], states: dict[str, dict[str, Any]]) -> str:
    prompt = resolve_prompt(pair, states)
    messages = prompt.get("messages") or []
    context_lines = []
    for msg in messages[-10:]:
        role = msg.get("role")
        content = flatten_content(msg.get("content"))
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            content = content + "\nTOOL_CALLS: " + json.dumps(tool_calls, ensure_ascii=False)
        context_lines.append(f"{role}: {compact_text(content, 1000)}")

    rejected = (pair.get("rejected") or {}).get("text") or ""
    return (
        "Prior context:\n"
        f"{compact_text(chr(10).join(context_lines), 4500)}\n\n"
        "Original assistant response:\n"
        f"{compact_text(rejected, 2500)}\n\n"
        "User feedback:\n"
        f"{compact_text(str(pair.get('feedback_text') or ''), 2500)}\n\n"
        f"Feedback type: {pair.get('feedback_type')}\n\n"
        "Write only the revised assistant response."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hpr-pairs", type=Path, required=True)
    p.add_argument("--feedback-instances", type=Path, default=None)
    p.add_argument("--output-hpr-pairs", type=Path, required=True)
    p.add_argument("--output-feedback-instances", type=Path, default=None)
    p.add_argument("--model", default="qwen3-4b")
    p.add_argument("--base-url", default=os.environ.get("HPR_BASE_URL", ""))
    p.add_argument("--api-key-env", default="HPR_API_KEY")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--splits", nargs="+", default=["train_update", "train", "dev"])
    p.add_argument("--feedback-types", nargs="+", default=sorted(LOCAL_TYPES))
    p.add_argument("--trajectories", type=Path, default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args(argv)


def stream_chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_retries: int,
) -> str:
    try:
        import httpx
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("HPR regeneration requires openai and httpx packages") from e

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(trust_env=False),
    )

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                stream=True,
            )
            parts: list[str] = []
            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if not delta:
                    continue
                piece = getattr(delta, "content", None)
                if piece:
                    parts.append(piece)
            text = "".join(parts).strip()
            if text:
                return text
            raise RuntimeError("empty completion")
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(min(2**attempt, 8))
                continue
            raise RuntimeError(f"HPR regeneration failed after retries: {last_error}") from e
    raise RuntimeError(f"HPR regeneration failed: {last_error}")


def regenerate_one(
    pair: dict[str, Any],
    *,
    args: argparse.Namespace,
    api_key: str,
    states: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if args.skip_existing and pair.get("chosen"):
        out = dict(pair)
        out["construction"] = pair.get("construction") or "existing"
        return out
    messages = [
        {"role": "system", "content": HPR_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_for_pair(pair, states)},
    ]
    text = stream_chat_completion(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        messages=messages,
        max_retries=args.max_retries,
    )
    out = dict(pair)
    out["chosen"] = {"text": text}
    out["generator_model"] = args.model
    out["construction"] = "hindsight_regenerated"
    return out


def update_instances(
    instances: list[dict[str, Any]],
    regenerated_by_instance: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for row in instances:
        inst_id = str(row.get("instance_id"))
        pair = regenerated_by_instance.get(inst_id)
        if not pair:
            out.append(row)
            continue
        row = dict(row)
        chosen = pair.get("chosen")
        rejected = pair.get("rejected")
        row["hpr"] = {
            "accepted": bool(chosen and rejected),
            "chosen": chosen,
            "rejected": rejected,
            "judge_model": pair.get("generator_model"),
            "judge_score": 1.0 if chosen and rejected else 0.0,
            "skip_reason": None if chosen and rejected else "regeneration_failed",
        }
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"missing API key env var: {args.api_key_env}")

    pairs = read_jsonl(args.hpr_pairs)
    allowed_splits = set(args.splits)
    allowed_feedback_types = set(args.feedback_types)
    states = build_state_index(args.trajectories)
    selected = [
        pair
        for pair in pairs
        if pair.get("feedback_type") in allowed_feedback_types
        and pair.get("split") in allowed_splits
        and (pair.get("rejected") or {}).get("text")
    ]
    if args.limit > 0:
        selected = selected[: args.limit]

    regenerated: list[dict[str, Any]] = []
    selected_ids = {id(pair) for pair in selected}
    if args.max_workers <= 1:
        for i, pair in enumerate(selected, 1):
            try:
                out = regenerate_one(pair, args=args, api_key=api_key, states=states)
            except Exception as e:
                if not args.continue_on_error:
                    raise
                out = dict(pair)
                out["chosen"] = None
                out["skip_reason"] = f"regeneration_error: {e}"
            regenerated.append(out)
            chosen = out.get("chosen") or {}
            chosen_text = str(chosen.get("text") or "")
            print(
                json.dumps(
                    {
                        "i": i,
                        "instance_id": pair.get("instance_id"),
                        "split": pair.get("split"),
                        "feedback_type": pair.get("feedback_type"),
                        "chosen_chars": len(chosen_text),
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            future_map = {
                executor.submit(regenerate_one, pair, args=args, api_key=api_key, states=states): (i, pair)
                for i, pair in enumerate(selected, 1)
            }
            for future in as_completed(future_map):
                i, pair = future_map[future]
                try:
                    out = future.result()
                except Exception as e:
                    if not args.continue_on_error:
                        raise
                    out = dict(pair)
                    out["chosen"] = None
                    out["skip_reason"] = f"regeneration_error: {e}"
                regenerated.append(out)
                chosen = out.get("chosen") or {}
                chosen_text = str(chosen.get("text") or "")
                print(
                    json.dumps(
                        {
                            "i": i,
                            "instance_id": pair.get("instance_id"),
                            "split": pair.get("split"),
                            "feedback_type": pair.get("feedback_type"),
                            "chosen_chars": len(chosen_text),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    # Keep non-selected pairs in the output as explicit skipped rows.
    regenerated_by_pair_id = {row.get("pair_id"): row for row in regenerated}
    output_pairs: list[dict[str, Any]] = []
    for pair in pairs:
        if pair.get("pair_id") in regenerated_by_pair_id:
            output_pairs.append(regenerated_by_pair_id[pair.get("pair_id")])
        elif id(pair) not in selected_ids:
            out = dict(pair)
            out.setdefault("skip_reason", "not_selected_for_regeneration")
            output_pairs.append(out)

    n_pairs = write_jsonl(args.output_hpr_pairs, output_pairs)

    n_instances = None
    if args.feedback_instances and args.output_feedback_instances:
        instances = read_jsonl(args.feedback_instances)
        regenerated_by_instance = {
            str(pair.get("instance_id")): pair
            for pair in output_pairs
            if pair.get("chosen") and pair.get("rejected")
        }
        updated_instances = update_instances(instances, regenerated_by_instance)
        n_instances = write_jsonl(args.output_feedback_instances, updated_instances)

    print(
        json.dumps(
            {
                "input_hpr_pairs": str(args.hpr_pairs),
                "output_hpr_pairs": str(args.output_hpr_pairs),
                "output_feedback_instances": str(args.output_feedback_instances)
                if args.output_feedback_instances
                else None,
                "pairs_written": n_pairs,
                "instances_written": n_instances,
                "regenerated": len(regenerated),
                "model": args.model,
                "max_workers": args.max_workers,
                "feedback_types": args.feedback_types,
                "states_indexed": len(states),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
