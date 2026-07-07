#!/usr/bin/env python3
"""Route feedback instances with either rule labels or an OpenAI-compatible LLM.

The script never stores API keys. For online labeling, pass credentials through
environment variables, for example:

  HPR_API_KEY=... python data/scripts/route_feedback_labels.py \
    --provider oneapi --model qwen3-max \
    --input feedback_instances.jsonl --output feedback_instances.qwen_router.jsonl
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable


FEEDBACK_TYPES = [
    "local_preference",
    "local_correction",
    "tool_api_outcome",
    "delayed_trajectory_outcome",
    "neutral",
]


ROUTER_SYSTEM_PROMPT = """\
You are a feedback router for multi-turn agent training.

Classify the feedback instance into exactly one label:

local_preference:
  The user is steering how the immediately preceding assistant response should
  be framed, ranked, ordered, worded, or selected. Use this label whenever the
  feedback clearly prefers one of several plausible realizations of the prior
  response and that preference is supported by earlier context. Be generous:
  if the message is a concrete preference about the prior response rather than
  a fresh request, prefer local_preference over neutral.

local_correction:
  The user is revising, fixing, or restating the immediately preceding
  assistant response or action. This includes explicit corrections, re-dos,
  reversals, replacements, and concrete follow-up changes to the prior turn.
  Look for signals such as "no", "actually", "that's wrong", "you missed",
  "I meant", "should have", "instead", or a direct contradiction of the
  previous response.

tool_api_outcome:
  The next state is a tool, API, shell, browser, or environment result. It is
  not a human preference unless the user explicitly critiques the response.

delayed_trajectory_outcome:
  The signal is final task success/failure, tests passed/failed, issue resolved,
  or completion of the whole trajectory.

neutral:
  The feedback is unrelated, truly ambiguous, or a newly introduced
  requirement that was not supported by prior context. Also label routine task
  progress as neutral: user confirmations, yes/no approvals, providing
  requested information, selecting one of the options requested by the
  assistant, saying thanks, or stopping the conversation are neutral unless
  they explicitly correct or critique the previous assistant response. A
  task-content choice is neutral when the assistant explicitly asked the user
  to choose, confirm, provide an ID, provide a reason, or provide a payment
  method.

Decision boundary:
  - "Yes, proceed", "I confirm", or "Use my card" after the assistant asks for
    confirmation is neutral.
  - "Use reservation QKRY03" after the assistant asks which reservation to
    modify is neutral, not local_preference.
  - "Choose option 2" or "I pick the first one" after the assistant asks the
    user to choose is neutral, not local_preference.
  - "I prefer the cheaper option" is neutral if cheapness was not previously
    stated or implied; otherwise route it as local_preference.
  - "I asked for the cheaper option; you showed expensive ones" is
    local_correction.
  - "Show cheaper options first, not fastest ones" is local_preference only
    when the earlier context already made price relevant.

Return compact JSON only:
{"feedback_type": "...", "confidence": 0.0-1.0, "rationale": "..."}
"""


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def compact_text(text: str, limit: int = 2400) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n...[truncated]...\n{tail}"


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


def instance_prompt(row: dict[str, Any]) -> str:
    action = row.get("action") or {}
    next_state = row.get("next_state") or {}
    return (
        "Classify this feedback instance.\n\n"
        f"Benchmark: {row.get('benchmark')}\n"
        f"Domain: {row.get('domain')}\n"
        f"Current split: {row.get('split')}\n\n"
        "Previous assistant action:\n"
        f"{compact_text(str(action.get('text') or ''))}\n\n"
        "Tool calls in previous action:\n"
        f"{compact_text(json.dumps(action.get('tool_calls') or [], ensure_ascii=False))}\n\n"
        "Next state role:\n"
        f"{next_state.get('role')}\n\n"
        "Next state / feedback text:\n"
        f"{compact_text(flatten_content(next_state.get('content')))}\n\n"
        "Important: classify only whether this feedback instance should route "
        "to HPR/local preference or correction, PPO/delayed reward, tool/API "
        "outcome, or be excluded."
    )


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def normalize_label(obj: dict[str, Any] | None) -> tuple[str, float | None, str]:
    if not obj:
        return "neutral", None, "parse_failed"
    label = str(obj.get("feedback_type") or obj.get("label") or "").strip()
    if label not in FEEDBACK_TYPES:
        label = "neutral"
    conf_raw = obj.get("confidence")
    try:
        conf = float(conf_raw)
    except (TypeError, ValueError):
        conf = None
    if conf is not None:
        conf = max(0.0, min(1.0, conf))
    rationale = str(obj.get("rationale") or obj.get("reason") or "")[:800]
    return label, conf, rationale


def stream_chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_retries: int,
    request_timeout: float,
) -> str:
    try:
        import httpx
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("online routing requires openai and httpx packages") from e

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(trust_env=False, timeout=request_timeout),
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
                time.sleep(min(2 ** attempt, 8))
                continue
            raise RuntimeError(f"router call failed after retries: {last_error}") from e

    raise RuntimeError(f"router call failed: {last_error}")


def route_oneapi(row: dict[str, Any], args: argparse.Namespace) -> tuple[str, float | None, str]:
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"missing API key env var: {args.api_key_env}")
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": instance_prompt(row)},
    ]
    text = stream_chat_completion(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        messages=messages,
        max_retries=args.max_retries,
        request_timeout=args.request_timeout,
    )
    return normalize_label(parse_json_object(text))


def route_rule(row: dict[str, Any]) -> tuple[str, float | None, str]:
    label = row.get("gold_feedback_type") or row.get("router_feedback_type") or "neutral"
    if label not in FEEDBACK_TYPES:
        label = "neutral"
    return str(label), 1.0, "copied_existing_rule_label"


def structural_label(row: dict[str, Any]) -> tuple[str, float, str] | None:
    label = row.get("gold_feedback_type") or row.get("router_feedback_type")
    if label in {"tool_api_outcome", "delayed_trajectory_outcome"}:
        return str(label), 1.0, "preserved_structural_outcome_label"
    next_state = row.get("next_state") or {}
    if next_state.get("role") == "tool":
        return "tool_api_outcome", 1.0, "preserved_tool_role_label"
    return None


def apply_label(row: dict[str, Any], *, label_field: str, model: str,
                label: str, confidence: float | None, rationale: str) -> dict[str, Any]:
    out = dict(row)
    if label_field == "router":
        out["router_feedback_type"] = label
        out["router_model"] = model
        out["router_confidence"] = confidence
        out["router_rationale"] = rationale
    elif label_field == "gold":
        out["gold_feedback_type"] = label
        out["gold_label_source"] = model
        out["gold_rationale"] = rationale
    else:
        raise ValueError(f"unknown label_field: {label_field}")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--provider", choices=["rule", "oneapi"], default="rule")
    p.add_argument("--label-field", choices=["router", "gold"], default="router")
    p.add_argument("--model", default=os.environ.get("QWEN_ROUTER_MODEL", "qwen3-max"))
    p.add_argument("--base-url", default=os.environ.get("HPR_BASE_URL", ""))
    p.add_argument("--api-key-env", default="HPR_API_KEY")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--preserve-structural-labels", action="store_true",
                   help="Keep tool/final outcome labels deterministic and use the LLM only for natural-language local feedback.")
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--progress-file", type=Path, default=None)
    p.add_argument("--events-file", type=Path, default=None)
    p.add_argument("--progress-interval", type=float, default=30.0)
    return p.parse_args(argv)


def row_key(row: dict[str, Any], fallback_index: int) -> str:
    return str(row.get("instance_id") or row.get("raw_id") or fallback_index)


def load_existing(path: Path, *, label_field: str, model: str) -> tuple[set[str], Counter[str]]:
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    if not path.exists():
        return seen, counts
    for i, row in enumerate(read_jsonl(path), 1):
        if label_field == "router" and row.get("router_model") != model:
            continue
        if label_field == "gold" and row.get("gold_label_source") != model:
            continue
        label = row.get(f"{label_field}_feedback_type") or "unknown"
        seen.add(row_key(row, i))
        counts[str(label)] += 1
    return seen, counts


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def label_row(row: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], str, float | None, str]:
    try:
        structural = structural_label(row) if args.preserve_structural_labels else None
        if structural is not None:
            label, conf, rationale = structural
        elif args.skip_existing and args.label_field == "router" and row.get("router_model") == args.model:
            label = str(row.get("router_feedback_type") or "neutral")
            conf = row.get("router_confidence")
            rationale = "skipped_existing_input_label"
        elif args.provider == "rule":
            label, conf, rationale = route_rule(row)
        else:
            label, conf, rationale = route_oneapi(row, args)
    except Exception as exc:
        if not args.continue_on_error:
            raise
        label = "neutral"
        conf = None
        rationale = f"router_error:{type(exc).__name__}:{str(exc)[:500]}"

    labeled = apply_label(
        row,
        label_field=args.label_field,
        model=args.model,
        label=label,
        confidence=conf,
        rationale=rationale,
    )
    return labeled, label, conf, rationale


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = list(read_jsonl(args.input))
    if args.limit > 0:
        rows = rows[: args.limit]

    progress_file = args.progress_file or args.output.with_suffix(".progress.json")
    events_file = args.events_file or args.output.with_suffix(".events.jsonl")
    started = time.time()
    started_at = utc_now()
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    if args.resume:
        seen, counts = load_existing(args.output, label_field=args.label_field, model=args.model)

    completed_existing = len(seen)
    completed_new = 0
    last_progress = 0.0

    def snapshot(current: dict[str, Any] | None = None) -> dict[str, Any]:
        completed = completed_existing + completed_new
        return {
            "input": str(args.input),
            "output": str(args.output),
            "provider": args.provider,
            "model": args.model,
            "label_field": args.label_field,
            "started_at": started_at,
            "updated_at": utc_now(),
            "elapsed_seconds": round(time.time() - started, 1),
            "rows_total": len(rows),
            "completed_existing": completed_existing,
            "completed_new": completed_new,
            "completed": completed,
            "remaining": max(0, len(rows) - completed),
            "by_label": dict(sorted(counts.items())),
            "current": current,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    append_event(events_file, {"event": "run_start", "time": started_at, **snapshot()})
    write_progress(progress_file, snapshot())

    mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8") as f:
        pending: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for i, row in enumerate(rows, 1):
            key = row_key(row, i)
            if key in seen:
                continue
            pending.append(
                (
                    row,
                    {
                        "i": i,
                        "instance_id": row.get("instance_id"),
                        "split": row.get("split"),
                        "domain": row.get("domain"),
                        "task_id": row.get("task_id"),
                    },
                )
            )

        if args.max_workers <= 1:
            iterator = ((row, current) for row, current in pending)
        else:
            iterator = None

        if args.max_workers <= 1:
            for row, current in iterator:
                labeled, label, conf, _ = label_row(row, args)
                f.write(json.dumps(labeled, ensure_ascii=False, sort_keys=True) + "\n")
                f.flush()
                completed_new += 1
                counts[str(label)] += 1
                event = {
                    "event": "labeled",
                    "time": utc_now(),
                    **current,
                    "label_field": args.label_field,
                    "label": label,
                    "confidence": conf,
                }
                append_event(events_file, event)
                print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True)
                now = time.time()
                if now - last_progress >= args.progress_interval:
                    write_progress(progress_file, snapshot(current))
                    last_progress = now
        else:
            with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
                future_map = {
                    executor.submit(label_row, row, args): current for row, current in pending
                }
                for future in as_completed(future_map):
                    current = future_map[future]
                    labeled, label, conf, _ = future.result()
                    f.write(json.dumps(labeled, ensure_ascii=False, sort_keys=True) + "\n")
                    f.flush()
                    completed_new += 1
                    counts[str(label)] += 1
                    event = {
                        "event": "labeled",
                        "time": utc_now(),
                        **current,
                        "label_field": args.label_field,
                        "label": label,
                        "confidence": conf,
                    }
                    append_event(events_file, event)
                    print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True)
                    now = time.time()
                    if now - last_progress >= args.progress_interval:
                        write_progress(progress_file, snapshot(current))
                        last_progress = now

    final_snapshot = snapshot()
    write_progress(progress_file, final_snapshot)
    append_event(events_file, {"event": "run_finish", "time": utc_now(), **final_snapshot})
    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "rows": completed_existing + completed_new,
        "provider": args.provider,
        "model": args.model,
        "label_field": args.label_field,
        "max_workers": args.max_workers,
        "progress_file": str(progress_file),
        "events_file": str(events_file),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
