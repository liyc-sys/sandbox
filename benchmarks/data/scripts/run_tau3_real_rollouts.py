#!/usr/bin/env python3
"""Run tau3-bench text rollouts through the tau2-bench simulator.

The script consumes the fixed split files under ``data/raw/tau3_bench`` and
writes raw tau2 ``SimulationRun`` records plus a compact summary. It is meant
to be used both as a smoke test and as the first stage of the real experiment
pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "OPENCLAW_KTO_RUN_ROOT",
        str(ROOT / "runs"),
    )
)

SPLIT_TO_FILE_SUFFIX = {
    "train_update": "train_update",
    "dev": "dev",
    "reported_test": "reported_test",
}

EVALUATION_TYPES = {
    "all": "ALL",
    "env": "ENV",
    "communicate": "COMMUNICATE",
    "action": "ACTION",
    "all_ignore_basis": "ALL_IGNORE_BASIS",
    "all_with_nl_assertions": "ALL_WITH_NL_ASSERTIONS",
}


class TaskTimeoutError(TimeoutError):
    pass


USER_SIM_GUARDRAIL_MARKER = "<openclaw_user_simulator_guardrails>"

USER_SIM_GUARDRAILS = f"""
{USER_SIM_GUARDRAIL_MARKER}
You are simulating the customer/user, never the service agent.

Critical role constraints:
- Speak only as the customer described in <scenario>. Do not offer help, ask for
  an order number/reservation id/phone number as a service representative, or
  summarize what "the user" should do.
- In your first reply, state your own request in first person using only the
  scenario information. Do not output ###STOP###, ###TRANSFER###, or
  ###OUT-OF-SCOPE### in the first reply.
- Use ###STOP### only after the agent has actually satisfied the scenario.
- Use ###TRANSFER### only when the scenario calls for human transfer or the
  agent has transferred you. Use ###OUT-OF-SCOPE### only when the agent truly
  leaves the supported task scope.
- If the agent asks for information that the scenario says you know, provide it
  as the customer. If the scenario says you do not know it, say you do not know.
</openclaw_user_simulator_guardrails>
""".strip()


ASSISTANT_LIKE_FIRST_USER_PATTERNS = [
    re.compile(r"\bi can help\b", re.I),
    re.compile(r"\bi'?d be happy to help\b", re.I),
    re.compile(r"\bi’d be happy to help\b", re.I),
    re.compile(r"\bto assist you\b", re.I),
    re.compile(r"\blet me (?:check|look up|pull up|verify)\b", re.I),
    re.compile(r"\bi found (?:your|the|that)\b", re.I),
    re.compile(
        r"\bcould you please (?:share|provide|confirm|give me)\b.{0,80}"
        r"\b(?:order|reservation|account|phone|email|zip|name|id|number)\b",
        re.I,
    ),
]


class alarm_timeout:
    def __init__(self, seconds: int):
        self.seconds = seconds
        self.previous_handler = None

    def __enter__(self):
        if self.seconds <= 0:
            return self

        def _handle_timeout(signum, frame):
            raise TaskTimeoutError(f"task timed out after {self.seconds}s")

        self.previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seconds > 0:
            signal.alarm(0)
            if self.previous_handler is not None:
                signal.signal(signal.SIGALRM, self.previous_handler)
        return False


def ensure_tau2_import_path() -> None:
    """Allow running from a source checkout without pip installing tau2."""
    src = ROOT / "vendor" / "tau2-bench" / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def apply_tau2_streaming_compat_patch() -> None:
    """Patch tau2's LLM helper for endpoints that only support stream=true.

    Some OpenAI-compatible gateways reject non-streaming chat completions. tau2
    expects a regular AssistantMessage, so this wrapper asks the gateway for a
    stream and folds content/tool-call deltas back into the same tau2 message
    contract. Reasoning-only deltas are intentionally ignored.
    """
    import httpx
    from openai import OpenAI

    import tau2.agent.llm_agent as llm_agent_module
    import tau2.evaluator.auth_classifier as auth_classifier_module
    import tau2.evaluator.evaluator_nl_assertions as evaluator_nl_assertions_module
    import tau2.evaluator.hallucination_reviewer as hallucination_reviewer_module
    import tau2.evaluator.review_llm_judge as review_llm_judge_module
    import tau2.evaluator.review_llm_judge_user_only as review_llm_judge_user_only_module
    import tau2.user.user_simulator as user_simulator_module
    import tau2.utils.llm_utils as llm_utils
    from tau2.data_model.message import AssistantMessage, ToolCall

    def _merge_tool_call_delta(
        tool_calls: dict[int, dict[str, Any]], delta_call: dict[str, Any]
    ) -> None:
        index = int(delta_call.get("index") or 0)
        acc = tool_calls.setdefault(
            index,
            {
                "id": "",
                "name": "",
                "arguments": "",
                "type": "function",
            },
        )
        if delta_call.get("id"):
            acc["id"] = str(delta_call["id"])
        if delta_call.get("type"):
            acc["type"] = str(delta_call["type"])
        function = delta_call.get("function") or {}
        if function.get("name"):
            acc["name"] = str(function["name"])
        if function.get("arguments"):
            acc["arguments"] += str(function["arguments"])

    def _streaming_generate(
        model: str,
        messages: list[Any],
        tools: list[Any] | None = None,
        tool_choice: str | None = None,
        call_name: str | None = None,
        **kwargs: Any,
    ) -> AssistantMessage:
        llm_utils.validate_message_history(messages)
        if kwargs.get("num_retries") is None:
            kwargs["num_retries"] = llm_utils.DEFAULT_MAX_RETRIES
        max_attempts = int(kwargs.pop("num_retries") or 0) + 1
        api_key = kwargs.pop("api_key", None)
        base_url = kwargs.pop("api_base", None) or kwargs.pop("base_url", None)
        timeout = kwargs.pop("timeout", None)
        temperature = kwargs.pop("temperature", None)
        # Do not forward token caps by default. Qwen3 thinking models may spend
        # a large prefix in reasoning_content before emitting final content/tool
        # calls; a small cap can truncate the usable assistant action. Retry and
        # triage runs may pass an explicit max_completion_tokens to bound a
        # pathological generation.
        kwargs.pop("max_tokens", None)
        max_completion_tokens = kwargs.pop("max_completion_tokens", None)
        kwargs.pop("stream", None)
        kwargs.pop("stream_options", None)

        litellm_messages = llm_utils.to_litellm_messages(messages)
        tools_schema = [tool.openai_schema for tool in tools] if tools else None
        if tools_schema and tool_choice is None:
            tool_choice = "auto"

        start_time = time.perf_counter()
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(trust_env=False, timeout=timeout or 120.0),
        )

        last_error: BaseException | None = None
        for attempt in range(max_attempts):
            request_messages = list(litellm_messages)
            content_parts: list[str] = []
            reasoning_chars = 0
            tool_call_parts: dict[int, dict[str, Any]] = {}
            finish_reason = "stop"
            usage: dict[str, int] | None = None
            response_id = ""
            created = int(time.time())
            response_model = model

            try:
                create_kwargs: dict[str, Any] = {
                    "model": model.removeprefix("openai/"),
                    "messages": request_messages,
                    "stream": True,
                }
                if temperature is not None:
                    create_kwargs["temperature"] = temperature
                if max_completion_tokens is not None:
                    create_kwargs["max_completion_tokens"] = max_completion_tokens
                if tools_schema:
                    create_kwargs["tools"] = tools_schema
                    create_kwargs["tool_choice"] = tool_choice or "auto"
                if timeout is not None:
                    create_kwargs["timeout"] = timeout

                stream = client.chat.completions.create(**create_kwargs)
                for chunk in stream:
                    data = chunk.model_dump() if hasattr(chunk, "model_dump") else {}
                    response_id = data.get("id") or response_id
                    created = int(data.get("created") or created)
                    response_model = data.get("model") or response_model
                    if data.get("usage"):
                        raw_usage = data["usage"]
                        usage = {
                            "completion_tokens": int(raw_usage.get("completion_tokens") or 0),
                            "prompt_tokens": int(raw_usage.get("prompt_tokens") or 0),
                        }
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    if choice.get("finish_reason"):
                        finish_reason = str(choice["finish_reason"])
                    delta = choice.get("delta") or {}
                    for reasoning_key in ("reasoning_content", "reasoning"):
                        if delta.get(reasoning_key):
                            reasoning_chars += len(str(delta[reasoning_key]))
                    if delta.get("content"):
                        content_parts.append(str(delta["content"]))
                    for delta_call in delta.get("tool_calls") or []:
                        _merge_tool_call_delta(tool_call_parts, delta_call)
            except TaskTimeoutError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt + 1 < max_attempts:
                    time.sleep(min(2**attempt, 8))
                    continue
                raise

            tool_calls: list[ToolCall] | None = None
            if tool_call_parts:
                tool_calls = []
                for index in sorted(tool_call_parts):
                    raw = tool_call_parts[index]
                    arguments_text = raw.get("arguments") or "{}"
                    try:
                        arguments = json.loads(arguments_text)
                    except json.JSONDecodeError:
                        arguments = {"_raw_arguments": arguments_text}
                    tool_calls.append(
                        ToolCall(
                            id=raw.get("id") or "",
                            name=raw.get("name") or "",
                            arguments=arguments,
                        )
                    )

            text = "".join(content_parts).strip() or None
            if text or tool_calls:
                raw_data = {
                    "id": response_id or f"stream-{int(time.time() * 1000)}",
                    "created": created,
                    "model": response_model,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "finish_reason": finish_reason,
                            "index": 0,
                            "message": {
                                "content": text,
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.name,
                                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                                        },
                                    }
                                    for tc in tool_calls or []
                                ]
                                or None,
                            },
                        }
                    ],
                }
                generation_time_seconds = time.perf_counter() - start_time
                return AssistantMessage(
                    role="assistant",
                    content=text,
                    tool_calls=tool_calls,
                    cost=0.0,
                    usage=usage,
                    raw_data=raw_data,
                    generation_time_seconds=generation_time_seconds,
                )

            last_error = RuntimeError(
                "streaming completion returned no content or tool calls "
                f"(finish_reason={finish_reason}, reasoning_chars={reasoning_chars})"
            )
            if attempt + 1 < max_attempts:
                time.sleep(min(2**attempt, 8))
                continue
            raise last_error

        raise RuntimeError(
            f"streaming completion failed without usable output: {last_error}"
        )

    llm_utils.generate = _streaming_generate
    llm_agent_module.generate = _streaming_generate
    user_simulator_module.generate = _streaming_generate
    evaluator_nl_assertions_module.generate = _streaming_generate
    auth_classifier_module.generate = _streaming_generate
    hallucination_reviewer_module.generate = _streaming_generate
    review_llm_judge_module.generate = _streaming_generate
    review_llm_judge_user_only_module.generate = _streaming_generate


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
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield obj


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def compact_error(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc)[:4000],
    }


def compact_text(text: str, limit: int = 500) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def append_user_simulator_guardrails() -> None:
    """Add local guardrails to tau2's user simulator prompt at runtime."""
    ensure_tau2_import_path()
    import tau2.user.user_simulator as user_simulator_module

    system_prompt = getattr(user_simulator_module, "SYSTEM_PROMPT", "")
    if USER_SIM_GUARDRAIL_MARKER in system_prompt:
        return
    user_simulator_module.SYSTEM_PROMPT = (
        system_prompt.rstrip()
        + "\n\n"
        + USER_SIM_GUARDRAILS
        + "\n"
    )


def simulation_messages(rec: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((rec.get("simulation") or {}).get("messages") or []))


def first_user_message(rec: dict[str, Any]) -> str:
    for message in simulation_messages(rec):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def joined_user_messages(rec: dict[str, Any]) -> str:
    return "\n".join(
        str(message.get("content") or "")
        for message in simulation_messages(rec)
        if message.get("role") == "user"
    )


def assistant_tool_call_names(rec: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for message in simulation_messages(rec):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            name = call.get("name")
            if not name and isinstance(call.get("function"), dict):
                name = call["function"].get("name")
            if name:
                names.append(str(name))
    return names


def detect_user_simulator_artifacts(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Return likely user-simulator artifacts that should not count as eval.

    These checks are deliberately conservative. They target cases where the
    simulated user either ends the task before the agent has a meaningful turn
    or slips into the service-agent role.
    """
    summary = rec.get("summary") or {}
    num_messages = int(summary.get("num_messages") or len(simulation_messages(rec)))
    first_user = first_user_message(rec)
    first_user_lower = first_user.lower()
    all_user = joined_user_messages(rec).lower()
    tool_names = assistant_tool_call_names(rec)
    artifacts: list[dict[str, Any]] = []

    if num_messages <= 2 and "###stop###" in all_user:
        artifacts.append(
            {
                "type": "first_turn_stop",
                "detail": compact_text(first_user),
            }
        )
    if num_messages <= 4 and "###transfer###" in all_user:
        artifacts.append(
            {
                "type": "early_transfer",
                "detail": compact_text(first_user),
            }
        )
    if num_messages <= 4 and "###out-of-scope###" in all_user:
        artifacts.append(
            {
                "type": "early_out_of_scope",
                "detail": compact_text(first_user),
            }
        )
    if num_messages <= 4 and not tool_names:
        artifacts.append(
            {
                "type": "no_agent_chance",
                "detail": f"num_messages={num_messages}; first_user={compact_text(first_user)}",
            }
        )
    for pattern in ASSISTANT_LIKE_FIRST_USER_PATTERNS:
        if pattern.search(first_user_lower):
            artifacts.append(
                {
                    "type": "assistant_like_user_first_turn",
                    "detail": compact_text(first_user),
                }
            )
            break

    return artifacts


def annotate_user_simulator_artifacts(rec: dict[str, Any]) -> dict[str, Any]:
    artifacts = detect_user_simulator_artifacts(rec)
    if artifacts:
        rec["user_simulator_artifacts"] = artifacts
    return rec


def artifact_retry_seed(base_seed: int, attempt: int, row_index: int) -> int:
    # Keep retries deterministic but far away from the original seed stream.
    return int(base_seed) + 100_000 * int(attempt) + int(row_index)


def load_split_rows(
    *,
    data_dir: Path,
    domains: list[str],
    splits: list[str],
    limit_per_split: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in domains:
        for split in splits:
            suffix = SPLIT_TO_FILE_SUFFIX[split]
            path = data_dir / f"{domain}_{suffix}_tasks.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
            split_rows = list(read_jsonl(path))
            if limit_per_split > 0:
                split_rows = split_rows[:limit_per_split]
            for row in split_rows:
                out = dict(row)
                out["domain"] = domain
                out["split"] = split
                rows.append(out)
    return rows


def load_error_rows(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in read_jsonl(path):
        if rec.get("status") == "ok":
            continue
        raw_task = rec.get("raw_task")
        if not isinstance(raw_task, dict):
            continue
        row = dict(raw_task)
        row["domain"] = rec.get("domain") or row.get("domain")
        row["split"] = rec.get("split") or row.get("split")
        row["_retry_task_id"] = rec.get("task_id")
        row["_retry_seed"] = rec.get("seed")
        row["_previous_error"] = rec.get("error")
        rows.append(row)
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def load_error_row_at(path: Path, index: int) -> list[dict[str, Any]]:
    if index < 0:
        raise ValueError("--retry-index must be non-negative")
    failures = load_error_rows(path)
    if index >= len(failures):
        raise IndexError(f"--retry-index {index} out of range for {len(failures)} failures")
    return [failures[index]]


def task_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("domain") or ""),
        str(row.get("split") or ""),
        str(row.get("task_id") or ""),
    )


def completed_task_keys(path: Path) -> set[tuple[str, str, str]]:
    done: set[tuple[str, str, str]] = set()
    for rec in read_jsonl(path):
        if rec.get("status") != "ok":
            continue
        if rec.get("user_simulator_artifacts"):
            continue
        done.add(
            (
                str(rec.get("domain") or ""),
                str(rec.get("split") or ""),
                str(rec.get("task_id") or ""),
            )
        )
    return done


def _manifest_keys_from_items(
    *,
    items: list[Any],
    default_split: str,
) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        domain = str(item.get("domain") or "")
        split = str(item.get("split") or default_split)
        task_id = str(item.get("task_id") or "")
        if domain and split and task_id:
            keys.add((domain, split, task_id))
    return keys


def _manifest_keys_from_by_domain(
    *,
    by_domain: dict[str, Any],
    default_split: str,
) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for domain, task_ids in by_domain.items():
        if not isinstance(task_ids, list):
            continue
        for task_id in task_ids:
            keys.add((str(domain), default_split, str(task_id)))
    return keys


def apply_task_manifest(
    rows: list[dict[str, Any]],
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter selected rows with a small JSON manifest.

    The manifest may specify explicit clean/include tasks or excluded tasks.
    We use this to keep the tau3 reported-test denominator fixed after
    dropping connection-failed trajectories.
    """
    with path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: expected JSON object")

    default_split = str(manifest.get("split") or "reported_test")
    include_keys = _manifest_keys_from_items(
        items=list(manifest.get("include_tasks") or manifest.get("clean_tasks") or []),
        default_split=default_split,
    )
    include_by_domain = (
        manifest.get("include_task_ids_by_domain")
        or manifest.get("clean_task_ids_by_domain")
        or {}
    )
    if isinstance(include_by_domain, dict):
        include_keys |= _manifest_keys_from_by_domain(
            by_domain=include_by_domain,
            default_split=default_split,
        )

    exclude_keys = _manifest_keys_from_items(
        items=list(manifest.get("exclude_tasks") or manifest.get("excluded_tasks") or []),
        default_split=default_split,
    )
    exclude_by_domain = (
        manifest.get("exclude_task_ids_by_domain")
        or manifest.get("excluded_task_ids_by_domain")
        or {}
    )
    if isinstance(exclude_by_domain, dict):
        exclude_keys |= _manifest_keys_from_by_domain(
            by_domain=exclude_by_domain,
            default_split=default_split,
        )

    before = len(rows)
    if include_keys:
        rows = [row for row in rows if task_key(row) in include_keys]
    if exclude_keys:
        rows = [row for row in rows if task_key(row) not in exclude_keys]

    expected_count = manifest.get("expected_count", manifest.get("clean_completed"))
    if expected_count is not None and len(rows) != int(expected_count):
        raise ValueError(
            f"{path}: expected {expected_count} selected rows after filtering, found {len(rows)}"
        )

    stats = {
        "path": str(path),
        "name": manifest.get("name"),
        "before": before,
        "after": len(rows),
        "include_keys": len(include_keys),
        "exclude_keys": len(exclude_keys),
        "expected_count": expected_count,
        "policy": manifest.get("policy"),
    }
    return rows, stats


def run_one_task(
    *,
    row: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
    save_dir: Path,
) -> dict[str, Any]:
    ensure_tau2_import_path()
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    if args.user_guardrails:
        append_user_simulator_guardrails()
    if args.force_streaming:
        apply_tau2_streaming_compat_patch()
    from tau2.data_model.simulation import TextRunConfig
    from tau2.data_model.tasks import Task
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.run import run_single_task

    task = Task.model_validate(row["metadata"])
    domain = str(row["domain"])
    llm_args_agent = {
        "temperature": args.agent_temperature,
        "api_base": args.agent_api_base,
        "api_key": args.agent_api_key,
        "timeout": args.llm_timeout,
    }
    if args.agent_max_completion_tokens > 0:
        llm_args_agent["max_completion_tokens"] = args.agent_max_completion_tokens
    llm_args_user = {
        "temperature": args.user_temperature,
        "api_base": args.user_api_base,
        "api_key": args.user_api_key,
        "timeout": args.llm_timeout,
    }
    if args.user_max_completion_tokens > 0:
        llm_args_user["max_completion_tokens"] = args.user_max_completion_tokens
    config = TextRunConfig(
        domain=domain,
        agent=args.agent,
        user=args.user,
        llm_agent=args.agent_model,
        llm_user=args.user_model,
        llm_args_agent=llm_args_agent,
        llm_args_user=llm_args_user,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        timeout=args.task_timeout_seconds if args.task_timeout_seconds > 0 else None,
        seed=seed,
        num_trials=1,
        max_concurrency=1,
        log_level=args.log_level,
        verbose_logs=args.verbose_logs,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        hallucination_retries=0,
    )
    evaluation_type = getattr(EvaluationType, EVALUATION_TYPES[args.evaluation_type])
    with alarm_timeout(args.task_timeout_seconds):
        simulation = run_single_task(
            config,
            task,
            seed=seed,
            evaluation_type=evaluation_type,
            save_dir=save_dir,
            verbose_logs=args.verbose_logs,
            auto_review=False,
        )
    sim_json = simulation.model_dump(mode="json")
    reward_info = sim_json.get("reward_info") or {}
    return {
        "status": "ok",
        "domain": domain,
        "split": row["split"],
        "task_id": str(task.id),
        "task_index": row.get("index"),
        "seed": seed,
        "raw_task": row,
        "simulation": sim_json,
        "summary": {
            "simulation_id": sim_json.get("id"),
            "termination_reason": sim_json.get("termination_reason"),
            "reward": reward_info.get("reward"),
            "duration": sim_json.get("duration"),
            "agent_cost": sim_json.get("agent_cost"),
            "user_cost": sim_json.get("user_cost"),
            "num_messages": len(sim_json.get("messages") or []),
        },
    }


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_status = Counter(str(r.get("status")) for r in records)
    by_split = Counter(str(r.get("split")) for r in records)
    by_domain = Counter(str(r.get("domain")) for r in records)
    artifact_counts: Counter[str] = Counter()
    records_with_artifacts = 0
    rewards: dict[str, list[float]] = defaultdict(list)
    failures: list[dict[str, Any]] = []
    for rec in records:
        artifacts = rec.get("user_simulator_artifacts") or []
        if artifacts:
            records_with_artifacts += 1
            for artifact in artifacts:
                artifact_counts[str(artifact.get("type") or "unknown")] += 1
        if rec.get("status") != "ok":
            failures.append(
                {
                    "domain": rec.get("domain"),
                    "split": rec.get("split"),
                    "task_id": rec.get("task_id"),
                    "error": rec.get("error"),
                }
            )
            continue
        summary = rec.get("summary") or {}
        reward = summary.get("reward")
        if reward is not None:
            key = f"{rec.get('domain')}::{rec.get('split')}"
            rewards[key].append(float(reward))

    reward_summary = {}
    for key, values in sorted(rewards.items()):
        reward_summary[key] = {
            "count": len(values),
            "mean": sum(values) / len(values) if values else None,
            "successes": sum(1 for v in values if v > 0),
        }

    return {
        "records": len(records),
        "by_status": dict(sorted(by_status.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_domain": dict(sorted(by_domain.items())),
        "records_with_user_simulator_artifacts": records_with_artifacts,
        "user_simulator_artifact_counts": dict(sorted(artifact_counts.items())),
        "reward_summary": reward_summary,
        "failures": failures[:20],
    }


def split_overlap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids_by_split: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        ids_by_split[str(row["split"])].add((str(row["domain"]), str(row["task_id"])))
    out: dict[str, Any] = {}
    splits = sorted(ids_by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            overlap = sorted(ids_by_split[a] & ids_by_split[b])
            out[f"{a}__{b}"] = [f"{d}:{tid}" for d, tid in overlap]
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domains", nargs="+", default=["airline", "retail", "telecom"])
    p.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(SPLIT_TO_FILE_SUFFIX),
        default=["train_update", "dev", "reported_test"],
    )
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "tau3_bench")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "rollouts")
    p.add_argument("--run-name", default="")
    p.add_argument("--limit-per-split", type=int, default=0)
    p.add_argument("--retry-errors-from", type=Path, default=None,
                   help="Previous raw_simulations.jsonl; only rows with status != ok are rerun.")
    p.add_argument("--retry-index", type=int, default=None,
                   help="Rerun only the Nth failed row from --retry-errors-from.")
    p.add_argument("--retry-limit", type=int, default=0)
    p.add_argument("--skip-completed-from", type=Path, default=None,
                   help="Previous raw_simulations.jsonl; rows with status == ok are skipped.")
    p.add_argument("--task-manifest", type=Path, default=None,
                   help="Optional JSON manifest that includes or excludes task ids before running.")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Split selected rows into this many deterministic shards.")
    p.add_argument("--shard-index", type=int, default=0,
                   help="Run only rows whose post-skip index belongs to this shard.")
    p.add_argument("--agent", default="llm_agent")
    p.add_argument("--user", default="user_simulator")
    p.add_argument("--agent-model", default=os.environ.get("TAU3_AGENT_MODEL", "openai/qwen3-4b"))
    p.add_argument("--user-model", default=os.environ.get("TAU3_USER_MODEL", "openai/qwen3-4b"))
    p.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL", ""))
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--agent-api-base", default=None)
    p.add_argument("--user-api-base", default=None)
    p.add_argument("--agent-api-key-env", default=None)
    p.add_argument("--user-api-key-env", default=None)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--max-errors", type=int, default=8)
    p.add_argument("--max-retries", type=int, default=1)
    p.add_argument("--retry-delay", type=float, default=2.0)
    p.add_argument("--llm-timeout", type=float, default=120.0)
    p.add_argument("--task-timeout-seconds", type=int, default=420)
    p.add_argument("--evaluation-type", choices=sorted(EVALUATION_TYPES), default="all")
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--agent-temperature", type=float, default=0.0)
    p.add_argument("--user-temperature", type=float, default=0.0)
    p.add_argument("--agent-max-completion-tokens", type=int, default=0,
                   help="Optional agent max_completion_tokens; 0 leaves it unset.")
    p.add_argument("--user-max-completion-tokens", type=int, default=0,
                   help="Optional user max_completion_tokens; 0 leaves it unset.")
    p.add_argument("--log-level", default="ERROR")
    p.add_argument("--verbose-logs", action="store_true")
    p.add_argument("--force-streaming", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--user-guardrails",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append stricter role constraints to the tau2 user simulator prompt.",
    )
    p.add_argument(
        "--reject-user-artifacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reject likely user-simulator artifacts and resample the task with a new seed.",
    )
    p.add_argument(
        "--artifact-max-attempts",
        type=int,
        default=3,
        help="Maximum rollout attempts per task when --reject-user-artifacts is enabled.",
    )
    p.add_argument(
        "--keep-artifact-attempts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write rejected artifact attempts to artifact_attempts.jsonl for auditability.",
    )
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.agent_api_base = args.agent_api_base or args.api_base
    args.user_api_base = args.user_api_base or args.api_base
    args.agent_api_key_env = args.agent_api_key_env or args.api_key_env
    args.user_api_key_env = args.user_api_key_env or args.api_key_env
    args.agent_api_key = os.environ.get(args.agent_api_key_env, "").strip()
    args.user_api_key = os.environ.get(args.user_api_key_env, "").strip()
    if not args.agent_api_key and not args.dry_run:
        raise SystemExit(f"missing agent API key env var: {args.agent_api_key_env}")
    if not args.user_api_key and not args.dry_run:
        raise SystemExit(f"missing user API key env var: {args.user_api_key_env}")

    if args.retry_errors_from is not None:
        if args.retry_index is not None:
            rows = load_error_row_at(args.retry_errors_from, args.retry_index)
        else:
            rows = load_error_rows(args.retry_errors_from, limit=args.retry_limit)
        if not rows:
            raise SystemExit(f"no failed records found in {args.retry_errors_from}")
    else:
        rows = load_split_rows(
            data_dir=args.data_dir,
            domains=args.domains,
            splits=args.splits,
            limit_per_split=args.limit_per_split,
        )
    for selection_index, row in enumerate(rows):
        row["_selection_index"] = selection_index

    task_manifest_stats = None
    if args.task_manifest is not None:
        rows, task_manifest_stats = apply_task_manifest(rows, args.task_manifest)

    skipped_completed = 0
    if args.skip_completed_from is not None:
        done = completed_task_keys(args.skip_completed_from)
        before = len(rows)
        rows = [row for row in rows if task_key(row) not in done]
        skipped_completed = before - len(rows)

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.num_shards > 1:
        rows = [
            row
            for post_skip_index, row in enumerate(rows)
            if post_skip_index % args.num_shards == args.shard_index
        ]

    overlap = split_overlap(rows)
    bad_overlap = {k: v for k, v in overlap.items() if v}
    if bad_overlap:
        raise SystemExit(f"split task overlap detected: {bad_overlap}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"tau3_real_{timestamp}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "data/scripts/run_tau3_real_rollouts.py",
        "output_dir": str(output_dir),
        "domains": args.domains,
        "splits": args.splits,
        "limit_per_split": args.limit_per_split,
        "retry_errors_from": str(args.retry_errors_from) if args.retry_errors_from else None,
        "retry_index": args.retry_index,
        "retry_limit": args.retry_limit,
        "skip_completed_from": str(args.skip_completed_from) if args.skip_completed_from else None,
        "skipped_completed": skipped_completed,
        "task_manifest": str(args.task_manifest) if args.task_manifest else None,
        "task_manifest_stats": task_manifest_stats,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "agent": args.agent,
        "user": args.user,
        "agent_model": args.agent_model,
        "user_model": args.user_model,
        "api_base": args.api_base,
        "agent_api_base": args.agent_api_base,
        "user_api_base": args.user_api_base,
        "agent_api_key_env": args.agent_api_key_env,
        "user_api_key_env": args.user_api_key_env,
        "max_steps": args.max_steps,
        "max_errors": args.max_errors,
        "force_streaming": args.force_streaming,
        "user_guardrails": args.user_guardrails,
        "reject_user_artifacts": args.reject_user_artifacts,
        "artifact_max_attempts": args.artifact_max_attempts,
        "keep_artifact_attempts": args.keep_artifact_attempts,
        "evaluation_type": args.evaluation_type,
        "task_timeout_seconds": args.task_timeout_seconds,
        "agent_max_completion_tokens": args.agent_max_completion_tokens,
        "user_max_completion_tokens": args.user_max_completion_tokens,
        "seed": args.seed,
        "num_tasks": len(rows),
        "split_overlap": overlap,
        "dry_run": args.dry_run,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    if args.dry_run:
        preview = [
            {
                "domain": row["domain"],
                "split": row["split"],
                "task_id": str(row["task_id"]),
                "index": row.get("index"),
            }
            for row in rows
        ]
        write_jsonl(output_dir / "selected_tasks.jsonl", preview)
        print(json.dumps({"output_dir": str(output_dir), "selected": len(preview)}, indent=2))
        return 0

    records: list[dict[str, Any]] = []
    artifact_attempt_records: list[dict[str, Any]] = []
    raw_path = output_dir / "raw_simulations.jsonl"
    artifact_attempts_path = output_dir / "artifact_attempts.jsonl"
    if args.artifact_max_attempts < 1:
        raise SystemExit("--artifact-max-attempts must be >= 1")
    for i, row in enumerate(rows, 1):
        base_seed = int(row.get("_retry_seed") or (args.seed + int(row.get("_selection_index", i - 1))))
        task_id = str(row.get("task_id"))
        print(
            json.dumps(
                {
                    "event": "start",
                    "i": i,
                    "n": len(rows),
                    "domain": row.get("domain"),
                    "split": row.get("split"),
                    "task_id": task_id,
                    "seed": base_seed,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        start = time.time()
        rejected_attempts: list[dict[str, Any]] = []
        rec: dict[str, Any] | None = None
        max_attempts = args.artifact_max_attempts if args.reject_user_artifacts else 1
        for attempt in range(1, max_attempts + 1):
            seed = base_seed if attempt == 1 else artifact_retry_seed(base_seed, attempt - 1, i)
            print(
                json.dumps(
                    {
                        "event": "attempt_start",
                        "i": i,
                        "n": len(rows),
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "domain": row.get("domain"),
                        "split": row.get("split"),
                        "task_id": task_id,
                        "seed": seed,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            try:
                candidate = run_one_task(
                    row=row,
                    args=args,
                    seed=seed,
                    save_dir=output_dir / "tau2_logs",
                )
                candidate["artifact_attempt"] = attempt
                candidate["artifact_max_attempts"] = max_attempts
                candidate = annotate_user_simulator_artifacts(candidate)
                artifacts = candidate.get("user_simulator_artifacts") or []
                if args.reject_user_artifacts and artifacts and attempt < max_attempts:
                    rejected_attempts.append(candidate)
                    artifact_attempt_records.append(candidate)
                    print(
                        json.dumps(
                            {
                                "event": "artifact_reject",
                                "i": i,
                                "attempt": attempt,
                                "task_id": task_id,
                                "seed": seed,
                                "artifacts": artifacts,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    if args.keep_artifact_attempts:
                        write_jsonl(artifact_attempts_path, artifact_attempt_records)
                    continue
                if args.reject_user_artifacts and artifacts:
                    candidate["status"] = "artifact"
                    candidate["error"] = {
                        "type": "UserSimulatorArtifact",
                        "message": "artifact persisted after max attempts",
                        "artifacts": artifacts,
                    }
                rec = candidate
                break
            except Exception as exc:
                rec = {
                    "status": "error",
                    "domain": row.get("domain"),
                    "split": row.get("split"),
                    "task_id": task_id,
                    "task_index": row.get("index"),
                    "seed": seed,
                    "raw_task": row,
                    "artifact_attempt": attempt,
                    "artifact_max_attempts": max_attempts,
                    "error": compact_error(exc),
                }
                if not args.continue_on_error:
                    records.append(rec)
                    write_jsonl(raw_path, records)
                    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
                        json.dump(build_summary(records), f, ensure_ascii=False, indent=2, sort_keys=True)
                        f.write("\n")
                    raise
                break
        if rec is None:
            rec = {
                "status": "error",
                "domain": row.get("domain"),
                "split": row.get("split"),
                "task_id": task_id,
                "task_index": row.get("index"),
                "seed": base_seed,
                "raw_task": row,
                "artifact_attempt": max_attempts,
                "artifact_max_attempts": max_attempts,
                "error": {
                    "type": "MissingAttemptRecord",
                    "message": "no rollout record was produced",
                },
            }
        if rejected_attempts:
            rec["rejected_artifact_attempts"] = [
                {
                    "attempt": item.get("artifact_attempt"),
                    "seed": item.get("seed"),
                    "artifacts": item.get("user_simulator_artifacts"),
                    "summary": item.get("summary"),
                }
                for item in rejected_attempts
            ]
        rec["wall_time_seconds"] = round(time.time() - start, 3)
        records.append(rec)
        write_jsonl(raw_path, records)
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(build_summary(records), f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        print(
            json.dumps(
                {
                    "event": "finish",
                    "i": i,
                    "status": rec.get("status"),
                    "summary": rec.get("summary"),
                    "error": rec.get("error"),
                    "wall_time_seconds": rec.get("wall_time_seconds"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = build_summary(records)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
