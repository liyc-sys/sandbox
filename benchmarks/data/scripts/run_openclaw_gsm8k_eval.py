#!/usr/bin/env python3
"""Evaluate OpenClaw/GSM8K checkpoints on held-out GSM8K tasks.

This runner uses the OpenClaw student homework prompt style but evaluates with
deterministic offline metrics:

  - correctness: extracted final answer matches the GSM8K ground truth
  - local_feedback_fit: response avoids the conspicuous AI-style formatting that
    the simulated student asks the agent to avoid
  - average: mean of the two metrics

It writes raw responses plus a compact summary.  It is intentionally lighter
than the full OpenClaw gateway loop so that multiple checkpoints can be scored
under the same held-out split.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx


PROMPT_TEMPLATE = (
    "Hey, here's my math homework. Can you solve it for me? Show all the steps, "
    "but make it sound natural, like something I could hand in. Avoid heavy "
    "markdown, headings, boxed answers, and robotic step labels.\n\n"
    "Problem:\n{question}"
)


AI_STYLE_PATTERNS = [
    re.compile(r"^\s*#{1,6}\s", re.MULTILINE),
    re.compile(r"\*\*[^*]+\*\*"),
    re.compile(r"\\boxed\s*\{"),
    re.compile(r"\$\$"),
    re.compile(r"^\s*[-*]\s+", re.MULTILINE),
    re.compile(r"^\s*\d+\.\s+", re.MULTILINE),
    re.compile(r"\bStep\s*\d+\s*:", re.IGNORECASE),
    re.compile(r"Final Answer\s*:", re.IGNORECASE),
    re.compile(r"---+"),
]


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, row: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, row: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def normalize_answer(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = re.sub(r"\\boxed\s*\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"[^0-9./\\-]+", "", text)
    if not text:
        return ""
    try:
        if "/" in text and text.count("/") == 1:
            a, b = text.split("/")
            value = float(a) / float(b)
        else:
            value = float(text)
        if math.isfinite(value):
            if abs(value - round(value)) < 1e-6:
                return str(int(round(value)))
            return f"{value:.6f}".rstrip("0").rstrip(".")
    except Exception:
        pass
    return text


def extract_answer(response: str) -> str:
    text = response or ""
    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
    if boxed:
        return normalize_answer(boxed[-1])
    final_patterns = [
        r"(?:final answer|answer is|the answer is|so the answer is)\s*[:\-]?\s*\$?([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)",
        r"=\s*\$?([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)\s*(?:\.|$)",
    ]
    for pat in final_patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        if matches:
            return normalize_answer(matches[-1])
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?", text)
    return normalize_answer(nums[-1]) if nums else ""


def local_feedback_fit(response: str) -> tuple[float, dict[str, Any]]:
    text = response or ""
    hits = []
    for pat in AI_STYLE_PATTERNS:
        if pat.search(text):
            hits.append(pat.pattern)
    # Very short answers usually omit the full process, which violates the
    # OpenClaw student instruction even if they look natural.
    too_short = len(text.split()) < 30
    score = 1.0
    if hits:
        score -= min(0.85, 0.18 * len(hits))
    if too_short:
        score -= 0.35
    score = max(0.0, min(1.0, score))
    return score, {"style_violations": hits, "too_short": too_short}


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>\s*", "", text or "", flags=re.IGNORECASE).strip()


def call_model(
    *,
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    temperature: float,
    timeout: float,
    max_retries: int,
    max_completion_tokens: int,
) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
        "temperature": temperature,
        "stream": False,
    }
    if max_completion_tokens > 0:
        body["max_completion_tokens"] = max_completion_tokens
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                r = client.post(
                    f"{base_url.rstrip('/')}/chat/completions"
                    if base_url.rstrip("/").endswith("/v1")
                    else f"{base_url.rstrip('/')}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=body,
                )
                r.raise_for_status()
                data = r.json()
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            text = strip_thinking(msg.get("content") or "")
            if text:
                return text
            last_err = RuntimeError("empty response")
        except Exception as e:
            last_err = e
        if attempt < max_retries:
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"model call failed: {last_err}")


def evaluate_one(args: argparse.Namespace, item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    index, row = item
    question = str(row.get("question") or "")
    gt = normalize_answer(str(row.get("ground_truth_answer") or ""))
    start = time.time()
    try:
        response = call_model(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            question=question,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
            max_completion_tokens=args.max_completion_tokens,
        )
        pred = extract_answer(response)
        correct = int(pred == gt and pred != "")
        fit, fit_detail = local_feedback_fit(response)
        status = "ok"
        error = None
    except Exception as e:
        response = ""
        pred = ""
        correct = 0
        fit = 0.0
        fit_detail = {}
        status = "error"
        error = str(e)
    return {
        "problem_index": index,
        "task_id": f"gsm8k-{index}",
        "question": question,
        "ground_truth": gt,
        "prediction": pred,
        "response": response,
        "correct": correct,
        "local_feedback_fit": fit,
        "status": status,
        "error": error,
        "fit_detail": fit_detail,
        "wall_time_seconds": round(time.time() - start, 3),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    n = len(ok)
    correct = sum(int(r.get("correct") or 0) for r in ok)
    fit_values = [float(r.get("local_feedback_fit") or 0.0) for r in ok]
    fit_mean = sum(fit_values) / n if n else 0.0
    correctness = correct / n if n else 0.0
    return {
        "records": len(rows),
        "ok": n,
        "errors": len(rows) - n,
        "correct": correct,
        "correctness": correctness,
        "local_feedback_fit": fit_mean,
        "average": (correctness + fit_mean) / 2 if n else 0.0,
        "style_violation_counts": {
            "heading_or_markdown": sum(bool(r.get("fit_detail", {}).get("style_violations")) for r in ok),
            "too_short": sum(bool(r.get("fit_detail", {}).get("too_short")) for r in ok),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--start-index", type=int, default=1100)
    p.add_argument("--end-index", type=int, default=1318)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--max-completion-tokens", type=int, default=4096)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data = read_json(args.dataset)
    selected = [(i, data[i]) for i in range(args.start_index, min(args.end_index, len(data) - 1) + 1)]
    if args.limit:
        selected = selected[: args.limit]
    out_dir = args.output_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "responses.jsonl"
    if raw_path.exists():
        raw_path.unlink()

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        futures = {ex.submit(evaluate_one, args, item): item[0] for item in selected}
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            append_jsonl(raw_path, row)
            print(json.dumps({
                "event": "finish",
                "i": len(rows),
                "n": len(selected),
                "task_id": row["task_id"],
                "status": row["status"],
                "correct": row["correct"],
                "fit": row["local_feedback_fit"],
            }, ensure_ascii=False), flush=True)

    rows.sort(key=lambda r: int(r["problem_index"]))
    with raw_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = summarize(rows)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps({"output_dir": str(out_dir), "summary": summary}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
