#!/usr/bin/env python3
"""Judge SWE-bench Lite patches with an LLM proxy evaluator.

This is not the official SWE-bench Docker test evaluator.  It asks a strong
model to compare the candidate patch against the issue, tests, and gold patch,
and emits a plausibility label for analysis when Docker execution is not
available.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx


PROMPT_TEMPLATE = """You are evaluating a candidate patch for a SWE-bench Lite issue.

Decide whether the candidate patch would plausibly resolve the issue. You may
use the gold patch and test patch as references, but do not require an exact
match. A candidate can pass if it implements the same fix with a different
structure. Penalize patches that are empty, non-diff text, modify only tests, do
not address the described bug, or appear likely to break existing behavior.

Return only valid JSON with this schema:
{{
  "plausibly_resolves": true or false,
  "score": number from 0 to 1,
  "rationale": "one concise sentence",
  "candidate_files": ["path", "..."],
  "gold_files": ["path", "..."]
}}

Repository: {repo}
Instance id: {instance_id}
Base commit: {base_commit}

Issue:
{problem_statement}

Fail-to-pass tests:
{fail_to_pass}

Pass-to-pass tests:
{pass_to_pass}

Gold patch:
{gold_patch}

Test patch:
{test_patch}

Candidate patch:
{candidate_patch}
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def truncate(text: str, limit: int) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("judge did not return a JSON object")
    return obj


def patch_files(patch: str) -> list[str]:
    return sorted({b for _, b in re.findall(r"^diff --git a/(.*?) b/(.*?)$", patch or "", flags=re.MULTILINE)})


def call_judge(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": args.judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if args.temperature >= 0:
        body["temperature"] = args.temperature
    if args.max_completion_tokens > 0:
        body["max_completion_tokens"] = args.max_completion_tokens
    url = (
        f"{args.base_url.rstrip('/')}/chat/completions"
        if args.base_url.rstrip("/").endswith("/v1")
        else f"{args.base_url.rstrip('/')}/v1/chat/completions"
    )
    last_err: Exception | None = None
    for attempt in range(args.max_retries + 1):
        try:
            with httpx.Client(timeout=args.timeout, trust_env=False) as client:
                resp = client.post(
                    url,
                    headers={"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"},
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
            text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            return extract_json(text)
        except Exception as e:
            last_err = e
        if attempt < args.max_retries:
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"judge call failed: {last_err}")


def judge_one(args: argparse.Namespace, item: tuple[int, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    idx, cand, gold = item
    patch = cand.get("patch") or ""
    start = time.time()
    try:
        if not cand.get("patch_valid"):
            verdict = {
                "plausibly_resolves": False,
                "score": 0.0,
                "rationale": "Candidate is not a valid git-style patch.",
                "candidate_files": patch_files(patch),
                "gold_files": patch_files(gold.get("patch") or ""),
            }
            status = "ok"
            error = None
        else:
            prompt = PROMPT_TEMPLATE.format(
                repo=gold.get("repo") or cand.get("repo") or "",
                instance_id=gold.get("instance_id") or cand.get("instance_id") or "",
                base_commit=gold.get("base_commit") or cand.get("base_commit") or "",
                problem_statement=truncate(gold.get("problem_statement") or cand.get("problem_statement") or "", args.max_field_chars),
                fail_to_pass=truncate(json.dumps(gold.get("FAIL_TO_PASS") or [], ensure_ascii=False), 3000),
                pass_to_pass=truncate(json.dumps(gold.get("PASS_TO_PASS") or [], ensure_ascii=False), 3000),
                gold_patch=truncate(gold.get("patch") or "", args.max_patch_chars),
                test_patch=truncate(gold.get("test_patch") or "", args.max_patch_chars),
                candidate_patch=truncate(patch, args.max_patch_chars),
            )
            verdict = call_judge(args, prompt)
            status = "ok"
            error = None
        resolves = bool(verdict.get("plausibly_resolves"))
        score = float(verdict.get("score") or 0.0)
    except Exception as e:
        verdict = {}
        resolves = False
        score = 0.0
        status = "error"
        error = str(e)
    return {
        "row_index": idx,
        "instance_id": cand.get("instance_id"),
        "repo": cand.get("repo"),
        "candidate_patch_valid": bool(cand.get("patch_valid")),
        "plausibly_resolves": resolves,
        "score": score,
        "verdict": verdict,
        "status": status,
        "error": error,
        "wall_time_seconds": round(time.time() - start, 3),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    valid = [r for r in ok if r.get("candidate_patch_valid")]
    resolved = [r for r in ok if r.get("plausibly_resolves")]
    scores = [float(r.get("score") or 0.0) for r in ok]
    return {
        "records": len(rows),
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "valid_patches": len(valid),
        "valid_patch_rate": len(valid) / len(ok) if ok else 0.0,
        "plausibly_resolved": len(resolved),
        "proxy_resolved_rate": len(resolved) / len(ok) if ok else 0.0,
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--judge-model", default="claude-opus-4-7")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=-1.0,
                   help="Set >=0 to send temperature. Default omits it for Claude gateway compatibility.")
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--max-completion-tokens", type=int, default=1200)
    p.add_argument("--max-field-chars", type=int, default=12000)
    p.add_argument("--max-patch-chars", type=int, default=20000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    gold_rows = {r.get("instance_id"): r for r in read_jsonl(args.dataset)}
    cand_rows = read_jsonl(args.candidates)
    selected = [(i, c, gold_rows.get(c.get("instance_id"), {})) for i, c in enumerate(cand_rows)]
    out_dir = args.output_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "judgments.jsonl"
    if raw_path.exists():
        raw_path.unlink()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        futures = {ex.submit(judge_one, args, item): item[0] for item in selected}
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            append_jsonl(raw_path, row)
            print(json.dumps({
                "event": "judge_finish",
                "i": len(rows),
                "n": len(selected),
                "instance_id": row.get("instance_id"),
                "status": row.get("status"),
                "resolved": row.get("plausibly_resolves"),
                "score": row.get("score"),
            }, ensure_ascii=False), flush=True)
    rows.sort(key=lambda r: int(r["row_index"]))
    with raw_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = summarize(rows)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps({"output_dir": str(out_dir), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
