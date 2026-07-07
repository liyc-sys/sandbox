#!/usr/bin/env python3
"""Generate candidate patches for SWE-bench Lite proxy evaluation."""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx


PROMPT_TEMPLATE = """You are a software engineering agent.

Given a GitHub issue, produce a minimal unified diff patch that fixes the issue.
Return only a git-style unified diff beginning with diff --git. Do not include
markdown fences, explanation, tests unless the issue requires changing tests, or
any text outside the patch.

Repository: {repo}
Base commit: {base_commit}
Instance id: {instance_id}

Issue:
{problem_statement}
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>\s*", "", text or "", flags=re.IGNORECASE).strip()


def extract_patch(text: str) -> str:
    text = strip_thinking(text)
    fenced = re.search(r"```(?:diff|patch)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    idx = text.find("diff --git ")
    if idx >= 0:
        text = text[idx:].strip()
    return text.strip()


def patch_valid(patch: str) -> tuple[bool, dict[str, Any]]:
    if not patch:
        return False, {"reason": "empty"}
    has_diff = "diff --git " in patch
    has_old = "--- a/" in patch or "--- /dev/null" in patch
    has_new = "+++ b/" in patch or "+++ /dev/null" in patch
    files = re.findall(r"^diff --git a/(.*?) b/(.*?)$", patch, flags=re.MULTILINE)
    valid = has_diff and has_old and has_new and bool(files)
    return valid, {
        "has_diff": has_diff,
        "has_old": has_old,
        "has_new": has_new,
        "files": sorted({b for _, b in files}),
        "chars": len(patch),
    }


def call_model(args: argparse.Namespace, row: dict[str, Any]) -> str:
    body: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    repo=row.get("repo") or "",
                    base_commit=row.get("base_commit") or "",
                    instance_id=row.get("instance_id") or "",
                    problem_statement=row.get("problem_statement") or "",
                ),
            }
        ],
        "temperature": args.temperature,
        "stream": False,
    }
    if args.max_completion_tokens > 0:
        body["max_completion_tokens"] = args.max_completion_tokens

    last_err: Exception | None = None
    url = (
        f"{args.base_url.rstrip('/')}/chat/completions"
        if args.base_url.rstrip("/").endswith("/v1")
        else f"{args.base_url.rstrip('/')}/v1/chat/completions"
    )
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
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            text = strip_thinking(msg.get("content") or "")
            if text:
                return text
            last_err = RuntimeError("empty response")
        except Exception as e:
            last_err = e
        if attempt < args.max_retries:
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"model call failed: {last_err}")


def evaluate_one(args: argparse.Namespace, item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    index, row = item
    start = time.time()
    try:
        response = call_model(args, row)
        patch = extract_patch(response)
        valid, detail = patch_valid(patch)
        status = "ok"
        error = None
    except Exception as e:
        response = ""
        patch = ""
        valid = False
        detail = {"reason": "call_error"}
        status = "error"
        error = str(e)
    return {
        "row_index": index,
        "instance_id": row.get("instance_id"),
        "repo": row.get("repo"),
        "base_commit": row.get("base_commit"),
        "problem_statement": row.get("problem_statement"),
        "raw_response": response,
        "patch": patch,
        "patch_valid": valid,
        "patch_detail": detail,
        "status": status,
        "error": error,
        "wall_time_seconds": round(time.time() - start, 3),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    valid = [r for r in ok if r.get("patch_valid")]
    return {
        "records": len(rows),
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "valid_patches": len(valid),
        "valid_patch_rate": len(valid) / len(ok) if ok else 0.0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--max-completion-tokens", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data = read_jsonl(args.dataset)
    if args.limit:
        data = data[: args.limit]
    selected = list(enumerate(data))
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
                "instance_id": row.get("instance_id"),
                "status": row.get("status"),
                "patch_valid": row.get("patch_valid"),
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
