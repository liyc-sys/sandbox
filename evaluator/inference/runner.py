"""8 并发跑 inference，落 predictions.jsonl。"""
from __future__ import annotations
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from ..types import Prediction
from .oneapi_client import build_messages, call_model

DEFAULT_WORKERS = 8


def _build_prompt_for_sample(sample: dict) -> str:
    """把统一 sample 的 prompt 拼成模型可读字符串。多选题把 choices 也拼上。"""
    p = sample.get("prompt", {}) or {}
    text = p.get("text", "") or ""
    choices = p.get("choices") or []
    labels = p.get("choice_labels") or []
    if choices:
        if labels and len(labels) == len(choices):
            opts = "\n".join(f"{lab}. {ch}" for lab, ch in zip(labels, choices))
        else:
            opts = "\n".join(f"{i}. {ch}" for i, ch in enumerate(choices))
        text = f"{text}\n\nOptions:\n{opts}"
    instr = p.get("instruction")
    if instr:
        text = f"{text}\n\n{instr}"
    return text


def _image_paths(sample: dict) -> list[str]:
    return [m["path"] for m in sample.get("media", []) if m.get("type") == "image" and m.get("path")]


def run_inference(
    samples: list[dict],
    model: str,
    framework_root: Path,
    out_path: Path,
    temperature: float = 0.0,
    workers: int = DEFAULT_WORKERS,
) -> list[Prediction]:
    """并发跑推理，结果同时写到 out_path 和返回值。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _do(sample: dict) -> Prediction:
        uid = sample["uid"]
        try:
            prompt_text = _build_prompt_for_sample(sample)
            imgs = _image_paths(sample)
            msgs = build_messages(prompt_text, imgs, framework_root)
            text, usage, err = call_model(msgs, model=model, temperature=temperature)
            return Prediction(uid=uid, raw_response=text, error=err, usage=usage)
        except Exception as e:  # noqa: BLE001
            return Prediction(uid=uid, raw_response="", error=f"runner error: {e}")

    results: list[Prediction] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_do, s): s["uid"] for s in samples}
        with open(out_path, "w") as f:
            for fut in as_completed(futures):
                pred = fut.result()
                results.append(pred)
                f.write(json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")
                f.flush()
    # 保持与输入顺序一致
    order = {s["uid"]: i for i, s in enumerate(samples)}
    results.sort(key=lambda p: order.get(p.uid, 0))
    return results


def load_samples(jsonl_path: Path, limit: int | None = None) -> list[dict]:
    out: list[dict] = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
