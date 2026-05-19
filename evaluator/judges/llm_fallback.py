"""统一 LLM judge：当规则 judge 抽不出 / 判 wrong 时由它出最终判决。

输入：record、模型 raw_response、规则解析结果
输出：correct / incorrect / unknown，附简短理由
"""
from __future__ import annotations
import json
import re
from pathlib import Path

from ..inference.oneapi_client import call_model, build_messages

DEFAULT_FALLBACK_MODEL = "doubao-seed-1-6-lite-251015"

JUDGE_PROMPT_TEMPLATE = """You are a strict grader for a vision benchmark. \
Decide whether the model's answer is correct.

Question / prompt given to the model:
{prompt}

Reference answer (ground truth):
{gt}

Model's raw answer:
{pred}

Rules:
- Output a JSON object with two fields: "verdict" and "reason".
- "verdict" must be exactly one of: "correct", "incorrect".
- "reason" is a short explanation (<= 30 words).
- Be strict on factual / numeric correctness, but tolerant of formatting differences \
(case, punctuation, equivalent expressions, equivalent units, alternative valid phrasings).
"""


def _gt_string(record: dict) -> str:
    ans = record.get("answer", {}) or {}
    if ans.get("text"):
        return str(ans["text"])
    if ans.get("choice"):
        return str(ans["choice"])
    if ans.get("value") is not None:
        return str(ans["value"])
    if ans.get("aliases"):
        return " | ".join(map(str, ans["aliases"]))
    return json.dumps(ans, ensure_ascii=False)


def _prompt_string(record: dict) -> str:
    p = record.get("prompt", {}) or {}
    text = p.get("text", "")
    choices = p.get("choices") or []
    labels = p.get("choice_labels") or []
    if choices:
        if labels and len(labels) == len(choices):
            opts = "\n".join(f"{lab}. {ch}" for lab, ch in zip(labels, choices))
        else:
            opts = "\n".join(f"- {ch}" for ch in choices)
        text = f"{text}\nOptions:\n{opts}"
    return text


def _parse_verdict(text: str) -> tuple[bool | None, str]:
    """从 LLM 输出抽 verdict。返回 (is_correct, reason)。抽不出 → (None, raw)。"""
    if not text:
        return None, ""
    # 优先 JSON 解析
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            v = str(obj.get("verdict", "")).strip().lower()
            r = str(obj.get("reason", "")).strip()
            if v == "correct":
                return True, r
            if v == "incorrect":
                return False, r
    except Exception:  # noqa: BLE001
        pass
    # 退化：直接找关键词
    low = text.lower()
    if "correct" in low and "incorrect" not in low:
        return True, text.strip()[:200]
    if "incorrect" in low or "wrong" in low:
        return False, text.strip()[:200]
    return None, text.strip()[:200]


def llm_judge(
    record: dict,
    raw_response: str,
    model: str = DEFAULT_FALLBACK_MODEL,
    framework_root: Path | None = None,
    include_image: bool = False,
) -> tuple[bool | None, str]:
    """让 LLM 判断模型答案是否正确。返回 (is_correct, reason)。

    include_image=True 时把样本图也一起喂给 judge（grounding 类需要）。
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        prompt=_prompt_string(record),
        gt=_gt_string(record),
        pred=raw_response or "<empty>",
    )
    if include_image and framework_root is not None:
        imgs = [m["path"] for m in record.get("media", []) if m.get("type") == "image" and m.get("path")]
        msgs = build_messages(prompt, imgs, framework_root)
    else:
        msgs = [{"role": "user", "content": prompt}]
    text, _usage, err = call_model(msgs, model=model)
    if err:
        return None, f"LLM error: {err}"
    return _parse_verdict(text)
