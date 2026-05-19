"""推理 client：把 unified sample 转成 oneapi 多模态 message 并调用。

依赖外部 oneapi.py：/Users/liyc/Desktop/oneapi.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any

# 让 import oneapi 可用
ONEAPI_DIR = "/Users/liyc/Desktop"
if ONEAPI_DIR not in sys.path:
    sys.path.insert(0, ONEAPI_DIR)

import oneapi  # noqa: E402

REASONING_MODEL_KEYS = ("thinking", "reasoning", "deepseek-v4-pro", "o1", "o3")


def is_reasoning_model(model: str) -> bool:
    m = model.lower()
    return any(k in m for k in REASONING_MODEL_KEYS)


def build_messages(prompt_text: str, image_paths: list[str], framework_root: Path) -> list[dict]:
    """把 prompt + 图像路径 拼成 oneapi 消息。

    image_paths 中的相对路径会以 framework_root 为基拼接。
    """
    content: list[dict] = []
    for p in image_paths:
        path = Path(p)
        if not path.is_absolute():
            path = framework_root / p
        b64 = oneapi.encode_image(str(path))
        if b64 is None:
            continue
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def call_model(messages: list[dict], model: str, temperature: float = 0.0) -> tuple[str, dict | None, str | None]:
    """调 oneapi。返回 (text, usage, error)。"""
    # 注意：oneapi.gpt_api_call 当前签名只接受 (messages, model, max_tokens)
    # temperature 不在签名里 —— 走默认。如需控温，需要改外部 oneapi.py。
    # 这里保留 temperature 参数仅用于未来扩展，先不传。
    _ = temperature
    if is_reasoning_model(model):
        resp = oneapi.gpt_api_call(messages, model=model)
    else:
        resp = oneapi.gpt_api_call(messages, model=model)
    if isinstance(resp, str):
        return "", None, resp
    try:
        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        usage = None
        if getattr(resp, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }
        return text, usage, None
    except Exception as e:  # noqa: BLE001
        return "", None, f"parse error: {e}"
