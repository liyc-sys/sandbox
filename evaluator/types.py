from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class JudgeResult:
    """单个样本的评测结果。

    is_correct / score 二选一（连续指标用 score，离散用 is_correct）。
    used_fallback / fallback_overturned 用来汇总 fallback rate。
    """
    is_correct: bool | None = None
    score: float | None = None
    parsed_prediction: Any = None
    metric_name: str = ""
    judge_mode: str = ""
    used_fallback: bool = False
    fallback_overturned: bool = False
    rule_verdict: bool | None = None  # 规则原本的判断（fallback 触发前）
    rule_extracted: bool = True       # 规则是否成功抽取
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Prediction:
    """单个样本的推理产物。"""
    uid: str
    raw_response: str
    error: str | None = None
    usage: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)
