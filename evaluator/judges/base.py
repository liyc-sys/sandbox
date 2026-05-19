"""所有 benchmark judge 的基类。

子类只需实现 _rule_judge(record, raw_response) -> (parsed, rule_correct, extracted)
基类负责按"规则抽不到 OR 规则判 wrong → LLM fallback"统一调度。

OmniDocBench 这种连续指标 benchmark 不走基类，直接独立实现。
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

from ..types import JudgeResult
from .llm_fallback import llm_judge, DEFAULT_FALLBACK_MODEL


class BaseJudge:
    benchmark: str = ""
    metric_name: str = "Accuracy"
    judge_mode: str = "exact_match"
    fallback_include_image: bool = False  # grounding 类设为 True

    def __init__(
        self,
        framework_root: Path,
        fallback_model: str = DEFAULT_FALLBACK_MODEL,
        enable_fallback: bool = True,
    ):
        self.framework_root = framework_root
        self.fallback_model = fallback_model
        self.enable_fallback = enable_fallback

    # 子类必须实现：
    #   返回 (parsed_prediction, rule_is_correct_or_None_if_extract_failed)
    #   extract_failed 时 rule_is_correct 必须返回 None
    def _rule_judge(self, record: dict, raw_response: str) -> tuple[Any, bool | None]:
        raise NotImplementedError

    def judge(self, record: dict, raw_response: str) -> JudgeResult:
        parsed, rule_verdict = self._rule_judge(record, raw_response)
        rule_extracted = rule_verdict is not None

        # 规则成功且为 correct → 直接采纳
        if rule_extracted and rule_verdict is True:
            return JudgeResult(
                is_correct=True,
                parsed_prediction=parsed,
                metric_name=self.metric_name,
                judge_mode=self.judge_mode,
                used_fallback=False,
                rule_verdict=True,
                rule_extracted=True,
            )

        # 规则抽不出 / 规则判 wrong → fallback
        if not self.enable_fallback:
            return JudgeResult(
                is_correct=bool(rule_verdict) if rule_extracted else False,
                parsed_prediction=parsed,
                metric_name=self.metric_name,
                judge_mode=self.judge_mode,
                used_fallback=False,
                rule_verdict=rule_verdict,
                rule_extracted=rule_extracted,
            )

        llm_verdict, llm_reason = llm_judge(
            record,
            raw_response,
            model=self.fallback_model,
            framework_root=self.framework_root,
            include_image=self.fallback_include_image,
        )

        # LLM 也无法判定 → 当 wrong 处理（不算对）
        if llm_verdict is None:
            return JudgeResult(
                is_correct=False,
                parsed_prediction=parsed,
                metric_name=self.metric_name,
                judge_mode="llm_as_judge",
                used_fallback=True,
                rule_verdict=rule_verdict,
                rule_extracted=rule_extracted,
                meta={"llm_reason": llm_reason, "llm_unknown": True},
            )

        return JudgeResult(
            is_correct=llm_verdict,
            parsed_prediction=parsed,
            metric_name=self.metric_name,
            judge_mode="llm_as_judge",
            used_fallback=True,
            fallback_overturned=(rule_extracted and rule_verdict is False and llm_verdict is True),
            rule_verdict=rule_verdict,
            rule_extracted=rule_extracted,
            meta={"llm_reason": llm_reason},
        )
