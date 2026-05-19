"""每个 benchmark 一个 judge 类。

覆盖走规则 + LLM fallback 的 benchmark；OmniDocBench 单独在另一个模块。
HallusionBench 这里只做单题 judge，pair 聚合在 reports 阶段处理。
"""
from __future__ import annotations
import re
from typing import Any

from .base import BaseJudge
from ..metrics.primitives import (
    text_exact_match,
    extract_choice_letter,
    extract_yes_no,
    extract_number,
    numeric_close,
    anls,
    iou_xyxy,
    extract_bbox,
    normalize_text,
)


# ---------- 公共：多选题字母 judge ----------
class _MultipleChoiceMixin:
    def _mc_rule(self, record: dict, raw_response: str) -> tuple[Any, bool | None]:
        labels = (record.get("prompt", {}) or {}).get("choice_labels") or list("ABCDEFGHIJ")
        gt = (record.get("answer", {}) or {}).get("choice")
        if not gt:
            # 兜底：从 text 字段拿
            gt = (record.get("answer", {}) or {}).get("text")
        gt = (gt or "").strip().upper()
        letter = extract_choice_letter(raw_response, valid=labels)
        if letter is None:
            return None, None
        return letter, (letter == gt)


class MMMUJudge(_MultipleChoiceMixin, BaseJudge):
    benchmark = "mmmu"
    metric_name = "Accuracy"
    judge_mode = "multiple_choice_parse"

    def _rule_judge(self, record, raw_response):
        # MMMU 大部分是多选；少量 open-ended
        ans = record.get("answer", {}) or {}
        if ans.get("type") == "choice" or (record.get("prompt", {}) or {}).get("choices"):
            return self._mc_rule(record, raw_response)
        # open-ended：归一化 + alias
        gt = ans.get("text") or ""
        aliases = ans.get("aliases") or []
        if not raw_response.strip():
            return None, None
        ok = text_exact_match(raw_response, gt, aliases)
        return raw_response.strip(), ok


class RealWorldQAJudge(BaseJudge):
    benchmark = "realworldqa"
    metric_name = "Accuracy"
    judge_mode = "normalized_exact_match"

    def _rule_judge(self, record, raw_response):
        ans = record.get("answer", {}) or {}
        gt = ans.get("text") or ""
        aliases = ans.get("aliases") or []
        # RealWorldQA 很多是字母 A/B/C 形式（题干带选项），优先抽字母
        choices_in_prompt = re.findall(
            r"^\s*([A-D])[\.\)]", (record.get("prompt", {}) or {}).get("text", ""), re.MULTILINE
        )
        if choices_in_prompt or (gt and len(gt) == 1 and gt.upper() in "ABCDEFGH"):
            letter = extract_choice_letter(raw_response, valid=list("ABCDEFGH"))
            if letter is not None:
                return letter, (letter == gt.strip().upper())
        if not raw_response.strip():
            return None, None
        ok = text_exact_match(raw_response, gt, aliases)
        return raw_response.strip(), ok


class VLMsAreBlindJudge(BaseJudge):
    benchmark = "vlmsareblind"
    metric_name = "Accuracy"
    judge_mode = "normalized_exact_match"

    def _rule_judge(self, record, raw_response):
        ans = record.get("answer", {}) or {}
        gt = ans.get("text") or ""
        aliases = ans.get("aliases") or []
        if not raw_response.strip():
            return None, None
        # 先试字母（很多题是选项）
        if gt and len(gt) == 1 and gt.upper() in "ABCDEFGH":
            letter = extract_choice_letter(raw_response, valid=list("ABCDEFGH"))
            if letter is not None:
                return letter, (letter == gt.upper())
        ok = text_exact_match(raw_response, gt, aliases)
        return raw_response.strip(), ok


class HallusionBenchJudge(BaseJudge):
    """单题 yes/no judge；qAcc 聚合在 reports 阶段做。"""
    benchmark = "hallusionbench"
    metric_name = "qAcc"  # 报告时按 pair 聚合
    judge_mode = "yes_no"

    def _rule_judge(self, record, raw_response):
        gt_raw = (record.get("answer", {}) or {}).get("text") or ""
        gt = gt_raw.strip()
        if gt not in ("0", "1"):
            return None, None
        pred = extract_yes_no(raw_response)
        if pred is None:
            return None, None
        return pred, (str(pred) == gt)


class ChartQAProJudge(BaseJudge):
    benchmark = "chartqapro"
    metric_name = "Accuracy"
    judge_mode = "mixed"

    # 模型常用 "**Answer**: X" / "Answer: X" / "answer: X" / 末尾加粗 "**X**" 收尾。
    # 优先按显式 Answer 标签，其次按 \boxed / 最后一行加粗。
    _ANSWER_PATTERNS = [
        re.compile(r"\*{0,2}\s*answer\s*\*{0,2}\s*[:：]?\s*\*{0,2}\s*(.+?)\s*\*{0,2}\s*$",
                   re.IGNORECASE | re.MULTILINE),
        re.compile(r"\\boxed\{\s*([^{}]+?)\s*\}"),
    ]

    def _extract_final_answer(self, raw_response: str) -> str:
        if not raw_response:
            return ""
        # 倒序找：从最后一行开始往前找 Answer 标签
        for pat in self._ANSWER_PATTERNS:
            matches = list(pat.finditer(raw_response))
            if matches:
                # 取最后一个命中的（最贴近模型最终答案）
                ans = matches[-1].group(1).strip()
                # 去掉两端引号、句号、加粗
                ans = ans.strip("*").strip("\"'`").rstrip(".。")
                return ans
        return raw_response.strip()

    def _rule_judge(self, record, raw_response):
        ans = record.get("answer", {}) or {}
        meta = record.get("metadata", {}) or {}
        gt = ans.get("text") or ""
        if not raw_response.strip():
            return None, None
        qtype = (meta.get("question_type") or "").lower()

        final = self._extract_final_answer(raw_response)

        # 年份类：exact match
        years = meta.get("year") or []
        is_year = isinstance(years, list) and any(str(y).upper() == "YES" for y in years)
        if is_year:
            return final, text_exact_match(final, gt, ans.get("aliases"))

        # Fact Checking / Multi Choice
        if qtype in ("fact checking", "multi choice", "multi-choice", "multiple choice"):
            return final, text_exact_match(final, gt, ans.get("aliases"))

        # 试数值
        gt_num = None
        try:
            gt_num = float(str(gt).replace(",", ""))
        except (ValueError, TypeError):
            pass
        if gt_num is not None:
            pred_num = extract_number(final) or extract_number(raw_response)
            if pred_num is None:
                return None, None
            return pred_num, numeric_close(pred_num, gt_num)

        # 文本：先 exact 试一下（带 aliases），再 ANLS 兜底
        if text_exact_match(final, gt, ans.get("aliases")):
            return final, True
        score = anls(normalize_text(final), normalize_text(gt))
        return final, (score > 0.5)



class MathVistaJudge(BaseJudge):
    benchmark = "mathvista"
    metric_name = "Accuracy"
    judge_mode = "mixed"

    def _rule_judge(self, record, raw_response):
        ans = record.get("answer", {}) or {}
        meta = record.get("metadata", {}) or {}
        gt = ans.get("text") or ""
        if not raw_response.strip():
            return None, None

        qtype = meta.get("question_type", "")
        atype = meta.get("answer_type", "")
        choices = (record.get("prompt", {}) or {}).get("choices") or []

        if qtype == "multi_choice" and choices:
            labels = (record.get("prompt", {}) or {}).get("choice_labels") or list("ABCDEFGH")
            letter = extract_choice_letter(raw_response, valid=labels)
            if letter is None:
                return None, None
            # gt 可能是 choice 文本而非字母
            if gt.strip().upper() in labels:
                return letter, (letter == gt.strip().upper())
            try:
                idx = labels.index(letter)
                return letter, (idx < len(choices) and normalize_text(choices[idx]) == normalize_text(gt))
            except ValueError:
                return letter, False

        if atype in ("integer", "float"):
            pred = extract_number(raw_response)
            if pred is None:
                return None, None
            try:
                gt_num = float(gt)
            except ValueError:
                return None, None
            if atype == "integer":
                return pred, int(round(pred)) == int(round(gt_num))
            precision = meta.get("precision") or 1
            try:
                p = round(float(pred), int(precision)) if precision else round(float(pred))
                g = round(float(gt_num), int(precision)) if precision else round(float(gt_num))
                return pred, abs(p - g) < 10 ** (-(int(precision) + 1))
            except Exception:
                return pred, abs(pred - gt_num) < 1e-3

        # list / 其它：归一化 exact
        return raw_response.strip(), text_exact_match(raw_response, gt, ans.get("aliases"))


class PhyXJudge(_MultipleChoiceMixin, BaseJudge):
    benchmark = "phyx_openended"
    metric_name = "Accuracy"
    judge_mode = "mixed"

    def _rule_judge(self, record, raw_response):
        ans = record.get("answer", {}) or {}
        if ans.get("type") == "choice" or (record.get("prompt", {}) or {}).get("choices"):
            return self._mc_rule(record, raw_response)
        # 开放题 → 数值
        if not raw_response.strip():
            return None, None
        gt_num = extract_number(ans.get("text") or "")
        pred_num = extract_number(raw_response)
        if gt_num is not None and pred_num is not None:
            return pred_num, numeric_close(pred_num, gt_num)
        return raw_response.strip(), text_exact_match(raw_response, ans.get("text") or "", ans.get("aliases"))


class VisuLogicJudge(_MultipleChoiceMixin, BaseJudge):
    benchmark = "visulogic"
    metric_name = "Accuracy"
    judge_mode = "multiple_choice_parse"

    def _rule_judge(self, record, raw_response):
        return self._mc_rule(record, raw_response)


class BabyVisionJudge(BaseJudge):
    benchmark = "babyvision"
    metric_name = "Accuracy"
    judge_mode = "mixed"

    def _rule_judge(self, record, raw_response):
        ans = record.get("answer", {}) or {}
        if not raw_response.strip():
            return None, None

        if ans.get("type") == "choice":
            labels = (record.get("prompt", {}) or {}).get("choice_labels") or list("ABCDEF")
            gt_choice = (ans.get("choice") or "").strip().upper()
            letter = extract_choice_letter(raw_response, valid=labels)
            if letter is not None:
                return letter, (letter == gt_choice)
            gt_text = ans.get("text") or ""
            return raw_response.strip(), text_exact_match(raw_response, gt_text, ans.get("aliases"))

        gt = ans.get("text") or ""
        return raw_response.strip(), text_exact_match(raw_response, gt, ans.get("aliases"))


class CountBenchJudge(BaseJudge):
    benchmark = "countbench"
    metric_name = "Accuracy"
    judge_mode = "integer_count"

    def _rule_judge(self, record, raw_response):
        if not raw_response.strip():
            return None, None
        ans = record.get("answer", {}) or {}
        gt = ans.get("value")
        if gt is None:
            gt = ans.get("text")
        pred = extract_number(raw_response)
        if pred is None:
            return None, None
        try:
            gt_num = float(gt)
        except (TypeError, ValueError):
            return pred, False
        return pred, int(round(pred)) == int(round(gt_num))


class RefCOCOJudge(BaseJudge):
    benchmark = "refcoco_avg"
    metric_name = "Acc@0.5"
    judge_mode = "iou_at_0.5"
    fallback_include_image = True

    def _rule_judge(self, record, raw_response):
        gt_bbox = (record.get("answer", {}) or {}).get("bbox")
        if not gt_bbox or len(gt_bbox) != 4:
            return None, None
        pred_bbox = extract_bbox(raw_response)
        if pred_bbox is None:
            return None, None
        # 兼容 0-1000 归一化坐标：若所有值都 ≤ 1000 而 gt 用绝对像素，需要按图缩放
        # 这里按绝对像素直接算；坐标系归一化由 inference 阶段保证
        meta_img = (record.get("media") or [{}])[0]
        w = meta_img.get("width")
        h = meta_img.get("height")
        # 自适应：如果模型坐标都在 [0, 1000]，按 1000 归一化反推
        if w and h and max(pred_bbox) <= 1001:
            pred_bbox = [
                pred_bbox[0] / 1000.0 * w,
                pred_bbox[1] / 1000.0 * h,
                pred_bbox[2] / 1000.0 * w,
                pred_bbox[3] / 1000.0 * h,
            ]
        elif w and h and max(pred_bbox) <= 1.001:
            pred_bbox = [
                pred_bbox[0] * w,
                pred_bbox[1] * h,
                pred_bbox[2] * w,
                pred_bbox[3] * h,
            ]
        score = iou_xyxy(pred_bbox, gt_bbox)
        return {"pred_bbox": pred_bbox, "iou": score}, score >= 0.5


JUDGE_REGISTRY: dict[str, type[BaseJudge]] = {
    "mmmu": MMMUJudge,
    "realworldqa": RealWorldQAJudge,
    "vlmsareblind": VLMsAreBlindJudge,
    "hallusionbench": HallusionBenchJudge,
    "chartqapro": ChartQAProJudge,
    "mathvista": MathVistaJudge,
    "phyx_openended": PhyXJudge,
    "visulogic": VisuLogicJudge,
    "babyvision": BabyVisionJudge,
    "countbench": CountBenchJudge,
    "refcoco_avg": RefCOCOJudge,
}
