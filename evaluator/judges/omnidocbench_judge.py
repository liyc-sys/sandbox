"""OmniDocBench judge：v1.5 Overall = [(1-TextNED)*100 + TableTEDS + FormulaCDM] / 3。

不接 LLM fallback。三个子指标各有独立打分。
- TextNED：纯文本块归一化 Levenshtein 距离
- TableTEDS：表格 Tree Edit Distance Similarity（这里实现一个简化版，基于 HTML 标签序列）
- FormulaCDM：公式 Character Detection Matching（简化版：基于字符集 F1）

为不引入第三方依赖，这里走简化版指标；后续可替换为官方实现。
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

from ..types import JudgeResult
from ..metrics.primitives import _levenshtein


def _ned(pred: str, gt: str) -> float:
    p = (pred or "").strip()
    g = (gt or "").strip()
    if not p and not g:
        return 0.0
    return _levenshtein(p, g) / max(len(p), len(g), 1)


def _tag_sequence(html: str) -> list[str]:
    return re.findall(r"<\s*/?\s*([a-zA-Z]+)[^>]*>", html or "")


def _teds_simple(pred_html: str, gt_html: str) -> float:
    """简化 TEDS：基于标签序列 + 文本归一化的 1 - 编辑距离比例。"""
    if not gt_html and not pred_html:
        return 1.0
    if not gt_html or not pred_html:
        return 0.0
    pt = _tag_sequence(pred_html)
    gt = _tag_sequence(gt_html)
    tag_dist = _levenshtein(" ".join(pt), " ".join(gt))
    tag_score = 1.0 - tag_dist / max(len(" ".join(pt)), len(" ".join(gt)), 1)
    pred_text = re.sub(r"<[^>]+>", " ", pred_html).split()
    gt_text = re.sub(r"<[^>]+>", " ", gt_html).split()
    text_dist = _levenshtein(" ".join(pred_text), " ".join(gt_text))
    text_score = 1.0 - text_dist / max(len(" ".join(pred_text)), len(" ".join(gt_text)), 1)
    score = 0.5 * tag_score + 0.5 * text_score
    return max(0.0, min(1.0, score))


def _cdm_simple(pred_latex: str, gt_latex: str) -> float:
    """简化 CDM：去空白后字符集 F1。"""
    p = re.sub(r"\s+", "", pred_latex or "")
    g = re.sub(r"\s+", "", gt_latex or "")
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    sp, sg = set(p), set(g)
    if not sp or not sg:
        return 0.0
    inter = sp & sg
    prec = len(inter) / len(sp)
    rec = len(inter) / len(sg)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _extract_gt_blocks(record: dict) -> dict:
    """从 unified sample 的 answer.structured.layout_dets 抽出文本/表格/公式 GT。"""
    structured = (record.get("answer", {}) or {}).get("structured") or {}
    dets = structured.get("layout_dets", []) if isinstance(structured, dict) else []
    text_blocks: list[str] = []
    table_blocks: list[str] = []
    formula_blocks: list[str] = []
    for d in dets:
        ctype = d.get("category_type", "")
        text = d.get("text", "") or ""
        if "equation" in ctype:
            formula_blocks.append(text)
        elif "table" in ctype:
            # 表格 GT 通常是 HTML
            table_blocks.append(text)
        elif "text" in ctype or "title" in ctype or "list" in ctype or "caption" in ctype:
            text_blocks.append(text)
    return {
        "text": "\n".join(t for t in text_blocks if t),
        "tables": table_blocks,
        "formulas": formula_blocks,
    }


def _split_pred(raw_response: str) -> dict:
    """把模型输出（markdown）粗暴拆成 text / tables(html) / formulas(latex)。"""
    if not raw_response:
        return {"text": "", "tables": [], "formulas": []}
    tables = re.findall(r"<table.*?</table>", raw_response, re.DOTALL | re.IGNORECASE)
    # 抓 $$...$$ 和 \[...\]
    formulas = re.findall(r"\$\$(.+?)\$\$", raw_response, re.DOTALL)
    formulas += re.findall(r"\\\[(.+?)\\\]", raw_response, re.DOTALL)
    # text 部分：去掉表格 / 公式
    text = raw_response
    for t in tables:
        text = text.replace(t, " ")
    text = re.sub(r"\$\$.+?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\[.+?\\\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    return {"text": text, "tables": tables, "formulas": [f.strip() for f in formulas]}


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


class OmniDocBenchJudge:
    benchmark = "omnidocbench"
    metric_name = "Overall"
    judge_mode = "structured_doc_eval"

    def __init__(self, framework_root: Path, **kwargs):
        self.framework_root = framework_root

    def judge(self, record: dict, raw_response: str) -> JudgeResult:
        gt = _extract_gt_blocks(record)
        pred = _split_pred(raw_response)

        text_ned = _ned(pred["text"], gt["text"])

        # 表格：逐对最优匹配（小数据量直接配对）
        if gt["tables"] and pred["tables"]:
            n = min(len(gt["tables"]), len(pred["tables"]))
            table_scores = [_teds_simple(pred["tables"][i], gt["tables"][i]) for i in range(n)]
            table_teds = _avg(table_scores) * 100.0
        elif not gt["tables"]:
            table_teds = 100.0  # 无表则该项满分
        else:
            table_teds = 0.0

        if gt["formulas"] and pred["formulas"]:
            n = min(len(gt["formulas"]), len(pred["formulas"]))
            formula_scores = [_cdm_simple(pred["formulas"][i], gt["formulas"][i]) for i in range(n)]
            formula_cdm = _avg(formula_scores) * 100.0
        elif not gt["formulas"]:
            formula_cdm = 100.0
        else:
            formula_cdm = 0.0

        overall = ((1.0 - text_ned) * 100.0 + table_teds + formula_cdm) / 3.0

        return JudgeResult(
            is_correct=None,
            score=overall,
            parsed_prediction=None,
            metric_name="Overall",
            judge_mode="structured_doc_eval",
            used_fallback=False,
            rule_extracted=True,
            meta={
                "text_ned": text_ned,
                "table_teds": table_teds,
                "formula_cdm": formula_cdm,
                "n_gt_tables": len(gt["tables"]),
                "n_gt_formulas": len(gt["formulas"]),
            },
        )
