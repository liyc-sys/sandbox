"""把 per-sample 的 JudgeResult 聚合成 benchmark 报告。

特殊处理：
- HallusionBench：单题 → question pair 聚合 → qAcc
- OmniDocBench：取所有样本 score 平均
- RefCOCO：8 split 的 Acc@0.5 平均（如果数据里有 split 字段）
- 其它：accuracy
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

from ..types import JudgeResult


def _fallback_stats(results: list[JudgeResult]) -> dict:
    n = len(results) or 1
    used = sum(1 for r in results if r.used_fallback)
    overturned = sum(1 for r in results if r.fallback_overturned)
    return {
        "fallback_rate": used / n,
        "fallback_overturn_rate": overturned / n,
    }


def _accuracy_report(benchmark: str, metric_name: str, results: list[JudgeResult]) -> dict:
    n = len(results) or 1
    correct = sum(1 for r in results if r.is_correct)
    fb = _fallback_stats(results)
    return {
        "benchmark": benchmark,
        "main_metric_name": metric_name,
        "main_metric_value": correct / n,
        "n_samples": len(results),
        "fallback_rate": fb["fallback_rate"],
        "fallback_overturn_rate": fb["fallback_overturn_rate"],
    }


def _hallusionbench_report(records: list[dict], results: list[JudgeResult]) -> dict:
    """qAcc：按 (category, subcategory, set_id, question_id) 分组（官方 utils.py:215 定义），
    pair 内全对算 1。figure_id 不进 key —— 它区分原图 vs 变体图，是 pair 的成员维度。"""
    pair_map: dict[tuple, list[bool]] = defaultdict(list)
    for rec, res in zip(records, results):
        meta = rec.get("metadata", {}) or {}
        key = (meta.get("category"), meta.get("subcategory"),
               str(meta.get("set_id")), str(meta.get("question_id")))
        pair_map[key].append(bool(res.is_correct))
    pair_correct = [all(v) for v in pair_map.values()]
    qacc = sum(pair_correct) / max(len(pair_correct), 1)
    fb = _fallback_stats(results)
    return {
        "benchmark": "hallusionbench",
        "main_metric_name": "qAcc",
        "main_metric_value": qacc,
        "n_samples": len(results),
        "n_pairs": len(pair_correct),
        "fallback_rate": fb["fallback_rate"],
        "fallback_overturn_rate": fb["fallback_overturn_rate"],
    }


def _omnidocbench_report(results: list[JudgeResult]) -> dict:
    if not results:
        return {"benchmark": "omnidocbench", "main_metric_name": "Overall", "main_metric_value": 0.0, "n_samples": 0}
    overall = sum(r.score for r in results if r.score is not None) / max(len(results), 1)
    text_ned = sum(r.meta.get("text_ned", 0) for r in results) / max(len(results), 1)
    table_teds = sum(r.meta.get("table_teds", 0) for r in results) / max(len(results), 1)
    formula_cdm = sum(r.meta.get("formula_cdm", 0) for r in results) / max(len(results), 1)
    return {
        "benchmark": "omnidocbench",
        "main_metric_name": "Overall",
        "main_metric_value": overall,
        "n_samples": len(results),
        "fallback_rate": 0.0,
        "fallback_overturn_rate": 0.0,
        "subscores": {
            "text_ned": text_ned,
            "table_teds": table_teds,
            "formula_cdm": formula_cdm,
        },
    }


def _refcoco_report(records: list[dict], results: list[JudgeResult]) -> dict:
    """按 split 字段（来源 metadata.refcoco_split 或 record.subset）平均后再总平均。"""
    by_split: dict[str, list[bool]] = defaultdict(list)
    for rec, res in zip(records, results):
        split = (rec.get("metadata", {}) or {}).get("refcoco_split") or rec.get("subset") or "all"
        by_split[split].append(bool(res.is_correct))
    per_split = {k: sum(v) / len(v) for k, v in by_split.items() if v}
    avg = sum(per_split.values()) / max(len(per_split), 1)
    fb = _fallback_stats(results)
    return {
        "benchmark": "refcoco_avg",
        "main_metric_name": "Acc@0.5",
        "main_metric_value": avg,
        "n_samples": len(results),
        "per_split_accuracy": per_split,
        "fallback_rate": fb["fallback_rate"],
        "fallback_overturn_rate": fb["fallback_overturn_rate"],
    }


def aggregate(
    benchmark: str,
    records: list[dict],
    results: list[JudgeResult],
    metric_name: str = "Accuracy",
) -> dict:
    if benchmark == "hallusionbench":
        return _hallusionbench_report(records, results)
    if benchmark == "omnidocbench":
        return _omnidocbench_report(results)
    if benchmark == "refcoco_avg":
        return _refcoco_report(records, results)
    return _accuracy_report(benchmark, metric_name, results)


def write_report(report: dict, out_path: Path, per_sample: list[dict] | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if per_sample is not None:
        ps_path = out_path.with_suffix(".per_sample.jsonl")
        with open(ps_path, "w") as f:
            for r in per_sample:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
