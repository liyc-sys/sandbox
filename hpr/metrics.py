"""Router diagnostics against offline gold feedback-type labels (paper Table 3).

Metrics:
  type accuracy    fraction of scored instances with predicted type == gold;
  local F1         local_preference + local_correction as the positive class;
  delayed F1       delayed_trajectory_outcome as the positive class;
  delayed -> HPR   fraction of gold delayed-outcome instances incorrectly
                   routed to the pairwise (local) branch.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .types import LOCAL_TYPES

DELAYED_TYPE = "delayed_trajectory_outcome"


def _f1(tp: int, fp: int, fn: int) -> Optional[float]:
    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    if tp + fp == 0 or tp + fn == 0:
        return None
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def router_diagnostics(
    labels: List[Tuple[str, str]]
) -> Dict[str, Optional[float]]:
    """labels: list of (predicted_type, gold_type) over scored instances."""
    n = len(labels)
    if n == 0:
        return {
            "n": 0,
            "type_accuracy": None,
            "local_f1": None,
            "delayed_f1": None,
            "delayed_to_hpr": None,
        }
    correct = sum(1 for pred, gold in labels if pred == gold)

    local_tp = sum(1 for p, g in labels if p in LOCAL_TYPES and g in LOCAL_TYPES)
    local_fp = sum(1 for p, g in labels if p in LOCAL_TYPES and g not in LOCAL_TYPES)
    local_fn = sum(1 for p, g in labels if p not in LOCAL_TYPES and g in LOCAL_TYPES)

    delayed_tp = sum(1 for p, g in labels if p == DELAYED_TYPE and g == DELAYED_TYPE)
    delayed_fp = sum(1 for p, g in labels if p == DELAYED_TYPE and g != DELAYED_TYPE)
    delayed_fn = sum(1 for p, g in labels if p != DELAYED_TYPE and g == DELAYED_TYPE)

    gold_delayed = sum(1 for _, g in labels if g == DELAYED_TYPE)
    delayed_into_hpr = sum(
        1 for p, g in labels if g == DELAYED_TYPE and p in LOCAL_TYPES
    )

    return {
        "n": n,
        "type_accuracy": correct / n,
        "local_f1": _f1(local_tp, local_fp, local_fn),
        "delayed_f1": _f1(delayed_tp, delayed_fp, delayed_fn),
        "delayed_to_hpr": (delayed_into_hpr / gold_delayed) if gold_delayed else None,
    }
