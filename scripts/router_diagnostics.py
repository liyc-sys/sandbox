#!/usr/bin/env python3
"""CLI: score router labels against offline gold feedback-type labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpr.metrics import router_diagnostics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--routed", type=Path, required=True)
    p.add_argument("--pred-field", default="router_feedback_type")
    p.add_argument("--gold-field", default="gold_feedback_type")
    args = p.parse_args()

    labels = []
    with args.routed.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gold = row.get(args.gold_field)
            pred = row.get(args.pred_field)
            if gold and pred:
                labels.append((str(pred), str(gold)))
    print(json.dumps(router_diagnostics(labels), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
