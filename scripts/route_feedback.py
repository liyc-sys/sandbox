#!/usr/bin/env python3
"""CLI: label feedback instances with the HPR router.

Input: JSONL of feedback instances (see hpr.types.FeedbackInstance fields).
Output: the same rows with router_* fields attached.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpr.llm import MockBackend, OpenAICompatBackend
from hpr.router import DEFAULT_CONFIDENCE_THRESHOLD, FeedbackRouter
from hpr.types import FeedbackInstance


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--provider", choices=["openai", "mock"], default="openai")
    p.add_argument("--model", default="qwen3-4b")
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key-env", default="HPR_API_KEY")
    p.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    p.add_argument("--preserve-structural-labels", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    if args.provider == "mock":
        backend = MockBackend()
    else:
        backend = OpenAICompatBackend(
            model=args.model, base_url=args.base_url, api_key_env=args.api_key_env
        )
    router = FeedbackRouter(
        backend,
        model_name=args.model,
        confidence_threshold=args.confidence_threshold,
        preserve_structural_labels=args.preserve_structural_labels,
    )

    n = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as fin, args.output.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            inst = FeedbackInstance.from_dict(row)
            decision = router.route(inst)
            out = dict(row)
            out["router_feedback_type"] = decision.feedback_type
            out["router_label"] = decision.label
            out["router_critique"] = decision.critique
            out["router_hint"] = decision.hint
            out["router_confidence"] = decision.confidence
            out["router_rationale"] = decision.rationale
            out["router_model"] = decision.router_model
            fout.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    print(json.dumps({"input": str(args.input), "output": str(args.output), "rows": n}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
