#!/usr/bin/env python3
"""CLI: compile routed feedback instances into offline artifacts.

Reads the output of scripts/route_feedback.py and writes three JSONL files:
pairwise_artifacts.jsonl, scalar_artifacts.jsonl, neutral_logs.jsonl.
Local types trigger hindsight regeneration of the chosen response.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpr.compile import ArtifactCompiler
from hpr.llm import MockBackend, OpenAICompatBackend
from hpr.regenerate import (
    DEFAULT_NUM_CANDIDATES,
    DEFAULT_TEMPERATURE,
    HindsightRegenerator,
)
from hpr.types import FeedbackInstance, RouterDecision


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--routed", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--provider", choices=["openai", "mock"], default="openai")
    p.add_argument("--regen-model", default="qwen3-4b")
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key-env", default="HPR_API_KEY")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--num-candidates", type=int, default=DEFAULT_NUM_CANDIDATES)
    p.add_argument("--rejudge", action="store_true",
                   help="Re-judge regenerated candidates and drop pairs that fail.")
    p.add_argument("--force-branch", choices=["pairwise", "scalar"], default=None,
                   help="Pairwise-Only / Scalar-Only baseline compilation.")
    args = p.parse_args()

    if args.provider == "mock":
        backend = MockBackend()
    else:
        backend = OpenAICompatBackend(
            model=args.regen_model, base_url=args.base_url, api_key_env=args.api_key_env
        )
    regenerator = HindsightRegenerator(
        backend,
        model_name=args.regen_model,
        temperature=args.temperature,
        num_candidates=args.num_candidates,
        rejudge=args.rejudge,
    )
    compiler = ArtifactCompiler(regenerator=regenerator, force_branch=args.force_branch)

    routed = []
    with args.routed.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            inst = FeedbackInstance.from_dict(row)
            decision = RouterDecision(
                feedback_type=str(row.get("router_feedback_type") or "neutral"),
                label=str(row.get("router_label") or "neutral"),
                critique=str(row.get("router_critique") or ""),
                hint=str(row.get("router_hint") or ""),
                confidence=row.get("router_confidence"),
                rationale=str(row.get("router_rationale") or ""),
                router_model=str(row.get("router_model") or ""),
            )
            routed.append((inst, decision))

    artifacts = compiler.compile(routed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pairwise_artifacts.jsonl": artifacts.pairwise,
        "scalar_artifacts.jsonl": artifacts.scalar,
        "neutral_logs.jsonl": artifacts.neutral,
    }
    for name, rows in outputs.items():
        with (args.output_dir / name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    print(json.dumps({"output_dir": str(args.output_dir), **artifacts.counts()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
