#!/usr/bin/env python3
"""CLI: routed offline training from compiled artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpr.train import RoutedTrainer, TrainConfig, samples_from_artifacts


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--artifacts-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--lambda-pairwise", type=float, default=1.0)
    p.add_argument("--lambda-scalar", type=float, default=1.0)
    p.add_argument("--hpr-sft-coef", type=float, default=0.05)
    p.add_argument("--positive-sft-coef", type=float, default=0.02)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    samples = samples_from_artifacts(
        read_jsonl(args.artifacts_dir / "pairwise_artifacts.jsonl"),
        read_jsonl(args.artifacts_dir / "scalar_artifacts.jsonl"),
    )
    config = TrainConfig(
        max_length=args.max_length,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        grad_accum=args.grad_accum,
        seed=args.seed,
        lambda_pairwise=args.lambda_pairwise,
        lambda_scalar=args.lambda_scalar,
        hpr_sft_coef=args.hpr_sft_coef,
        positive_sft_coef=args.positive_sft_coef,
        device=args.device,
    )
    trainer = RoutedTrainer(model, tokenizer, config, args.output_dir)
    summary = trainer.train(samples)
    final_dir = trainer.save("final")
    print(json.dumps({"final_dir": str(final_dir), **{k: v for k, v in summary.items() if k != "config"}}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
