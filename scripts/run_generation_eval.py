#!/usr/bin/env python3
"""Generation benchmark runner.

Examples:
  python3 scripts/run_generation_eval.py --benchmark geneval --model bagel --phase prepare
  python3 scripts/run_generation_eval.py --all --model bagel --phase full --backend existing --backend-option source_root=/path/to/images --scorer torchumm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluator.generation.benchmarks import BENCHMARKS
from evaluator.generation.runner import run_generation_benchmark, run_generation_suite


def _parse_backend_options(values: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"--backend-option expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() in ("true", "false"):
            out[key] = value.lower() == "true"
        else:
            try:
                out[key] = int(value)
            except ValueError:
                out[key] = value
    return out


def _parse_key_value_options(values: list[str] | None, option_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"{option_name} expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() in ("true", "false"):
            out[key] = value.lower() == "true"
        else:
            try:
                out[key] = int(value)
            except ValueError:
                try:
                    out[key] = float(value)
                except ValueError:
                    out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run image-generation benchmarks.")
    parser.add_argument("--benchmark", help="single generation benchmark")
    parser.add_argument("--all", action="store_true", help="run all generation benchmarks")
    parser.add_argument("--model", required=True, help="model/backbone name used in output paths")
    parser.add_argument(
        "--phase",
        choices=["prepare", "generate", "score", "evaluate", "full"],
        default="prepare",
        help="prepare writes manifests only; generate runs inference; score/evaluate runs scorer; full runs generate+score",
    )
    parser.add_argument(
        "--backend",
        default="manifest",
        choices=["manifest", "hope_manifest", "placeholder", "existing", "command"],
        help="generation execution backend",
    )
    parser.add_argument(
        "--scorer",
        default="existing",
        choices=["existing", "torchumm", "command", "fixture", "none"],
        help="evaluation scorer backend; torchumm actively runs official/TorchUMM scorers when dependencies are present",
    )
    parser.add_argument("--run-id", help="reuse or set run id")
    parser.add_argument("--limit", type=int, default=None, help="limit samples for smoke runs")
    parser.add_argument(
        "--backend-option",
        action="append",
        help="backend option as key=value; repeatable",
    )
    parser.add_argument(
        "--scorer-option",
        action="append",
        help="scorer option as key=value; repeatable, e.g. torchumm_root=/path/to/TorchUMM",
    )
    parser.add_argument(
        "--benchmark-option",
        action="append",
        help="benchmark option as key=value; repeatable, e.g. local_json=benchmarks/generation/ueval/test.json",
    )
    parser.add_argument(
        "--scorer-command",
        help="external scorer command template; implies --scorer command unless --scorer is set explicitly",
    )
    args = parser.parse_args()

    if not args.benchmark and not args.all:
        parser.error("specify --benchmark or --all")

    phase = "score" if args.phase == "evaluate" else args.phase
    backend_options = _parse_backend_options(args.backend_option)
    scorer_options = _parse_key_value_options(args.scorer_option, "--scorer-option")
    benchmark_options = _parse_key_value_options(args.benchmark_option, "--benchmark-option")
    names = sorted(BENCHMARKS) if args.all else [args.benchmark]

    if args.all:
        reports = run_generation_suite(
            benchmark_names=names,
            model=args.model,
            framework_root=ROOT,
            phase=phase,
            backend_name=args.backend,
            scorer_name=args.scorer,
            run_id=args.run_id,
            limit=args.limit,
            backend_options=backend_options,
            scorer_options=scorer_options,
            benchmark_options=benchmark_options,
            scorer_command=args.scorer_command,
        )
        for report in reports:
            value = report.main_metric_value
            value_text = "pending" if value is None else f"{value:.4f}"
            print(f"[{report.benchmark}] {report.main_metric_name}: {value_text} -> {report.summary_path}")
        return

    report = run_generation_benchmark(
        benchmark_name=args.benchmark,
        model=args.model,
        framework_root=ROOT,
        phase=phase,
        backend_name=args.backend,
        scorer_name=args.scorer,
        run_id=args.run_id,
        limit=args.limit,
        backend_options=backend_options,
        scorer_options=scorer_options,
        benchmark_options=benchmark_options,
        scorer_command=args.scorer_command,
    )
    value = report.main_metric_value
    value_text = "pending" if value is None else f"{value:.4f}"
    print(f"[{report.benchmark}] {report.main_metric_name}: {value_text} -> {report.summary_path}")


if __name__ == "__main__":
    main()
