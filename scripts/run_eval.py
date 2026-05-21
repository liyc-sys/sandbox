"""统一视觉评测：load 归一化 jsonl → inference → judge → 报告。

用法示例:
  python3 scripts/run_eval.py --benchmark realworldqa --model doubao-seed-1-6-vision-250815
  python3 scripts/run_eval.py --all --model gemini-3-pro-preview --limit 30
  python3 scripts/run_eval.py --benchmark realworldqa --model doubao-seed-1-6-vision-250815 --phase inference
  python3 scripts/run_eval.py --benchmark realworldqa --model doubao-seed-1-6-vision-250815 --phase judge
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluator.inference.runner import run_inference, load_samples
from evaluator.judges.benchmark_judges import JUDGE_REGISTRY
from evaluator.judges.omnidocbench_judge import OmniDocBenchJudge
from evaluator.judges.llm_fallback import DEFAULT_FALLBACK_MODEL
from evaluator.reports.aggregate import aggregate, write_report
from evaluator.generation.utils import write_jsonl


# 各 benchmark 默认归一化 jsonl 路径
DEFAULT_DATA = {
    "mmmu": "data/normalized/mmmu/validation.jsonl",
    "realworldqa": "data/normalized/realworldqa/test.jsonl",
    "vlmsareblind": "data/normalized/vlmsareblind/valid.jsonl",
    "hallusionbench": "data/normalized/hallusionbench/test.jsonl",
    "omnidocbench": "data/normalized/omnidocbench/train.jsonl",
    "chartqapro": "data/normalized/chartqapro/test.jsonl",
    "mathvista": "data/normalized/mathvista/testmini.jsonl",
    "phyx_openended": "data/normalized/phyx_openended/test_mini.jsonl",
    "babyvision": "data/normalized/babyvision/train.jsonl",
    "countbench": "data/normalized/countbench/test.jsonl",
    "refcoco_avg": "data/normalized/refcoco_avg/all8.jsonl",
    "visulogic": "data/normalized/visulogic/test.jsonl",
}


def get_judge(benchmark: str, framework_root: Path, fallback_model: str, enable_fallback: bool):
    if benchmark == "omnidocbench":
        return OmniDocBenchJudge(framework_root=framework_root)
    cls = JUDGE_REGISTRY[benchmark]
    return cls(
        framework_root=framework_root,
        fallback_model=fallback_model,
        enable_fallback=enable_fallback,
    )


def _write_samples(path: Path, samples: list[dict]) -> None:
    write_jsonl(path, samples)


def _read_predictions(path: Path) -> list[dict]:
    preds: list[dict] = []
    if not path.exists():
        return preds
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))
    return preds


def run_one_benchmark(
    benchmark: str,
    model: str,
    framework_root: Path,
    phase: str,
    limit: int | None,
    fallback_model: str,
    enable_fallback: bool,
    workers: int,
    judge_workers: int,
) -> dict:
    print(f"\n=== [{benchmark}] model={model} phase={phase} limit={limit} ===")
    data_path = framework_root / DEFAULT_DATA[benchmark]
    if not data_path.exists():
        print(f"  data file missing: {data_path} — skip")
        return {"benchmark": benchmark, "error": f"data file missing: {data_path}"}
    out_dir = framework_root / "assets" / "output" / benchmark / model.replace("/", "_")
    sample_path = out_dir / "samples.jsonl"
    pred_path = out_dir / "predictions.jsonl"

    if phase == "judge" and sample_path.exists():
        samples = load_samples(sample_path)
    else:
        samples = load_samples(data_path, limit=limit)
    if not samples:
        print(f"  no samples in {data_path}")
        return {"benchmark": benchmark, "error": "no samples"}
    print(f"  loaded {len(samples)} samples from {data_path}")

    pred_by_uid: dict[str, str] = {}
    preds: list[object] = []
    n_err = 0
    inference_elapsed = 0.0
    if phase in ("inference", "full"):
        _write_samples(sample_path, samples)
        t0 = time.time()
        preds = run_inference(
            samples=samples,
            model=model,
            framework_root=framework_root,
            out_path=pred_path,
            workers=workers,
        )
        inference_elapsed = time.time() - t0
        n_err = sum(1 for p in preds if p.error)
        print(f"  inference done in {inference_elapsed:.1f}s ({n_err} errors)  -> {pred_path}")
        pred_by_uid = {p.uid: p.raw_response for p in preds}
    else:
        if not pred_path.exists():
            print(f"  predictions missing: {pred_path} — run phase=inference first")
            return {"benchmark": benchmark, "error": f"predictions missing: {pred_path}"}
        preds = _read_predictions(pred_path)
        pred_by_uid = {str(p.get("uid", "")): str(p.get("raw_response", "")) for p in preds}
        print(f"  loaded {len(pred_by_uid)} predictions from {pred_path}")

    if phase == "inference":
        judge_name = get_judge(benchmark, framework_root, fallback_model, enable_fallback).metric_name
        report = {
            "benchmark": benchmark,
            "main_metric_name": "Inference only",
            "main_metric_value": None,
            "n_samples": len(samples),
            "fallback_rate": 0.0,
            "fallback_overturn_rate": 0.0,
            "model": model,
            "fallback_model": fallback_model if enable_fallback and benchmark != "omnidocbench" else None,
            "phase": phase,
            "inference_elapsed_sec": inference_elapsed,
            "n_inference_errors": n_err,
            "predictions_path": str(pred_path),
            "judge_metric_name": judge_name,
        }
        report_path = out_dir / "report.json"
        write_report(report, report_path, per_sample=[{"uid": s["uid"], "raw_response": pred_by_uid.get(s["uid"], "")} for s in samples])
        print(f"  report -> {report_path}")
        print(f"  >>> inference-only: wrote {len(samples)} samples")
        return report

    # judge（也并发：fallback 可能调 LLM，IO 密集）
    judge = get_judge(benchmark, framework_root, fallback_model, enable_fallback)
    t0 = time.time()

    def _do(rec):
        return rec["uid"], judge.judge(rec, pred_by_uid.get(rec["uid"], ""))

    results_by_uid: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=judge_workers) as ex:
        futs = [ex.submit(_do, rec) for rec in samples]
        for fut in as_completed(futs):
            uid, res = fut.result()
            results_by_uid[uid] = res
    results = [results_by_uid[s["uid"]] for s in samples]
    print(f"  judge done in {time.time() - t0:.1f}s")

    # 3) 聚合
    report = aggregate(benchmark, samples, results, metric_name=judge.metric_name)
    report["model"] = model
    report["fallback_model"] = fallback_model if enable_fallback and benchmark != "omnidocbench" else None
    report["phase"] = phase
    report["predictions_path"] = str(pred_path)
    if phase in ("inference", "full"):
        report["inference_elapsed_sec"] = inference_elapsed
        report["n_inference_errors"] = n_err
    report_path = out_dir / "report.json"
    per_sample = [
        {
            "uid": s["uid"],
            "raw_response": pred_by_uid.get(s["uid"], ""),
            **r.to_dict(),
        }
        for s, r in zip(samples, results)
    ]
    write_report(report, report_path, per_sample=per_sample)
    print(f"  report -> {report_path}")
    print(f"  >>> {report['main_metric_name']} = {report['main_metric_value']:.4f}  "
          f"(fallback={report.get('fallback_rate', 0):.2%}, "
          f"overturn={report.get('fallback_overturn_rate', 0):.2%})")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", help="single benchmark to run")
    p.add_argument("--all", action="store_true", help="run all configured benchmarks")
    p.add_argument("--model", required=True, help="inference model id (oneapi)")
    p.add_argument(
        "--phase",
        choices=["inference", "judge", "full"],
        default="full",
        help="inference only, judge only, or full pipeline",
    )
    p.add_argument("--judge-model", default=DEFAULT_FALLBACK_MODEL,
                   help=f"LLM fallback judge model (default: {DEFAULT_FALLBACK_MODEL})")
    p.add_argument("--limit", type=int, default=30,
                   help="default 30 (smoke run); use --full for all samples")
    p.add_argument("--full", action="store_true", help="ignore --limit, run full splits")
    p.add_argument("--workers", type=int, default=8, help="inference concurrency (default 8)")
    p.add_argument("--judge-workers", type=int, default=8, help="judge concurrency (default 8)")
    p.add_argument("--no-fallback", action="store_true", help="disable LLM fallback")
    args = p.parse_args()

    if not args.benchmark and not args.all:
        p.error("specify --benchmark or --all")

    benches = list(DEFAULT_DATA.keys()) if args.all else [args.benchmark]
    framework_root = ROOT
    limit = None if args.full else args.limit

    summary = []
    for b in benches:
        if b not in DEFAULT_DATA:
            print(f"unknown benchmark: {b}", file=sys.stderr)
            continue
        rep = run_one_benchmark(
            benchmark=b,
            model=args.model,
            framework_root=framework_root,
            phase=args.phase,
            limit=limit,
            fallback_model=args.judge_model,
            enable_fallback=not args.no_fallback,
            workers=args.workers,
            judge_workers=args.judge_workers,
        )
        summary.append(rep)

    summary_path = framework_root / "assets" / "output" / f"summary_{args.model.replace('/', '_')}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n=== summary -> {summary_path} ===")


if __name__ == "__main__":
    main()
