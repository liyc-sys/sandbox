from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .backends import get_backend
from .benchmarks import get_generation_benchmark
from .scorers import get_scorer
from .types import BenchmarkReport, GenerationResult, GenerationSample
from .utils import ensure_dir, make_run_id, read_jsonl, sanitize_component, write_json, write_jsonl, run_command


def _load_generation_configs(framework_root: Path) -> dict[str, Any]:
    config_path = framework_root / "configs" / "generation_benchmarks.json"
    if not config_path.exists():
        return {"benchmarks": {}}
    import json

    return json.loads(config_path.read_text(encoding="utf-8"))


def _run_dir(framework_root: Path, benchmark: str, model: str, run_id: str) -> Path:
    return (
        framework_root
        / "assets"
        / "output"
        / "generation"
        / sanitize_component(benchmark)
        / sanitize_component(model.replace("/", "_"))
        / sanitize_component(run_id)
    )


def _sample_output_dir(run_dir: Path, sample: GenerationSample) -> Path:
    return ensure_dir(run_dir / "raw_outputs" / sanitize_component(sample.uid))


def _write_samples(run_dir: Path, samples: list[GenerationSample]) -> Path:
    path = run_dir / "samples.jsonl"
    write_jsonl(path, [s.to_dict() for s in samples])
    return path


def _read_samples(run_dir: Path) -> list[GenerationSample]:
    path = run_dir / "samples.jsonl"
    if not path.exists():
        return []
    samples: list[GenerationSample] = []
    for row in read_jsonl(path):
        samples.append(
            GenerationSample(
                uid=str(row.get("uid", "")),
                benchmark=str(row.get("benchmark", "")),
                prompt=str(row.get("prompt", "")),
                stage=str(row.get("stage", "generate")),
                expected_outputs=int(row.get("expected_outputs") or 1),
                task=str(row.get("task", "")),
                media=list(row.get("media") or []),
                metadata=dict(row.get("metadata") or {}),
                answer=dict(row.get("answer") or {}),
                extra=dict(row.get("extra") or {}),
            )
        )
    return samples


def _write_predictions(run_dir: Path, results: list[GenerationResult]) -> Path:
    path = run_dir / "predictions.jsonl"
    write_jsonl(path, [r.to_dict() for r in results])
    return path


def _write_request_manifests(
    run_dir: Path,
    samples: list[GenerationSample],
    framework_root: Path,
    backend_name: str,
    backend_options: dict[str, Any] | None,
) -> dict[str, Any]:
    requests_root = ensure_dir(run_dir / "requests")
    rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_dir = ensure_dir(requests_root / sanitize_component(sample.uid))
        request_path = sample_dir / f"{sanitize_component(sample.uid)}.request.json"
        payload = {
            "uid": sample.uid,
            "benchmark": sample.benchmark,
            "stage": sample.stage,
            "prompt": sample.prompt,
            "expected_outputs": sample.expected_outputs,
            "task": sample.task,
            "media": sample.media,
            "metadata": sample.metadata,
            "output_dir": str((run_dir / "raw_outputs" / sanitize_component(sample.uid)).relative_to(framework_root)),
            "backend": backend_name,
            "backend_options": backend_options or {},
        }
        write_json(request_path, payload)
        rows.append(
            {
                "uid": sample.uid,
                "request_path": str(request_path.relative_to(framework_root)),
                "expected_outputs": sample.expected_outputs,
                "output_dir": payload["output_dir"],
            }
        )
    requests_index = run_dir / "requests.jsonl"
    write_jsonl(requests_index, rows)
    return {
        "requests_root": str(requests_root.relative_to(framework_root)),
        "requests_index": str(requests_index.relative_to(framework_root)),
        "n_requests": len(rows),
    }


def _run_external_scorer(
    command: str | None,
    framework_root: Path,
    run_dir: Path,
    benchmark: str,
    model: str,
    run_id: str,
) -> dict[str, Any]:
    if not command:
        return {}
    context = {
        "framework_root": str(framework_root),
        "run_dir": str(run_dir),
        "benchmark": benchmark,
        "model": model,
        "run_id": run_id,
    }
    from .utils import format_template

    rendered = format_template(command, context)
    env = os.environ.copy()
    env.update({k: str(v) for k, v in context.items()})
    proc = run_command(rendered, cwd=framework_root, env=env)
    log = {
        "command": rendered,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-8000:],
        "stderr": proc.stderr[-8000:],
    }
    write_json(run_dir / "external_scorer.log.json", log)
    return log


def run_generation_benchmark(
    benchmark_name: str,
    model: str,
    framework_root: Path,
    phase: str = "prepare",
    backend_name: str = "manifest",
    scorer_name: str = "existing",
    run_id: str | None = None,
    limit: int | None = None,
    backend_options: dict[str, Any] | None = None,
    scorer_options: dict[str, Any] | None = None,
    benchmark_options: dict[str, Any] | None = None,
    scorer_command: str | None = None,
) -> BenchmarkReport:
    cfg_root = _load_generation_configs(framework_root)
    benchmark_cfg = (cfg_root.get("benchmarks", {}) or {}).get(benchmark_name, {})
    benchmark_cfg = dict(benchmark_cfg)
    if benchmark_options:
        benchmark_cfg.update(benchmark_options)
    benchmark = get_generation_benchmark(benchmark_name, framework_root, benchmark_cfg)
    backend = get_backend(backend_name)
    run_id = run_id or make_run_id(model)
    run_dir = ensure_dir(_run_dir(framework_root, benchmark_name, model, run_id))

    existing_samples = _read_samples(run_dir) if phase == "score" else []
    try:
        samples = existing_samples or benchmark.load_samples(limit=limit)
    except Exception as exc:  # noqa: BLE001
        report = BenchmarkReport(
            benchmark=benchmark_name,
            display_name=getattr(benchmark, "display_name", benchmark_name),
            model=model,
            run_id=run_id,
            phase=phase,
            backend=backend.name,
            n_samples=0,
            main_metric_name=getattr(benchmark, "main_metric_name", ""),
            main_metric_value=None,
            summary_path=str((run_dir / "report.json").relative_to(framework_root)),
            meta={"error": f"failed to load generation samples: {exc}"},
        )
        write_json(run_dir / "manifest.json", {
            "benchmark": benchmark_name,
            "display_name": getattr(benchmark, "display_name", benchmark_name),
            "model": model,
            "run_id": run_id,
            "phase": phase,
            "backend": backend.name,
            "scorer": scorer_name,
            "n_samples": 0,
            "run_dir": str(run_dir.relative_to(framework_root)),
            "error": report.meta["error"],
        })
        write_json(run_dir / "report.json", report.to_dict())
        return report
    samples_path = run_dir / "samples.jsonl" if existing_samples else _write_samples(run_dir, samples)

    manifest = {
        "benchmark": benchmark_name,
        "display_name": benchmark.display_name,
        "model": model,
        "run_id": run_id,
        "phase": phase,
        "backend": backend.name,
        "scorer": scorer_name,
        "n_samples": len(samples),
        "samples_path": str(samples_path.relative_to(framework_root)),
        "run_dir": str(run_dir.relative_to(framework_root)),
    }
    write_json(run_dir / "manifest.json", manifest)

    if phase == "prepare":
        options = dict(benchmark_cfg.get("backend_options", {}) or {})
        if backend_options:
            options.update(backend_options)
        prepare_artifacts = _write_request_manifests(
            run_dir=run_dir,
            samples=samples,
            framework_root=framework_root,
            backend_name=backend.name,
            backend_options=options,
        )
        manifest["prepare_artifacts"] = prepare_artifacts
        write_json(run_dir / "manifest.json", manifest)

    results: list[GenerationResult] = []
    if phase in ("generate", "full"):
        options = dict(benchmark_cfg.get("backend_options", {}) or {})
        if backend_options:
            options.update(backend_options)
        for sample in samples:
            result = backend.generate(
                sample=sample,
                output_dir=_sample_output_dir(run_dir, sample),
                framework_root=framework_root,
                options=options,
            )
            results.append(result)
        predictions_path = _write_predictions(run_dir, results)
        materialized = benchmark.materialize_outputs(
            run_dir=run_dir,
            samples=samples,
            results=[r.to_dict() for r in results],
        )
        write_json(
            run_dir / "generation_summary.json",
            {
                "n_samples": len(samples),
                "n_results": len(results),
                "status_counts": {
                    status: sum(1 for r in results if r.status == status)
                    for status in sorted({r.status for r in results})
                },
                "predictions_path": str(predictions_path.relative_to(framework_root)),
                "materialized": materialized,
            },
        )

    if phase in ("score", "full"):
        options = dict(benchmark_cfg.get("scorer_options", {}) or {})
        if scorer_options:
            options.update(scorer_options)
        if scorer_command:
            options["command"] = scorer_command
            if scorer_name == "existing":
                scorer_name = "command"
        for key in ("data_root", "local_json", "local_cache", "local_data", "local_source", "dataset_cache", "hf_dataset", "domains"):
            if key in benchmark_cfg and key not in options:
                options[key] = benchmark_cfg[key]
        scorer = get_scorer(scorer_name)
        scorer_result = scorer.score(
            benchmark=benchmark_name,
            run_dir=run_dir,
            framework_root=framework_root,
            model=model,
            run_id=run_id,
            benchmark_cfg=benchmark_cfg,
            options=options,
        )
        manifest["scorer_result"] = scorer_result.to_dict()
        write_json(run_dir / "manifest.json", manifest)

    report = benchmark.score(
        run_dir=run_dir,
        samples=samples,
        phase=phase,
        model=model,
        run_id=run_id,
        backend_name=backend.name,
    )
    if "scorer_result" in manifest:
        report.meta["scorer_result"] = manifest["scorer_result"]
    report.summary_path = str((run_dir / "report.json").relative_to(framework_root))
    write_json(run_dir / "report.json", report.to_dict())
    return report


def run_generation_suite(
    benchmark_names: list[str],
    model: str,
    framework_root: Path,
    phase: str = "prepare",
    backend_name: str = "manifest",
    scorer_name: str = "existing",
    run_id: str | None = None,
    limit: int | None = None,
    backend_options: dict[str, Any] | None = None,
    scorer_options: dict[str, Any] | None = None,
    benchmark_options: dict[str, Any] | None = None,
    scorer_command: str | None = None,
) -> list[BenchmarkReport]:
    suite_run_id = run_id or make_run_id(model)
    reports: list[BenchmarkReport] = []
    for name in benchmark_names:
        reports.append(
            run_generation_benchmark(
                benchmark_name=name,
                model=model,
                framework_root=framework_root,
                phase=phase,
                backend_name=backend_name,
                scorer_name=scorer_name,
                run_id=suite_run_id,
                limit=limit,
                backend_options=backend_options,
                scorer_options=scorer_options,
                benchmark_options=benchmark_options,
                scorer_command=scorer_command,
            )
        )
    summary_path = (
        framework_root
        / "assets"
        / "output"
        / "generation"
        / f"summary_{sanitize_component(model.replace('/', '_'))}_{sanitize_component(suite_run_id)}.json"
    )
    write_json(summary_path, [r.to_dict() for r in reports])
    return reports
