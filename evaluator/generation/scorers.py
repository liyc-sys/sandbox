from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .utils import ensure_dir, format_template, read_jsonl, run_command, write_json


@dataclass
class ScorerResult:
    name: str
    status: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    logs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "artifacts": self.artifacts,
            "error": self.error,
            "logs": self.logs,
        }


class GenerationScorer(Protocol):
    name: str

    def score(
        self,
        benchmark: str,
        run_dir: Path,
        framework_root: Path,
        model: str,
        run_id: str,
        benchmark_cfg: dict[str, Any],
        options: dict[str, Any],
    ) -> ScorerResult:
        ...


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _path_from_option(
    options: dict[str, Any],
    key: str,
    default: str | None,
    framework_root: Path,
) -> Path | None:
    value = options.get(key)
    if value is None:
        value = default
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = framework_root / path
    return path


def _resolved_option_string(options: dict[str, Any], key: str, framework_root: Path) -> str | None:
    value = options.get(key)
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = framework_root / path
    return str(path)


def _run_logged(
    command: str | list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    proc = run_command(command, cwd=cwd, env=env)
    return {
        "command": command if isinstance(command, str) else [str(x) for x in command],
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "stdout": proc.stdout[-12000:],
        "stderr": proc.stderr[-12000:],
    }


def _copy_if_exists(src: Path, dst: Path) -> Path | None:
    if not src.exists():
        return None
    ensure_dir(dst.parent)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return dst


def _write_ueval_dataset_from_samples(run_dir: Path) -> Path | None:
    samples_path = run_dir / "samples.jsonl"
    if not samples_path.exists():
        return None
    rows: list[dict[str, Any]] = []
    for sample in read_jsonl(samples_path):
        metadata = sample.get("metadata")
        row = dict(metadata) if isinstance(metadata, dict) else {}
        row.setdefault("id", sample.get("uid"))
        row.setdefault("prompt", sample.get("prompt", ""))
        row.setdefault("question", sample.get("prompt", ""))
        if sample.get("task") and not row.get("task") and not row.get("task_type"):
            row["task"] = sample["task"]
        rows.append(row)
    if not rows:
        return None
    dataset_dir = ensure_dir(run_dir / "ueval_dataset")
    dataset_path = dataset_dir / "test.json"
    write_json(dataset_path, rows)
    return dataset_path


class NoopScorer:
    name = "none"

    def score(
        self,
        benchmark: str,
        run_dir: Path,
        framework_root: Path,
        model: str,
        run_id: str,
        benchmark_cfg: dict[str, Any],
        options: dict[str, Any],
    ) -> ScorerResult:
        return ScorerResult(name=self.name, status="skipped", error="scorer disabled")


class ExistingScorer:
    name = "existing"

    def score(
        self,
        benchmark: str,
        run_dir: Path,
        framework_root: Path,
        model: str,
        run_id: str,
        benchmark_cfg: dict[str, Any],
        options: dict[str, Any],
    ) -> ScorerResult:
        return ScorerResult(name=self.name, status="ready")


class CommandScorer:
    name = "command"

    def score(
        self,
        benchmark: str,
        run_dir: Path,
        framework_root: Path,
        model: str,
        run_id: str,
        benchmark_cfg: dict[str, Any],
        options: dict[str, Any],
    ) -> ScorerResult:
        template = options.get("command") or benchmark_cfg.get("scorer_command")
        if not template:
            return ScorerResult(
                name=self.name,
                status="error",
                error="command scorer requires scorer option command=...",
            )
        context = {
            "framework_root": str(framework_root),
            "run_dir": str(run_dir),
            "benchmark": benchmark,
            "model": model,
            "run_id": run_id,
        }
        command = format_template(str(template), context)
        env = os.environ.copy()
        env.update({k: str(v) for k, v in context.items()})
        extra_env = options.get("env")
        if isinstance(extra_env, dict):
            env.update({str(k): str(v) for k, v in extra_env.items()})
        log = _run_logged(command, cwd=framework_root, env=env)
        write_json(run_dir / "scorer_command.log.json", log)
        return ScorerResult(
            name=self.name,
            status="completed" if log["returncode"] == 0 else "error",
            artifacts={"log": _relative(run_dir / "scorer_command.log.json", framework_root)},
            error=None if log["returncode"] == 0 else "command scorer failed",
            logs=log,
        )


class FixtureScorer:
    """Deterministic contract scorer for local smoke tests.

    It does not claim official benchmark validity. It writes the same artifact
    shapes that the real scorer adapters consume, so runner/report contracts can
    be tested without downloading detector/VLM judge models.
    """

    name = "fixture"

    def score(
        self,
        benchmark: str,
        run_dir: Path,
        framework_root: Path,
        model: str,
        run_id: str,
        benchmark_cfg: dict[str, Any],
        options: dict[str, Any],
    ) -> ScorerResult:
        score = float(options.get("score", 1.0))
        artifacts: dict[str, Any] = {}
        if benchmark == "dpg_bench":
            path = run_dir / "final_score.json"
            write_json(path, {"benchmark": benchmark, "final_score": score, "fixture": True})
            artifacts["final_score"] = _relative(path, framework_root)
        elif benchmark == "geneval":
            results_path = run_dir / "results.jsonl"
            summary_path = run_dir / "score_summary.json"
            rows = []
            samples_path = run_dir / "samples.jsonl"
            if samples_path.exists():
                import json

                for line in samples_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        sample = json.loads(line)
                        rows.append(
                            {
                                "tag": sample.get("task", "fixture"),
                                "metadata": sample.get("metadata", {}),
                                "correct": score > 0,
                            }
                        )
            from .utils import write_jsonl

            write_jsonl(results_path, rows or [{"tag": "fixture", "metadata": {}, "correct": score > 0}])
            write_json(summary_path, {"scores": {"overall": score}, "fixture": True})
            artifacts.update(
                {
                    "results": _relative(results_path, framework_root),
                    "score_summary": _relative(summary_path, framework_root),
                }
            )
        elif benchmark == "wise":
            results_dir = ensure_dir(run_dir / "Results")
            from .utils import write_jsonl

            for name in (
                "cultural_common_sense",
                "spatio-temporal_reasoning",
                "natural_science",
            ):
                write_jsonl(
                    results_dir / f"{name}_scores.jsonl",
                    [
                        {
                            "prompt_id": 1,
                            "consistency": score * 2,
                            "realism": score * 2,
                            "aesthetic_quality": score * 2,
                            "fixture": True,
                        }
                    ],
                )
            artifacts["results_dir"] = _relative(results_dir, framework_root)
        elif benchmark == "ueval":
            path = run_dir / "eval_results.json"
            write_json(path, {"summary": {"overall_avg_rate": score}, "results": [], "fixture": True})
            artifacts["eval_results"] = _relative(path, framework_root)
        else:
            return ScorerResult(name=self.name, status="error", error=f"unsupported benchmark: {benchmark}")
        return ScorerResult(name=self.name, status="completed", artifacts=artifacts)


class TorchUMMScorer:
    name = "torchumm"

    def score(
        self,
        benchmark: str,
        run_dir: Path,
        framework_root: Path,
        model: str,
        run_id: str,
        benchmark_cfg: dict[str, Any],
        options: dict[str, Any],
    ) -> ScorerResult:
        torchumm_root = _path_from_option(
            options,
            "torchumm_root",
            benchmark_cfg.get("torchumm_root") or "/private/tmp/TorchUMM",
            framework_root,
        )
        if torchumm_root is None or not torchumm_root.is_dir():
            return ScorerResult(
                name=self.name,
                status="error",
                error="torchumm scorer requires torchumm_root pointing to a TorchUMM checkout",
            )
        if benchmark == "dpg_bench":
            return self._score_dpg(run_dir, framework_root, torchumm_root, options)
        if benchmark == "geneval":
            return self._score_geneval(run_dir, framework_root, torchumm_root, options)
        if benchmark == "wise":
            return self._score_wise(run_dir, framework_root, torchumm_root, options)
        if benchmark == "ueval":
            return self._score_ueval(run_dir, framework_root, torchumm_root, options)
        return ScorerResult(name=self.name, status="error", error=f"unsupported benchmark: {benchmark}")

    def _score_dpg(
        self,
        run_dir: Path,
        framework_root: Path,
        torchumm_root: Path,
        options: dict[str, Any],
    ) -> ScorerResult:
        script = torchumm_root / "eval" / "generation" / "dpg_bench" / "compute_dpg_bench.py"
        shell_script = torchumm_root / "eval" / "generation" / "dpg_bench" / "dist_eval.sh"
        image_dir = run_dir / "images"
        if not image_dir.is_dir():
            return ScorerResult(name=self.name, status="error", error=f"DPG images dir missing: {image_dir}")
        if not script.exists() and not shell_script.exists():
            return ScorerResult(name=self.name, status="error", error=f"DPG scorer not found under {torchumm_root}")

        resolution = str(options.get("resolution", 512))
        eval_python = options.get("eval_python")
        logs: dict[str, Any]
        if eval_python and script.exists():
            command: list[str] = [
                str(Path(str(eval_python)).expanduser()),
                str(script),
                "--image-root-path",
                str(image_dir),
                "--resolution",
                resolution,
                "--pic-num",
                str(options.get("pic_num", 4)),
                "--vqa-model",
                str(options.get("vqa_model", "mplug")),
            ]
            logs = _run_logged(command, cwd=torchumm_root / "eval" / "generation")
        else:
            if not shell_script.exists():
                return ScorerResult(name=self.name, status="error", error=f"DPG dist_eval.sh not found: {shell_script}")
            logs = _run_logged(["bash", str(shell_script), str(image_dir), resolution], cwd=torchumm_root / "eval" / "generation")

        write_json(run_dir / "torchumm_scorer.log.json", logs)
        if logs["returncode"] != 0:
            return ScorerResult(
                name=self.name,
                status="error",
                error="DPG TorchUMM scorer failed",
                artifacts={"log": _relative(run_dir / "torchumm_scorer.log.json", framework_root)},
                logs=logs,
            )

        result_files = sorted(image_dir.glob("dpg-bench_*_results.txt"), key=lambda p: p.stat().st_mtime)
        score = None
        result_file = result_files[-1] if result_files else None
        if result_file:
            for line in result_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("DPG-Bench score:"):
                    try:
                        score = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        score = None
                    break
        final_score = run_dir / "final_score.json"
        write_json(
            final_score,
            {
                "result_file": _relative(result_file, framework_root) if result_file else "",
                "final_score": score,
            },
        )
        return ScorerResult(
            name=self.name,
            status="completed" if score is not None else "error",
            artifacts={
                "final_score": _relative(final_score, framework_root),
                "result_file": _relative(result_file, framework_root) if result_file else "",
                "log": _relative(run_dir / "torchumm_scorer.log.json", framework_root),
            },
            error=None if score is not None else "DPG scorer ran but no DPG-Bench score was found",
            logs=logs,
        )

    def _score_geneval(
        self,
        run_dir: Path,
        framework_root: Path,
        torchumm_root: Path,
        options: dict[str, Any],
    ) -> ScorerResult:
        eval_dir = torchumm_root / "eval" / "generation" / "geneval" / "evaluation"
        evaluate_script = eval_dir / "evaluate_images.py"
        summary_script = eval_dir / "summary_scores.py"
        images_dir = run_dir / "images"
        if not images_dir.is_dir():
            return ScorerResult(name=self.name, status="error", error=f"GenEval images dir missing: {images_dir}")
        if not evaluate_script.exists() or not summary_script.exists():
            return ScorerResult(name=self.name, status="error", error=f"GenEval scorer scripts missing under {eval_dir}")

        eval_python = str(Path(str(options.get("eval_python", "python3"))).expanduser())
        model_path = str(options.get("model_path", "./"))
        results_file = run_dir / "results.jsonl"
        cmd: list[str] = [
            eval_python,
            str(evaluate_script),
            str(images_dir),
            "--outfile",
            str(results_file),
            "--model-path",
            model_path,
        ]
        model_config = options.get("model_config")
        if model_config:
            cmd.extend(["--model-config", str(model_config)])
        logs = {"evaluate_images": _run_logged(cmd, cwd=eval_dir)}
        if logs["evaluate_images"]["returncode"] != 0:
            write_json(run_dir / "torchumm_scorer.log.json", logs)
            return ScorerResult(
                name=self.name,
                status="error",
                error="GenEval evaluate_images.py failed",
                artifacts={"log": _relative(run_dir / "torchumm_scorer.log.json", framework_root)},
                logs=logs,
            )
        summary_cmd = [eval_python, str(summary_script), str(results_file)]
        logs["summary_scores"] = _run_logged(summary_cmd, cwd=eval_dir)
        score_summary = run_dir / "score_summary.json"
        summary: dict[str, Any] = {
            "benchmark": "geneval",
            "images_dir": str(images_dir),
            "results_file": str(results_file),
            "scores": {},
        }
        for line in str(logs["summary_scores"].get("stdout", "")).splitlines():
            stripped = line.strip()
            if "Overall score" in stripped and ":" in stripped:
                try:
                    summary["scores"]["overall"] = float(stripped.split(":")[-1].strip())
                except ValueError:
                    pass
        write_json(score_summary, summary)
        write_json(run_dir / "torchumm_scorer.log.json", logs)
        ok = logs["summary_scores"]["returncode"] == 0 and "overall" in summary["scores"]
        return ScorerResult(
            name=self.name,
            status="completed" if ok else "error",
            artifacts={
                "results_file": _relative(results_file, framework_root),
                "score_summary": _relative(score_summary, framework_root),
                "log": _relative(run_dir / "torchumm_scorer.log.json", framework_root),
            },
            error=None if ok else "GenEval scorer ran but no overall score was parsed",
            logs=logs,
        )

    def _score_wise(
        self,
        run_dir: Path,
        framework_root: Path,
        torchumm_root: Path,
        options: dict[str, Any],
    ) -> ScorerResult:
        script = torchumm_root / "eval" / "generation" / "wise" / "vlm_eval.py"
        if not script.exists():
            return ScorerResult(name=self.name, status="error", error=f"WISE vlm_eval.py missing: {script}")
        image_dir = run_dir / "images"
        if not image_dir.is_dir():
            return ScorerResult(name=self.name, status="error", error=f"WISE images dir missing: {image_dir}")
        data_root = _path_from_option(options, "data_root", None, framework_root)
        if data_root is None:
            data_root = framework_root / "benchmarks" / "generation" / "wise" / "data"
        if not data_root.is_dir():
            return ScorerResult(name=self.name, status="error", error=f"WISE data_root missing: {data_root}")
        results_dir = ensure_dir(run_dir / "Results")
        eval_python = str(Path(str(options.get("eval_python", "python3"))).expanduser())
        cmd = [
            eval_python,
            str(script),
            "--data_root",
            str(data_root),
            "--image_dir",
            str(image_dir),
            "--output_dir",
            str(results_dir),
            "--model_name",
            str(options.get("model_name", "Qwen/Qwen2.5-VL-72B-Instruct")),
            "--max_new_tokens",
            str(options.get("max_new_tokens", 512)),
            "--max_retries",
            str(options.get("max_retries", 2)),
        ]
        attn_impl = options.get("attn_implementation")
        if attn_impl:
            cmd.extend(["--attn_implementation", str(attn_impl)])
        env = os.environ.copy()
        src_dir = str(torchumm_root / "src")
        env["PYTHONPATH"] = src_dir if not env.get("PYTHONPATH") else f"{src_dir}:{env['PYTHONPATH']}"
        logs = {"vlm_eval": _run_logged(cmd, cwd=torchumm_root, env=env)}

        calculate = torchumm_root / "eval" / "generation" / "wise" / "Calculate.py"
        if calculate.exists() and options.get("run_calculate", True):
            score_files = [
                results_dir / "cultural_common_sense_scores.jsonl",
                results_dir / "spatio-temporal_reasoning_scores.jsonl",
                results_dir / "natural_science_scores.jsonl",
            ]
            existing = [str(p) for p in score_files if p.exists()]
            if existing:
                logs["calculate"] = _run_logged([eval_python, str(calculate), *existing, "--category", "all"], cwd=torchumm_root)
        write_json(run_dir / "torchumm_scorer.log.json", logs)
        score_files_found = list(results_dir.glob("*_scores.jsonl"))
        ok = logs["vlm_eval"]["returncode"] == 0 and bool(score_files_found)
        return ScorerResult(
            name=self.name,
            status="completed" if ok else "error",
            artifacts={
                "results_dir": _relative(results_dir, framework_root),
                "score_files": [_relative(p, framework_root) for p in score_files_found],
                "log": _relative(run_dir / "torchumm_scorer.log.json", framework_root),
            },
            error=None if ok else "WISE VLM scorer failed or produced no *_scores.jsonl files",
            logs=logs,
        )

    def _score_ueval(
        self,
        run_dir: Path,
        framework_root: Path,
        torchumm_root: Path,
        options: dict[str, Any],
    ) -> ScorerResult:
        script = torchumm_root / "eval" / "generation" / "ueval" / "run_scoring.py"
        if not script.exists():
            return ScorerResult(name=self.name, status="error", error=f"UEval run_scoring.py missing: {script}")
        model_outputs = run_dir / "model_outputs.json"
        if not model_outputs.exists():
            return ScorerResult(name=self.name, status="error", error=f"UEval model_outputs.json missing: {model_outputs}")

        cfg_path = run_dir / "ueval_torchumm_score_config.json"
        cfg: dict[str, Any] = {
            "ueval": {
                "out_dir": str(run_dir),
                "hf_dataset": str(options.get("hf_dataset", "zlab-princeton/UEval")),
                "domains": options.get("domains", "all"),
                "scoring": {
                    "text_model": str(options.get("text_model", "Qwen/Qwen3-32B")),
                    "vl_model": str(options.get("vl_model", "Qwen/Qwen2.5-VL-72B-Instruct")),
                    "text_field": str(options.get("text_field", "text_answer")),
                    "image_field": str(options.get("image_field", "image_answer")),
                },
            }
        }
        if options.get("local_cache"):
            cfg["ueval"]["local_cache"] = _resolved_option_string(options, "local_cache", framework_root)
        for local_key in ("local_json", "local_data", "local_source", "dataset_cache"):
            local_value = _resolved_option_string(options, local_key, framework_root)
            if local_value:
                cfg["ueval"]["local_data"] = local_value
                break
        if not cfg["ueval"].get("local_cache") and not cfg["ueval"].get("local_data"):
            local_dataset = _write_ueval_dataset_from_samples(run_dir)
            if local_dataset:
                cfg["ueval"]["local_data"] = str(local_dataset)
        if options.get("limit") is not None:
            cfg["ueval"]["scoring"]["limit"] = int(options["limit"])
        write_json(cfg_path, cfg)
        eval_python = str(Path(str(options.get("eval_python", "python3"))).expanduser())
        env = os.environ.copy()
        src_dir = str(torchumm_root / "src")
        env["PYTHONPATH"] = src_dir if not env.get("PYTHONPATH") else f"{src_dir}:{env['PYTHONPATH']}"
        logs = _run_logged([eval_python, str(script), "--config", str(cfg_path)], cwd=torchumm_root, env=env)
        write_json(run_dir / "torchumm_scorer.log.json", logs)
        eval_results = run_dir / "eval_results.json"
        ok = logs["returncode"] == 0 and eval_results.exists()
        return ScorerResult(
            name=self.name,
            status="completed" if ok else "error",
            artifacts={
                "config": _relative(cfg_path, framework_root),
                "eval_results": _relative(eval_results, framework_root) if eval_results.exists() else "",
                "log": _relative(run_dir / "torchumm_scorer.log.json", framework_root),
            },
            error=None if ok else "UEval TorchUMM scorer failed or produced no eval_results.json",
            logs=logs,
        )


SCORERS: dict[str, type[GenerationScorer]] = {
    "none": NoopScorer,
    "existing": ExistingScorer,
    "command": CommandScorer,
    "fixture": FixtureScorer,
    "torchumm": TorchUMMScorer,
}


def get_scorer(name: str | None) -> GenerationScorer:
    key = (name or "existing").strip().lower()
    if key not in SCORERS:
        raise KeyError(f"unknown generation scorer: {name}")
    return SCORERS[key]()
