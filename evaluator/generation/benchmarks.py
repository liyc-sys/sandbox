from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from .types import BenchmarkReport, GenerationSample
from .utils import ensure_dir, glob_sorted, numeric_key, read_json, read_jsonl, write_json, write_jsonl


def _records_from_json_like(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, dict) or hasattr(item, "items")]
    if isinstance(raw, dict):
        for key in ("data", "items", "entries", "examples", "prompts", "test"):
            value = raw.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, dict) or hasattr(item, "items")]
        return [dict(raw)]
    return []


def _read_record_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(dict(json.loads(line)))
        return rows
    if suffix == ".json":
        return _records_from_json_like(json.loads(path.read_text(encoding="utf-8")))
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("reading parquet requires pyarrow") from exc
        table = pq.read_table(path)
        return [dict(row) for row in table.to_pylist()]
    return []


def _read_local_dataset_source(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        rows = _read_record_file(path)
        if rows:
            return rows
        raise FileNotFoundError(f"unsupported UEval data file: {path}")

    if not path.is_dir():
        raise FileNotFoundError(f"UEval local source not found: {path}")

    preferred_files = [
        path / "test.json",
        path / "test.jsonl",
        path / "test.parquet",
        path / "data" / "test.json",
        path / "data" / "test.jsonl",
        path / "data" / "test.parquet",
        path / "data.json",
        path / "data.jsonl",
        path / "data.parquet",
    ]
    for candidate in preferred_files:
        if candidate.is_file():
            rows = _read_record_file(candidate)
            if rows:
                return rows

    try:
        from datasets import load_from_disk

        loaded = load_from_disk(str(path))
        if isinstance(loaded, dict):
            if "test" in loaded:
                loaded = loaded["test"]
            elif loaded:
                loaded = next(iter(loaded.values()))
        return [dict(item) for item in loaded]
    except Exception:
        pass

    try:
        from datasets import load_dataset

        loaded = load_dataset(str(path), split="test")
        return [dict(item) for item in loaded]
    except Exception:
        pass

    rows: list[dict[str, Any]] = []
    for candidate in sorted(path.glob("*.jsonl")) + sorted(path.glob("*.json")) + sorted(path.glob("*.csv")) + sorted(path.glob("*.parquet")):
        rows.extend(_read_record_file(candidate))
    if rows:
        return rows
    raise FileNotFoundError(f"UEval local source directory is empty or unsupported: {path}")


_UEVAL_LOCAL_SOURCE_KEYS = (
    "local_json",
    "local_cache",
    "local_data",
    "local_source",
    "dataset_cache",
)


class GenerationBenchmark:
    name = ""
    display_name = ""
    main_metric_name = ""

    def __init__(self, framework_root: Path, cfg: dict[str, Any]):
        self.framework_root = framework_root
        self.cfg = cfg

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = self.framework_root / path
        return path

    def load_samples(self, limit: int | None = None) -> list[GenerationSample]:
        raise NotImplementedError

    def materialize_outputs(
        self,
        run_dir: Path,
        samples: list[GenerationSample],
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {}

    def score(
        self,
        run_dir: Path,
        samples: list[GenerationSample],
        phase: str,
        model: str,
        run_id: str,
        backend_name: str,
    ) -> BenchmarkReport:
        raise NotImplementedError

    def report_from_summary(
        self,
        run_dir: Path,
        model: str,
        run_id: str,
        phase: str,
        backend_name: str,
        summary: dict[str, Any],
        n_samples: int,
    ) -> BenchmarkReport:
        return BenchmarkReport(
            benchmark=self.name,
            display_name=self.display_name,
            model=model,
            run_id=run_id,
            phase=phase,
            backend=backend_name,
            n_samples=n_samples,
            main_metric_name=self.main_metric_name,
            main_metric_value=summary.get("main_metric_value"),
            summary_path=str((run_dir / "report.json").relative_to(self.framework_root)),
            artifacts=summary.get("artifacts", {}),
        )


class DPGBench(GenerationBenchmark):
    name = "dpg_bench"
    display_name = "DPG Bench"
    main_metric_name = "DPG-Bench score"

    def load_samples(self, limit: int | None = None) -> list[GenerationSample]:
        prompts_dir = self.resolve_path(self.cfg.get("prompts_dir", "benchmarks/generation/dpg_bench/prompts"))
        prompt_files = sorted(glob_sorted(prompts_dir, "*.txt"), key=numeric_key)
        if limit is not None:
            prompt_files = prompt_files[:limit]
        samples: list[GenerationSample] = []
        for path in prompt_files:
            prompt = path.read_text(encoding="utf-8").strip()
            if not prompt:
                continue
            samples.append(
                GenerationSample(
                    uid=path.stem,
                    benchmark=self.name,
                    prompt=prompt,
                    expected_outputs=int(self.cfg.get("images_per_prompt", 4)),
                    metadata={"prompt_file": str(path), "torchumm_layout": "2x2_grid"},
                )
            )
        return samples

    def materialize_outputs(
        self,
        run_dir: Path,
        samples: list[GenerationSample],
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        images_dir = ensure_dir(run_dir / "images")
        by_uid = {r.get("uid"): r for r in results}
        written: list[str] = []
        for sample in samples:
            result = by_uid.get(sample.uid, {})
            image_paths = result.get("image_paths") or []
            if not image_paths:
                continue
            abs_images = [
                self.resolve_path(p)
                for p in image_paths
                if self.resolve_path(p).is_file()
            ]
            target = images_dir / f"{sample.uid}.png"
            if len(abs_images) >= 4:
                with Image.open(abs_images[0]) as base:
                    w, h = base.size
                grid = Image.new("RGB", (w * 2, h * 2))
                for i, img_path in enumerate(abs_images[:4]):
                    with Image.open(img_path) as im:
                        grid.paste(im.convert("RGB"), ((i % 2) * w, (i // 2) * h))
                grid.save(target)
            else:
                with Image.open(abs_images[0]) as im:
                    im.convert("RGB").save(target)
            written.append(str(target.relative_to(self.framework_root)))
        return {"images_dir": str(images_dir.relative_to(self.framework_root)), "grids": written}

    def _read_score(self, run_dir: Path) -> tuple[float | None, dict[str, Any]]:
        final_json = run_dir / "final_score.json"
        if final_json.exists():
            data = read_json(final_json)
            value = data.get("final_score")
            if value is None:
                value = data.get("score")
            try:
                return float(value), {"score_file": str(final_json.relative_to(self.framework_root)), "raw": data}
            except (TypeError, ValueError):
                return None, {"score_file": str(final_json.relative_to(self.framework_root)), "raw": data}
        result_files = sorted((run_dir / "images").glob("dpg-bench_*_results.txt"))
        if not result_files:
            result_files = sorted(run_dir.glob("dpg-bench_*_results.txt"))
        for path in result_files[-1:]:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("DPG-Bench score:"):
                    try:
                        return float(line.split(":", 1)[1].strip()), {
                            "score_file": str(path.relative_to(self.framework_root))
                        }
                    except ValueError:
                        pass
        return None, {}

    def score(
        self,
        run_dir: Path,
        samples: list[GenerationSample],
        phase: str,
        model: str,
        run_id: str,
        backend_name: str,
    ) -> BenchmarkReport:
        value, artifacts = self._read_score(run_dir)
        report = BenchmarkReport(
            benchmark=self.name,
            display_name=self.display_name,
            model=model,
            run_id=run_id,
            phase=phase,
            backend=backend_name,
            n_samples=len(samples),
            main_metric_name=self.main_metric_name,
            main_metric_value=value,
            artifacts=artifacts,
            meta={"score_status": "found" if value is not None else "missing_external_scorer_output"},
        )
        write_json(run_dir / "report.json", report.to_dict())
        return report


class GenEvalBench(GenerationBenchmark):
    name = "geneval"
    display_name = "GenEval"
    main_metric_name = "Overall score"

    def load_samples(self, limit: int | None = None) -> list[GenerationSample]:
        metadata_path = self.resolve_path(
            self.cfg.get("metadata_path", "benchmarks/generation/geneval/evaluation_metadata.jsonl")
        )
        rows = read_jsonl(metadata_path)
        if limit is not None:
            rows = rows[:limit]
        samples: list[GenerationSample] = []
        for idx, row in enumerate(rows):
            samples.append(
                GenerationSample(
                    uid=f"{idx:05d}",
                    benchmark=self.name,
                    prompt=str(row.get("prompt", "")),
                    expected_outputs=int(self.cfg.get("images_per_prompt", 4)),
                    task=str(row.get("tag", "")),
                    metadata=row,
                )
            )
        return samples

    def materialize_outputs(
        self,
        run_dir: Path,
        samples: list[GenerationSample],
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        images_dir = ensure_dir(run_dir / "images")
        by_uid = {r.get("uid"): r for r in results}
        prompt_dirs: list[str] = []
        for sample in samples:
            prompt_dir = ensure_dir(images_dir / sample.uid)
            samples_dir = ensure_dir(prompt_dir / "samples")
            write_jsonl(prompt_dir / "metadata.jsonl", [sample.metadata])
            result = by_uid.get(sample.uid, {})
            image_paths = result.get("image_paths") or []
            for idx, image_path in enumerate(image_paths[: sample.expected_outputs]):
                src = self.resolve_path(image_path)
                if not src.is_file():
                    continue
                dst = samples_dir / f"{idx:05d}.png"
                with Image.open(src) as im:
                    im.convert("RGB").save(dst)
            prompt_dirs.append(str(prompt_dir.relative_to(self.framework_root)))
        return {"images_dir": str(images_dir.relative_to(self.framework_root)), "prompt_dirs": prompt_dirs}

    def _summarize_results(self, results_path: Path) -> dict[str, Any]:
        rows = read_jsonl(results_path)
        if not rows:
            return {"overall": None}
        n = len(rows)
        correct_images = sum(1 for r in rows if r.get("correct"))
        by_metadata: dict[str, list[bool]] = defaultdict(list)
        by_tag: dict[str, list[bool]] = defaultdict(list)
        for row in rows:
            key = json.dumps(row.get("metadata"), sort_keys=True, ensure_ascii=False)
            by_metadata[key].append(bool(row.get("correct")))
            by_tag[str(row.get("tag", ""))].append(bool(row.get("correct")))
        task_scores = {
            tag: sum(vals) / len(vals)
            for tag, vals in by_tag.items()
            if tag and vals
        }
        overall = sum(task_scores.values()) / len(task_scores) if task_scores else None
        return {
            "total_images": n,
            "total_prompts": len(by_metadata),
            "correct_image_rate": correct_images / n,
            "correct_prompt_rate": sum(any(v) for v in by_metadata.values()) / max(len(by_metadata), 1),
            "task_scores": task_scores,
            "overall": overall,
        }

    def score(
        self,
        run_dir: Path,
        samples: list[GenerationSample],
        phase: str,
        model: str,
        run_id: str,
        backend_name: str,
    ) -> BenchmarkReport:
        results_path = run_dir / "results.jsonl"
        score_summary_path = run_dir / "score_summary.json"
        summary: dict[str, Any] = {}
        if score_summary_path.exists():
            summary = read_json(score_summary_path)
            scores = summary.get("scores") if isinstance(summary.get("scores"), dict) else summary
            value = scores.get("overall") if isinstance(scores, dict) else None
        elif results_path.exists():
            summary = self._summarize_results(results_path)
            value = summary.get("overall")
            write_json(score_summary_path, summary)
        else:
            value = None
        report = BenchmarkReport(
            benchmark=self.name,
            display_name=self.display_name,
            model=model,
            run_id=run_id,
            phase=phase,
            backend=backend_name,
            n_samples=len(samples),
            main_metric_name=self.main_metric_name,
            main_metric_value=float(value) if value is not None else None,
            artifacts={
                "results_file": str(results_path.relative_to(self.framework_root)) if results_path.exists() else "",
                "score_summary": str(score_summary_path.relative_to(self.framework_root)) if score_summary_path.exists() else "",
            },
            meta={"score_summary": summary, "score_status": "found" if value is not None else "missing_external_scorer_output"},
        )
        write_json(run_dir / "report.json", report.to_dict())
        return report


class WISEBench(GenerationBenchmark):
    name = "wise"
    display_name = "WISE"
    main_metric_name = "WiScore"
    data_files = [
        ("cultural_common_sense.json", "cultural_common_sense"),
        ("spatio-temporal_reasoning.json", "spatio-temporal_reasoning"),
        ("natural_science.json", "natural_science"),
    ]

    def _data_root(self) -> Path:
        root = self.resolve_path(self.cfg.get("data_root", "benchmarks/generation/wise/data"))
        if (root / "data").is_dir() and not (root / self.data_files[0][0]).is_file():
            root = root / "data"
        return root

    def load_samples(self, limit: int | None = None) -> list[GenerationSample]:
        root = self._data_root()
        rows: list[dict[str, Any]] = []
        missing: list[str] = []
        for filename, category in self.data_files:
            path = root / filename
            if not path.exists():
                missing.append(str(path))
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in data:
                item = dict(item)
                item["_wise_file"] = filename
                item["_wise_category_file"] = category
                rows.append(item)
        if missing and not rows:
            raise FileNotFoundError(
                "WISE data files are missing. Expected cultural_common_sense.json, "
                f"spatio-temporal_reasoning.json, natural_science.json under {root}."
            )
        rows.sort(key=lambda x: int(x["prompt_id"]))
        if limit is not None:
            rows = rows[:limit]
        return [
            GenerationSample(
                uid=str(row["prompt_id"]),
                benchmark=self.name,
                prompt=str(row.get("Prompt", "")),
                expected_outputs=1,
                task=str(row.get("_wise_category_file", "")),
                metadata=row,
            )
            for row in rows
        ]

    def materialize_outputs(
        self,
        run_dir: Path,
        samples: list[GenerationSample],
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        images_dir = ensure_dir(run_dir / "images")
        by_uid = {r.get("uid"): r for r in results}
        written: list[str] = []
        for sample in samples:
            image_paths = (by_uid.get(sample.uid, {}) or {}).get("image_paths") or []
            if not image_paths:
                continue
            src = self.resolve_path(image_paths[0])
            if not src.is_file():
                continue
            dst = images_dir / f"{sample.uid}.png"
            with Image.open(src) as im:
                im.convert("RGB").save(dst)
            written.append(str(dst.relative_to(self.framework_root)))
        return {"images_dir": str(images_dir.relative_to(self.framework_root)), "images": written}

    @staticmethod
    def _wiscore(consistency: float, realism: float, aesthetic_quality: float) -> float:
        return (0.7 * consistency + 0.2 * realism + 0.1 * aesthetic_quality) / 2

    def _score_files(self, run_dir: Path) -> list[Path]:
        results_dir = run_dir / "Results"
        paths = [results_dir / f"{category}_scores.jsonl" for _, category in self.data_files]
        return [p for p in paths if p.exists()]

    def _summarize_scores(self, score_files: list[Path]) -> dict[str, Any]:
        by_category: dict[str, list[float]] = defaultdict(list)
        rows_total = 0
        for path in score_files:
            for row in read_jsonl(path):
                pid = row.get("prompt_id")
                if not isinstance(pid, int):
                    continue
                try:
                    score = self._wiscore(
                        float(row["consistency"]),
                        float(row["realism"]),
                        float(row["aesthetic_quality"]),
                    )
                except (TypeError, ValueError, KeyError):
                    continue
                rows_total += 1
                if 1 <= pid <= 400:
                    by_category["CULTURE"].append(score)
                elif 401 <= pid <= 567:
                    by_category["TIME"].append(score)
                elif 568 <= pid <= 700:
                    by_category["SPACE"].append(score)
                elif 701 <= pid <= 800:
                    by_category["BIOLOGY"].append(score)
                elif 801 <= pid <= 900:
                    by_category["PHYSICS"].append(score)
                elif 901 <= pid <= 1000:
                    by_category["CHEMISTRY"].append(score)
        avg = {k: sum(v) / len(v) for k, v in by_category.items() if v}
        weights = {
            "CULTURE": 0.4,
            "TIME": 0.167,
            "SPACE": 0.133,
            "BIOLOGY": 0.1,
            "PHYSICS": 0.1,
            "CHEMISTRY": 0.1,
        }
        if all(k in avg for k in weights):
            overall = sum(weights[k] * avg[k] for k in weights)
        elif avg:
            overall = sum(avg.values()) / len(avg)
        else:
            overall = None
        return {
            "overall_wiscore": overall,
            "category_scores": avg,
            "n_scored": rows_total,
        }

    def score(
        self,
        run_dir: Path,
        samples: list[GenerationSample],
        phase: str,
        model: str,
        run_id: str,
        backend_name: str,
    ) -> BenchmarkReport:
        score_files = self._score_files(run_dir)
        summary = self._summarize_scores(score_files) if score_files else {}
        summary_path = run_dir / "summary.json"
        if summary:
            write_json(summary_path, summary)
        value = summary.get("overall_wiscore")
        report = BenchmarkReport(
            benchmark=self.name,
            display_name=self.display_name,
            model=model,
            run_id=run_id,
            phase=phase,
            backend=backend_name,
            n_samples=len(samples),
            main_metric_name=self.main_metric_name,
            main_metric_value=float(value) if value is not None else None,
            artifacts={
                "score_files": [str(p.relative_to(self.framework_root)) for p in score_files],
                "summary": str(summary_path.relative_to(self.framework_root)) if summary_path.exists() else "",
            },
            meta={"score_summary": summary, "score_status": "found" if value is not None else "missing_vlm_score_files"},
        )
        write_json(run_dir / "report.json", report.to_dict())
        return report


class UEvalBench(GenerationBenchmark):
    name = "ueval"
    display_name = "UEval"
    main_metric_name = "Overall average rate"

    def _load_dataset(self) -> list[dict[str, Any]]:
        for key in _UEVAL_LOCAL_SOURCE_KEYS:
            value = self.cfg.get(key)
            if not value:
                continue
            path = self.resolve_path(value)
            if not path.exists() and self.cfg.get("fallback_to_hf", True):
                continue
            try:
                return _read_local_dataset_source(path)
            except FileNotFoundError:
                if self.cfg.get("fallback_to_hf", True):
                    continue
                raise

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("UEval loading requires datasets, or set local_json/local_cache in config.") from exc

        ds = load_dataset(str(self.cfg.get("hf_dataset", "zlab-princeton/UEval")), split="test")
        return [dict(item) for item in ds]

    def load_samples(self, limit: int | None = None) -> list[GenerationSample]:
        rows = self._load_dataset()
        domains_value = self.cfg.get("domains")
        if isinstance(domains_value, str) and domains_value.strip() and domains_value.strip() != "all":
            domains = {d.strip().lower() for d in domains_value.split(",") if d.strip()}
        elif isinstance(domains_value, list) and domains_value:
            domains = {str(d).lower() for d in domains_value}
        else:
            domains = set()
        if domains:
            rows = [r for r in rows if str(r.get("task") or r.get("task_type", "")).lower() in domains]
        if limit is not None:
            rows = rows[:limit]
        samples: list[GenerationSample] = []
        for idx, row in enumerate(rows):
            item_id = str(row.get("id", idx))
            prompt = str(row.get("prompt") or row.get("question") or "")
            samples.append(
                GenerationSample(
                    uid=item_id,
                    benchmark=self.name,
                    prompt=prompt,
                    stage="understanding_generation",
                    expected_outputs=int(self.cfg.get("images_per_prompt", 1)),
                    task=str(row.get("task") or row.get("task_type", "")),
                    metadata=row,
                )
            )
        return samples

    def materialize_outputs(
        self,
        run_dir: Path,
        samples: list[GenerationSample],
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        images_dir = ensure_dir(run_dir / "images")
        by_uid = {r.get("uid"): r for r in results}
        model_outputs: list[dict[str, Any]] = []
        for sample in samples:
            result = by_uid.get(sample.uid, {}) or {}
            image_paths_out: list[str] = []
            for idx, image_path in enumerate(result.get("image_paths") or []):
                src = self.resolve_path(image_path)
                if not src.is_file():
                    continue
                suffix = "" if idx == 0 else f"_{idx}"
                dst = images_dir / f"{sample.uid}{suffix}.png"
                with Image.open(src) as im:
                    im.convert("RGB").save(dst)
                image_paths_out.append(str(dst.relative_to(run_dir)))
            model_outputs.append(
                {
                    "id": sample.uid,
                    "prompt": sample.prompt,
                    "task_type": sample.task,
                    "question_type": sample.metadata.get("question_type", ""),
                    "text_answer": result.get("text_answer", ""),
                    "image_answer": image_paths_out,
                }
            )
        model_outputs_path = run_dir / "model_outputs.json"
        write_json(model_outputs_path, model_outputs)
        return {
            "model_outputs": str(model_outputs_path.relative_to(self.framework_root)),
            "images_dir": str(images_dir.relative_to(self.framework_root)),
        }

    def _summarize_eval_results(self, eval_path: Path) -> dict[str, Any]:
        data = read_json(eval_path)
        summary = data.get("summary") if isinstance(data, dict) else {}
        if isinstance(summary, dict):
            return summary
        return {}

    def score(
        self,
        run_dir: Path,
        samples: list[GenerationSample],
        phase: str,
        model: str,
        run_id: str,
        backend_name: str,
    ) -> BenchmarkReport:
        eval_path = run_dir / "eval_results.json"
        summary_path = run_dir / "eval_summary.json"
        summary = self._summarize_eval_results(eval_path) if eval_path.exists() else {}
        if summary:
            write_json(summary_path, summary)
        value = summary.get("overall_avg_rate")
        report = BenchmarkReport(
            benchmark=self.name,
            display_name=self.display_name,
            model=model,
            run_id=run_id,
            phase=phase,
            backend=backend_name,
            n_samples=len(samples),
            main_metric_name=self.main_metric_name,
            main_metric_value=float(value) if value is not None else None,
            artifacts={
                "model_outputs": str((run_dir / "model_outputs.json").relative_to(self.framework_root))
                if (run_dir / "model_outputs.json").exists()
                else "",
                "eval_results": str(eval_path.relative_to(self.framework_root)) if eval_path.exists() else "",
                "eval_summary": str(summary_path.relative_to(self.framework_root)) if summary_path.exists() else "",
            },
            meta={"score_summary": summary, "score_status": "found" if value is not None else "missing_eval_results"},
        )
        write_json(run_dir / "report.json", report.to_dict())
        return report


BENCHMARKS: dict[str, type[GenerationBenchmark]] = {
    "dpg_bench": DPGBench,
    "geneval": GenEvalBench,
    "wise": WISEBench,
    "ueval": UEvalBench,
}


def get_generation_benchmark(name: str, framework_root: Path, cfg: dict[str, Any]) -> GenerationBenchmark:
    key = name.strip().lower()
    if key not in BENCHMARKS:
        raise KeyError(f"unknown generation benchmark: {name}")
    return BENCHMARKS[key](framework_root, cfg)
