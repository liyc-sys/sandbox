from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageDraw

from .types import GenerationResult, GenerationSample
from .utils import ensure_dir, find_images, format_template, read_json, run_command, write_json


class GenerationBackend(Protocol):
    name: str

    def generate(
        self,
        sample: GenerationSample,
        output_dir: Path,
        framework_root: Path,
        options: dict[str, Any],
    ) -> GenerationResult:
        ...


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_path(value: str | Path, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _extract_text_answer(payload: dict[str, Any]) -> str:
    for key in ("text_answer", "text", "answer", "response", "output", "generated_text"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""


def _extract_image_paths(payload: dict[str, Any], roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for key in ("image_paths", "images", "image_answer", "saved_paths", "output_path", "image_path"):
        for item in _ensure_list(payload.get(key)):
            if isinstance(item, str) and item:
                path = Path(item).expanduser()
                if path.is_absolute():
                    paths.append(path)
                    continue
                for root in roots:
                    candidate = root / path
                    if candidate.exists():
                        paths.append(candidate)
                        break
                else:
                    paths.append(roots[0] / path)
    return paths


def _read_response_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def _copy_images_to_output(
    candidates: list[Path],
    output_dir: Path,
    framework_root: Path,
    sample: GenerationSample,
) -> list[str]:
    image_paths: list[str] = []
    seen: set[Path] = set()
    for idx, src in enumerate(candidates[: max(sample.expected_outputs, 1)]):
        if not src.is_file():
            continue
        src = src.resolve()
        if src in seen:
            continue
        seen.add(src)
        if src.parent.resolve() == output_dir.resolve():
            image_paths.append(_relative(src, framework_root))
            continue
        suffix = "" if sample.expected_outputs == 1 else f"_{idx:02d}"
        dst = output_dir / f"{sample.uid}{suffix}{src.suffix.lower() or '.png'}"
        shutil.copy2(src, dst)
        image_paths.append(_relative(dst, framework_root))
    return image_paths


class ManifestBackend:
    """Create HOPE-friendly request manifests without running image generation."""

    name = "manifest"

    def generate(
        self,
        sample: GenerationSample,
        output_dir: Path,
        framework_root: Path,
        options: dict[str, Any],
    ) -> GenerationResult:
        ensure_dir(output_dir)
        request_path = output_dir / f"{sample.uid}.request.json"
        payload = {
            "uid": sample.uid,
            "benchmark": sample.benchmark,
            "stage": sample.stage,
            "prompt": sample.prompt,
            "expected_outputs": sample.expected_outputs,
            "task": sample.task,
            "media": sample.media,
            "metadata": sample.metadata,
            "output_dir": str(output_dir),
            "backend_options": options,
        }
        write_json(request_path, payload)
        return GenerationResult(
            uid=sample.uid,
            benchmark=sample.benchmark,
            stage=sample.stage,
            status="prepared",
            request_path=_relative(request_path, framework_root),
            output_dir=_relative(output_dir, framework_root),
        )


class PlaceholderBackend:
    """Deterministic local smoke backend that writes simple PNGs."""

    name = "placeholder"

    def generate(
        self,
        sample: GenerationSample,
        output_dir: Path,
        framework_root: Path,
        options: dict[str, Any],
    ) -> GenerationResult:
        ensure_dir(output_dir)
        width = int(options.get("width", 512))
        height = int(options.get("height", 512))
        n = max(int(sample.expected_outputs or 1), 1)
        image_paths: list[str] = []
        for idx in range(n):
            suffix = "" if n == 1 else f"_{idx:02d}"
            out_path = output_dir / f"{sample.uid}{suffix}.png"
            if out_path.exists() and not options.get("overwrite", False):
                image_paths.append(_relative(out_path, framework_root))
                continue
            image = Image.new("RGB", (width, height), (236, 238, 242))
            draw = ImageDraw.Draw(image)
            draw.rectangle((18, 18, width - 18, height - 18), outline=(70, 84, 101), width=3)
            text = f"{sample.benchmark}\n{sample.uid}\n{sample.prompt[:160]}"
            draw.multiline_text((32, 32), text, fill=(28, 35, 43), spacing=8)
            image.save(out_path)
            image_paths.append(_relative(out_path, framework_root))
        return GenerationResult(
            uid=sample.uid,
            benchmark=sample.benchmark,
            stage=sample.stage,
            status="completed",
            output_dir=_relative(output_dir, framework_root),
            image_paths=image_paths,
        )


class ExistingOutputBackend:
    """Map pre-generated images/text outputs into the framework output layout."""

    name = "existing"

    def generate(
        self,
        sample: GenerationSample,
        output_dir: Path,
        framework_root: Path,
        options: dict[str, Any],
    ) -> GenerationResult:
        ensure_dir(output_dir)
        source_root_value = options.get("source_root") or options.get("input_dir")
        if not source_root_value:
            return GenerationResult(
                uid=sample.uid,
                benchmark=sample.benchmark,
                stage=sample.stage,
                status="error",
                output_dir=_relative(output_dir, framework_root),
                error="existing backend requires source_root/input_dir",
            )
        source_root = _as_path(source_root_value, framework_root)
        response_payload: dict[str, Any] = {}
        response_path: Path | None = None
        for response_name in (
            f"{sample.uid}.json",
            f"{sample.uid}.response.json",
            f"{sample.uid}_response.json",
            f"{sample.uid}/response.json",
            f"{sample.uid}/output.json",
        ):
            candidate = source_root / response_name
            if candidate.is_file():
                response_path = candidate
                response_payload = _read_response_payload(candidate)
                break

        image_roots = [source_root]
        if response_path is not None:
            image_roots.insert(0, response_path.parent)
        candidates: list[Path] = _extract_image_paths(response_payload, image_roots)
        for pattern in (
            f"{sample.uid}.png",
            f"{sample.uid}.jpg",
            f"{sample.uid}.jpeg",
            f"{sample.uid}.webp",
            f"{sample.uid}_*.png",
            f"{sample.uid}-*.png",
        ):
            candidates.extend(source_root.glob(pattern))
        if not candidates:
            direct_dir = source_root / sample.uid
            if direct_dir.is_dir():
                candidates.extend(find_images(direct_dir))
        candidates = sorted({p.resolve() for p in candidates})
        if not candidates:
            return GenerationResult(
                uid=sample.uid,
                benchmark=sample.benchmark,
                stage=sample.stage,
                status="missing",
                output_dir=_relative(output_dir, framework_root),
                error=f"no existing images found for {sample.uid} under {source_root}",
            )
        image_paths = _copy_images_to_output(candidates, output_dir, framework_root, sample)
        return GenerationResult(
            uid=sample.uid,
            benchmark=sample.benchmark,
            stage=sample.stage,
            status="completed",
            output_dir=_relative(output_dir, framework_root),
            text_answer=_extract_text_answer(response_payload),
            image_paths=image_paths,
            response_path=_relative(response_path, framework_root) if response_payload else "",
            meta={"source_root": str(source_root), "response": response_payload},
        )


class CommandBackend:
    """Run a user-provided command per sample.

    The command may use placeholders such as {prompt}, {uid}, {output_dir},
    {expected_outputs}, and {request_json}. It must write generated images under
    output_dir, or write a JSON response path via the command itself.
    """

    name = "command"

    def generate(
        self,
        sample: GenerationSample,
        output_dir: Path,
        framework_root: Path,
        options: dict[str, Any],
    ) -> GenerationResult:
        ensure_dir(output_dir)
        command_template = options.get("command")
        if not command_template:
            return GenerationResult(
                uid=sample.uid,
                benchmark=sample.benchmark,
                stage=sample.stage,
                status="error",
                output_dir=_relative(output_dir, framework_root),
                error="command backend requires options.command",
            )
        request_path = output_dir / f"{sample.uid}.request.json"
        write_json(
            request_path,
            {
                "uid": sample.uid,
                "benchmark": sample.benchmark,
                "prompt": sample.prompt,
                "expected_outputs": sample.expected_outputs,
                "metadata": sample.metadata,
                "media": sample.media,
            },
        )
        context = {
            "uid": sample.uid,
            "benchmark": sample.benchmark,
            "prompt": sample.prompt,
            "expected_outputs": sample.expected_outputs,
            "output_dir": str(output_dir),
            "request_json": str(request_path),
            "framework_root": str(framework_root),
        }
        command = format_template(str(command_template), context)
        env = os.environ.copy()
        extra_env = options.get("env")
        if isinstance(extra_env, dict):
            env.update({str(k): str(v) for k, v in extra_env.items()})
        proc = run_command(command, cwd=framework_root, env=env)
        response_payload: dict[str, Any] = {}
        response_path_value = options.get("response_path")
        response_candidates = []
        if response_path_value:
            response_candidates.append(_as_path(format_template(str(response_path_value), context), framework_root))
        response_candidates.extend(
            [
                output_dir / "response.json",
                output_dir / "output.json",
                output_dir / f"{sample.uid}.json",
            ]
        )
        response_path = next((p for p in response_candidates if p.is_file()), None)
        if response_path:
            response_payload = _read_response_payload(response_path)

        image_roots = [response_path.parent if response_path else output_dir, output_dir, framework_root]
        image_candidates = _extract_image_paths(response_payload, image_roots) + find_images(output_dir)
        images = [_relative(p, framework_root) for p in image_candidates if p.is_file()]
        images = list(dict.fromkeys(images))
        status = "completed" if proc.returncode == 0 and images else "error"
        return GenerationResult(
            uid=sample.uid,
            benchmark=sample.benchmark,
            stage=sample.stage,
            status=status,
            request_path=_relative(request_path, framework_root),
            response_path=_relative(response_path, framework_root) if response_path else "",
            output_dir=_relative(output_dir, framework_root),
            text_answer=_extract_text_answer(response_payload),
            image_paths=images,
            error=None if status == "completed" else (proc.stderr or proc.stdout or "command produced no images"),
            meta={
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
                "response": response_payload,
            },
        )


class HopeManifestBackend(ManifestBackend):
    """HOPE placeholder: emit the submission manifest that company tooling can consume."""

    name = "hope_manifest"


BACKENDS: dict[str, type[GenerationBackend]] = {
    "manifest": ManifestBackend,
    "hope_manifest": HopeManifestBackend,
    "placeholder": PlaceholderBackend,
    "existing": ExistingOutputBackend,
    "command": CommandBackend,
}


def get_backend(name: str) -> GenerationBackend:
    key = (name or "manifest").strip().lower()
    if key not in BACKENDS:
        raise KeyError(f"unknown generation backend: {name}")
    return BACKENDS[key]()
