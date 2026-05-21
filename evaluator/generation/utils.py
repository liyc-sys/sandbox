from __future__ import annotations

import base64
import json
import re
import shlex
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sanitize_component(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    return value.strip("_") or "run"


def make_run_id(prefix: str = "run") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{sanitize_component(prefix)}_{stamp}_{uuid.uuid4().hex[:6]}"


def glob_sorted(path: Path, pattern: str) -> list[Path]:
    return sorted(path.glob(pattern), key=lambda p: (p.name))


def numeric_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.isdigit():
        return (0, f"{int(stem):08d}")
    return (1, stem)


def find_images(path: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    images = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(images, key=lambda p: (p.stat().st_mtime, p.name))


def image_to_data_url(image_path: Path, max_w: int | None = 2048) -> str:
    mime = "image/png"
    with Image.open(image_path) as im:
        if max_w and im.width > max_w:
            ratio = max_w / float(im.width)
            new_height = max(1, int(im.height * ratio))
            im = im.resize((max_w, new_height), Image.Resampling.LANCZOS)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            save_format = "PNG"
        else:
            save_format = "JPEG"
        if save_format == "JPEG" and im.mode != "RGB":
            im = im.convert("RGB")
        from io import BytesIO

        buffer = BytesIO()
        im.save(buffer, format=save_format)
        mime = f"image/{save_format.lower()}"
        data = buffer.getvalue()
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"


def image_to_bytes(image_path: Path) -> bytes:
    return image_path.read_bytes()


def copy_image(src: Path, dst: Path) -> Path:
    ensure_dir(dst.parent)
    with Image.open(src) as im:
        im.save(dst)
    return dst


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def format_template(template: str, context: dict[str, Any]) -> str:
    values = {k: str(v) for k, v in context.items()}

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return values.get(key, match.group(0))

    return re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", replace, template)


def run_command(
    command: str | list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    if isinstance(command, str):
        argv = command if command.strip() else ""
        return subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            env=env,
            shell=True,
            check=False,
            text=True,
            capture_output=True,
        )
    return subprocess.run(
        [str(x) for x in command],
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )


def split_csv_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    try:
        return shlex.split(text)
    except ValueError:
        return [text]
