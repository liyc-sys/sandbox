#!/usr/bin/env python3

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import zipfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "benchmarks.json"
RAW_ROOT = ROOT / "data" / "raw"
NORM_ROOT = ROOT / "data" / "normalized"


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def sanitize_name(text):
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def guess_choice_labels(count):
    return [chr(ord("A") + i) for i in range(count)]


def normalize_choices(value):
    if value is None:
      return []
    if isinstance(value, list):
      return [str(x) for x in value]
    if isinstance(value, dict):
      keys = sorted(value.keys())
      return [str(value[k]) for k in keys]
    if isinstance(value, str):
      value = value.strip()
      if value.startswith("[") and value.endswith("]"):
          try:
              parsed = json.loads(value.replace("'", "\""))
              if isinstance(parsed, list):
                  return [str(x) for x in parsed]
          except Exception:
              pass
      return [value]
    return [str(value)]


def save_pil_image(image, out_path):
    ensure_dir(out_path.parent)
    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGB")
    image.save(out_path)
    return {
        "type": "image",
        "path": str(out_path.relative_to(ROOT)),
        "width": image.size[0],
        "height": image.size[1],
        "page_index": None,
        "sha256": sha256_of_file(out_path),
    }


def save_unknown_image(obj, out_path):
    ensure_dir(out_path.parent)
    if isinstance(obj, Image.Image):
        return save_pil_image(obj, out_path)
    if isinstance(obj, (bytes, bytearray)):
        with Image.open(io.BytesIO(bytes(obj))) as image:
            width, height = image.size
            suffix = ".png" if image.format is None else "." + image.format.lower()
            dst = out_path.with_suffix(suffix)
            ensure_dir(dst.parent)
            image.save(dst)
        return {
            "type": "image",
            "path": str(dst.relative_to(ROOT)),
            "width": width,
            "height": height,
            "page_index": None,
            "sha256": sha256_of_file(dst),
        }
    if isinstance(obj, dict):
        if obj.get("path"):
            src = Path(obj["path"])
            suffix = src.suffix or out_path.suffix or ".png"
            dst = out_path.with_suffix(suffix)
            ensure_dir(dst.parent)
            shutil.copy2(src, dst)
            with Image.open(dst) as image:
                width, height = image.size
            return {
                "type": "image",
                "path": str(dst.relative_to(ROOT)),
                "width": width,
                "height": height,
                "page_index": None,
                "sha256": sha256_of_file(dst),
            }
        if obj.get("bytes"):
            data = obj["bytes"]
            if isinstance(data, list):
                data = bytes(data)
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                suffix = ".png" if image.format is None else "." + image.format.lower()
                dst = out_path.with_suffix(suffix)
                ensure_dir(dst.parent)
                image.save(dst)
            return {
                "type": "image",
                "path": str(dst.relative_to(ROOT)),
                "width": width,
                "height": height,
                "page_index": None,
                "sha256": sha256_of_file(dst),
            }
    raise TypeError(f"Unsupported image object: {type(obj)!r}")


def write_jsonl(records, out_path):
    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_output_path(benchmark_name, split, limit=None):
    filename = f"{split}.jsonl" if limit is None else f"{split}.sample_{limit}.jsonl"
    return NORM_ROOT / benchmark_name / filename


def base_record(benchmark, split, subset, task_type, text, source):
    return {
        "uid": "",
        "benchmark": benchmark,
        "subset": subset,
        "split": split,
        "task_type": task_type,
        "prompt": {
            "text": text,
            "choices": [],
            "choice_labels": [],
            "instruction": None,
        },
        "media": [],
        "answer": {
            "type": "text",
            "text": None,
            "choice": None,
            "value": None,
            "aliases": [],
            "bbox": None,
            "structured": None,
        },
        "metadata": {},
        "source": source,
    }


def dataset_rows(dataset_id, split, config_name=None):
    kwargs = {"split": split}
    if config_name is not None:
        kwargs["name"] = config_name
    return load_dataset(dataset_id, **kwargs)


def inspect_dataset(dataset_id, split, config_name=None):
    rows = dataset_rows(dataset_id, split, config_name=config_name)
    sample = rows[0]
    print("dataset_id:", dataset_id)
    print("config:", config_name)
    print("split:", split)
    print("rows:", len(rows))
    print("keys:", sorted(sample.keys()))
    print(json.dumps({k: type(v).__name__ for k, v in sample.items()}, ensure_ascii=False, indent=2))


def prepare_realworldqa(name, cfg, inspect_only=False, limit=None):
    split = cfg["preferred_split"]
    rows = dataset_rows(cfg["dataset_id"], split)
    if inspect_only:
        inspect_dataset(cfg["dataset_id"], split)
        return
    raw_dir = RAW_ROOT / name / split
    norm_records = []
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        uid = row.get("id") or f"{name}_{split}_{idx}"
        source = {
            "source_type": cfg["source_type"],
            "dataset_id": cfg["dataset_id"],
            "repo_id": None,
            "config": None,
            "split": split,
            "raw_index": idx,
            "license": cfg.get("license"),
        }
        rec = base_record(name, split, None, cfg["task_type"], row["question"], source)
        rec["uid"] = str(uid)
        rec["media"].append(save_unknown_image(row["image"], raw_dir / "images" / f"{sanitize_name(str(uid))}.png"))
        answer = str(row["answer"])
        rec["answer"]["type"] = "text"
        rec["answer"]["text"] = answer
        rec["answer"]["aliases"] = [answer]
        norm_records.append(rec)
    write_jsonl(norm_records, resolve_output_path(name, split, limit=limit))


def prepare_vlmsareblind(name, cfg, inspect_only=False, limit=None):
    split = cfg["preferred_split"]
    rows = dataset_rows(cfg["dataset_id"], split)
    if inspect_only:
        inspect_dataset(cfg["dataset_id"], split)
        return
    raw_dir = RAW_ROOT / name / split
    norm_records = []
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        uid = f"{name}_{split}_{idx}"
        source = {
            "source_type": cfg["source_type"],
            "dataset_id": cfg["dataset_id"],
            "repo_id": None,
            "config": None,
            "split": split,
            "raw_index": idx,
            "license": cfg.get("license"),
        }
        prompt = row.get("prompt") or row.get("question") or ""
        rec = base_record(name, split, None, cfg["task_type"], prompt, source)
        rec["uid"] = uid
        rec["media"].append(save_unknown_image(row["image"], raw_dir / "images" / f"{uid}.png"))
        gt = str(row.get("groundtruth", ""))
        rec["answer"]["type"] = "text"
        rec["answer"]["text"] = gt
        rec["answer"]["aliases"] = [gt]
        rec["metadata"] = {
            "task": row.get("task"),
            "source_metadata": row.get("metadata"),
        }
        norm_records.append(rec)
    write_jsonl(norm_records, resolve_output_path(name, split, limit=limit))


def prepare_mathvista(name, cfg, inspect_only=False, limit=None):
    split = cfg["preferred_split"]
    rows = dataset_rows(cfg["dataset_id"], split)
    if inspect_only:
        inspect_dataset(cfg["dataset_id"], split)
        return
    raw_dir = RAW_ROOT / name / split
    norm_records = []
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        uid = str(row.get("pid", f"{name}_{split}_{idx}"))
        source = {
            "source_type": cfg["source_type"],
            "dataset_id": cfg["dataset_id"],
            "repo_id": None,
            "config": "default",
            "split": split,
            "raw_index": idx,
            "license": cfg.get("license"),
        }
        prompt_text = row.get("query") or row.get("question") or ""
        rec = base_record(name, split, None, cfg["task_type"], prompt_text, source)
        rec["uid"] = uid
        rec["media"].append(save_unknown_image(row["decoded_image"], raw_dir / "images" / f"{uid}.png"))
        choices = normalize_choices(row.get("choices"))
        rec["prompt"]["choices"] = choices
        rec["prompt"]["choice_labels"] = guess_choice_labels(len(choices))
        rec["answer"]["type"] = "choice" if row.get("question_type") == "multi_choice" else "text"
        rec["answer"]["text"] = str(row.get("answer", ""))
        rec["answer"]["aliases"] = [str(row.get("answer", ""))]
        rec["metadata"] = {
            "question_type": row.get("question_type"),
            "answer_type": row.get("answer_type"),
            "unit": row.get("unit"),
            "precision": row.get("precision"),
            "mathvista_metadata": row.get("metadata"),
        }
        norm_records.append(rec)
    write_jsonl(norm_records, resolve_output_path(name, split, limit=limit))


def prepare_chartqapro(name, cfg, inspect_only=False, limit=None):
    split = cfg["preferred_split"]
    rows = dataset_rows(cfg["dataset_id"], split)
    if inspect_only:
        inspect_dataset(cfg["dataset_id"], split)
        return
    raw_dir = RAW_ROOT / name / split
    norm_records = []
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        uid = f"{name}_{split}_{idx}"
        source = {
            "source_type": cfg["source_type"],
            "dataset_id": cfg["dataset_id"],
            "repo_id": None,
            "config": None,
            "split": split,
            "raw_index": idx,
            "license": cfg.get("license"),
        }
        question = row.get("Question")
        if isinstance(question, list):
            question = question[0] if question else ""
        answer = row.get("Answer")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        rec = base_record(name, split, None, cfg["task_type"], str(question or ""), source)
        rec["uid"] = uid
        rec["media"].append(save_unknown_image(row["image"], raw_dir / "images" / f"{uid}.png"))
        rec["answer"]["type"] = "text"
        rec["answer"]["text"] = str(answer or "")
        rec["answer"]["aliases"] = [str(answer or "")]
        rec["metadata"] = {
            "question_type": row.get("Question Type"),
            "year": row.get("Year"),
            "paragraph": row.get("Paragraph"),
        }
        norm_records.append(rec)
    write_jsonl(norm_records, resolve_output_path(name, split, limit=limit))


def prepare_mmmu(name, cfg, inspect_only=False, limit=None):
    split = cfg["preferred_split"]
    configs = cfg["preferred_configs"]
    all_records = []
    total_written = 0
    for config_name in configs:
        rows = dataset_rows(cfg["dataset_id"], split, config_name=config_name)
        if inspect_only:
            inspect_dataset(cfg["dataset_id"], split, config_name=config_name)
            return
        raw_dir = RAW_ROOT / name / split / config_name
        for idx, row in enumerate(rows):
            if limit is not None and total_written >= limit:
                break
            uid = str(row.get("id", f"{name}_{config_name}_{split}_{idx}"))
            source = {
                "source_type": cfg["source_type"],
                "dataset_id": cfg["dataset_id"],
                "repo_id": None,
                "config": config_name,
                "split": split,
                "raw_index": idx,
                "license": cfg.get("license"),
            }
            rec = base_record(name, split, config_name, cfg["task_type"], row.get("question", ""), source)
            rec["uid"] = uid
            options = normalize_choices(row.get("options"))
            rec["prompt"]["choices"] = options
            rec["prompt"]["choice_labels"] = guess_choice_labels(len(options))
            for image_idx in range(1, 8):
                key = f"image_{image_idx}"
                image_obj = row.get(key)
                if image_obj is None:
                    continue
                try:
                    rec["media"].append(
                        save_unknown_image(image_obj, raw_dir / "images" / f"{sanitize_name(uid)}_{image_idx}.png")
                    )
                except Exception:
                    continue
            rec["answer"]["type"] = "choice"
            rec["answer"]["choice"] = str(row.get("answer", ""))
            rec["answer"]["text"] = str(row.get("answer", ""))
            rec["answer"]["aliases"] = [str(row.get("answer", ""))]
            rec["metadata"] = {
                "img_type": row.get("img_type"),
                "question_type": row.get("question_type"),
                "subfield": row.get("subfield"),
                "topic_difficulty": row.get("topic_difficulty"),
                "explanation": row.get("explanation"),
            }
            all_records.append(rec)
            total_written += 1
        if limit is not None and total_written >= limit:
            break
    write_jsonl(all_records, resolve_output_path(name, split, limit=limit))


def prepare_phyx(name, cfg, inspect_only=False, limit=None):
    split = cfg["preferred_split"]
    config_name = cfg.get("preferred_configs", [None])[0]
    rows = dataset_rows(cfg["dataset_id"], split, config_name=config_name)
    if inspect_only:
        inspect_dataset(cfg["dataset_id"], split, config_name=config_name)
        return
    raw_dir = RAW_ROOT / name / split
    norm_records = []
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        uid = str(row.get("id", f"{name}_{split}_{idx}"))
        source = {
            "source_type": cfg["source_type"],
            "dataset_id": cfg["dataset_id"],
            "repo_id": None,
            "config": config_name,
            "split": split,
            "raw_index": idx,
            "license": cfg.get("license"),
        }
        prompt = row.get("question_description_simplified") or row.get("question_description") or row.get("question") or ""
        rec = base_record(name, split, None, cfg["task_type"], str(prompt), source)
        rec["uid"] = uid
        rec["media"].append(save_unknown_image(row["image"], raw_dir / "images" / f"{uid}.png"))
        choices = normalize_choices(row.get("options"))
        rec["prompt"]["choices"] = choices
        rec["prompt"]["choice_labels"] = guess_choice_labels(len(choices))
        rec["answer"]["type"] = "choice" if choices else "text"
        rec["answer"]["text"] = str(row.get("answer", ""))
        rec["answer"]["choice"] = str(row.get("answer", "")) if choices else None
        rec["answer"]["aliases"] = [str(row.get("answer", ""))]
        rec["metadata"] = {
            "question_description": row.get("question_description"),
            "question_description_simplified": row.get("question_description_simplified"),
            "image_caption": row.get("image_caption"),
            "category": row.get("category"),
            "subfield": row.get("subfield"),
            "reasoning_type": row.get("reasoning_type"),
        }
        norm_records.append(rec)
    write_jsonl(norm_records, resolve_output_path(name, split, limit=limit))


def prepare_babyvision(name, cfg, inspect_only=False, limit=None):
    split = cfg["preferred_split"]
    rows = dataset_rows(cfg["dataset_id"], split)
    if inspect_only:
        inspect_dataset(cfg["dataset_id"], split)
        return
    raw_dir = RAW_ROOT / name / split
    norm_records = []
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        task_id = row.get("taskId")
        uid = f"{name}_{split}_{task_id if task_id is not None else idx}"
        source = {
            "source_type": cfg["source_type"],
            "dataset_id": cfg["dataset_id"],
            "repo_id": None,
            "config": None,
            "split": split,
            "raw_index": idx,
            "license": cfg.get("license"),
        }
        rec = base_record(name, split, None, cfg["task_type"], str(row.get("question") or ""), source)
        rec["uid"] = uid
        rec["media"].append(save_unknown_image(row["image"], raw_dir / "images" / f"{sanitize_name(uid)}.png"))

        choices = normalize_choices(row.get("options"))
        choices = [c for c in choices if c]
        rec["prompt"]["choices"] = choices
        rec["prompt"]["choice_labels"] = guess_choice_labels(len(choices))

        choice_ans = row.get("choiceAns")
        has_choice_answer = choice_ans is not None and str(choice_ans).strip() != ""
        if choices and has_choice_answer:
            try:
                choice_idx = int(choice_ans)
            except (TypeError, ValueError):
                choice_idx = -1
            labels = rec["prompt"]["choice_labels"]
            choice_label = labels[choice_idx] if 0 <= choice_idx < len(labels) else str(choice_ans)
            choice_text = choices[choice_idx] if 0 <= choice_idx < len(choices) else str(choice_ans)
            rec["answer"]["type"] = "choice"
            rec["answer"]["choice"] = choice_label
            rec["answer"]["text"] = choice_text
            rec["answer"]["aliases"] = [choice_label, choice_text]
            rec["metadata"]["choice_index"] = choice_idx
        else:
            answer = row.get("blankAns")
            if isinstance(answer, list):
                aliases = [str(x) for x in answer if x is not None and str(x).strip()]
                answer_text = aliases[0] if aliases else ""
            else:
                answer_text = str(answer or "")
                aliases = [answer_text] if answer_text else []
            rec["answer"]["type"] = "text"
            rec["answer"]["text"] = answer_text
            rec["answer"]["aliases"] = aliases

        rec["metadata"].update({
            "task_id": task_id,
            "status": row.get("status"),
            "type": row.get("type"),
            "subtype": row.get("subtype"),
            "ans_type": row.get("ansType"),
            "chain_of_thought": row.get("coT"),
        })
        norm_records.append(rec)
    write_jsonl(norm_records, resolve_output_path(name, split, limit=limit))


def prepare_countbench(name, cfg, inspect_only=False, limit=None):
    split = cfg["preferred_split"]
    rows = dataset_rows(cfg["dataset_id"], split)
    if inspect_only:
        inspect_dataset(cfg["dataset_id"], split)
        return
    raw_dir = RAW_ROOT / name / split
    norm_records = []
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        uid = f"{name}_{split}_{idx}"
        source = {
            "source_type": cfg["source_type"],
            "dataset_id": cfg["dataset_id"],
            "repo_id": None,
            "config": None,
            "split": split,
            "raw_index": idx,
            "license": cfg.get("license"),
        }
        rec = base_record(name, split, None, cfg["task_type"], str(row.get("question") or ""), source)
        rec["uid"] = uid
        rec["prompt"]["instruction"] = "Answer with a single integer."
        rec["media"].append(save_unknown_image(row["image"], raw_dir / "images" / f"{uid}.png"))
        number = row.get("number")
        rec["answer"]["type"] = "number"
        rec["answer"]["text"] = str(number)
        rec["answer"]["value"] = number
        rec["answer"]["aliases"] = [str(number)]
        rec["metadata"] = {
            "caption": row.get("text"),
        }
        norm_records.append(rec)
    write_jsonl(norm_records, resolve_output_path(name, split, limit=limit))


def _coco_xywh_to_xyxy(b):
    """RefCOCO bbox 是 COCO 风格 [x, y, w, h]，转成 [x1,y1,x2,y2]。"""
    if not b or len(b) != 4:
        return []
    x, y, w, h = b
    return [float(x), float(y), float(x) + float(w), float(y) + float(h)]


def prepare_refcoco(name, cfg, inspect_only=False, limit=None):
    """传统 grounding 评测：8 split (RefCOCO val/testA/testB + RefCOCO+ val/testA/testB + RefCOCOg val/test)。

    关键点：
    - 原 dataset 的 `question` 字段是 captioning 用的固定 prompt，弃用
    - referring expression 在 `answer` 字段（list of str），取第 0 条作 prompt，其余进 aliases
    - bbox 是 COCO 风格 [x, y, w, h]，需转 [x1, y1, x2, y2]
    - 用 streaming 边下边读，避免一次性把所有 split 读进内存被 OOM
    """
    PROMPT_TPL = (
        "Please provide the bounding box coordinates [x1, y1, x2, y2] in absolute "
        "image pixel coordinates of the region this sentence describes: <ref>{expr}</ref>"
    )
    all_records = []
    for sub in cfg["subdatasets"]:
        ds_id = sub["dataset_id"]
        prefix = sub["split_alias_prefix"]
        for split in sub["splits"]:
            split_alias = f"{prefix}_{split}"
            print(f"  loading {ds_id} split={split} alias={split_alias}", flush=True)
            rows = load_dataset(ds_id, split=split, streaming=True)
            if inspect_only:
                first = next(iter(rows))
                print("keys:", sorted(first.keys()))
                return
            raw_dir = RAW_ROOT / name / split_alias / "images"
            n_written = 0
            for idx, row in enumerate(rows):
                if limit is not None and idx >= limit:
                    break
                qid = row.get("question_id", f"{idx}")
                uid = f"{name}_{split_alias}_{qid}"
                source = {
                    "source_type": cfg["source_type"],
                    "dataset_id": ds_id,
                    "repo_id": None,
                    "config": None,
                    "split": split,
                    "raw_index": idx,
                    "license": cfg.get("license"),
                }
                expressions = [str(x) for x in (row.get("answer") or []) if x]
                expr = expressions[0] if expressions else ""
                prompt_text = PROMPT_TPL.format(expr=expr)
                rec = base_record(name, split_alias, prefix, cfg["task_type"], prompt_text, source)
                rec["uid"] = uid
                try:
                    rec["media"].append(
                        save_unknown_image(row["image"], raw_dir / f"{sanitize_name(uid)}.png")
                    )
                except Exception as e:
                    print(f"    warn: image save fail uid={uid} err={e}", flush=True)
                    continue
                bbox_xyxy = _coco_xywh_to_xyxy(row.get("bbox") or [])
                rec["answer"]["type"] = "bbox"
                rec["answer"]["text"] = expr or None
                rec["answer"]["aliases"] = expressions
                rec["answer"]["bbox"] = bbox_xyxy
                rec["metadata"] = {
                    "refcoco_split": split_alias,
                    "expressions": expressions,
                    "file_name": row.get("file_name"),
                    "iscrowd": row.get("iscrowd"),
                    "original_bbox_xywh": list(row.get("bbox") or []),
                }
                all_records.append(rec)
                n_written += 1
                if n_written % 200 == 0:
                    print(f"    {split_alias}: {n_written} rows...", flush=True)
            print(f"    -> {n_written} rows from {ds_id}/{split}", flush=True)
    out_split = "all8" if limit is None else f"all8.sample_per_split_{limit}"
    write_jsonl(all_records, NORM_ROOT / name / f"{out_split}.jsonl")
    print(f"  total normalized: {len(all_records)} -> {NORM_ROOT / name / f'{out_split}.jsonl'}", flush=True)


def prepare_visulogic(name, cfg, inspect_only=False, limit=None):
    split = cfg["preferred_split"]
    rows = dataset_rows(cfg["dataset_id"], split)
    if inspect_only:
        inspect_dataset(cfg["dataset_id"], split)
        return
    raw_dir = RAW_ROOT / name / split
    norm_records = []
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        uid = str(row.get("id", f"{name}_{split}_{idx}"))
        source = {
            "source_type": cfg["source_type"],
            "dataset_id": cfg["dataset_id"],
            "repo_id": None,
            "config": None,
            "split": split,
            "raw_index": idx,
            "license": cfg.get("license"),
        }
        prompt_text = row.get("question") or row.get("query") or row.get("prompt") or ""
        rec = base_record(name, split, None, cfg["task_type"], str(prompt_text), source)
        rec["uid"] = uid
        image_obj = row.get("image") or row.get("images")
        if isinstance(image_obj, list):
            for image_idx, item in enumerate(image_obj):
                rec["media"].append(save_unknown_image(item, raw_dir / "images" / f"{uid}_{image_idx}.png"))
        else:
            rec["media"].append(save_unknown_image(image_obj, raw_dir / "images" / f"{uid}.png"))
        choices = normalize_choices(row.get("options") or row.get("choices"))
        rec["prompt"]["choices"] = choices
        rec["prompt"]["choice_labels"] = guess_choice_labels(len(choices))
        answer = row.get("answer") or row.get("label") or row.get("groundtruth")
        rec["answer"]["type"] = "choice" if choices else "text"
        rec["answer"]["text"] = str(answer or "")
        rec["answer"]["choice"] = str(answer or "") if choices else None
        rec["answer"]["aliases"] = [str(answer or "")]
        rec["metadata"] = {k: v for k, v in row.items() if k not in {"image", "images", "question", "query", "prompt", "options", "choices", "answer", "label", "groundtruth"}}
        norm_records.append(rec)
    write_jsonl(norm_records, resolve_output_path(name, split, limit=limit))


def find_first_existing(base, candidates):
    for candidate in candidates:
        path = base / candidate
        if path.exists():
            return path
    return None


def prepare_hallusionbench(name, cfg, inspect_only=False, limit=None):
    raw_repo = RAW_ROOT / name / "snapshot"
    ensure_dir(raw_repo)
    snapshot_path = Path(
        snapshot_download(
            repo_id=cfg["repo_id"],
            repo_type="dataset",
            local_dir=raw_repo,
            local_dir_use_symlinks=False,
        )
    )
    ann_path = find_first_existing(snapshot_path, cfg["annotation_file_candidates"])
    if ann_path is None:
        raise FileNotFoundError("HallusionBench annotation json not found in snapshot")
    image_zip = find_first_existing(snapshot_path, cfg["image_archive_candidates"])
    images_dir = RAW_ROOT / name / "images"
    if image_zip is not None and not images_dir.exists():
        ensure_dir(images_dir)
        with zipfile.ZipFile(image_zip, "r") as zf:
            zf.extractall(images_dir)
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    if inspect_only:
        print("hallusionbench_keys:", sorted(data[0].keys()))
        return
    split = cfg["preferred_split"]
    norm_records = []
    for idx, row in enumerate(data):
        if limit is not None and idx >= limit:
            break
        uid = str(row.get("id", f"{name}_{split}_{idx}"))
        source = {
            "source_type": cfg["source_type"],
            "dataset_id": None,
            "repo_id": cfg["repo_id"],
            "config": None,
            "split": split,
            "raw_index": idx,
            "license": cfg.get("license"),
        }
        question = row.get("question") or row.get("query") or row.get("prompt") or row.get("text") or ""
        rec = base_record(name, split, None, cfg["task_type"], str(question), source)
        rec["uid"] = uid
        image_path = None
        image_name = row.get("filename")
        if image_name:
            image_name = str(image_name).replace("./", "")
            image_path = snapshot_path / image_name
            if not image_path.exists():
                image_path = next(snapshot_path.rglob(Path(image_name).name), None)
        elif str(row.get("visual_input")) == "1":
            category = str(row.get("category", "")).strip()
            subcategory = str(row.get("subcategory", "")).strip()
            set_id = str(row.get("set_id", "")).strip()
            figure_id = str(row.get("figure_id", "")).strip()
            candidate_rel = f"hallusion_bench/{category}/{subcategory}/{set_id}_{figure_id}.png"
            candidate_paths = [
                snapshot_path / candidate_rel,
                snapshot_path / "examples" / f"{set_id}_{figure_id}.png",
                snapshot_path / "examples" / f"{set_id}_{figure_id}" / "0.png",
            ]
            for candidate in candidate_paths:
                if candidate.exists():
                    image_path = candidate
                    break
        if image_path is not None and image_path.exists():
            rec["media"].append(
                {
                    "type": "image",
                    "path": str(image_path.relative_to(ROOT)),
                    "width": None,
                    "height": None,
                    "page_index": None,
                    "sha256": sha256_of_file(image_path),
                }
            )
        answer = row.get("answer") or row.get("label") or row.get("gt_answer")
        rec["answer"]["type"] = "text"
        rec["answer"]["text"] = str(answer or "")
        rec["answer"]["aliases"] = [str(answer or "")]
        rec["metadata"] = {k: v for k, v in row.items() if k not in {"question", "query", "prompt", "text", "answer", "label", "gt_answer"}}
        norm_records.append(rec)
    write_jsonl(norm_records, resolve_output_path(name, split, limit=limit))


def prepare_omnidocbench(name, cfg, inspect_only=False, limit=None):
    raw_repo = RAW_ROOT / name / "snapshot"
    ensure_dir(raw_repo)
    snapshot_path = Path(
        snapshot_download(
            repo_id=cfg["repo_id"],
            repo_type="dataset",
            local_dir=raw_repo,
            local_dir_use_symlinks=False,
        )
    )
    ann_path = find_first_existing(snapshot_path, cfg["annotation_file_candidates"])
    if ann_path is None:
        raise FileNotFoundError("OmniDocBench annotation json not found in snapshot")
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    if inspect_only:
        print("omnidocbench_keys:", sorted(data[0].keys()))
        return
    split = cfg["preferred_split"]
    images_dir = snapshot_path / "images"
    norm_records = []
    for idx, row in enumerate(data):
        if limit is not None and idx >= limit:
            break
        uid = str(row.get("page_id", row.get("id", f"{name}_{split}_{idx}")))
        source = {
            "source_type": cfg["source_type"],
            "dataset_id": None,
            "repo_id": cfg["repo_id"],
            "config": None,
            "split": split,
            "raw_index": idx,
            "license": cfg.get("license"),
        }
        prompt = "Parse this document page and return structured document elements, reading order, OCR text, formulas, and tables."
        rec = base_record(name, split, None, cfg["task_type"], prompt, source)
        rec["uid"] = uid
        page_info = row.get("page_info") or {}
        image_name = (
            page_info.get("image_path")
            or row.get("image")
            or row.get("image_path")
            or row.get("file_name")
        )
        if image_name:
            image_path = next(images_dir.rglob(str(image_name)), None)
            if image_path is not None:
                rec["media"].append(
                    {
                        "type": "image",
                        "path": str(image_path.relative_to(ROOT)),
                        "width": None,
                        "height": None,
                        "page_index": None,
                        "sha256": sha256_of_file(image_path),
                    }
                )
        rec["answer"]["type"] = "structured"
        rec["answer"]["structured"] = row
        rec["metadata"] = {
            "document_parse": True,
            "page_info": page_info,
        }
        norm_records.append(rec)
    write_jsonl(norm_records, resolve_output_path(name, split, limit=limit))


PREPARERS = {
    "mmmu": prepare_mmmu,
    "realworldqa": prepare_realworldqa,
    "vlmsareblind": prepare_vlmsareblind,
    "hallusionbench": prepare_hallusionbench,
    "omnidocbench": prepare_omnidocbench,
    "chartqapro": prepare_chartqapro,
    "mathvista": prepare_mathvista,
    "phyx_openended": prepare_phyx,
    "babyvision": prepare_babyvision,
    "countbench": prepare_countbench,
    "refcoco_avg": prepare_refcoco,
    "visulogic": prepare_visulogic,
}


def main():
    parser = argparse.ArgumentParser(description="Download and normalize the selected visual benchmarks.")
    parser.add_argument("--benchmark", choices=sorted(PREPARERS.keys()))
    parser.add_argument("--all", action="store_true", help="Prepare all configured benchmarks.")
    parser.add_argument("--inspect-only", action="store_true", help="Inspect schema / keys only, do not write normalized files.")
    parser.add_argument("--limit", type=int, help="Only normalize the first N samples.")
    args = parser.parse_args()

    if not args.all and not args.benchmark:
        parser.error("Please pass --benchmark <name> or --all")

    config = load_config()["benchmarks"]
    targets = sorted(PREPARERS.keys()) if args.all else [args.benchmark]
    for target in targets:
        print(f"=== preparing {target} ===")
        PREPARERS[target](target, config[target], inspect_only=args.inspect_only, limit=args.limit)
        print(f"=== done {target} ===")


if __name__ == "__main__":
    main()
