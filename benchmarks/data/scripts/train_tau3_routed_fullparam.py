#!/usr/bin/env python3
"""Full-parameter tau3 routed-feedback training.

This is the first formal tau3 trainer for the HPR/RL routing experiments. It
uses only train_update examples for optimization and dev examples for sanity
metrics. reported_test is deliberately ignored.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


LOCAL_TYPES = {"local_preference", "local_correction"}
NON_NEUTRAL_TYPES = {
    "local_preference",
    "local_correction",
    "tool_api_outcome",
    "delayed_trajectory_outcome",
}
DELAYED_TYPE = "delayed_trajectory_outcome"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(content)


def compact(text: str, limit: int = 2000) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def build_state_index(trajectories_path: Path) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for traj in read_jsonl(trajectories_path):
        traj_id = str(traj.get("trajectory_id"))
        for turn in traj.get("turns") or []:
            turn_id = turn.get("turn_id")
            key = f"{traj_id}:turn:{turn_id}:state"
            state = turn.get("state") or {}
            if isinstance(state, dict):
                states[key] = state
    return states


def resolve_prompt(prompt: Any, row: dict[str, Any], states: dict[str, dict[str, Any]]) -> dict[str, Any] | str:
    if isinstance(prompt, dict) and prompt.get("messages"):
        return prompt
    state_ref = None
    if isinstance(prompt, dict):
        state_ref = prompt.get("state_ref")
    state_ref = state_ref or row.get("state_ref")
    if state_ref and state_ref in states:
        return states[state_ref]
    traj_id = row.get("trajectory_id")
    turn_id = row.get("turn_id")
    if traj_id is not None and turn_id is not None:
        key = f"{traj_id}:turn:{turn_id}:state"
        if key in states:
            return states[key]
    return prompt if prompt is not None else ""


def flatten_prompt(prompt: Any, *, max_messages: int = 14) -> str:
    if isinstance(prompt, dict):
        messages = prompt.get("messages")
        if isinstance(messages, list) and messages:
            lines: list[str] = []
            for msg in messages[-max_messages:]:
                role = msg.get("role", "unknown")
                content = compact(flatten_content(msg.get("content")), 1400)
                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    content += "\nTOOL_CALLS: " + compact(json.dumps(tool_calls, ensure_ascii=False), 1200)
                if content:
                    lines.append(f"{role}: {content}")
            return "\n".join(lines)
        latest = prompt.get("latest_user_message")
        if latest:
            return str(latest)
    return str(prompt or "")


def text_field(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("text") or "")
    return str(value or "")


def parse_reward(row: dict[str, Any]) -> float | None:
    ppo = row.get("ppo") or {}
    value = ppo.get("reward")
    if value is None:
        raw = row.get("feedback_text")
        try:
            value = (json.loads(raw) if isinstance(raw, str) else raw or {}).get("reward")
        except Exception:
            value = None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def selected_feedback_type(row: dict[str, Any], field: str) -> str:
    if field == "gold":
        return str(row.get("gold_feedback_type") or "")
    if field == "router":
        return str(row.get("router_feedback_type") or row.get("gold_feedback_type") or "")
    return str(row.get("router_feedback_type") or row.get("gold_feedback_type") or "")


def action_text(row: dict[str, Any], mode: str) -> str:
    action = row.get("action")
    if isinstance(action, dict):
        text = str(action.get("text") or "").strip()
        raw = str(action.get("raw_text") or "").strip()
        if mode == "raw":
            return raw
        if mode == "text_or_raw":
            return text or raw
        return text
    return text_field(action).strip()


def scalar_reward(row: dict[str, Any], feedback_type: str) -> float | None:
    reward = parse_reward(row)
    if reward is not None:
        return reward
    label = row.get("local_label")
    if label is None:
        return None
    try:
        return 1.0 if float(label) > 0 else 0.0
    except (TypeError, ValueError):
        return None


@dataclass
class TrainSample:
    sample_id: str
    kind: str
    split: str
    benchmark: str
    domain: str
    task_id: str
    prompt: str
    chosen: str = ""
    rejected: str = ""
    response: str = ""
    reward: float | None = None
    feedback_type: str = ""
    ref_chosen_logp: float | None = None
    ref_rejected_logp: float | None = None
    ref_response_logp: float | None = None


def hpr_prompt(pair: dict[str, Any], states: dict[str, dict[str, Any]]) -> str:
    prompt_obj = resolve_prompt(pair.get("prompt"), pair, states)
    context = flatten_prompt(prompt_obj)
    feedback = str(pair.get("feedback_text") or "")
    return (
        "You are an agent being adapted from local user feedback.\n\n"
        f"Conversation context:\n{context}\n\n"
        f"Local feedback:\n{feedback}\n\n"
        "Write the assistant response that should have been given at that turn.\n"
        "Assistant:"
    )


def delayed_prompt(row: dict[str, Any], states: dict[str, dict[str, Any]]) -> str:
    prompt_obj = resolve_prompt(row.get("prompt"), row, states)
    context = flatten_prompt(prompt_obj)
    return (
        "You are a task-oriented assistant. Continue the conversation by taking "
        "the next assistant action.\n\n"
        f"Conversation context:\n{context}\n\n"
        "Assistant:"
    )


def build_samples(
    *,
    hpr_pairs_path: Path,
    feedback_instances_path: Path,
    trajectories_path: Path,
    hpr_types: set[str],
    scalar_types: set[str],
    scalar_feedback_field: str,
    scalar_action_mode: str,
) -> tuple[list[TrainSample], list[TrainSample], dict[str, Any]]:
    states = build_state_index(trajectories_path)
    train: list[TrainSample] = []
    dev: list[TrainSample] = []
    skipped: dict[str, int] = {"hpr_empty": 0, "delayed_empty": 0, "delayed_missing_reward": 0}

    for pair in read_jsonl(hpr_pairs_path):
        split = str(pair.get("split") or "")
        if split not in {"train_update", "dev"}:
            continue
        feedback_type = str(pair.get("feedback_type") or "")
        if feedback_type not in hpr_types:
            continue
        chosen = text_field(pair.get("chosen")).strip()
        rejected = text_field(pair.get("rejected")).strip()
        if not chosen or not rejected:
            skipped["hpr_empty"] += 1
            continue
        sample = TrainSample(
            sample_id=str(pair.get("pair_id") or pair.get("instance_id")),
            kind="hpr",
            split=split,
            benchmark=str(pair.get("benchmark") or "tau3_bench"),
            domain=str(pair.get("domain") or ""),
            task_id=str(pair.get("task_id") or ""),
            prompt=hpr_prompt(pair, states),
            chosen=chosen,
            rejected=rejected,
            feedback_type=feedback_type,
        )
        (train if split == "train_update" else dev).append(sample)

    for row in read_jsonl(feedback_instances_path):
        split = str(row.get("split") or "")
        if split not in {"train_update", "dev"}:
            continue
        feedback_type = selected_feedback_type(row, scalar_feedback_field)
        if feedback_type not in scalar_types:
            continue
        reward = scalar_reward(row, feedback_type)
        if reward is None:
            skipped["delayed_missing_reward"] += 1
            continue
        response = action_text(row, scalar_action_mode)
        if not response:
            skipped["delayed_empty"] += 1
            continue
        sample = TrainSample(
            sample_id=str(row.get("instance_id")),
            kind="delayed",
            split=split,
            benchmark=str(row.get("benchmark") or "tau3_bench"),
            domain=str(row.get("domain") or ""),
            task_id=str(row.get("task_id") or ""),
            prompt=delayed_prompt(row, states),
            response=response,
            reward=1.0 if reward > 0 else 0.0,
            feedback_type=feedback_type,
        )
        (train if split == "train_update" else dev).append(sample)

    summary = {
        "train": count_samples(train),
        "dev": count_samples(dev),
        "skipped": skipped,
        "states_indexed": len(states),
    }
    return train, dev, summary


def count_samples(samples: list[TrainSample]) -> dict[str, Any]:
    out: dict[str, Any] = {"total": len(samples), "by_kind": {}, "by_domain": {}}
    for s in samples:
        out["by_kind"][s.kind] = out["by_kind"].get(s.kind, 0) + 1
        out["by_domain"][s.domain] = out["by_domain"].get(s.domain, 0) + 1
    return out


def encode_with_response_mask(tokenizer, prompt: str, response: str, max_length: int) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    response_ids = tokenizer("\n" + response + (tokenizer.eos_token or ""), add_special_tokens=False).input_ids
    if len(prompt_ids) + len(response_ids) > max_length:
        keep_response = min(len(response_ids), max_length // 2)
        keep_prompt = max_length - keep_response
        prompt_ids = prompt_ids[-keep_prompt:]
        response_ids = response_ids[:keep_response]
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def sequence_logprob(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    logits = outputs.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~mask, 0)
    token_logps = torch.gather(F.log_softmax(logits, dim=-1), -1, safe_labels.unsqueeze(-1)).squeeze(-1)
    denom = mask.sum(dim=-1).clamp_min(1)
    return (token_logps * mask).sum(dim=-1) / denom


@torch.no_grad()
def precompute_reference(
    *,
    model,
    tokenizer,
    samples: list[TrainSample],
    max_length: int,
    device: torch.device,
    log_path: Path,
) -> None:
    model.eval()
    for i, sample in enumerate(samples, 1):
        if sample.kind == "hpr":
            chosen = move_batch(encode_with_response_mask(tokenizer, sample.prompt, sample.chosen, max_length), device)
            rejected = move_batch(encode_with_response_mask(tokenizer, sample.prompt, sample.rejected, max_length), device)
            sample.ref_chosen_logp = float(sequence_logprob(model, chosen).detach().cpu())
            sample.ref_rejected_logp = float(sequence_logprob(model, rejected).detach().cpu())
        else:
            batch = move_batch(encode_with_response_mask(tokenizer, sample.prompt, sample.response, max_length), device)
            sample.ref_response_logp = float(sequence_logprob(model, batch).detach().cpu())
        if i == 1 or i % 25 == 0 or i == len(samples):
            append_jsonl(log_path, {"event": "reference_progress", "i": i, "total": len(samples), "time": time.time()})
    model.train()


def sample_loss(
    *,
    model,
    tokenizer,
    sample: TrainSample,
    max_length: int,
    device: torch.device,
    beta: float,
    hpr_sft_coef: float,
    positive_sft_coef: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if sample.kind == "hpr":
        chosen = move_batch(encode_with_response_mask(tokenizer, sample.prompt, sample.chosen, max_length), device)
        rejected = move_batch(encode_with_response_mask(tokenizer, sample.prompt, sample.rejected, max_length), device)
        chosen_logp = sequence_logprob(model, chosen)
        rejected_logp = sequence_logprob(model, rejected)
        pi_margin = chosen_logp - rejected_logp
        ref_margin = float(sample.ref_chosen_logp or 0.0) - float(sample.ref_rejected_logp or 0.0)
        pref_loss = -F.logsigmoid(beta * (pi_margin - ref_margin)).mean()
        sft_loss = model(
            input_ids=chosen["input_ids"],
            attention_mask=chosen["attention_mask"],
            labels=chosen["labels"],
            use_cache=False,
        ).loss
        loss = pref_loss + hpr_sft_coef * sft_loss
        metrics = {
            "loss": float(loss.detach().cpu()),
            "pref_loss": float(pref_loss.detach().cpu()),
            "sft_loss": float(sft_loss.detach().cpu()),
            "margin": float(pi_margin.detach().cpu()),
        }
        return loss, metrics

    batch = move_batch(encode_with_response_mask(tokenizer, sample.prompt, sample.response, max_length), device)
    logp = sequence_logprob(model, batch)
    ref_logp = float(sample.ref_response_logp or 0.0)
    sign = 1.0 if (sample.reward or 0.0) > 0.5 else -1.0
    kto_loss = -F.logsigmoid(beta * sign * (logp - ref_logp)).mean()
    if sign > 0 and positive_sft_coef > 0:
        sft_loss = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
        ).loss
        loss = kto_loss + positive_sft_coef * sft_loss
        sft_value = float(sft_loss.detach().cpu())
    else:
        loss = kto_loss
        sft_value = 0.0
    metrics = {
        "loss": float(loss.detach().cpu()),
        "kto_loss": float(kto_loss.detach().cpu()),
        "reward": float(sample.reward or 0.0),
        "logp_delta": float((logp - ref_logp).detach().cpu()),
        "sft_loss": sft_value,
    }
    return loss, metrics


@torch.no_grad()
def evaluate(
    *,
    model,
    tokenizer,
    samples: list[TrainSample],
    max_length: int,
    device: torch.device,
    beta: float,
    hpr_sft_coef: float,
    positive_sft_coef: float,
    limit: int,
) -> dict[str, Any]:
    model.eval()
    use = samples[:limit] if limit > 0 else samples
    losses: list[float] = []
    hpr_correct = 0
    hpr_total = 0
    delayed_pos_delta: list[float] = []
    delayed_neg_delta: list[float] = []
    for sample in use:
        loss, metrics = sample_loss(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            max_length=max_length,
            device=device,
            beta=beta,
            hpr_sft_coef=hpr_sft_coef,
            positive_sft_coef=positive_sft_coef,
        )
        losses.append(float(loss.detach().cpu()))
        if sample.kind == "hpr":
            hpr_total += 1
            if metrics.get("margin", 0.0) > 0:
                hpr_correct += 1
        else:
            if sample.reward and sample.reward > 0.5:
                delayed_pos_delta.append(metrics.get("logp_delta", 0.0))
            else:
                delayed_neg_delta.append(metrics.get("logp_delta", 0.0))
    model.train()
    return {
        "n": len(use),
        "loss": sum(losses) / max(1, len(losses)),
        "hpr_pref_acc": hpr_correct / hpr_total if hpr_total else None,
        "hpr_total": hpr_total,
        "delayed_pos_delta": sum(delayed_pos_delta) / max(1, len(delayed_pos_delta)) if delayed_pos_delta else None,
        "delayed_neg_delta": sum(delayed_neg_delta) / max(1, len(delayed_neg_delta)) if delayed_neg_delta else None,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--hpr-pairs", type=Path, required=True)
    p.add_argument("--feedback-instances", type=Path, required=True)
    p.add_argument("--trajectories", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-length", type=int, default=1536)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--hpr-sft-coef", type=float, default=0.05)
    p.add_argument("--positive-sft-coef", type=float, default=0.02)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--dev-limit", type=int, default=0)
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--hpr-types", nargs="*", default=sorted(LOCAL_TYPES),
                   help="Feedback types consumed by the HPR pairwise branch; empty disables HPR.")
    p.add_argument("--scalar-types", nargs="*", default=[DELAYED_TYPE],
                   help="Feedback types consumed by the scalar binary branch; empty disables scalar updates.")
    p.add_argument("--scalar-feedback-field", choices=["router", "gold"], default="router")
    p.add_argument("--scalar-action-mode", choices=["text", "raw", "text_or_raw"], default="text")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "train_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    train_samples, dev_samples, data_summary = build_samples(
        hpr_pairs_path=args.hpr_pairs,
        feedback_instances_path=args.feedback_instances,
        trajectories_path=args.trajectories,
        hpr_types=set(args.hpr_types),
        scalar_types=set(args.scalar_types),
        scalar_feedback_field=args.scalar_feedback_field,
        scalar_action_mode=args.scalar_action_mode,
    )
    if not train_samples:
        raise SystemExit("no train_update samples were constructed")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    manifest = {
        "version": "tau3-routed-fullparam-v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "model_path": str(args.model_path),
        "hpr_pairs": str(args.hpr_pairs),
        "feedback_instances": str(args.feedback_instances),
        "trajectories": str(args.trajectories),
        "output_dir": str(args.output_dir),
        "objective": "HPR DPO-style pair loss plus delayed-outcome KTO-style binary loss with initial-policy reference log-probabilities",
        "split_policy": "optimize train_update only; dev used only for sanity metrics; reported_test is not read",
        "data_summary": data_summary,
        "hyperparameters": {
            "max_length": args.max_length,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "beta": args.beta,
            "grad_accum": args.grad_accum,
            "hpr_sft_coef": args.hpr_sft_coef,
            "positive_sft_coef": args.positive_sft_coef,
            "seed": args.seed,
            "hpr_types": args.hpr_types,
            "scalar_types": args.scalar_types,
            "scalar_feedback_field": args.scalar_feedback_field,
            "scalar_action_mode": args.scalar_action_mode,
        },
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    append_jsonl(metrics_path, {"event": "manifest", **manifest})

    all_ref = train_samples + dev_samples
    precompute_reference(
        model=model,
        tokenizer=tokenizer,
        samples=all_ref,
        max_length=args.max_length,
        device=device,
        log_path=metrics_path,
    )
    if dev_samples:
        dev_metrics = evaluate(
            model=model,
            tokenizer=tokenizer,
            samples=dev_samples,
            max_length=args.max_length,
            device=device,
            beta=args.beta,
            hpr_sft_coef=args.hpr_sft_coef,
            positive_sft_coef=args.positive_sft_coef,
            limit=args.dev_limit,
        )
        append_jsonl(metrics_path, {"event": "dev_before_train", **dev_metrics, "time": time.time()})

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    global_step = 0
    optimizer_step = 0
    running: list[dict[str, float]] = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_samples)
        optimizer.zero_grad(set_to_none=True)
        for idx, sample in enumerate(train_samples, 1):
            loss, metrics = sample_loss(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                max_length=args.max_length,
                device=device,
                beta=args.beta,
                hpr_sft_coef=args.hpr_sft_coef,
                positive_sft_coef=args.positive_sft_coef,
            )
            (loss / args.grad_accum).backward()
            global_step += 1
            running.append(metrics)
            did_step = global_step % args.grad_accum == 0 or idx == len(train_samples)
            if did_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if optimizer_step == 1 or optimizer_step % args.log_every == 0:
                    mean_loss = sum(m["loss"] for m in running) / max(1, len(running))
                    append_jsonl(metrics_path, {
                        "event": "train_progress",
                        "epoch": epoch,
                        "idx": idx,
                        "train_total": len(train_samples),
                        "global_step": global_step,
                        "optimizer_step": optimizer_step,
                        "mean_loss": mean_loss,
                        "last_kind": sample.kind,
                        "last_domain": sample.domain,
                        "grad_norm": float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm),
                        "elapsed_sec": round(time.time() - start_time, 2),
                    })
                    running.clear()
        if dev_samples:
            dev_metrics = evaluate(
                model=model,
                tokenizer=tokenizer,
                samples=dev_samples,
                max_length=args.max_length,
                device=device,
                beta=args.beta,
                hpr_sft_coef=args.hpr_sft_coef,
                positive_sft_coef=args.positive_sft_coef,
                limit=args.dev_limit,
            )
            append_jsonl(metrics_path, {"event": "dev_after_epoch", "epoch": epoch, **dev_metrics, "time": time.time()})
        if args.save_every_epoch:
            epoch_dir = args.output_dir / f"epoch_{epoch}"
            model.save_pretrained(epoch_dir, safe_serialization=True, max_shard_size="4GB")
            tokenizer.save_pretrained(epoch_dir)

    final_dir = args.output_dir / "final"
    model.save_pretrained(final_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(final_dir)
    shutil.copy2(args.output_dir / "manifest.json", final_dir / "train_manifest.json")
    append_jsonl(metrics_path, {
        "event": "finished",
        "final_dir": str(final_dir),
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "elapsed_sec": round(time.time() - start_time, 2),
    })
    print(json.dumps({"final_dir": str(final_dir), "optimizer_step": optimizer_step}, ensure_ascii=False))
    if math.isnan(global_step):
        raise SystemExit("unreachable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
