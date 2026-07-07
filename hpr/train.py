"""Offline routed trainer over compiled artifacts (paper §3.5).

The trainer consumes static pairwise and scalar artifacts; feedback instances
from the same trajectory may contribute to different losses. The reference
model is a frozen copy of the initial policy whose response log-probabilities
are precomputed once, so no second model has to stay resident during updates
and no policy-versioned rollout or old-logprob replay state is required.

  L_route(theta) = E_x [ sum_z q(z|x) * lambda_z * l_z(theta; x) ]

Hard routing (the default artifact compilation) makes q one-hot, so each
artifact carries exactly one branch loss weighted by lambda_pairwise or
lambda_scalar. Neutral logs never reach the trainer.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .losses import (
    dpo_pair_loss,
    encode_with_response_mask,
    kto_scalar_loss,
    render_prompt,
    sequence_logprob,
    sft_loss,
)


@dataclass
class TrainConfig:
    max_length: int = 8192
    epochs: int = 3
    lr: float = 1e-5
    weight_decay: float = 0.01
    beta: float = 0.1  # alpha in the paper's loss notation
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0
    lambda_pairwise: float = 1.0
    lambda_scalar: float = 1.0
    grad_accum: int = 16  # paper batch size 16, accumulated per-sample
    seed: int = 13
    hpr_sft_coef: float = 0.05
    positive_sft_coef: float = 0.02
    max_grad_norm: float = 1.0
    log_every: int = 5
    device: str = "cuda"

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TrainSample:
    sample_id: str
    kind: str  # "pairwise" | "scalar"
    prompt: str
    chosen: str = ""
    rejected: str = ""
    response: str = ""
    y: int = 0
    feedback_type: str = ""
    ref_chosen_logp: Optional[float] = None
    ref_rejected_logp: Optional[float] = None
    ref_response_logp: Optional[float] = None


def samples_from_artifacts(
    pairwise: List[Dict[str, Any]], scalar: List[Dict[str, Any]]
) -> List[TrainSample]:
    samples: List[TrainSample] = []
    for row in pairwise:
        chosen = str(row.get("chosen") or "").strip()
        rejected = str(row.get("rejected") or "").strip()
        if not chosen or not rejected:
            continue
        samples.append(
            TrainSample(
                sample_id=str(row.get("pair_id") or row.get("instance_id")),
                kind="pairwise",
                prompt=render_prompt(
                    row.get("context") or [],
                    str(row.get("feedback_text") or ""),
                    kind="pairwise",
                ),
                chosen=chosen,
                rejected=rejected,
                feedback_type=str(row.get("feedback_type") or ""),
            )
        )
    for row in scalar:
        response = str(row.get("response") or "").strip()
        y = int(row.get("y") or 0)
        if not response or y == 0:
            continue
        samples.append(
            TrainSample(
                sample_id=str(row.get("artifact_id") or row.get("instance_id")),
                kind="scalar",
                prompt=render_prompt(row.get("context") or [], kind="scalar"),
                response=response,
                y=y,
                feedback_type=str(row.get("feedback_type") or ""),
            )
        )
    return samples


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


class RoutedTrainer:
    def __init__(self, model, tokenizer, config: TrainConfig, output_dir: Path) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "train_metrics.jsonl"
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def _move(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}

    def _encode(self, prompt: str, response: str) -> Dict[str, torch.Tensor]:
        return self._move(
            encode_with_response_mask(
                self.tokenizer, prompt, response, self.config.max_length
            )
        )

    @torch.no_grad()
    def precompute_reference(self, samples: List[TrainSample]) -> None:
        """Score every artifact once under the frozen initial policy."""
        self.model.eval()
        for i, s in enumerate(samples, 1):
            if s.kind == "pairwise":
                s.ref_chosen_logp = float(
                    sequence_logprob(self.model, self._encode(s.prompt, s.chosen)).cpu()
                )
                s.ref_rejected_logp = float(
                    sequence_logprob(self.model, self._encode(s.prompt, s.rejected)).cpu()
                )
            else:
                s.ref_response_logp = float(
                    sequence_logprob(self.model, self._encode(s.prompt, s.response)).cpu()
                )
            if i == 1 or i % 25 == 0 or i == len(samples):
                append_jsonl(
                    self.metrics_path,
                    {"event": "reference_progress", "i": i, "total": len(samples)},
                )
        self.model.train()

    def sample_loss(self, s: TrainSample):
        cfg = self.config
        if s.kind == "pairwise":
            chosen = self._encode(s.prompt, s.chosen)
            rejected = self._encode(s.prompt, s.rejected)
            loss, metrics = dpo_pair_loss(
                sequence_logprob(self.model, chosen),
                sequence_logprob(self.model, rejected),
                s.ref_chosen_logp or 0.0,
                s.ref_rejected_logp or 0.0,
                beta=cfg.beta,
            )
            loss = cfg.lambda_pairwise * loss
            if cfg.hpr_sft_coef > 0:
                extra = sft_loss(self.model, chosen)
                loss = loss + cfg.hpr_sft_coef * extra
                metrics["sft_loss"] = float(extra.detach().cpu())
        else:
            batch = self._encode(s.prompt, s.response)
            loss, metrics = kto_scalar_loss(
                sequence_logprob(self.model, batch),
                s.ref_response_logp or 0.0,
                s.y,
                beta=cfg.beta,
                desirable_weight=cfg.desirable_weight,
                undesirable_weight=cfg.undesirable_weight,
            )
            loss = cfg.lambda_scalar * loss
            if s.y > 0 and cfg.positive_sft_coef > 0:
                extra = sft_loss(self.model, batch)
                loss = loss + cfg.positive_sft_coef * extra
                metrics["sft_loss"] = float(extra.detach().cpu())
        metrics["loss"] = float(loss.detach().cpu())
        metrics["kind"] = s.kind
        return loss, metrics

    def train(self, samples: List[TrainSample]) -> Dict[str, Any]:
        cfg = self.config
        if not samples:
            raise ValueError("no training samples were constructed")
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        manifest = {
            "objective": (
                "pairwise HPR DPO-style loss plus scalar KTO-style outcome loss "
                "with frozen-initial-policy reference log-probabilities"
            ),
            "config": cfg.to_dict(),
            "n_samples": len(samples),
            "n_pairwise": sum(1 for s in samples if s.kind == "pairwise"),
            "n_scalar": sum(1 for s in samples if s.kind == "scalar"),
        }
        append_jsonl(self.metrics_path, {"event": "manifest", **manifest})

        self.precompute_reference(samples)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )
        self.model.train()

        global_step = 0
        optimizer_step = 0
        running: List[float] = []
        start = time.time()
        for epoch in range(1, cfg.epochs + 1):
            random.shuffle(samples)
            optimizer.zero_grad(set_to_none=True)
            for idx, s in enumerate(samples, 1):
                loss, metrics = self.sample_loss(s)
                (loss / cfg.grad_accum).backward()
                global_step += 1
                running.append(metrics["loss"])
                if global_step % cfg.grad_accum == 0 or idx == len(samples):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.max_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1
                    if optimizer_step == 1 or optimizer_step % cfg.log_every == 0:
                        append_jsonl(
                            self.metrics_path,
                            {
                                "event": "train_progress",
                                "epoch": epoch,
                                "optimizer_step": optimizer_step,
                                "mean_loss": sum(running) / max(1, len(running)),
                                "grad_norm": float(grad_norm),
                                "elapsed_sec": round(time.time() - start, 2),
                            },
                        )
                        running = []

        summary = {
            "event": "finished",
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "elapsed_sec": round(time.time() - start, 2),
        }
        append_jsonl(self.metrics_path, summary)
        return {**manifest, **summary}

    def save(self, subdir: str = "final") -> Path:
        target = self.output_dir / subdir
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(target, safe_serialization=True)
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(target)
        return target
