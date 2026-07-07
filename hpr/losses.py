"""Pairwise HPR loss and scalar KTO-style loss (paper §3.4, §3.5).

Both branches score only the response/action tokens of the routed feedback
instance; context tokens are masked out. rho_theta(a, s) is the policy /
reference log-ratio computed from length-normalized sequence log-probabilities
under a frozen copy of the initial policy:

  pairwise:  l_HPR    = -log sigma(alpha * (rho(a+, s) - rho(a-, s)))
  scalar:    l_scalar = -log sigma(alpha * y * rho(a, s)),  y in {+1, -1}

Small SFT coefficients on the chosen / desirable response are kept as optional
stabilizers, matching the reference training runs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F


def render_prompt(context: List[Dict[str, Any]], feedback_text: str = "", kind: str = "pairwise") -> str:
    """Flatten an artifact context into the training prompt string s_t."""
    lines = []
    for msg in (context or [])[-14:]:
        role = msg.get("role", "unknown")
        content = msg.get("content")
        if isinstance(content, list):
            content = " ".join(
                str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content
            )
        content = " ".join(str(content or "").split())
        if len(content) > 1400:
            content = content[:700] + " ...[truncated]... " + content[-700:]
        if content:
            lines.append("%s: %s" % (role, content))
    rendered = "\n".join(lines)
    if kind == "pairwise":
        return (
            "You are an agent being adapted from local user feedback.\n\n"
            "Conversation context:\n%s\n\n"
            "Local feedback:\n%s\n\n"
            "Write the assistant response that should have been given at that turn.\n"
            "Assistant:" % (rendered, feedback_text)
        )
    return (
        "You are a task-oriented assistant. Continue the conversation by taking "
        "the next assistant action.\n\n"
        "Conversation context:\n%s\n\n"
        "Assistant:" % rendered
    )


def encode_with_response_mask(
    tokenizer, prompt: str, response: str, max_length: int
) -> Dict[str, torch.Tensor]:
    """Tokenize prompt+response; labels mask the prompt tokens with -100."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    response_ids = tokenizer(
        "\n" + response + (tokenizer.eos_token or ""), add_special_tokens=False
    ).input_ids
    if len(prompt_ids) + len(response_ids) > max_length:
        keep_response = min(len(response_ids), max_length // 2)
        keep_prompt = max_length - keep_response
        prompt_ids = prompt_ids[-keep_prompt:]
        response_ids = response_ids[:keep_response]
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "attention_mask": torch.tensor([[1] * len(input_ids)], dtype=torch.long),
    }


def sequence_logprob(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Length-normalized log-probability of the unmasked (response) tokens."""
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    logits = outputs.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~mask, 0)
    token_logps = torch.gather(
        F.log_softmax(logits.float(), dim=-1), -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    denom = mask.sum(dim=-1).clamp_min(1)
    return (token_logps * mask).sum(dim=-1) / denom


def dpo_pair_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    ref_chosen_logp: float,
    ref_rejected_logp: float,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """l_HPR = -log sigma(alpha * d_t), d_t = rho(a+, s) - rho(a-, s)."""
    pi_margin = chosen_logp - rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp
    loss = -F.logsigmoid(beta * (pi_margin - ref_margin)).mean()
    metrics = {
        "pref_loss": float(loss.detach().cpu()),
        "margin": float(pi_margin.detach().cpu()),
        "ref_margin": float(ref_margin),
    }
    return loss, metrics


def kto_scalar_loss(
    logp: torch.Tensor,
    ref_logp: float,
    y: int,
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """l_scalar = -log sigma(alpha * y * rho(a, s)), y in {+1, -1}."""
    sign = 1.0 if y > 0 else -1.0
    weight = desirable_weight if y > 0 else undesirable_weight
    loss = weight * -F.logsigmoid(beta * sign * (logp - ref_logp)).mean()
    metrics = {
        "kto_loss": float(loss.detach().cpu()),
        "y": float(sign),
        "logp_delta": float((logp - ref_logp).detach().cpu()),
    }
    return loss, metrics


def sft_loss(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Cross-entropy on the response tokens (context masked)."""
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        use_cache=False,
    ).loss
