"""Frozen prompt-based feedback router R(x_t) (paper §3.1, Appendix B).

The deployable router is a frozen same-scale LLM (Qwen3-4B in the paper). It
returns the structured object of Table 7: feedback_type, label, critique,
hint, confidence. Local preference / correction are routed to the pairwise
branch only when the preferred behavior was inferable from the pre-feedback
context; newly revealed preferences and low-confidence decisions are neutral
for the current update. Delayed task success / final test failure is never
routed to the pairwise branch.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .llm import LLMBackend
from .types import FEEDBACK_TYPES, FeedbackInstance, RouterDecision

DEFAULT_CONFIDENCE_THRESHOLD = 0.65

ROUTER_SYSTEM_PROMPT = """\
You are a feedback router for multi-turn agent training.

Classify the feedback instance into exactly one label:

local_preference:
  The user is steering how the immediately preceding assistant response should
  be framed, ranked, ordered, worded, or selected. Use this label whenever the
  feedback clearly prefers one of several plausible realizations of the prior
  response and that preference is supported by earlier context. Be generous:
  if the message is a concrete preference about the prior response rather than
  a fresh request, prefer local_preference over neutral.

local_correction:
  The user is revising, fixing, or restating the immediately preceding
  assistant response or action. This includes explicit corrections, re-dos,
  reversals, replacements, and concrete follow-up changes to the prior turn.
  Look for signals such as "no", "actually", "that's wrong", "you missed",
  "I meant", "should have", "instead", or a direct contradiction of the
  previous response.

tool_api_outcome:
  The next state is a tool, API, shell, browser, or environment result. It is
  not a human preference unless the user explicitly critiques the response.

delayed_trajectory_outcome:
  The signal is final task success/failure, tests passed/failed, issue
  resolved, or completion of the whole trajectory. Never route this label to
  the pairwise branch: it judges the trajectory outcome without isolating a
  reliable local alternative.

neutral:
  The feedback is unrelated, truly ambiguous, or a newly introduced
  requirement that was not supported by prior context. Also label routine task
  progress as neutral: user confirmations, yes/no approvals, providing
  requested information, selecting one of the options requested by the
  assistant, saying thanks, or stopping the conversation are neutral unless
  they explicitly correct or critique the previous assistant response. A
  task-content choice is neutral when the assistant explicitly asked the user
  to choose, confirm, provide an ID, provide a reason, or provide a payment
  method.

Decision boundary:
  - "Yes, proceed", "I confirm", or "Use my card" after the assistant asks for
    confirmation is neutral.
  - "Use reservation QKRY03" after the assistant asks which reservation to
    modify is neutral, not local_preference.
  - "Choose option 2" or "I pick the first one" after the assistant asks the
    user to choose is neutral, not local_preference.
  - "I prefer the cheaper option" is neutral if cheapness was not previously
    stated or implied; otherwise route it as local_preference.
  - "I asked for the cheaper option; you showed expensive ones" is
    local_correction.
  - "Show cheaper options first, not fastest ones" is local_preference only
    when the earlier context already made price relevant.

For local_preference and local_correction also produce:
  - label: whether the logged response was desirable or undesirable under the
    feedback;
  - critique: one short sentence naming the local issue;
  - hint: a short training-only instruction that, prepended to the original
    context, would let the assistant produce the corrected response. The hint
    must only use information available in the pre-feedback context plus the
    feedback itself.

Return compact JSON only:
{"feedback_type": "...", "label": "desirable|undesirable|neutral",
 "critique": "...", "hint": "...", "confidence": 0.0-1.0, "rationale": "..."}
"""


def compact_text(text: str, limit: int = 2400) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2:]
    return head + "\n...[truncated]...\n" + tail


def render_context(context: List[Dict[str, Any]], max_messages: int = 12) -> str:
    lines = []
    for msg in (context or [])[-max_messages:]:
        role = msg.get("role", "unknown")
        content = msg.get("content")
        if isinstance(content, list):
            content = " ".join(
                str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content
            )
        lines.append("%s: %s" % (role, compact_text(str(content or ""), 1200)))
    return "\n".join(lines)


def instance_prompt(inst: FeedbackInstance) -> str:
    return (
        "Classify this feedback instance.\n\n"
        "Benchmark: %s\nDomain: %s\nCurrent split: %s\n\n"
        "Pre-feedback context:\n%s\n\n"
        "Previous assistant action:\n%s\n\n"
        "Tool calls in previous action:\n%s\n\n"
        "Next state role:\n%s\n\n"
        "Next state / feedback text:\n%s\n\n"
        "Important: classify only whether this feedback instance should route "
        "to the pairwise local branch, the scalar outcome branch, a tool/API "
        "outcome, or be excluded."
        % (
            inst.benchmark,
            inst.domain,
            inst.split,
            render_context(inst.context),
            compact_text(inst.action_text),
            compact_text(json.dumps(inst.action_tool_calls or [], ensure_ascii=False)),
            inst.feedback_role,
            compact_text(str(inst.feedback_text or "")),
        )
    )


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def decision_from_json(obj: Optional[Dict[str, Any]], router_model: str) -> RouterDecision:
    if not obj:
        return RouterDecision(
            feedback_type="neutral", rationale="parse_failed", router_model=router_model
        )
    label = str(obj.get("feedback_type") or "").strip()
    if label not in FEEDBACK_TYPES:
        label = "neutral"
    conf: Optional[float]
    try:
        conf = float(obj.get("confidence"))
    except (TypeError, ValueError):
        conf = None
    if conf is not None:
        conf = max(0.0, min(1.0, conf))
    return RouterDecision(
        feedback_type=label,
        label=str(obj.get("label") or "neutral"),
        critique=str(obj.get("critique") or "")[:800],
        hint=str(obj.get("hint") or "")[:800],
        confidence=conf,
        rationale=str(obj.get("rationale") or obj.get("reason") or "")[:800],
        router_model=router_model,
    )


def structural_decision(inst: FeedbackInstance) -> Optional[RouterDecision]:
    """Deterministic environment rules for structurally typed feedback.

    Tool results and final-outcome records carry their type in the log
    structure; the LLM router is only needed for natural-language feedback.
    """
    if inst.feedback_role == "tool":
        return RouterDecision(
            feedback_type="tool_api_outcome",
            label="desirable" if inst.outcome_desirable else "undesirable",
            confidence=1.0,
            rationale="structural: tool-role next state",
            router_model="rule",
        )
    if inst.feedback_role == "environment" and inst.outcome_desirable is not None:
        return RouterDecision(
            feedback_type="delayed_trajectory_outcome",
            label="desirable" if inst.outcome_desirable else "undesirable",
            confidence=1.0,
            rationale="structural: trajectory-level outcome record",
            router_model="rule",
        )
    return None


class FeedbackRouter:
    """Routes feedback instances with a frozen LLM plus structural rules."""

    def __init__(
        self,
        backend: LLMBackend,
        model_name: str = "qwen3-4b",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        preserve_structural_labels: bool = True,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.preserve_structural_labels = preserve_structural_labels

    def route(self, inst: FeedbackInstance) -> RouterDecision:
        if self.preserve_structural_labels:
            structural = structural_decision(inst)
            if structural is not None:
                return structural
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": instance_prompt(inst)},
        ]
        texts = self.backend.complete(messages, temperature=0.0, n=1)
        decision = decision_from_json(parse_json_object(texts[0]), self.model_name)
        return decision.gated(self.confidence_threshold)
