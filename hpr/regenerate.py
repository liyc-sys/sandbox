"""Hindsight regeneration a_t^+ ~ pi(. | s_t (+) u_t) (paper §3.3).

The hint u_t is a retrospective, training-only signal: it is appended to the
generation context so a frozen same-scale generator can produce the response
the assistant should have given, but it is never included in the deployment
prompt or in the compiled artifact's context field. Candidates are sampled at
temperature 0.7 and an optional re-judge filter keeps a pair only when the
regenerated response actually satisfies the feedback.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .llm import LLMBackend
from .router import compact_text, parse_json_object, render_context
from .types import FeedbackInstance, RouterDecision

DEFAULT_TEMPERATURE = 0.7
DEFAULT_NUM_CANDIDATES = 4

HPR_SYSTEM_PROMPT = """\
You regenerate a corrected assistant response for hindsight preference training.

Given the prior conversation context, the original assistant response, and the
user's local feedback, write the response the assistant should have given at
that same turn. The new response must:
- satisfy the user's context-supported feedback;
- preserve all valid task-relevant content from the original response;
- avoid adding facts that were not available at that point;
- output only the revised assistant response, with no explanation.
"""

REJUDGE_SYSTEM_PROMPT = """\
You verify hindsight preference pairs for offline training.

Given the context, the user's local feedback, the original response, and a
candidate revised response, decide whether the candidate (a) satisfies the
feedback and (b) is at least as good as the original on the underlying task.
Return compact JSON only: {"accept": true|false, "score": 0.0-1.0,
"reason": "..."}
"""


def regeneration_prompt(inst: FeedbackInstance, decision: RouterDecision) -> str:
    hint_block = ""
    if decision.hint:
        hint_block = "Hindsight hint (training-only):\n%s\n\n" % compact_text(decision.hint, 800)
    return (
        "Prior context:\n%s\n\n"
        "Original assistant response:\n%s\n\n"
        "User feedback:\n%s\n\n"
        "%s"
        "Feedback type: %s\n\n"
        "Write only the revised assistant response."
        % (
            compact_text(render_context(inst.context, max_messages=10), 4500),
            compact_text(inst.action_text, 2500),
            compact_text(str(inst.feedback_text or ""), 2500),
            hint_block,
            decision.feedback_type,
        )
    )


def rejudge_prompt(inst: FeedbackInstance, candidate: str) -> str:
    return (
        "Context:\n%s\n\n"
        "User feedback:\n%s\n\n"
        "Original response:\n%s\n\n"
        "Candidate revised response:\n%s\n"
        % (
            compact_text(render_context(inst.context, max_messages=10), 3000),
            compact_text(str(inst.feedback_text or ""), 1500),
            compact_text(inst.action_text, 1500),
            compact_text(candidate, 2000),
        )
    )


def rejudge_score(obj: Optional[Dict[str, Any]]) -> float:
    if not obj or not obj.get("accept"):
        return 0.0
    try:
        return max(0.0, min(1.0, float(obj.get("score", 1.0))))
    except (TypeError, ValueError):
        return 1.0


class HindsightRegenerator:
    """Constructs a_t^+ candidates and optionally re-judges them."""

    def __init__(
        self,
        backend: LLMBackend,
        model_name: str = "qwen3-4b",
        temperature: float = DEFAULT_TEMPERATURE,
        num_candidates: int = DEFAULT_NUM_CANDIDATES,
        rejudge: bool = False,
        rejudge_backend: Optional[LLMBackend] = None,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.temperature = temperature
        self.num_candidates = max(1, num_candidates)
        self.rejudge = rejudge
        self.rejudge_backend = rejudge_backend or backend

    def candidates(self, inst: FeedbackInstance, decision: RouterDecision) -> List[str]:
        messages = [
            {"role": "system", "content": HPR_SYSTEM_PROMPT},
            {"role": "user", "content": regeneration_prompt(inst, decision)},
        ]
        texts = self.backend.complete(
            messages, temperature=self.temperature, n=self.num_candidates
        )
        cleaned = []
        for t in texts:
            t = re.sub(r"^\s*(assistant\s*:)?\s*", "", t, flags=re.IGNORECASE).strip()
            if t:
                cleaned.append(t)
        return cleaned

    def regenerate(
        self, inst: FeedbackInstance, decision: RouterDecision
    ) -> Optional[str]:
        """Return the accepted a_t^+, or None when no candidate survives."""
        cands = self.candidates(inst, decision)
        if not cands:
            return None
        if not self.rejudge:
            return cands[0]
        best_text, best_score = None, 0.0
        for cand in cands:
            messages = [
                {"role": "system", "content": REJUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": rejudge_prompt(inst, cand)},
            ]
            texts = self.rejudge_backend.complete(messages, temperature=0.0, n=1)
            score = rejudge_score(parse_json_object(texts[0]))
            if score > best_score:
                best_text, best_score = cand, score
        return best_text  # None when every candidate was rejected


def pair_record_debug(inst: FeedbackInstance, decision: RouterDecision, chosen: str) -> str:
    return json.dumps(
        {
            "instance_id": inst.instance_id,
            "feedback_type": decision.feedback_type,
            "chosen_chars": len(chosen or ""),
        },
        ensure_ascii=False,
    )
