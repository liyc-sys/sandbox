"""Feedback instance and offline artifact schemas (paper §3.1, Table 6, Table 7).

A feedback instance is x_t = (s_t, a_t, o_{t+1}, h_{<=t}). The router maps it
to one of five feedback types; the compiler turns routed instances into one of
three offline artifact records.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

FEEDBACK_TYPES = [
    "local_preference",
    "local_correction",
    "tool_api_outcome",
    "delayed_trajectory_outcome",
    "neutral",
]

LOCAL_TYPES = {"local_preference", "local_correction"}
OUTCOME_TYPES = {"tool_api_outcome", "delayed_trajectory_outcome"}
NON_NEUTRAL_TYPES = LOCAL_TYPES | OUTCOME_TYPES


def stable_id(*parts: Any, prefix: str = "") -> str:
    raw = "\n".join(str(p) for p in parts)
    return prefix + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class FeedbackInstance:
    """One logged next-state observation attached to the action it followed.

    `context` holds the pre-feedback context s_t as a list of chat messages
    ({"role": ..., "content": ...}); `action_text` is the logged response or
    action a_t; `feedback_text` is the next-state observation o_{t+1}.
    """

    instance_id: str
    context: List[Dict[str, Any]]
    action_text: str
    feedback_text: str
    feedback_role: str = "user"  # user | tool | environment
    action_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    benchmark: str = ""
    domain: str = ""
    split: str = ""
    task_id: str = ""
    trajectory_id: str = ""
    turn_id: Optional[int] = None
    # Optional outcome signal for scalar routing: True desirable, False not.
    outcome_desirable: Optional[bool] = None
    # Offline reference label for router diagnostics only (never trains).
    gold_feedback_type: Optional[str] = None

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "FeedbackInstance":
        known = {f: row.get(f) for f in cls.__dataclass_fields__ if f in row}
        known.setdefault("instance_id", stable_id(json.dumps(row, sort_keys=True)))
        return cls(**known)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RouterDecision:
    """Router output object (paper Table 7)."""

    feedback_type: str
    label: str = "neutral"  # desirable | undesirable | neutral
    critique: str = ""
    hint: str = ""  # training-only instruction u_t for a_t^+ generation
    confidence: Optional[float] = None
    rationale: str = ""
    router_model: str = ""

    def gated(self, threshold: float) -> "RouterDecision":
        """Low-confidence decisions are treated as neutral for the current update."""
        if (
            self.feedback_type != "neutral"
            and self.confidence is not None
            and self.confidence < threshold
        ):
            return RouterDecision(
                feedback_type="neutral",
                label="neutral",
                critique=self.critique,
                hint="",
                confidence=self.confidence,
                rationale="below_confidence_threshold: " + self.rationale,
                router_model=self.router_model,
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PairwiseArtifact:
    """Pairwise HPR artifact: (s_t, a_t^+, a_t) with provenance (Table 6)."""

    pair_id: str
    instance_id: str
    context: List[Dict[str, Any]]
    chosen: str  # hindsight-improved response a_t^+
    rejected: str  # original response a_t
    feedback_text: str
    feedback_type: str
    critique: str = ""
    hint: str = ""
    router_confidence: Optional[float] = None
    generator_model: str = ""
    construction: str = "hindsight_regenerated"
    benchmark: str = ""
    domain: str = ""
    split: str = ""
    task_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScalarArtifact:
    """Scalar outcome artifact: (s_t, a_t, y_t), y_t in {+1, -1} (paper §3.4)."""

    artifact_id: str
    instance_id: str
    context: List[Dict[str, Any]]
    response: str  # logged response or action a_t
    y: int  # +1 desirable, -1 undesirable
    feedback_type: str
    feedback_text: str = ""
    router_confidence: Optional[float] = None
    benchmark: str = ""
    domain: str = ""
    split: str = ""
    task_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NeutralLog:
    """Neutral / unsupported feedback: retained for audit, never trains."""

    log_id: str
    instance_id: str
    context: List[Dict[str, Any]]
    response: str
    feedback_text: str
    reason: str = ""
    router_confidence: Optional[float] = None
    benchmark: str = ""
    domain: str = ""
    split: str = ""
    task_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
