"""Offline Hindsight Preference Routing (HPR).

Typed offline interface for next-state agent feedback: local, context-supported
feedback becomes pairwise preference artifacts; delayed or aggregate outcomes
become scalar preference artifacts; unsupported feedback is logged only.
"""
from __future__ import annotations

from .types import (
    FEEDBACK_TYPES,
    LOCAL_TYPES,
    FeedbackInstance,
    NeutralLog,
    PairwiseArtifact,
    RouterDecision,
    ScalarArtifact,
)

__all__ = [
    "FEEDBACK_TYPES",
    "LOCAL_TYPES",
    "FeedbackInstance",
    "NeutralLog",
    "PairwiseArtifact",
    "RouterDecision",
    "ScalarArtifact",
]

__version__ = "1.0.0"
