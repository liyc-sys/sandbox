"""Compile routed feedback instances into offline artifacts (paper §3.2-§3.4).

Routing policy (paper Table 1):
  local_preference / local_correction -> pairwise HPR artifact
      (requires hindsight regeneration of the chosen response);
  tool_api_outcome                   -> scalar artifact (locally attributable
      validity signal, no trustworthy chosen alternative);
  delayed_trajectory_outcome         -> scalar artifact;
  neutral                            -> neutral log (no policy update).

`force_branch` recovers the paper's single-objective baselines:
  "pairwise" forces every non-neutral instance through hindsight regeneration
  (Pairwise-Only stress test); "scalar" compiles every non-neutral instance
  into a scalar artifact (Scalar-Only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .regenerate import HindsightRegenerator
from .types import (
    LOCAL_TYPES,
    NON_NEUTRAL_TYPES,
    FeedbackInstance,
    NeutralLog,
    PairwiseArtifact,
    RouterDecision,
    ScalarArtifact,
    stable_id,
)


@dataclass
class CompiledArtifacts:
    pairwise: List[PairwiseArtifact] = field(default_factory=list)
    scalar: List[ScalarArtifact] = field(default_factory=list)
    neutral: List[NeutralLog] = field(default_factory=list)
    skipped: Dict[str, int] = field(default_factory=dict)

    def counts(self) -> Dict[str, int]:
        out = {
            "pairwise": len(self.pairwise),
            "scalar": len(self.scalar),
            "neutral": len(self.neutral),
        }
        out.update({"skipped_" + k: v for k, v in self.skipped.items()})
        return out


def scalar_label(inst: FeedbackInstance, decision: RouterDecision) -> Optional[int]:
    """y_t in {+1, -1}; None when no usable outcome signal exists."""
    if inst.outcome_desirable is not None:
        return 1 if inst.outcome_desirable else -1
    if decision.label == "desirable":
        return 1
    if decision.label == "undesirable":
        return -1
    return None


class ArtifactCompiler:
    def __init__(
        self,
        regenerator: Optional[HindsightRegenerator] = None,
        force_branch: Optional[str] = None,  # None | "pairwise" | "scalar"
    ) -> None:
        if force_branch not in (None, "pairwise", "scalar"):
            raise ValueError("force_branch must be None, 'pairwise', or 'scalar'")
        self.regenerator = regenerator
        self.force_branch = force_branch

    def compile(
        self, routed: List[Tuple[FeedbackInstance, RouterDecision]]
    ) -> CompiledArtifacts:
        out = CompiledArtifacts(
            skipped={"regeneration_failed": 0, "missing_outcome_label": 0, "empty_action": 0}
        )
        for inst, decision in routed:
            ftype = decision.feedback_type
            if ftype == "neutral" or ftype not in NON_NEUTRAL_TYPES:
                out.neutral.append(self._neutral(inst, decision, decision.rationale))
                continue
            if not (inst.action_text or "").strip():
                out.skipped["empty_action"] += 1
                continue

            branch = self.force_branch or (
                "pairwise" if ftype in LOCAL_TYPES else "scalar"
            )
            if branch == "pairwise":
                artifact = self._pairwise(inst, decision)
                if artifact is None:
                    out.skipped["regeneration_failed"] += 1
                    out.neutral.append(
                        self._neutral(inst, decision, "regeneration_failed")
                    )
                else:
                    out.pairwise.append(artifact)
            else:
                y = scalar_label(inst, decision)
                if y is None:
                    out.skipped["missing_outcome_label"] += 1
                    out.neutral.append(
                        self._neutral(inst, decision, "missing_outcome_label")
                    )
                    continue
                out.scalar.append(
                    ScalarArtifact(
                        artifact_id=stable_id(inst.instance_id, ftype, prefix="scal_"),
                        instance_id=inst.instance_id,
                        context=inst.context,
                        response=inst.action_text,
                        y=y,
                        feedback_type=ftype,
                        feedback_text=inst.feedback_text,
                        router_confidence=decision.confidence,
                        benchmark=inst.benchmark,
                        domain=inst.domain,
                        split=inst.split,
                        task_id=inst.task_id,
                    )
                )
        return out

    def _pairwise(
        self, inst: FeedbackInstance, decision: RouterDecision
    ) -> Optional[PairwiseArtifact]:
        if self.regenerator is None:
            return None
        chosen = self.regenerator.regenerate(inst, decision)
        if not chosen or chosen.strip() == inst.action_text.strip():
            return None
        return PairwiseArtifact(
            pair_id=stable_id(inst.instance_id, decision.feedback_type, prefix="hpr_"),
            instance_id=inst.instance_id,
            context=inst.context,
            chosen=chosen,
            rejected=inst.action_text,
            feedback_text=inst.feedback_text,
            feedback_type=decision.feedback_type,
            critique=decision.critique,
            hint=decision.hint,
            router_confidence=decision.confidence,
            generator_model=self.regenerator.model_name,
            benchmark=inst.benchmark,
            domain=inst.domain,
            split=inst.split,
            task_id=inst.task_id,
        )

    @staticmethod
    def _neutral(
        inst: FeedbackInstance, decision: RouterDecision, reason: str
    ) -> NeutralLog:
        return NeutralLog(
            log_id=stable_id(inst.instance_id, "neutral", prefix="neut_"),
            instance_id=inst.instance_id,
            context=inst.context,
            response=inst.action_text,
            feedback_text=inst.feedback_text,
            reason=reason,
            router_confidence=decision.confidence,
            benchmark=inst.benchmark,
            domain=inst.domain,
            split=inst.split,
            task_id=inst.task_id,
        )
