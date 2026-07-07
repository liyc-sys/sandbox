"""End-to-end smoke test: route -> compile -> routed training -> diagnostics.

Runs fully offline: the router and regenerator use the deterministic mock
backend, and training uses a tiny randomly initialized GPT-2 with a toy
tokenizer. Verifies the pipeline shape, the loss wiring, and the router
diagnostics — not model quality.

Usage: python3 -m tests.test_pipeline
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpr.compile import ArtifactCompiler
from hpr.llm import MockBackend
from hpr.metrics import router_diagnostics
from hpr.regenerate import HindsightRegenerator
from hpr.router import FeedbackRouter
from hpr.train import RoutedTrainer, TrainConfig, samples_from_artifacts
from hpr.types import FeedbackInstance

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "toy_feedback_instances.jsonl"


class ToyTokenizer:
    """Word-hash tokenizer implementing the minimal protocol the trainer needs."""

    eos_token = "<eos>"
    pad_token = "<pad>"

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size

    def __call__(self, text, add_special_tokens: bool = False):
        ids = [2 + (hash(w) % (self.vocab_size - 2)) for w in str(text).split()]
        return SimpleNamespace(input_ids=ids)


def load_instances():
    rows = []
    with EXAMPLES.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return [FeedbackInstance.from_dict(r) for r in rows]


def main() -> int:
    instances = load_instances()
    assert len(instances) == 6

    # 1. Route.
    router = FeedbackRouter(MockBackend(), model_name="mock-router")
    routed = [(inst, router.route(inst)) for inst in instances]
    by_id = {inst.instance_id: d.feedback_type for inst, d in routed}
    assert by_id["toy-001"] == "local_preference", by_id
    assert by_id["toy-002"] == "local_correction", by_id
    assert by_id["toy-003"] == "neutral", by_id
    assert by_id["toy-004"] == "tool_api_outcome", by_id
    assert by_id["toy-005"] == "delayed_trajectory_outcome", by_id
    assert by_id["toy-006"] == "neutral", by_id

    # 2. Router diagnostics against gold labels.
    diag = router_diagnostics(
        [(d.feedback_type, inst.gold_feedback_type) for inst, d in routed]
    )
    assert diag["n"] == 6
    assert diag["type_accuracy"] == 1.0, diag
    assert diag["local_f1"] == 1.0 and diag["delayed_f1"] == 1.0, diag
    assert diag["delayed_to_hpr"] == 0.0, diag

    # 3. Compile artifacts (mock hindsight regeneration, re-judge off).
    regenerator = HindsightRegenerator(
        MockBackend(), model_name="mock-regen", num_candidates=2
    )
    artifacts = ArtifactCompiler(regenerator=regenerator).compile(routed)
    counts = artifacts.counts()
    assert counts["pairwise"] == 2, counts
    assert counts["scalar"] == 2, counts
    assert counts["neutral"] == 2, counts
    assert all(a.chosen and a.rejected and a.chosen != a.rejected for a in artifacts.pairwise)
    assert all(s.y in (-1, 1) for s in artifacts.scalar)

    # Scalar-Only / Pairwise-Only baseline compilation paths.
    scalar_only = ArtifactCompiler(force_branch="scalar").compile(routed)
    assert scalar_only.counts()["pairwise"] == 0
    assert scalar_only.counts()["scalar"] == 4  # locals compressed to scalar labels
    pairwise_only = ArtifactCompiler(
        regenerator=regenerator, force_branch="pairwise"
    ).compile(routed)
    assert pairwise_only.counts()["pairwise"] == 4  # every non-neutral forced to pairs

    # 4. Routed training smoke run on a tiny random model.
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    tokenizer = ToyTokenizer()
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=256, n_positions=192, n_embd=32, n_layer=2, n_head=2)
    )
    samples = samples_from_artifacts(
        [a.to_dict() for a in artifacts.pairwise],
        [s.to_dict() for s in artifacts.scalar],
    )
    assert len(samples) == 4

    with tempfile.TemporaryDirectory() as tmp:
        config = TrainConfig(
            max_length=160, epochs=1, lr=1e-4, grad_accum=2, device="cpu", log_every=1
        )
        trainer = RoutedTrainer(model, tokenizer, config, Path(tmp))
        summary = trainer.train(samples)
        assert summary["optimizer_step"] >= 2, summary
        assert all(s.ref_chosen_logp is not None or s.ref_response_logp is not None for s in samples)
        events = [
            json.loads(line)
            for line in (Path(tmp) / "train_metrics.jsonl").read_text().splitlines()
        ]
        losses = [e["mean_loss"] for e in events if e["event"] == "train_progress"]
        assert losses and all(torch.isfinite(torch.tensor(v)) for v in losses), losses

    print(json.dumps({"router_diagnostics": diag, "artifact_counts": counts,
                      "train_optimizer_steps": summary["optimizer_step"],
                      "status": "ok"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
