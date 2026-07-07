# Benchmark Splits, Manifests, and Experiment Pipeline

This directory pins the exact data splits used in the paper and ships the
experiment pipeline scripts for the three evaluation settings. It contains id
manifests and scripts only; original benchmark data is not redistributed and
is rebuilt from the official sources.

## Contents

```
data/split_manifest.json           canonical split definition for all three settings
data/splits/tau3_reported_test_clean89_manifest.json
                                   fixed tau3 evaluation set (89 tasks, ids listed)
data/scripts/                      materialize + integrity + experiment pipeline (below)
```

## Splits

| Setting | Train/update | Dev | Reported eval |
| --- | ---: | ---: | ---: |
| OpenClaw/GSM8K (protocol-specified) | GSM8K indices 0-999 | 1000-1099 | 1100-1318 |
| tau3-bench (airline/retail/telecom) | 152 tasks | 26 tasks | 89 tasks (clean89) |
| SWE-bench Lite | 45 issues | 15 issues | 240 issues |

For each tau3 domain, the first 15% of official train task ids after string
sort are dev; the remaining official train ids are train updates; the official
test split is the reported-test pool. Train/dev/reported-test ids are disjoint
within every domain. SWE-bench Lite is sorted by `instance_id`: first 15 dev,
next 45 train, remaining 240 reported eval.

All reported tau3 comparisons use the fixed 89-task evaluation set defined in
`data/splits/tau3_reported_test_clean89_manifest.json`. The manifest lists the
included and excluded task ids per domain; the set is identical for every
compared method. See the paper appendix for the evaluation-set construction.

## Rebuild and verify

```bash
# 1. tau3-bench task files (clone the official repo first)
git clone https://github.com/sierra-research/tau2-bench vendor/tau2-bench
python3 data/scripts/materialize_tau3_bench.py

# 2. SWE-bench Lite splits (requires: pip install datasets)
python3 data/scripts/materialize_swe_lite.py

# 3. Verify counts, id uniqueness, and train/dev/eval disjointness
python3 data/scripts/check_split_integrity.py
```

The integrity check asserts the exact counts above (152/26/100 for tau3,
45/15/240/300 for SWE-bench Lite) and that the clean89 manifest is consistent
with the rebuilt official test split.

## Experiment pipeline

All scripts live in `data/scripts/` and read/write under `data/` by default;
LLM-calling stages use an OpenAI-compatible endpoint configured through
`HPR_API_KEY` / `HPR_BASE_URL` (router, regeneration) or `OPENAI_API_KEY` /
`OPENAI_BASE_URL` (rollout user simulator).

tau3-bench (main mixed-feedback setting), in execution order:

1. `run_tau3_real_rollouts.py` — run tool-user trajectories through the
   tau2-bench simulator for a given policy endpoint and split; writes raw
   `SimulationRun` records. The two `qwen_tool_*.jinja` chat templates are the
   vLLM templates used for the Qwen3-4B policy.
2. `build_tau3_routed_data.py` — decompose trajectories into feedback
   instances with structural gold labels.
3. `route_feedback_labels.py` — label instances with the deployable router
   (`--provider oneapi`) or copy rule labels (`--provider rule`); use
   `--preserve-structural-labels` to keep tool/final outcomes deterministic.
4. `extract_hpr_pairs_from_router.py` — pair candidates from routed local
   feedback (`make_tau3_oracle_hpr_pairs.py` for gold-label pairs,
   `extract_hpr_pairs_all_feedback.py` for the Pairwise-Only stress test).
5. `regenerate_hpr_pairs.py` — hindsight regeneration of chosen responses.
6. `train_tau3_routed_fullparam.py` — routed offline training (pairwise
   DPO-style + scalar KTO-style) from the compiled artifacts.
7. Evaluate the trained checkpoint with `run_tau3_real_rollouts.py` on the
   `reported_test` split, restricted to the clean89 manifest.

OpenClaw-style personal-agent setting:

- `build_openclaw_gsm8k_artifacts.py` — compile student/teacher interaction
  feedback into routed artifacts.
- `run_openclaw_gsm8k_eval.py` — simulator-scored personalization evaluation.

SWE-bench Lite delayed-outcome setting:

- `resplit_existing_swe_lite.py` — deterministic 15/45/240 dev/train/eval split.
- `build_swe_lite_artifacts.py` — compile rollout feedback into artifacts.
- `run_swe_lite_patch_generation.py` — generate patches with a policy endpoint.
- `judge_swe_lite_patches.py` — patch validity / resolution scoring.
