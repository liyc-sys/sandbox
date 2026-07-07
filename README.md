# Offline Hindsight Preference Routing (HPR)

Companion code release for the paper **"Offline Hindsight Preference Routing
for OpenClaw-Style Agent Learning"** (under review).

HPR is a typed offline interface for next-state agent feedback. Each logged
feedback instance `x_t = (s_t, a_t, o_{t+1}, h_{<=t})` is routed by the
supervision it can validly support:

| Feedback instance                | Routed to                    | Artifact |
| -------------------------------- | ---------------------------- | -------- |
| context-supported preference     | pairwise HPR branch          | `(s_t, a_t^+, a_t)` |
| local correction                 | pairwise HPR branch          | `(s_t, a_t^+, a_t)` |
| local tool/API outcome           | scalar branch                | `(s_t, a_t, y_t)` |
| delayed trajectory outcome       | scalar branch                | `(s_t, a_t, y_t)` |
| newly revealed / neutral         | no update (logged only)      | neutral log |

Pairwise artifacts feed a DPO-style loss; scalar artifacts feed a KTO-style
loss; neutral logs never update the policy.

## Layout

```
hpr/
  types.py       feedback instance + artifact schemas (paper Table 6)
  llm.py         OpenAI-compatible backend + deterministic mock backend
  router.py      frozen prompt-based feedback router (paper §3.1, App. B)
  regenerate.py  hindsight regeneration a_t^+ ~ pi(. | s_t (+) u_t) (paper §3.3)
  compile.py     feedback instances -> pairwise/scalar/neutral artifacts
  losses.py      pairwise HPR loss + scalar KTO-style loss (paper §3.4, §3.5)
  train.py       offline routed trainer over compiled artifacts
  metrics.py     router diagnostics: type acc, local F1, delayed F1, delayed->HPR
scripts/
  route_feedback.py      CLI: label feedback instances with the router
  compile_artifacts.py   CLI: compile routed instances into artifacts
  train_routed.py        CLI: routed offline training from artifacts
  router_diagnostics.py  CLI: score router labels against gold labels
tests/
  test_pipeline.py       end-to-end smoke test (mock LLM + tiny random model)
examples/
  toy_feedback_instances.jsonl  six instances covering all five feedback types
benchmarks/
  README.md              data splits used in the paper: rebuild + verify
  data/split_manifest.json          canonical split definition (all settings)
  data/splits/tau3_reported_test_clean89_manifest.json
                          fixed 89-task tau3 evaluation set (ids listed)
  data/scripts/           materialize official data + split integrity check
```

## Method-to-code map

| Paper | Code |
| ----- | ---- |
| Eq. (1)-(2) router `R(x_t)` | `hpr/router.py` |
| §3.3 hindsight regeneration `a_t^+ ~ pi(. \| s_t (+) u_t)` | `hpr/regenerate.py` |
| §3.3 pair construction `D_pair = {(s_t, a_t^+, a_t)}` | `hpr/compile.py` |
| §3.4 scalar loss `-log sigma(alpha * y_t * rho_theta)` | `hpr/losses.py:kto_scalar_loss` |
| §3.5 pairwise loss `-log sigma(alpha * d_t)` | `hpr/losses.py:dpo_pair_loss` |
| §3.5 routed objective `L_route` | `hpr/train.py:RoutedTrainer` |
| Table 3 router diagnostics | `hpr/metrics.py` |
| Table 7 router output schema | `hpr/types.py:RouterDecision` |

Only response/action tokens of the routed instance are scored; context tokens
are masked (`hpr/losses.py:encode_with_response_mask`). The reference policy is
a frozen copy of the initial policy whose log-probabilities are precomputed
once before training (`hpr/train.py:precompute_reference`).

## Quickstart

Route, compile, and smoke-train on the bundled toy data without any API key or
GPU (uses the deterministic mock backend and a tiny random model):

```bash
python3 -m tests.test_pipeline
```

With a real OpenAI-compatible endpoint (router and regeneration default to the
same-scale Qwen3-4B backbone used in the paper):

```bash
export HPR_API_KEY=...
export HPR_BASE_URL=...   # OpenAI-compatible /v1 endpoint

python3 scripts/route_feedback.py \
  --input feedback_instances.jsonl \
  --output routed.jsonl \
  --provider openai --model qwen3-4b \
  --preserve-structural-labels

python3 scripts/compile_artifacts.py \
  --routed routed.jsonl \
  --output-dir artifacts/ \
  --provider openai --regen-model qwen3-4b

python3 scripts/train_routed.py \
  --model-path Qwen/Qwen3-4B \
  --artifacts-dir artifacts/ \
  --output-dir ckpt/hpr_routed

python3 scripts/router_diagnostics.py \
  --routed routed.jsonl --gold-field gold_feedback_type
```

## Default hyperparameters (paper Table 8)

| Parameter | Value |
| --------- | ----- |
| DPO / KTO beta (`alpha` in the paper) | 0.1 |
| KTO desirable / undesirable weights | 1.0 / 1.0 |
| learning rate | 1e-5 |
| training batch size (grad-accum micro-batches) | 16 |
| max sequence length | 8192 |
| hint regeneration temperature | 0.7 |
| hint candidates for rejection sampling | 4 |
| router confidence threshold | 0.65 |

## Notes

- The router, interpreter, and regeneration generator are *artifact
  constructors*, not teachers: the paper uses the same-scale Qwen3-4B backbone
  for all three and keeps them frozen across compared methods.
- Compiled artifacts are static JSONL records. They can be cached, audited,
  filtered, and reused across later updates without policy-versioned rollouts,
  old log-probabilities, or environment replay state.
- `Pairwise-Only` and `Scalar-Only` baselines from the paper are recovered by
  forcing all non-neutral types into one branch (see
  `scripts/compile_artifacts.py --force-branch`).
