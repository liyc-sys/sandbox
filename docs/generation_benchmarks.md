# Generation Benchmarks

这套 generation 流程和现有 QA 流程分开：`scripts/run_eval.py` 仍然负责回答类 benchmark；图片生成类 benchmark 走 `scripts/run_generation_eval.py`。

## 支持范围

- `dpg_bench`
- `geneval`
- `wise`
- `ueval`

实现参考了 TorchUMM 的目录约定，但不绑定 TorchUMM 的本地推理假设。公司电脑只能通过 HOPE 提交任务时，用 `manifest` / `hope_manifest` backend 先生成请求清单，再由 HOPE 工具执行并回填图片。

核心约束是推理和评测分离：

- `prepare`：只落 `samples.jsonl`、请求 JSON 和 `manifest.json`，不跑模型。
- `generate`：跑或接入生成模型，落 `predictions.jsonl` 和 benchmark-specific 输出布局，不跑 judge。
- `score` / `evaluate`：只跑 scorer 或读取 scorer artifacts；如果 run 目录已有 `samples.jsonl`，不会重新加载 HF 数据。
- `full`：本地一次性跑 `generate` + `score`，适合 smoke 或已具备本地 judge 环境的机器。

## 输出目录

```text
assets/output/generation/<benchmark>/<model>/<run_id>/
├── manifest.json
├── samples.jsonl
├── predictions.jsonl
├── generation_summary.json
├── report.json
├── raw_outputs/
└── benchmark-specific artifacts
```

各 benchmark 的兼容布局：

- DPG Bench：`images/<prompt_id>.png`，每个文件是 4 张图拼成的 2x2 grid；评分结果读 `final_score.json`。
- GenEval：`images/<00000>/metadata.jsonl` + `images/<00000>/samples/*.png`；评分结果读 `score_summary.json` 或 `results.jsonl`。
- WISE：`images/<prompt_id>.png`；评分结果读 `Results/*_scores.jsonl` 并汇总 WiScore。默认会查 `benchmarks/generation/wise/data/{cultural_common_sense.json,spatio-temporal_reasoning.json,natural_science.json}`；本地没有这些文件时，需要把 WISE 数据放到该目录，或改 `configs/generation_benchmarks.json` 里的 `data_root`。
- UEval：`model_outputs.json`；评分结果读 `eval_results.json`，并额外导出 `eval_summary.json`。默认先读 `benchmarks/generation/ueval/data`，该目录不存在或为空时才回退到 `zlab-princeton/UEval`。本地数据可用 `local_json`、`local_cache`、`local_data`、`local_source` 或 `dataset_cache` 指向 JSON / JSONL / CSV / Parquet、`test.json` 目录、`data/test.json` 目录或 `datasets.load_from_disk` 目录。

## 常用命令

只生成 HOPE/外部执行用请求清单：

```bash
python3 scripts/run_generation_eval.py --benchmark geneval --model bagel --phase prepare --backend hope_manifest --limit 5
```

全量只跑推理，不跑 judge：

```bash
python3 scripts/run_generation_eval.py --benchmark wise --model your_model --phase generate --backend command --backend-option command='your_generator --prompt "{prompt}" --out {output_dir}'
```

全量只跑评测，复用同一个 `run_id` 下已经生成好的图片 / `model_outputs.json`：

```bash
python3 scripts/run_generation_eval.py --benchmark wise --model your_model --phase score --run-id <run_id> --scorer torchumm --scorer-option torchumm_root=/private/tmp/TorchUMM
```

本地 smoke，不依赖真实生成模型：

```bash
python3 scripts/run_generation_eval.py --benchmark dpg_bench --model smoke --phase full --backend placeholder --limit 2
```

把已有图片目录纳入评测产物：

```bash
python3 scripts/run_generation_eval.py --benchmark wise --model your_model --phase full --backend existing --backend-option source_root=/path/to/generated/images
```

`command` backend 适合直接包一层你自己的生成器。命令执行后，框架会优先读取 `response.json` / `output.json` / `{uid}.json`，里面可以放 `text_answer`、`image_paths`、`images` 或 `image_answer`，然后自动复制图片并写回 `predictions.jsonl`。

评分阶段如果需要外部官方 scorer：

```bash
python3 scripts/run_generation_eval.py --benchmark geneval --model bagel --phase score --run-id <run_id> --scorer-command 'python /path/to/scorer.py --run-dir {run_dir}'
```

## UEval 本地数据

把 UEval 数据落到默认目录：

```bash
./.venv/bin/python scripts/sync_ueval_data.py --source zlab-princeton/UEval
```

如果 HF 在公司机不可达，可以先在可联网环境下载成 JSON / JSONL / Parquet 或 `datasets.save_to_disk()` 目录，再同步到默认布局：

```bash
./.venv/bin/python scripts/sync_ueval_data.py --source /path/to/UEval --output-dir benchmarks/generation/ueval/data
```

也可以完全不改配置，直接在命令行指定数据源：

```bash
python3 scripts/run_generation_eval.py --benchmark ueval --model bagel --phase generate --backend hope_manifest --benchmark-option local_json=/path/to/ueval_test.json
python3 scripts/run_generation_eval.py --benchmark ueval --model bagel --phase score --run-id <run_id> --scorer torchumm --benchmark-option local_json=/path/to/ueval_test.json
```

UEval 的 TorchUMM judge 是 Qwen 风格本地 scorer：文本 rubric 走 `Qwen/Qwen3-32B`，图像 rubric 走 `Qwen/Qwen2.5-VL-72B-Instruct`。这和官方 UEval README 里提到的 Gemini evaluator 不是同一个 judge；这里为了对齐 TorchUMM / Qwen 报告口径，默认采用 TorchUMM 的 Qwen scorer。

## HOPE 接入边界

当前代码不会假设 HOPE 的具体 CLI/API，因为公司环境的提交方式通常是内部工具。框架负责稳定地产生：

- `samples.jsonl`：benchmark 样本
- `*.request.json`：单样本请求
- `manifest.json`：本次运行清单
- benchmark-specific 输出目录：HOPE 回填图片后可直接 score

后续只需要把 HOPE 提交、轮询、下载结果封装成一个新的 backend，或用 `command` backend 包一层已有 HOPE CLI。
