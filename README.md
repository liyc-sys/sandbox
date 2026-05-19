# 视觉 Benchmark 评测框架

这套目录当前负责视觉 benchmark 的本地数据准备、统一 `JSONL` 归一化、模型推理和基础打分报告。模型调用通过仓库外部的 `oneapi.py` 接入，评测结果默认写入 `assets/output/`。

## 当前保留的 9 个 Benchmark

- `mmmu`
- `realworldqa`
- `vlmsareblind`
- `hallusionbench`
- `omnidocbench`
- `chartqapro`
- `mathvista`
- `phyx_openended`
- `visulogic`

## 不再纳入当前计划

- `babyvision`
- `countbench`
- `refcoco_avg` 全量正式版

说明：`refcoco_avg` 目前只保留了小样本探测文件 `data/normalized/refcoco_avg/all8.sample_per_split_5.jsonl`，不再修默认数据路径，也不再生成全量 `all8.jsonl`。

## 不包含的内容

- 长上下文文档理解 benchmark
- `ArcAGI1-Image`
- 更严格的官方评测脚本复刻
- `babyvision`、`countbench`、`refcoco_avg` 的后续补全

## 目录结构

```text
evaluate_framework/
├── README.md
├── visual-benchmark-shortlist.md
├── configs/
│   └── benchmarks.json
├── docs/
│   └── benchmark_notes.md
├── schemas/
│   └── unified_sample.schema.json
├── scripts/
│   ├── prepare_benchmarks.py
│   ├── run_eval.py
│   └── validate_normalized.py
├── evaluator/
│   ├── inference/
│   ├── judges/
│   └── reports/
├── data/
│   ├── raw/
│   └── normalized/
├── assets/
│   ├── output/
│   └── log/
└── benchmarks/
```

## 统一输出格式

归一化后的文件放在：

- `data/normalized/<benchmark>/<split>.jsonl`

每一行都是一个统一样本，核心字段包括：

- `uid`
- `benchmark`
- `split`
- `task_type`
- `prompt`
- `media`
- `answer`
- `metadata`
- `source`

详细字段定义见 [schemas/unified_sample.schema.json](schemas/unified_sample.schema.json)。

## 推荐使用方式

准备单个 benchmark：

```bash
python3 scripts/prepare_benchmarks.py --benchmark realworldqa
```

准备全部 benchmark：

```bash
python3 scripts/prepare_benchmarks.py --all
```

只做结构探测，不落完整数据：

```bash
python3 scripts/prepare_benchmarks.py --benchmark mmmu --inspect-only
```

校验归一化结果：

```bash
python3 scripts/validate_normalized.py --all
```

运行 30 条样本的 smoke eval：

```bash
python3 scripts/run_eval.py --benchmark realworldqa --model doubao-seed-1-6-250615 --limit 30
```

评测结果会写入：

- 单 benchmark 报告：`assets/output/<benchmark>/<model>/report.json`
- 单样本明细：`assets/output/<benchmark>/<model>/report.per_sample.jsonl`
- 当前命令的汇总：`assets/output/summary_<model>.json`

注意：`summary_<model>.json` 是每次 `run_eval.py` 命令的汇总文件。如果后来只单独跑一个 benchmark，它会覆盖成这一次命令的结果；完整历史仍然以各 benchmark 目录下的 `report.json` 为准。

## 设计原则

- 尽量保留 benchmark 原始语义，不为了统一而过度扁平化
- 多图样本用 `media` 数组统一表达
- OCR / 文档解析 / grounding 这类非标准 QA 任务也保留在同一 schema 下
- 对字段不稳定的数据集，适配器显式写规则，不依赖隐式猜测
- 原始下载和归一化输出解耦，便于后续复跑

## 说明

- 这套脚本优先对接公开 Hugging Face 数据源。
- 某些 benchmark 的公开格式会变化，例如 config 名、split 名或字段名。如果未来上游更新，优先改 `configs/benchmarks.json` 和 `scripts/prepare_benchmarks.py` 里的适配器。
- 对于 `PhyX`、`OmniDocBench` 这类任务形态更特殊的数据，已经在 `docs/benchmark_notes.md` 里记录了当前选型和 caveat。
