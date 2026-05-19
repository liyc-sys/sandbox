# Benchmark 数据源说明

## 总体说明

这份文档记录当前 12 个 benchmark 的数据源、选型理由和已知 caveat。评测主体如果后续由 `claudecode` 接入，优先直接消费 `data/normalized/` 下的统一 `JSONL`，不要直接耦合原始数据格式。

## 每个 Benchmark 的说明

### `mmmu`

- 当前数据源：`MMMU/MMMU`
- 理由：官方公开版本，字段稳定
- 说明：`MMMU` 在 Hugging Face 上是 30 个学科 config，不是单一配置；脚本会按 config 逐个拉取并合并成统一输出
- 当前默认 split：`validation`

### `realworldqa`

- 当前数据源：`xai-org/RealworldQA`
- 理由：报告中常见引用版本，HF viewer 字段清楚
- 当前默认 split：`test`
- 主要字段：`image` / `question` / `answer`

### `vlmsareblind`

- 当前数据源：`XAI/vlmsareblind`
- 当前默认 split：`valid`
- 主要字段：`task` / `image` / `prompt` / `groundtruth` / `metadata`

### `hallusionbench`

- 当前数据源：`rayguan/HallusionBench`
- 理由：官方 benchmark repo
- 说明：这份数据在 HF 上更像文件仓库，不一定适合直接 `load_dataset`
- 当前策略：用 `snapshot_download` 下载注释 JSON 和图像压缩包，再解析到统一格式

### `omnidocbench`

- 当前数据源：`opendatalab/OmniDocBench`
- 说明：这是文档解析 benchmark，不是普通 QA
- 当前策略：用固定文档解析指令作为 `prompt`，把页面图像和结构化标注一起转入统一格式
- 注意：这一项的 `answer` 会是 `structured` 类型，而不是纯文本答案

### `chartqapro`

- 当前数据源：`ahmed-masry/ChartQAPro`
- 当前默认 split：`test`
- 说明：图表类能力建议单独保留，不和普通 VQA 合并

### `mathvista`

- 当前数据源：`AI4Math/MathVista`
- 当前默认 split：`testmini`
- 理由：`testmini` 更适合作为本地框架调试起点，后续可无缝切到 `test`
- 说明：同时支持 `multi_choice` 和 `free_form`

### `phyx_openended`

- 当前数据源：`Cloudriver/PhyX`
- 当前默认 split：`test_mini`
- 说明：报告里写的是 `PhyX (openended)`，但 HF 上公开配置可能不是直接按 openended / mc 明确拆开
- 当前策略：优先保留所有和 open-ended 评测相关的文字字段；如果拿到的是多选字段，也一并保留

### `babyvision`

- 当前数据源：`UnipatAI/BabyVision`
- 当前默认 split：`train`
- 说明：公开数据是视觉推理题，字段包含 `question`、`options`、`choiceAns`、`blankAns`
- 当前策略：有 `choiceAns` 的样本按多选题处理，答案从 0-based index 映射到 A/B/C；否则按填空文本 exact match 处理

### `countbench`

- 当前数据源：`vikhyatk/CountBenchQA`
- 当前默认 split：`test`
- 说明：这是 CountBench 的 QA 版本，字段包含 `image`、`text`、`question`、`number`
- 当前策略：保留 caption 到 metadata，模型 prompt 使用 question，并要求输出单个整数；judge 用整数计数 accuracy

### `refcoco_avg`

- 当前数据源：`lmms-lab/RefCOCO`
- 理由：便于直接接 VLM 评测
- 注意：这不是最原始的 `RefCOCO/+/g` 官方打包格式，而是已经面向评测整理过的版本
- 当前策略：保留 `bbox`、`segmentation` 和 `answer` 里的多参考描述

### `visulogic`

- 当前 sample 数据源：`lscpku/VisuLogic`
- 原始 benchmark 仓：`VisuLogic/VisuLogic`
- 说明：原始仓当前公开 viewer 只暴露 `image`，不适合作为直接评测输入；sample 阶段改接 `lscpku/VisuLogic`
- 当前 viewer 可见字段：`image`、`question`、`label`、`id`、`tag`
- 当前策略：先用镜像源落 sample，后续如果要做正式全量版，再决定是否切回官方源或 project page 提供的完整数据

## 当前已验证的字段探测结果

以下 benchmark 已经完成一次 `inspect-only` 验证，字段结构基本清楚：

- `realworldqa`
- `mathvista`
- `vlmsareblind`
- `chartqapro`
- `phyx_openended`

其中：

- `chartqapro` 的图像字段是原始 `bytes`，脚本已兼容
- `mathvista` 使用 `decoded_image`
- `vlmsareblind` 的 `metadata` 当前是字符串，不是结构化对象

## 当前有风险或待确认项

- `visulogic`
  已找到可用镜像源，准备落 sample
- `refcoco_avg`
  数据可下载，但体量较大，字段探测还在进行中
- `hallusionbench`
  当前按快照仓处理，后续要确认注释文件名和图像压缩包布局
- `omnidocbench`
  当前按快照仓处理，后续要确认图像目录和结构化标注字段

## 后续建议

- 如果后续主体需要更严格的官方评测脚本，优先在统一格式之外额外保留每个 benchmark 的原始下载目录
- 如果某个 benchmark 上游字段有变化，不要直接改评测主体，优先只改 `prepare_benchmarks.py` 里的适配器
- 如果后续要加入视频或音频 benchmark，建议另开 schema 版本，不要强行并入这版视觉 schema
