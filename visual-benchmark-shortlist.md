# 视觉 Benchmark 推荐清单

## 范围

这份清单从我们前面讨论过的 `Qwen` 和 `Seed` 报告截图中，挑出 10 个适合做视觉大模型综合评测的 benchmark。

筛选约束如下：
- 尽量覆盖主要视觉能力类别
- 排除“长上下文文档理解”类 benchmark
- 不包含 `ArcAGI1-Image`
- 尽量减少能力重叠过高的 benchmark

## 推荐的 10 个 Benchmark

| Benchmark | 能力类别 | 选择原因 |
| --- | --- | --- |
| `MMMU` | 综合多模态推理 | 适合作为总盘 benchmark，覆盖学科广、任务类型杂，能反映模型整体视觉推理上限。 |
| `RealWorldQA` | 通用 VQA | 覆盖真实场景图片问答，比很多偏学术的数据集更接近日常使用场景。 |
| `VLMsAreBlind` | 感知与识别 | 适合检验模型是不是“真的看见了”，能暴露依赖语言先验猜答案的问题。 |
| `HallusionBench` | 幻觉与忠实性 | 用来测模型会不会凭空编造图中不存在的信息，这是视觉模型很关键的失效模式。 |
| `OmniDocBench 1.5` | 文档理解 | 相比纯 OCR benchmark，更接近完整文档理解，而不只是识别文字。 |
| `ChartQAPro` | 图表理解 | 图表理解和普通 VQA、文档 OCR 都不完全一样，值得单独覆盖。 |
| `MathVista` | 视觉数学推理 | 是视觉数学里非常核心的 benchmark，适合覆盖图形、数学表达和题目理解。 |
| `PhyX (openended)` | STEM 推理 | 用来补足“数学之外的科学推理”，避免整套 benchmark 过度偏向 VQA 和 OCR。 |
| `RefCOCO_avg` | 空间定位与指代 | 适合覆盖 spatial grounding，检验模型是否理解“哪个目标、哪个区域”。 |
| `VisuLogic` | 视觉谜题与抽象推理 | 用来补足抽象视觉逻辑能力，同时满足“不使用 `ArcAGI1-Image`”这个约束。 |

## 选择逻辑

这 10 个 benchmark 分别覆盖不同维度：

- `MMMU`：综合多模态能力总盘
- `RealWorldQA`：自然图像问答
- `VLMsAreBlind`：基础感知鲁棒性
- `HallusionBench`：视觉幻觉控制
- `OmniDocBench 1.5`：文档理解
- `ChartQAPro`：图表理解
- `MathVista`：视觉数学
- `PhyX (openended)`：科学 / STEM 推理
- `RefCOCO_avg`：空间 grounding
- `VisuLogic`：抽象视觉逻辑

## 为什么没有优先选这些

- `MMMU-Pro`：很有价值，但和 `MMMU` 重叠较高；在 10 个名额里保留一个总盘 benchmark 就够了。
- `SimpleVQA`、`MMStar`：都不错，但和 `RealWorldQA` 有一定重叠，所以把名额留给文档、空间和谜题类能力。
- `OCRBench` / `OCRBenchv2`：如果重点是 OCR，它们很适合；但如果目标是“综合视觉能力”，`OmniDocBench 1.5` 提供的文档级信号更完整。
- `DynaMath`：有价值，但和 `MathVista` 同属视觉数学；这里优先保留更核心的 `MathVista`。
- `CountBench`：如果你特别关心 counting，可以纳入；这里只是优先了覆盖更广的 `RefCOCO_avg`。
- `DUDE`、`MMLongBench`、`LongDocURL`、`MMLongBench-Doc`：属于长上下文文档理解，这次按要求排除，不视为核心视觉 benchmark。
- `ArcAGI1-Image`：按你的要求明确排除。

## 可选替换方案

如果你更看重 OCR，而不是 STEM：
- 用 `OCRBenchv2` 替换 `PhyX (openended)`

如果你更看重 counting，而不是抽象视觉逻辑：
- 用 `CountBench` 替换 `VisuLogic`

如果你更希望和两份报告的公共 benchmark 重合更多：
- 可以用 `SimpleVQA` 或 `MMStar` 替换 `OmniDocBench 1.5` 或 `VisuLogic`
