# Benchmark Judge Logic

这份文档整理当前 10 个视觉 benchmark 的官方评测方式，以及在本地统一评测框架里应如何落地 judge 逻辑。

目标不是复刻每个 benchmark 的完整官方 leaderboard 环境，而是明确：

- 每个 benchmark 的官方 judge / metric 是什么
- 你的本地评测主体应该输出什么
- 需要什么后处理、归一化、规则判分或额外 evaluator

---

## 全局规则（适用于除 OmniDocBench 外的所有 benchmark）

### 1. 所有 benchmark 必须有 LLM judge fallback

每个 benchmark 在主 judge（规则 / 数值 / ANLS / IoU 等）之外，都必须接入一个统一的 LLM judge 作为 fallback。

**fallback 触发条件（两种都触发）：**

1. 规则 judge **抽不出** 答案 — 例如多选题没抽到 A/B/C/D，开放题正则匹配失败
2. 规则 judge **判 wrong** — 即使抽到了答案，只要规则判定为错，也送 LLM 复核一次，由 LLM 给出最终判断

也就是说：**只有"规则抽到答案 且 规则判 correct"才直接采纳规则结果，其它情况一律 LLM 复核。**

这条策略的代价是 fallback 调用量上升，但能避免规则 judge 在归一化粒度过严时把语义正确的答案误判为错。

### 2. 报告里必须附 fallback rate

每个 benchmark 的最终报告除了主指标，还必须输出：

- `fallback_rate` = 走了 LLM judge 的样本数 / 总样本数
- `fallback_overturn_rate` = LLM 把规则原本判错的样本翻成对的比例（用来量化规则 judge 的低估幅度）

### 3. 例外：OmniDocBench

OmniDocBench 的主指标是 Text Edit Distance / Table TEDS / Formula CDM 这类连续分指标，本质上没有"抽不出 / 判 wrong"的概念，LLM judge 也不会更准。**OmniDocBench 不接 LLM fallback，fallback rate 永远为 0%。**

### 4. 每个 benchmark 只报告 1 个主指标

为了保持评测报告的可比性，每个 benchmark 只对外报告 1 个数值。其它子指标可以保留在内部日志里供 debug，但不进 headline。

各 benchmark 的"那 1 个指标"见各章节。

---

## 总览

| Benchmark | 任务类型 | 主指标（headline） | 主 judge 方式 | LLM fallback |
| --- | --- | --- | --- | --- |
| `MMMU` | 多模态综合推理 | Accuracy | 多选 exact match / 开放题规则解析 | ✅ |
| `RealWorldQA` | 自然图像 QA | Accuracy | 文本归一化 + exact match | ✅ |
| `VLMsAreBlind` | 感知 / 识别 | Accuracy | 文本归一化 + exact match | ✅ |
| `HallusionBench` | 幻觉 / 忠实性 | **Question Pair Accuracy (qAcc)** | yes/no 解析 + 按 question pair 聚合 | ✅ |
| `OmniDocBench` | 文档解析 | **Overall = [(1−TextED)×100 + TableTEDS + FormulaCDM] / 3** | 三个子指标各自计算后合成 | ❌（不适用） |
| `ChartQAPro` | 图表理解 | Accuracy（按题型混合 judge） | 数值容差 / ANLS / exact match | ✅ |
| `MathVista` | 视觉数学 | Accuracy | 答案抽取 + 类型化比较 | ✅ |
| `PhyX` | STEM / 物理推理 | Accuracy | 多选字母提取 / 数值规则匹配 | ✅ |
| `RefCOCO_avg` | 视觉 grounding | **Acc@0.5（IoU≥0.5 比例，8 split 平均）** | bbox 解析 + IoU 计算 | ✅ |
| `VisuLogic` | 视觉逻辑多选 | Accuracy | 选项字母提取 + 标签匹配 | ✅ |

---

## 1. MMMU

### 官方评测方式

官方 `main_parse_and_eval.py` 的核心流程是：

- 若题型是 `multiple-choice`
  - 先对模型原始输出做解析
  - 从输出中抽出一个选项
  - 再和标准答案比较
- 若题型是 `open`
  - 先调用 `parse_open_response`
  - 再做评测

官方入口代码明确区分：

- `parse_multi_choice_response(response, all_choices, index2ans)`
- `parse_open_response(response)`
- `evaluate(eval_samples)`

### 本地建议

按两段式实现：

1. `answer_extraction`
   - 多选：提取 `A/B/C/D...` 或直接匹配 option 文本
   - 开放题：抽取最终答案字符串
2. `judge`
   - 多选：标准化后 exact match
   - 开放题：规则归一化匹配

3. **LLM fallback**：抽取失败 OR 规则判 wrong → 送 LLM judge 复核

### 实现注意点

- `MMMU` 不是单一 split，而是多 subject config
- 不同题可能有多张图
- 本地框架里必须把"模型原始输出"和"解析后答案"分开保存
- 主指标：**Accuracy**（所有 subject 平均）

---

## 2. RealWorldQA

### 官方评测方式

`RealWorldQA` 本质是标准图像问答 benchmark，主指标是准确率。

- 输入：图像 + 问题
- 输出：文本答案
- 评测：预测答案和参考答案是否一致

### 本地建议

规则 judge 用轻量归一化即可：

- 小写化
- 去掉句末标点
- 去掉首尾空白
- 必要时去冠词和无意义前缀

然后做 exact match。

**LLM fallback**：归一化匹配失败 OR 判 wrong → LLM judge 复核（典型场景：模型答了同义但表述不同的答案）

### 实现注意点

- 同义答案多的情况，建议优先扩 alias 表，而不是无脑都丢给 LLM
- 主指标：**Accuracy**

---

## 3. VLMsAreBlind

### 官方评测方式

这类 benchmark 关注"模型是否真的看见了图像信息"，公开版本本质上仍然是监督答案比对，核心指标是准确率。

### 本地建议

judge 逻辑可以和 `RealWorldQA` 类似：

- 先文本归一化
- 再 exact match
- 规则失败 / 判 wrong → LLM fallback

### 实现注意点

- 某些题是非常短的标签型答案，归一化要谨慎，不要过度模糊匹配
- 主指标：**Accuracy**（按 task 子类的 accuracy 内部记录，但不进 headline）

---

## 4. HallusionBench

### 官方评测方式

官方 README 与 CVPR'24 paper 明确说明：

- benchmark 是 yes/no 问题
- `visual_input=1` 表示必须看图
- `visual_input=0` 表示纯文本也可答

官方给出了 5 个维度的 accuracy：

- `Question Pair Acc`（qAcc）
- `Figure Acc`（fAcc）
- `Easy Question Acc`
- `Hard Question Acc`
- `Question Acc`（aAcc，单题 accuracy）

### 主指标：Question Pair Accuracy (qAcc)

各家大模型 report 里报道 HallusionBench 时统一用 **Question Pair Accuracy**。例如官方 paper 中 GPT-4V 31.42% 这个 headline 数字就是 qAcc。

qAcc 是一个 consistency 测试：把同一 figure / 同一知识点下的多个变体问题打包成一个 pair，**该 pair 内所有题都答对才算这个 pair 对**，最后 pair 维度的 accuracy 就是 qAcc。

### 本地建议

判分流程：

1. **单题二分类 judge**
   - 模型输出规整成 `0/1`（0 = no，1 = yes）
   - 与 `gt_answer`（数据中已是 `0/1`）比较
   - LLM fallback：规则提取不到 yes/no OR 提到了但和 gt 不符 → LLM 复核
2. **pair 聚合**
   - 按 `set_id + figure_id + question_id` 关联同一 pair 的多题
   - pair 内全对 → 该 pair = 1，否则 = 0
3. **qAcc = mean(pair_correct)**

### 实现注意点

- `gt_answer` 是 `0/1` 字符串
- 单题 judge 阶段记录的 fallback rate 直接进报告
- aAcc / fAcc 等可在内部 log 保留，但不对外报告
- 主指标：**Question Pair Accuracy (qAcc)**

---

## 5. OmniDocBench

### 官方评测方式

OmniDocBench 是文档解析 benchmark，原生支持：

- End-to-end evaluation
- Layout detection
- Table recognition
- Formula recognition
- Text OCR

支持的指标包括：

- `Normalized Edit Distance` (NED)
- `BLEU`
- `METEOR`
- `TEDS`（Tree Edit Distance Similarity）
- `CDM`（Character Detection Matching）
- `COCODet`（mAP / mAR 等）

### 主指标：Overall（v1.5 公式）

```
Overall = [ (1 − TextEditDistance) × 100  +  TableTEDS  +  FormulaCDM ] / 3
```

- v1.5 版本 leaderboard 用这个单一数字排序，reading order 不进入 Overall
- 当前 leaderboard 上 Gemini-3-Flash 90.1% 用的就是 Overall

### 本地建议

需要单独实现 `OmniDocBenchJudge`，包含三件子事：

1. **Text** — 用模型输出的 markdown text 与 GT text 算 Normalized Edit Distance
2. **Table** — 解析模型输出中的 HTML/markdown 表，用 TEDS 算结构 + 内容相似度
3. **Formula** — 解析模型输出中的 LaTeX 公式，用 CDM 算

最后按公式合成 Overall。

### 实现注意点

- 当前 unified sample 里 `answer.structured` 已经保留了全标注（layout / OCR / table / formula）
- **不接 LLM fallback**：连续分指标本质上没有"抽不出 / 判 wrong"，LLM 也判不准
- 报告里 fallback rate 字段填 `0%` 或 `N/A`
- 这项不能和普通 VQA 共用 judge 接口
- 主指标：**Overall** 单值

---

## 6. ChartQAPro

### 官方评测方式

官方 `evaluate_predictions.py` 的 judge 分三类：

1. **数值题**
   - 先转 float
   - 若相对误差 `<= 5%`，判对
2. **文本题**
   - 用 `ANLS`
   - 阈值 `0.5`
3. **特定题型**
   - `Fact Checking`
   - `Multi Choice`
   - 年份类
   - 走 exact match

支持列表答案，对多元素答案逐项打分再平均。

### 本地建议

按官方逻辑直接实现：

- numeric → 容差 judge
- 文本 → ANLS
- `Fact Checking` / `Multi Choice` → exact match
- `Year == YES` → exact match

**LLM fallback**：抽不出有效预测 OR 任一规则判 wrong → LLM judge 复核

### 实现注意点

- 对 `Conversational` 题，官方对 `year_flags` 还做了特殊切片
- 不能简单地把所有题都当 exact match
- 主指标：**Accuracy**（按官方加权方式聚合的总分）

---

## 7. MathVista

### 官方评测方式

官方评测是"两阶段"：

1. 从模型完整回答里抽取最终答案
   - 规则优先
   - 不够时可用额外模型抽取
2. 按题型和答案类型判对错

官方 `calculate_score.py` 的核心逻辑：

- `question_type == multi_choice`
  - 如果输出是字母，映射到 choice 文本
  - 否则找最相似 choice
- `answer_type == integer`
  - 转整数后比较
- `answer_type == float`
  - 按 `precision` 四舍五入后比较
- `answer_type == list`
  - 直接字符串化比较

### 本地建议

本地拆成：

1. `extract_final_answer`
2. `normalize_by_answer_type`
3. `exact_match`
4. **LLM fallback**：规则抽不出 OR 类型化比较判 wrong → LLM judge 复核

### 实现注意点

- 不要直接拿原始长回答和参考答案比
- `MathVista` 的 judge 关键在"答案抽取"，规则抽不到时让 LLM 抽并判
- 主指标：**Accuracy**（按官方所有题型混合）

---

## 8. PhyX

### 官方评测方式

官方 README 明确写了两种 judge：

- `rule-based`
- `LLM-as-judge`

而且官方在 `VLMEvalKit` / `lmms-eval` 中都支持：

- `valid_type = STR` → 规则 judge
- `valid_type = LLM` → LLM 评审

### 本地建议

PhyX 本身的官方判分就是双轨设计，正好契合"规则 + LLM fallback"。

1. `rule judge`
   - 多选题：提取选项字母
   - 开放题：规则抽最终数值 / 短答案
2. `LLM fallback`（必走）
   - 规则抽取失败 OR 规则判 wrong → LLM judge

### 实现注意点

- 当前本地 sample 用的是 `Cloudriver/PhyX`
- 主指标：**Accuracy**

---

## 9. RefCOCO_avg

### 数据源切换说明

⚠️ **当前 normalized 出来的 `lmms-lab/RefCOCO` region captioning 版本作废**，需要切回传统 grounding 版本。

切换原因：各家 VLM technical report（Qwen-VL、InternVL、Kosmos-2、GPT-4V 等）报告 RefCOCO 时，统一是"输入 referring expression，输出 bbox，按 IoU 算 Acc@0.5"，不是 region captioning。

### 数据源

- 三个 dataset：**RefCOCO / RefCOCO+ / RefCOCOg**
- 8 个 split：
  - `refcoco`: val / testA / testB
  - `refcoco+`: val / testA / testB
  - `refcocog`: val / test

每条样本：

- 图像
- referring expression（自然语言描述某个区域）
- GT bbox `[x1, y1, x2, y2]`

### 主指标：Acc@0.5（8 split 平均）

```
Acc@0.5 = mean over 8 splits of  [ #(IoU(pred_bbox, gt_bbox) ≥ 0.5) / #samples ]
```

`refcoco_avg` 名字里的 `avg` 就是 8 个 split 上 Acc@0.5 的平均。

### Prompt 模板

InternVL 标准 prompt：

```
Please provide the bounding box coordinates of the region this sentence describes: <ref>{expression}</ref>
```

### 本地建议

判分流程：

1. **bbox 解析**
   - 从模型输出抽出 `[x1, y1, x2, y2]`
   - 兼容多种坐标格式（绝对像素 / 0-1000 归一化 / 0-1 归一化）
2. **IoU 计算**
   - IoU ≥ 0.5 → 1，否则 0
3. **LLM fallback**
   - bbox 解析不出来 OR IoU < 0.5 → 让 LLM 看原图 + 模型输出，判断模型描述的位置是否实际就是 GT 区域
   - 这一步是为了兜住"模型答了正确语义位置但坐标系或格式偏差"的情况

### 实现注意点

- 数据准备脚本 `prepare_benchmarks.py` 里的 RefCOCO adapter 需要重写
- `configs/benchmarks.json` 里 `dataset_id` 需要换成传统 grounding 数据源（lmms-lab 的另一份、或 jxu124/refcoco）
- schema 里 `answer.bbox` 需要正式启用
- 主指标：**Acc@0.5（8 split 平均）**

---

## 10. VisuLogic

### 官方评测方式

官方 `VisuLogic-Eval` 是典型多选题评测。关键不是直接拿整段输出比，而是从复杂回答中抽选项字母。

官方代码包含多种提取逻辑：

- 提取最后一个 `\boxed{...}`
- 提取 `<answer>...</answer>`
- 从自然语言尾句里找 `A/B/C/D`
- 必要时再走额外 judge

最终目标是把模型输出规整成一个选项字母，再做 accuracy 统计。

官方还会按 tag 做内部统计（Quantitative / Spatial / Positional / Attribute / Stylistic / Other），但 headline 还是单一 accuracy。

### 本地建议

1. `extract_option_letter`（多套规则尝试）
2. `compare_with_label`
3. **LLM fallback**：抽不到字母 OR 抽到但和 label 不符 → LLM judge 复核
4. `per_tag_accuracy` 内部记录，不进 headline

### 实现注意点

- 这类 benchmark 的核心是"答案抽取器"，不是直接 exact match 原文
- 模型输出长 CoT 时，必须保证最后答案抽取得稳
- 主指标：**Accuracy**

---

## 统一实现建议

### Judge API

建议给评测主体提供一个统一接口：

```python
judge(record, prediction, benchmark_name) -> JudgeResult
```

`JudgeResult` 字段：

- `is_correct`：bool（连续分指标用 None）
- `score`：float（连续分指标用具体数值）
- `parsed_prediction`：抽取后的答案
- `metric_name`：本次实际用的指标名
- `judge_mode`：取值见下
- `used_fallback`：bool — 是否走了 LLM fallback
- `fallback_overturned`：bool — LLM 是否把规则原本的"wrong"翻成"correct"
- `meta`：原始 raw 输出 / 中间产物

`judge_mode` 取值：

- `exact_match`
- `normalized_exact_match`
- `multiple_choice_parse`
- `numeric_tolerance`
- `anls`
- `iou_at_0.5`
- `structured_doc_eval`
- `llm_as_judge`（fallback 触发时）

### 报告字段

每个 benchmark 报告必出：

```
{
  "benchmark": "...",
  "main_metric_name": "...",     # 例如 "qAcc" / "Overall" / "Acc@0.5"
  "main_metric_value": 0.xxxx,
  "n_samples": N,
  "fallback_rate": 0.xxxx,        # OmniDocBench 永远 0
  "fallback_overturn_rate": 0.xx, # OmniDocBench 永远 0
}
```

---

## 参考来源

- `MMMU` 官方解析入口：`MMMU-Benchmark/MMMU`
- `MathVista` 官方 `evaluation/calculate_score.py` 与 `evaluation/extract_answer.py`
- `ChartQAPro` 官方 `evaluate_predictions.py`
- `PhyX` 官方 README 与 VLMEvalKit / lmms-eval 接入说明
- `RefCOCO` Acc@0.5 标准评测：InternVL 2.5 报告（arXiv:2412.05271）、Qwen-VL / Qwen2.5-VL 系列报告
- `VisuLogic-Eval` 官方评测代码
- `OmniDocBench` 官方 README + CVPR 2025 paper（v1.5 Overall 公式）
- `HallusionBench` 官方 README + CVPR'24 paper（qAcc 主指标）
