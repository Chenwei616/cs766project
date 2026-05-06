# Presentation Scripts & Outputs — 演示脚本与产出说明

本文档为中英文双语，汇总 Vireo 展示相关的四项任务、代码入口、产出路径及与原始需求对照时的注意事项。

This document is bilingual (Chinese / English). It summarizes the four presentation tasks, code entry points, output paths, and caveats relative to the original prompt wording.

---

## 1. 产出文件一览 | Output artifacts

| 内容 | 路径 |
|------|------|
| 三模型定性对比（1×5） | `results/qualitative_1x5_grid.png` |
| Depth 零张量消融（1×3） | `results/depth_zero_ablation.png` |
| Cross-attention 叠加热力图 | `results/cross_attention_overlay.png` |
| Scenario mIoU 数值表 | `results/scenario_miou_scores.csv` |
| Scenario mIoU 柱状图 | `results/scenario_miou_bar.png` |
| FC-CLIP / SED 中间缓存（可选） | `results/.baseline_cache/` |

预训练权重（若已下载）| Pretrained weights (if downloaded):

- `checkpoints/fcclip_convnext_large.pth`
- `checkpoints/sed_convnext_l.pth`

下载脚本 | Download script: `download_baselines.py`

---

## 2. 环境与依赖 | Environment

| 组件 | 说明 |
|------|------|
| **vireo**（conda） | 运行 Vireo / mmseg 推理、定性 grid、mIoU 主进程；工作目录需为 `DGSS/vireo_repro/Vireo`（以便解析 `open_vocab/cityscapes.json` 等相对路径）。 |
| **ov_baselines**（conda） | 运行 FC-CLIP、SED 的 Detectron2 推理；通过子进程调用 `detectron2_batch_infer.py`。 |
| **OV_BASELINE_PYTHON** | 指向 `ov_baselines` 环境的 `python`；或在 `run_all_presentations.py` 中自动探测 `conda …/envs/ov_baselines/bin/python`。 |

Baseline 环境安装可参考 | For baseline env setup see: `setup_baseline_env.sh`（含 PyTorch、Detectron2、FC-CLIP deformable ops 编译说明等）。

---

## 3. 一键复现 | One-shot regeneration

在项目根（或从任意目录）设置好路径后：

```bash
export OV_BASELINE_PYTHON=/path/to/miniconda3/envs/ov_baselines/bin/python   # 若未自动探测
cd /path/to/DGSS/vireo_repro/Vireo   # 实际运行时 run_all 会将 cwd 设为 Vireo
python /path/to/DGSS/cv_project/run_all_presentations.py
```

`run_all_presentations.py` 会：自动选择较空闲 GPU、设置 `PYTHONPATH`、依次调用四个脚本（定性 **20** 张、mIoU **50** 张）。

---

## 4. 四项任务与原始需求对照 | Four tasks vs. original prompts

### 4.1 Prompt 1 — 多模型定性（1×5）

**中文**

- **实现**：`qualitative_multimodel_grid.py`。列为 RGB \| GT \| FC-CLIP \| SED \| Our Model；Cityscapes 调色板；matplotlib；**300 DPI**。
- **数据**：从 ACDC `val` 中抽取 `night` / `rain` / `fog`，作为**恶劣天气 / 城市场景**展示子集（叙述上可对应「极端场景 / SVI 代理」，并非字面 Google Street View）。
- **Baseline**：FC-CLIP、SED 使用**官方仓库 + Detectron2 + 预训练权重**（`detectron2_batch_infer.py`），**不是** HuggingFace 或 MMSegmentation 字面配置；功能上仍为真实预训练对比。
- **注意**：默认 `--num-samples` 为 **20**；若曾用手工参数（例如 12）生成图，行数可能少于 20，需与 `run_all_presentations.py` 一致再跑一次以严格对齐「20 张」表述。

**English**

- **Implementation**: `qualitative_multimodel_grid.py`. Columns: RGB \| GT \| FC-CLIP \| SED \| Ours; Cityscapes palette; matplotlib; **300 DPI**.
- **Data**: ACDC `val` subsets `night` / `rain` / `fog` as **adverse-weather urban** samples (narrative fit for “hard cases”; not literal SVI tiles).
- **Baselines**: FC-CLIP and SED run via **official code + Detectron2 + released checkpoints**, **not** Hugging Face / MMSeg configs as written in the prompt; still genuine pretrained baselines.
- **Note**: Default `--num-samples` is **20**. If you ever rendered with fewer (e.g. 12), re-run with `--num-samples 20` to match the spec.

---

### 4.2 Prompt 2 — Depth 零张量消融（1×3）

**中文**

- **实现**：`depth_zero_ablation.py`。
- **布局**：默认从候选池自动挑选差异最明显的若干样本（`--topk`，默认 3），每行 RGB \| Normal Inference Mask \| Depth-Ablated Mask。
- **机制**：对 `backbone.vireo.forward_delta_feat` 打补丁，将 `depth_features` 替换为 `torch.zeros_like(depth_features)`。
- **高亮**：对 normal 与 ablated 差异区域做**填充 + 边缘**红色叠加，确保视觉上可见差异。
- **导出**：300 DPI。

**English**

- **Script**: `depth_zero_ablation.py`.
- **Layout**: RGB \| normal mask \| depth-ablated mask (**1×3**).
- **Mechanism**: monkey-patch `backbone.vireo.forward_delta_feat` and pass `torch.zeros_like(depth_features)`.
- **Highlight**: edge map on disagreement regions; red overlay on the ablated visualization.
- **Export**: 300 DPI.

---

### 4.3 Prompt 3 — Cross-attention 可视化

**中文**

- **实现**：`extract_cross_attention_map.py`。
- **流程**：hook 到 `decode_head`，提取最后一层 decoder 的 `mask_pred`（按 query）生成热力图；支持 `--query-index` 指定 query，或 `--pick-best` 从候选池自动选“峰值最明显”的样本；**COLORMAP_JET** + `addWeighted` 叠加。
- **说明**：该图是 decoder query 的空间响应热力图（比旧版“flatten 激活重排”更可解释），但仍不等同于逐 head 的完整 attention matrix 可视化。

**English**

- **Script**: `extract_cross_attention_map.py`.
- **Pipeline**: `forward_hook` on a configurable submodule (default cross-attention path); normalize tensor and reshape to a **heuristic** H×W heatmap; **JET** colormap + **`addWeighted`** blend.
- **Important**: This is **not** a rigorous attention-weight extraction from MHSA; it visualizes **module output** reshaped to spatial maps—safe wording: “response / activation heatmap (heuristic)”. Per-class maps (e.g. car vs. sign) are **not** implemented; use `--query-index` to vary which query slot is emphasized.

---

### 4.4 Prompt 4 — Scenario mIoU

**中文**

- **实现**：`scenario_miou_eval.py`。
- **样本数**：默认 **50** 张（与 prompt 一致可调）。
- **三模型**：FC-CLIP、SED（Detectron2 子进程）、Our Model（Vireo）。
- **指标**：混淆矩阵 → per-class IoU；默认导出 **Overall mIoU**（更直观的模型对比）。可通过 `--grouping seen_unseen` 切换为 Seen（0–9）/Unseen（10–18）分组。
- **图表**：pandas 输出 CSV；**matplotlib** 分组柱状图（prompt 允许 seaborn **或** matplotlib）；柱顶标注数值。
- **未实现**：按 **Day vs. Night** 分组（prompt 中「或」的替代方案）；若需要需扩展脚本。
- **无效预测**：GT 仍为 255 忽略；预测中 **255 或越界类别**在统计前剔除，避免混淆矩阵溢出（见 `presentation_common.compute_confusion_matrix`）。

**English**

- **Script**: `scenario_miou_eval.py`.
- **Sample size**: default **50** images.
- **Models**: FC-CLIP, SED (subprocess), Vireo.
- **Metrics**: confusion → IoU; **Seen** = trainIds **0–9**, **Unseen** = **10–18**.
- **Outputs**: pandas CSV + **matplotlib** grouped bar chart with value labels (seaborn optional per prompt).
- **Not implemented**: **Day vs. Night** stratification (the “or” branch in the prompt).
- **Invalid preds**: pixels with pred outside `[0, n_cls)` are masked out from metric accumulation (void / OOB handling).

---

## 5. FC-CLIP / SED 推理细节摘要 | FC-CLIP / SED inference notes

- **FC-CLIP**：配置 `third_party/fc-clip/.../fcclip_convnext_large_eval_cityscapes.yaml`；工作目录需为 **fc-clip 仓库根目录**（保证 `./fcclip/data/...` 相对路径）；panoptic 输出映射为 Cityscapes **trainId**。
- **SED**：`third_party/SED/configs/convnextL_768.yaml`；在 `detectron2_batch_infer.py` 中将 **SED 内置 `open_clip/src` 置于 `sys.path` 最前**，以支持 `encode_image(..., dense=True)`；测试文本类别见 `assets/cityscapes_ov_prompts.json`（19 类，与 trainId 对齐）。

---

## 6. 版本与诚信表述建议 | Versioning & honest wording

- 所有图为**展示导向**生成；随机种子、子集划分可在脚本参数中调整。
- 若论文/答辩需严格可复现性，请固定 `--seed`、记录 checkpoint 路径与 conda 版本。
- Prompt 3 热力图、Prompt 4 小样本 mIoU 均建议在片中标注为 **illustrative / subset evaluation**，避免过度声称「完整 benchmark 排名」。

---

*文档随 `cv_project` 脚本更新；最后涵盖范围：四项 prompt、产物路径、环境与复现、与字面需求的差异说明。*

*This README reflects the `cv_project` scripts as of the last update; it covers all four prompts, artifact paths, environment, reproduction, and deviations from literal prompt wording.*
