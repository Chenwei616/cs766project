# Data & checkpoints policy — 数据与权重说明

## English

This repository **does not redistribute**:

1. **Training / evaluation images or annotations** (e.g. ACDC adverse-weather dataset, Cityscapes-style labels used in our pipeline). Users must obtain datasets from their **official publishers**, accept the respective licenses, and place them under the paths expected by the Vireo config (see main README).

2. **Model checkpoints** (Vireo weights, DINO/DepthAnything conversions, FC-CLIP / SED pretrained weights). Users must either:
   - **Train** the model following the Vireo repository instructions, or  
   - **Download** publicly released weights from the original projects (FC-CLIP: authors’ Google Drive; SED: authors’ Google Drive) and place files according to `cv_project/download_baselines.py` / script defaults.

We provide **only source code and evaluation/visualization scripts** so that results remain reproducible *in principle*; **full reproduction requires GPUs, correct dependency versions, and user-supplied data and weights.**

---

## 中文

本代码发布 **不包含**：

1. **数据集**（如 ACDC 恶劣天气数据、与配置一致的标注）。请从 **官方渠道** 自行下载、遵守许可，并按 Vireo 配置中的路径放置。

2. **模型权重**（Vireo 训练 checkpoint、DINO/Depth 转换权重、FC-CLIP / SED 官方预训练权重）。使用者需 **自行训练**，或从 **原作者公布的链接**（如 Google Drive）下载后放入脚本默认或自定义路径。

我们仅提供 **源码与评测/可视化脚本**，使工作在形式上可复现；**实际跑通需 GPU、匹配的环境版本，以及用户自备的数据与权重。**
