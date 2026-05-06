# Presentation Scripts & Outputs

This document summarizes the four presentation tasks, code entry points, output paths, and caveats relative to the original prompt wording.

---

## 1. Output artifacts

| Artifact | Path |
|----------|------|
| Three-model qualitative grid (1×5) | `results/qualitative_1x5_grid.png` |
| Depth zero-tensor ablation (1×3)   | `results/depth_zero_ablation.png` |
| Cross-attention heatmap overlay    | `results/cross_attention_overlay.png` |
| Scenario mIoU values (CSV)         | `results/scenario_miou_scores.csv` |
| Scenario mIoU bar chart            | `results/scenario_miou_bar.png` |
| FC-CLIP / SED intermediate cache (optional) | `results/.baseline_cache/` |

Pretrained weights (if downloaded):

- `checkpoints/fcclip_convnext_large.pth`
- `checkpoints/sed_convnext_l.pth`

Download script: `download_baselines.py`

---

## 2. Environment

| Component | Notes |
|-----------|-------|
| **vireo** (conda) | Runs Vireo / mmseg inference, the qualitative grid, and the mIoU main process. The working directory must be `DGSS/vireo_repro/Vireo` so that relative paths like `open_vocab/cityscapes.json` resolve. |
| **ov_baselines** (conda) | Runs FC-CLIP and SED Detectron2 inference; invoked as a subprocess via `detectron2_batch_infer.py`. |
| **OV_BASELINE_PYTHON** | Points to the `python` binary inside the `ov_baselines` env. If unset, `run_all_presentations.py` auto-detects `conda …/envs/ov_baselines/bin/python`. |

For baseline env setup see `setup_baseline_env.sh` (PyTorch, Detectron2, FC-CLIP deformable-ops compilation, etc.).

---

## 3. One-shot regeneration

After exporting paths from any directory:

```bash
export OV_BASELINE_PYTHON=/path/to/miniconda3/envs/ov_baselines/bin/python   # if not auto-detected
cd /path/to/DGSS/vireo_repro/Vireo                                            # run_all sets cwd to Vireo at runtime
python /path/to/DGSS/cv_project/run_all_presentations.py
```

`run_all_presentations.py` picks the most idle GPU, sets `PYTHONPATH`, and runs the four scripts in sequence (qualitative: **20** images; mIoU: **50** images).

---

## 4. Four tasks vs. original prompts

### 4.1 Prompt 1 — Multi-model qualitative grid (1×5)

- **Implementation**: `qualitative_multimodel_grid.py`. Columns: RGB | GT | FC-CLIP | SED | Ours; Cityscapes palette; matplotlib; **300 DPI**.
- **Data**: ACDC `val` subsets `night` / `rain` / `fog` as adverse-weather urban samples (narrative fit for "hard cases"; not literal SVI tiles).
- **Baselines**: FC-CLIP and SED run via official code + Detectron2 + released checkpoints, **not** HuggingFace / MMSeg configs as written in the prompt; still genuine pretrained baselines.
- **Note**: Default `--num-samples` is **20**. If you ever rendered with fewer (e.g. 12), re-run with `--num-samples 20` to match the spec.

---

### 4.2 Prompt 2 — Depth zero-tensor ablation (1×3)

- **Script**: `depth_zero_ablation.py`.
- **Layout**: RGB | normal mask | depth-ablated mask (**1×3**).
- **Mechanism**: monkey-patch `backbone.vireo.forward_delta_feat` and pass `torch.zeros_like(depth_features)`.
- **Highlight**: edge map on disagreement regions; red overlay on the ablated visualization.
- **Export**: 300 DPI.

---

### 4.3 Prompt 3 — Cross-attention visualization

- **Script**: `extract_cross_attention_map.py`.
- **Pipeline**: `forward_hook` on a configurable submodule (default cross-attention path); normalize tensor and reshape to a heuristic H×W heatmap; JET colormap + `addWeighted` blend.
- **Important**: This is **not** a rigorous attention-weight extraction from MHSA; it visualizes module output reshaped to spatial maps. Safe wording: "response / activation heatmap (heuristic)". Per-class maps (e.g. car vs. sign) are **not** implemented; use `--query-index` to vary which query slot is emphasized.

---

### 4.4 Prompt 4 — Scenario mIoU

- **Script**: `scenario_miou_eval.py`.
- **Sample size**: default **50** images.
- **Models**: FC-CLIP, SED (subprocess), Vireo.
- **Metrics**: confusion → IoU; **Seen** = trainIds **0–9**, **Unseen** = **10–18**.
- **Outputs**: pandas CSV + matplotlib grouped bar chart with value labels (seaborn optional per prompt).
- **Not implemented**: Day-vs-Night stratification (the "or" branch in the prompt).
- **Invalid preds**: pixels with pred outside `[0, n_cls)` are masked out from metric accumulation (void / OOB handling).

---

## 5. FC-CLIP / SED inference notes

- **FC-CLIP**: config `third_party/fc-clip/.../fcclip_convnext_large_eval_cityscapes.yaml`; the working directory must be the fc-clip repo root (so `./fcclip/data/...` resolves); panoptic outputs are mapped to Cityscapes `trainId`.
- **SED**: `third_party/SED/configs/convnextL_768.yaml`; in `detectron2_batch_infer.py` we put SED's bundled `open_clip/src` at the front of `sys.path` so that `encode_image(..., dense=True)` resolves correctly; the test text categories live in `assets/cityscapes_ov_prompts.json` (19 classes, trainId-aligned).

---

## 6. Versioning & honest wording

- All figures are presentation-oriented; random seeds and subset splits can be tuned via script flags.
- For paper / defense rigor, fix `--seed`, record checkpoint paths, and pin conda versions.
- Prompt 3's heatmap and Prompt 4's small-sample mIoU should be labeled as **illustrative / subset evaluation** in slides — do not over-claim "full benchmark ranking".

---

*This README reflects the `cv_project` scripts as of the last update; it covers all four prompts, artifact paths, environment, reproduction, and deviations from literal prompt wording.*
