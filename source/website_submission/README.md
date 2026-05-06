# Presentation & Evaluation Code Release

This package contains **source code** for qualitative comparison, depth ablation visualization, attention overlays, and scenario mIoU plots used in our course project website and report.

---

## 1. What is included

| Component | Description |
|-----------|-------------|
| `presentation_common.py` | Shared loaders, Cityscapes coloring, confusion / IoU |
| `project_paths.py` | **`VIREO_REPO_ROOT`** resolution (no hard-coded machine paths) |
| `qualitative_multimodel_grid.py` | 1×5 grid vs FC-CLIP, SED, our model |
| `depth_zero_ablation.py` | Zero-depth branch ablation |
| `extract_cross_attention_map.py` | Hook-based heatmap overlay |
| `scenario_miou_eval.py` | Seen / Unseen mIoU + bar chart |
| `detectron2_batch_infer.py` | FC-CLIP / SED inference helpers |
| `run_all_presentations.py` | Orchestrates all figures |
| `download_baselines.py` | *Optional* helper to fetch **public** baseline URLs (run yourself) |
| `setup_baseline_env.sh` | Example conda recipe for Detectron2 baselines |
| `assets/` | Cityscapes open-vocab label JSON for SED |
| `third_party/` | Vendored pointers / clones / README — **see each `README`** |
| `docs/DATA_AND_CHECKPOINTS.md` | **Policy: no dataset / no checkpoints in zip** |

---

## 2. What is NOT included (deliberate exclusions)

We follow standard academic practice: **code release without hosting copyrighted datasets or multi‑GB weights.**

| Item | Policy |
|------|--------|
| **Datasets** | Not shipped. Obtain **ACDC**, **Cityscapes**, etc. from official sites; arrange paths to match Vireo configs. |
| **Checkpoints** | Not shipped. Train the model from the training repo **or** download authors’ public weights yourself. |
| **Baseline weights** | FC-CLIP / SED: use official Google Drive links referenced in upstream repos / `download_baselines.py`. |

**Teaching staff:** Full end-to-end reproduction requires GPU clusters, matching package versions, registered datasets, and trained or downloaded weights—we document steps but do not guarantee one-click reruns.

---

## 3. Layout expectations

Recommended monorepo layout (adjust `VIREO_REPO_ROOT` if different):

```text
your_workspace/
  vireo_repro/
    Vireo/                    # training repo: configs, vireo/, data/, work_dirs/, ...
  DGSS/
    cv_project/               # this package (presentation scripts)
```

Export:

```bash
export VIREO_REPO_ROOT=/absolute/path/to/Vireo
export PYTHONPATH="${VIREO_REPO_ROOT}:${PWD}/DGSS/cv_project:${PYTHONPATH}"
```

Run scripts with **cwd = `Vireo`** when configs use relative paths (e.g. `open_vocab/cityscapes.json`).

---

## 4. Quick sanity check (no data required)

```bash
python -c "from pathlib import Path; import sys; sys.path.insert(0,'.'); from project_paths import get_vireo_root; print(get_vireo_root())"
```

---

## 5. Building the downloadable zip

From this repo:

```bash
bash website_submission/scripts/pack_release.sh
```

Produces `cv_project_source_release.zip` next to `DGSS/` by default (custom path: first argument).  
The zip **excludes** `.git`, checkpoints, `.npy` caches, compiled `.so`, and large baseline binaries—while keeping **all `.py` / `.sh` / docs** so the archive looks complete.

---

## 6. License

Presentation scripts: **Apache-2.0** or project license as stated in the course repo (match your `LICENSE` file).  
Third-party code (FC-CLIP, SED, MMSeg) remains under **their** original licenses—do not remove upstream license headers.

---

## 7. Citation

Cite the original FC-CLIP, SED, MMSegmentation, and your Vireo technical report as required by the course.
