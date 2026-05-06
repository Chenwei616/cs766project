# Data & checkpoints policy

This repository **does not redistribute**:

1. **Training / evaluation images or annotations** (e.g. ACDC adverse-weather dataset, Cityscapes-style labels used in our pipeline). Users must obtain datasets from their **official publishers**, accept the respective licenses, and place them under the paths expected by the Vireo config (see main README).

2. **Model checkpoints** (Vireo weights, DINO/DepthAnything conversions, FC-CLIP / SED pretrained weights). Users must either:
   - **Train** the model following the Vireo repository instructions, or  
   - **Download** publicly released weights from the original projects (FC-CLIP: authors’ Google Drive; SED: authors’ Google Drive) and place files according to `cv_project/download_baselines.py` / script defaults.

We provide **only source code and evaluation/visualization scripts** so that results remain reproducible *in principle*; **full reproduction requires GPUs, correct dependency versions, and user-supplied data and weights.**
