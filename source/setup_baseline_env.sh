#!/usr/bin/env bash
# One-time: create conda env "ov_baselines" with PyTorch 2.4 + cu124, detectron2, then
# install FC-CLIP and SED requirements and compile FC-CLIP MSDeformAttn CUDA ops.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FC_CLIP="${ROOT}/third_party/fc-clip"
SED="${ROOT}/third_party/SED"

conda create -n ov_baselines python=3.10 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ov_baselines

pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install "detectron2==0.4" -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu124/torch2.4/index.html \
  || pip install "git+https://github.com/facebookresearch/detectron2.git@v0.4"

pip install opencv-python-headless tqdm scipy ftfy regex einops timm pillow imageio matplotlib pandas numpy gdown open_clip_torch

pip install -r "${FC_CLIP}/requirements.txt"
pip install -r "${SED}/requirements.txt"

if [[ -n "${CUDA_HOME:-}" ]]; then
  pushd "${FC_CLIP}/fcclip/modeling/pixel_decoder/ops" >/dev/null
  bash make.sh
  popd >/dev/null
else
  echo "[warn] CUDA_HOME unset; FC-CLIP deformable ops not built. Export CUDA_HOME and re-run make.sh under fcclip/modeling/pixel_decoder/ops"
fi

echo "[done] conda activate ov_baselines && export OV_BASELINE_PYTHON=\$(which python)"
