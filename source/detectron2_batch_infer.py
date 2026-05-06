#!/usr/bin/env python3
"""
Run FC-CLIP or SED (Detectron2) on a list of images; save Cityscapes trainId maps (H,W) int64 as .npy.

Requires a dedicated env with detectron2 + fc-clip / sed deps (see download_baselines.py / compile fc-clip ops).

Usage:
  python detectron2_batch_infer.py fcclip --weights PATH --images img1.png img2.png --out-dir OUT
  python detectron2_batch_infer.py sed --weights PATH --images img1.png img2.png --out-dir OUT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CV_PROJECT = Path(__file__).resolve().parent
FC_CLIP_ROOT = CV_PROJECT / "third_party/fc-clip"
SED_ROOT = CV_PROJECT / "third_party/SED"
FC_CLIP_CFG = FC_CLIP_ROOT / "configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_cityscapes.yaml"
SED_CFG = SED_ROOT / "configs/convnextL_768.yaml"
CITYSCAPES_PROMPTS = CV_PROJECT / "assets/cityscapes_ov_prompts.json"


def panoptic_to_train_ids(panoptic_seg, segments_info):
    """FC-CLIP panoptic ids -> Cityscapes trainId map (255 = ignore / empty)."""
    import numpy as np
    import torch

    if isinstance(panoptic_seg, torch.Tensor):
        pan = panoptic_seg.cpu().numpy().astype(np.int64)
    else:
        pan = np.asarray(panoptic_seg, dtype=np.int64)
    h, w = pan.shape
    sem = np.full((h, w), 255, dtype=np.int64)
    for s in segments_info:
        sid = int(s["id"])
        tid = int(s["category_id"])
        sem[pan == sid] = tid
    return sem


def run_fc_clip(weights: str, images: list[str], out_dir: Path, device: str) -> None:
    import os

    import numpy as np
    from detectron2.config import get_cfg
    from detectron2.data.detection_utils import read_image
    from detectron2.engine.defaults import DefaultPredictor
    from detectron2.projects.deeplab import add_deeplab_config

    # Dataset prompts use paths like `./fcclip/data/...` relative to fc-clip repo root (not demo/).
    os.chdir(FC_CLIP_ROOT)
    if str(FC_CLIP_ROOT) not in sys.path:
        sys.path.insert(0, str(FC_CLIP_ROOT))
    demo_dir = FC_CLIP_ROOT / "demo"
    if str(demo_dir) not in sys.path:
        sys.path.insert(0, str(demo_dir))

    import fcclip  # noqa: F401 — registers datasets
    from fcclip.config import add_fcclip_config, add_maskformer2_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_fcclip_config(cfg)
    cfg.merge_from_file(str(FC_CLIP_CFG))
    cfg.merge_from_list(
        [
            "MODEL.WEIGHTS",
            weights,
            "MODEL.DEVICE",
            device,
        ]
    )
    cfg.freeze()

    predictor = DefaultPredictor(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in images:
        img = read_image(path)
        outputs = predictor(img)
        if "panoptic_seg" in outputs:
            panoptic_seg, segments_info = outputs["panoptic_seg"]
            sem = panoptic_to_train_ids(panoptic_seg, segments_info)
        elif "sem_seg" in outputs:
            sem = outputs["sem_seg"].argmax(dim=0).cpu().numpy().astype(np.int64)
        else:
            raise RuntimeError(f"No panoptic_seg or sem_seg in outputs for {path}; keys={outputs.keys()}")

        stem = Path(path).stem
        np.save(out_dir / f"{stem}.npy", sem)


def run_sed(weights: str, images: list[str], out_dir: Path, device: str) -> None:
    import os

    import numpy as np
    from detectron2.config import get_cfg
    from detectron2.data.detection_utils import read_image
    from detectron2.engine.defaults import DefaultPredictor
    from detectron2.projects.deeplab import add_deeplab_config

    # SED training used a vendored `open_clip` with `encode_image(..., dense=True)`; shadow the PyPI package.
    sed_openclip_src = SED_ROOT / "open_clip" / "src"
    if str(sed_openclip_src) not in sys.path:
        sys.path.insert(0, str(sed_openclip_src))

    demo_dir = SED_ROOT / "demo"
    os.chdir(SED_ROOT)
    if str(SED_ROOT) not in sys.path:
        sys.path.insert(0, str(SED_ROOT))
    if str(demo_dir) not in sys.path:
        sys.path.insert(0, str(demo_dir))

    from sed import add_sed_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_sed_config(cfg)
    cfg.merge_from_file(str(SED_CFG))
    cfg.merge_from_list(
        [
            "MODEL.WEIGHTS",
            weights,
            "MODEL.DEVICE",
            device,
            "MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON",
            str(CITYSCAPES_PROMPTS.resolve()),
            "MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON",
            str((SED_ROOT / "datasets/coco.json").resolve()),
        ]
    )
    cfg.freeze()

    predictor = DefaultPredictor(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in images:
        img = read_image(path)
        outputs = predictor(img)
        if "sem_seg" not in outputs:
            raise RuntimeError(f"No sem_seg for {path}; keys={outputs.keys()}")
        sem = outputs["sem_seg"].argmax(dim=0).cpu().numpy().astype(np.int64)
        stem = Path(path).stem
        np.save(out_dir / f"{stem}.npy", sem)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("backend", choices=("fcclip", "sed"))
    p.add_argument("--weights", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    if args.backend == "fcclip":
        run_fc_clip(args.weights, args.images, args.out_dir, args.device)
    else:
        run_sed(args.weights, args.images, args.out_dir, args.device)

    print(f"[done] wrote npy under {args.out_dir}")


if __name__ == "__main__":
    main()
