import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmseg.apis import inference_model
from mmseg.utils import get_palette

from mmengine.runner.checkpoint import _load_checkpoint
from vireo.utils import init_model as init_vireo_model

from project_paths import get_vireo_root

# This package: presentation / evaluation scripts only (workspace root).
CV_PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = CV_PROJECT_ROOT / "results"

# Vireo repo: configs, data, checkpoints (read-only for these scripts). Override with VIREO_REPO_ROOT.
REPO_ROOT = get_vireo_root()
DEFAULT_VIREO_BACKBONE = REPO_ROOT / "checkpoints/dinov2_converted_depth.pth"
CITYSCAPES_PALETTE = np.asarray(get_palette("cityscapes"), dtype=np.uint8)


def _merge_converted_backbone_into_state_dict(state_dict: dict, backbone_weight_path: str) -> None:
    """Match `tools/visualize.py`: merge converted DINO weights under `backbone.*`."""
    converted = _load_checkpoint(backbone_weight_path, map_location="cpu")
    if isinstance(converted, dict) and "state_dict" in converted:
        converted = converted["state_dict"]
    prefix = {f"backbone.{k}": v for k, v in converted.items()}
    if "state_dict" in state_dict:
        state_dict["state_dict"].update(prefix)
    else:
        state_dict.update(prefix)


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda:0", help="Torch device.")
    parser.add_argument(
        "--vireo-config",
        default=str(
            REPO_ROOT / "configs/dinov2_domain/vireo_dinov2_mask2former_512x512_bs1x4_citys.py"
        ),
        help="Vireo config path.",
    )
    parser.add_argument(
        "--vireo-checkpoint",
        default=None,
        help="Vireo checkpoint path. If missing, auto-discovery is used.",
    )


def _discover_vireo_checkpoint(root: Path) -> Optional[Path]:
    candidates = sorted(
        (root / "work_dirs").glob("**/iter_*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_vireo_model(
    config_path: str,
    checkpoint_path: Optional[str],
    device: str,
    backbone_weight_path: Optional[str] = None,
) -> torch.nn.Module:
    ckpt = checkpoint_path
    if ckpt is None:
        guessed = _discover_vireo_checkpoint(REPO_ROOT)
        if guessed is None:
            raise FileNotFoundError("No Vireo iter_*.pth checkpoint found in work_dirs.")
        ckpt = str(guessed)
        print(f"[info] Auto-discovered Vireo checkpoint: {ckpt}")

    cfg = Config.fromfile(config_path)
    if "test_pipeline" not in cfg:
        cfg.test_pipeline = [
            dict(type="LoadImageFromFile"),
            dict(type="Resize", scale=(2048, 1024), keep_ratio=True),
            dict(type="PackSegInputs"),
        ]
    model = init_vireo_model(cfg, ckpt, device=device)

    # Training checkpoints often omit full DINO backbone; official inference merges
    # `checkpoints/dinov2_converted_depth.pth` (see `tools/visualize.py`). Without this,
    # predictions look like random multi-class noise.
    bb_path = backbone_weight_path
    if bb_path is None and DEFAULT_VIREO_BACKBONE.is_file():
        bb_path = str(DEFAULT_VIREO_BACKBONE)
    if bb_path is not None and Path(bb_path).is_file():
        sd = model.state_dict()
        _merge_converted_backbone_into_state_dict(sd, bb_path)
        model.load_state_dict(sd, strict=False)
        print(f"[info] Merged converted backbone weights: {bb_path}")
    elif bb_path is not None:
        print(f"[warn] Backbone file missing, skip merge: {bb_path}")

    model.eval()
    return model


def load_mmseg_model(config_path: str, checkpoint_path: str, device: str):
    from mmseg.apis import init_model

    model = init_model(config_path, checkpoint_path, device=device)
    model.eval()
    return model


def inference_label_map(model, image_path: str) -> np.ndarray:
    predict_seg = getattr(model, "predict_seg", None)
    if callable(predict_seg):
        return predict_seg(image_path)
    result = inference_model(model, image_path)
    pred = result.pred_sem_seg.data.squeeze(0).detach().cpu().numpy().astype(np.int64)
    return pred


def resolve_baseline_python(cli_path: Optional[str]) -> Optional[str]:
    """Python interpreter that can import detectron2 (for FC-CLIP / SED subprocess)."""
    if cli_path:
        return cli_path
    env = os.environ.get("OV_BASELINE_PYTHON")
    if env:
        return env
    try:
        subprocess.run(
            [sys.executable, "-c", "import detectron2"],
            check=True,
            capture_output=True,
        )
        return sys.executable
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class CachedDetectron2Baseline:
    """Caches per-image .npy trainId maps via detectron2_batch_infer.py subprocess."""

    def __init__(
        self,
        backend: str,
        weights: str,
        python_exe: str,
        cache_dir: Path,
        device: str = "cuda:0",
    ):
        self.backend = backend
        self.weights = str(Path(weights).resolve())
        self.python_exe = python_exe
        self.cache_dir = cache_dir
        self.device = device
        self.script = CV_PROJECT_ROOT / "detectron2_batch_infer.py"

    def prepare(self, image_paths: Sequence[str]) -> None:
        missing: List[str] = []
        for p in image_paths:
            stem = Path(p).stem
            if not (self.cache_dir / f"{stem}.npy").is_file():
                missing.append(str(Path(p).resolve()))
        if not missing:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.python_exe,
            str(self.script),
            self.backend,
            "--weights",
            self.weights,
            "--out-dir",
            str(self.cache_dir.resolve()),
            "--device",
            self.device,
            "--images",
            *missing,
        ]
        print("[run]", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(CV_PROJECT_ROOT))

    def predict_seg(self, image_path: str) -> np.ndarray:
        stem = Path(image_path).stem
        path = self.cache_dir / f"{stem}.npy"
        if not path.is_file():
            self.prepare([image_path])
        return np.load(str(path))


def default_baseline_cache_dir(backend: str, weights: str) -> Path:
    return RESULTS_DIR / ".baseline_cache" / backend / Path(weights).stem


def colorize_cityscapes(label_map: np.ndarray) -> np.ndarray:
    n = len(CITYSCAPES_PALETTE)
    safe = np.where((label_map < 0) | (label_map >= n), 0, label_map)
    label = np.clip(safe, 0, n - 1)
    return CITYSCAPES_PALETTE[label]


def read_rgb(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def read_gt_label(path: str) -> np.ndarray:
    gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if gt is None:
        raise FileNotFoundError(f"Cannot read GT mask: {path}")
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    return gt.astype(np.int64)


def resize_label_to_image(label_map: np.ndarray, h: int, w: int) -> np.ndarray:
    if label_map.shape[0] == h and label_map.shape[1] == w:
        return label_map
    return cv2.resize(label_map.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)


def resolve_pairs_from_acdc(
    acdc_root: str,
    splits: Sequence[str],
    scenarios: Sequence[str],
    limit: int,
) -> List[Tuple[str, str, str]]:
    root = Path(acdc_root)
    out: List[Tuple[str, str, str]] = []
    for split in splits:
        for scene in scenarios:
            img_dir = root / "rgb_anon" / scene / split
            gt_dir = root / "gt" / scene / split
            if not img_dir.exists() or not gt_dir.exists():
                continue
            for img_path in sorted(img_dir.rglob("*_rgb_anon.png")):
                rel = img_path.relative_to(img_dir)
                gt_name = img_path.name.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
                gt_path = (gt_dir / rel).with_name(gt_name)
                if gt_path.exists():
                    out.append((str(img_path), str(gt_path), scene))
                if len(out) >= limit:
                    return out
    return out[:limit]


def discover_acdc_scenarios(acdc_root: str, split: str = "val") -> List[str]:
    root = Path(acdc_root) / "rgb_anon"
    if not root.exists():
        return []
    scenarios: List[str] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if (d / split).exists():
            scenarios.append(d.name)
    return scenarios


def compute_confusion_matrix(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    pred_in_range = (pred >= 0) & (pred < num_classes)
    valid = (gt != ignore_index) & pred_in_range
    gt_v = gt[valid]
    pred_v = pred[valid]
    flat = gt_v * num_classes + pred_v
    hist = np.bincount(flat, minlength=num_classes * num_classes)
    return hist.reshape(num_classes, num_classes)


def iou_from_confusion(conf: np.ndarray) -> np.ndarray:
    diag = np.diag(conf).astype(np.float64)
    denom = conf.sum(axis=1) + conf.sum(axis=0) - diag
    iou = np.divide(diag, np.maximum(denom, 1.0), where=denom > 0)
    return iou
