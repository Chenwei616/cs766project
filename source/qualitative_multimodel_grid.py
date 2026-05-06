from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from presentation_common import (
    discover_acdc_scenarios,
    CV_PROJECT_ROOT,
    CachedDetectron2Baseline,
    REPO_ROOT,
    RESULTS_DIR,
    colorize_cityscapes,
    default_baseline_cache_dir,
    inference_label_map,
    load_mmseg_model,
    load_vireo_model,
    read_gt_label,
    read_rgb,
    resolve_baseline_python,
    resolve_pairs_from_acdc,
    resize_label_to_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description="1x5 qualitative grid.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--acdc-root",
        default=str(REPO_ROOT / "data/acdc"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["night", "rain", "fog", "snow"],
        help="Scenarios to sample from. Use 'all' to auto-discover available scenarios.",
    )
    parser.add_argument("--vireo-config", default=str(REPO_ROOT / "configs/dinov2_domain/vireo_dinov2_mask2former_512x512_bs1x4_citys.py"))
    parser.add_argument("--vireo-checkpoint", default=None)
    parser.add_argument("--fcclip-config", default=None, help="Unused if FC-CLIP Detectron2 weights are set.")
    parser.add_argument("--fcclip-checkpoint", default=None)
    parser.add_argument("--sed-config", default=None, help="Unused if SED Detectron2 weights are set.")
    parser.add_argument("--sed-checkpoint", default=None)
    parser.add_argument(
        "--baseline-python",
        default=None,
        help="Python with detectron2 + fc-clip/SED deps. Else OV_BASELINE_PYTHON or auto-detect.",
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "qualitative_1x5_grid.png"),
    )
    return parser.parse_args()


def _pick_ckpt(arg: str | None, default_path) -> str | None:
    if arg:
        return arg
    p = Path(default_path)
    return str(p) if p.is_file() else None


def main():
    args = parse_args()
    np.random.seed(args.seed)

    scenarios = args.scenarios
    if len(scenarios) == 1 and scenarios[0].lower() == "all":
        scenarios = discover_acdc_scenarios(args.acdc_root, args.split)
        if not scenarios:
            raise RuntimeError(f"No scenarios found in {args.acdc_root}/rgb_anon for split={args.split}")

    # Pick at least one sample per scenario for diversity, then fill the rest if needed.
    scenario_pools = {}
    for scene in scenarios:
        scene_pairs = resolve_pairs_from_acdc(
            acdc_root=args.acdc_root,
            splits=[args.split],
            scenarios=[scene],
            limit=max(50, args.num_samples * 3),
        )
        if scene_pairs:
            scenario_pools[scene] = scene_pairs
    if not scenario_pools:
        raise RuntimeError("No image/GT pairs found for requested scenarios.")

    pairs = []
    for scene in scenarios:
        pool = scenario_pools.get(scene, [])
        if not pool:
            continue
        pick = pool[np.random.randint(0, len(pool))]
        pairs.append(pick)

    target_n = min(args.num_samples, sum(len(v) for v in scenario_pools.values()))
    if target_n > len(pairs):
        all_pairs = []
        for v in scenario_pools.values():
            all_pairs.extend(v)
        chosen = {p[0] for p in pairs}
        remain = [p for p in all_pairs if p[0] not in chosen]
        if remain:
            extra_n = min(target_n - len(pairs), len(remain))
            extra_idx = np.random.choice(len(remain), extra_n, replace=False)
            pairs.extend([remain[i] for i in extra_idx])

    vireo_model = load_vireo_model(args.vireo_config, args.vireo_checkpoint, args.device)
    baseline_py = resolve_baseline_python(args.baseline_python)

    fc_ckpt = _pick_ckpt(args.fcclip_checkpoint, CV_PROJECT_ROOT / "checkpoints/fcclip_convnext_large.pth")
    if fc_ckpt and baseline_py:
        fcclip_model = CachedDetectron2Baseline(
            "fcclip",
            fc_ckpt,
            baseline_py,
            default_baseline_cache_dir("fcclip", fc_ckpt),
            device=args.device,
        )
        fcclip_model.prepare([p[0] for p in pairs])
        print(f"[info] FC-CLIP Detectron2 weights: {fc_ckpt}")
    elif args.fcclip_config and args.fcclip_checkpoint:
        fcclip_model = load_mmseg_model(args.fcclip_config, args.fcclip_checkpoint, args.device)
    else:
        if fc_ckpt and not baseline_py:
            print("[warn] FC-CLIP weights found but detectron2 env missing; use --baseline-python or OV_BASELINE_PYTHON.")
        else:
            print("[warn] FC-CLIP checkpoint missing; using Vireo as proxy.")
        fcclip_model = vireo_model

    sed_ckpt = _pick_ckpt(args.sed_checkpoint, CV_PROJECT_ROOT / "checkpoints/sed_convnext_l.pth")
    if sed_ckpt and baseline_py:
        sed_model = CachedDetectron2Baseline(
            "sed",
            sed_ckpt,
            baseline_py,
            default_baseline_cache_dir("sed", sed_ckpt),
            device=args.device,
        )
        sed_model.prepare([p[0] for p in pairs])
        print(f"[info] SED Detectron2 weights: {sed_ckpt}")
    elif args.sed_config and args.sed_checkpoint:
        sed_model = load_mmseg_model(args.sed_config, args.sed_checkpoint, args.device)
    else:
        if sed_ckpt and not baseline_py:
            print("[warn] SED weights found but detectron2 env missing; use --baseline-python or OV_BASELINE_PYTHON.")
        else:
            print("[warn] SED checkpoint missing; using Vireo as proxy.")
        sed_model = vireo_model

    n = len(pairs)
    fig, axes = plt.subplots(n, 5, figsize=(24, 3 * n), dpi=300)
    if n == 1:
        axes = axes[None, :]
    titles = ["RGB Input", "Ground Truth", "FC-CLIP Mask", "SED Mask", "Our Model Mask"]
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=12)

    for i, (img_path, gt_path, scene) in enumerate(pairs):
        rgb = read_rgb(img_path)
        h, w = rgb.shape[:2]
        gt = resize_label_to_image(read_gt_label(gt_path), h, w)
        fcclip_pred = resize_label_to_image(inference_label_map(fcclip_model, img_path), h, w)
        sed_pred = resize_label_to_image(inference_label_map(sed_model, img_path), h, w)
        vireo_pred = resize_label_to_image(inference_label_map(vireo_model, img_path), h, w)
        ims = [rgb, colorize_cityscapes(gt), colorize_cityscapes(fcclip_pred), colorize_cityscapes(sed_pred), colorize_cityscapes(vireo_pred)]
        for j, im in enumerate(ims):
            axes[i, j].imshow(im)
            axes[i, j].axis("off")
        axes[i, 0].set_ylabel(f"{scene}\n#{i+1}", fontsize=10)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"[done] saved: {args.output}")


if __name__ == "__main__":
    main()
