from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from presentation_common import (
    CV_PROJECT_ROOT,
    CachedDetectron2Baseline,
    CITYSCAPES_PALETTE,
    REPO_ROOT,
    RESULTS_DIR,
    compute_confusion_matrix,
    discover_acdc_scenarios,
    default_baseline_cache_dir,
    inference_label_map,
    iou_from_confusion,
    load_mmseg_model,
    load_vireo_model,
    read_gt_label,
    resolve_baseline_python,
    resolve_pairs_from_acdc,
    resize_label_to_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Scenario mIoU eval.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--acdc-root", default=str(REPO_ROOT / "data/acdc"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--scenarios", nargs="+", default=["night", "rain", "fog", "snow"])
    parser.add_argument(
        "--grouping",
        choices=["overall", "seen_unseen"],
        default="overall",
        help="overall: one bar/model; seen_unseen: two bars/model by class split.",
    )
    parser.add_argument("--vireo-config", default=str(REPO_ROOT / "configs/dinov2_domain/vireo_dinov2_mask2former_512x512_bs1x4_citys.py"))
    parser.add_argument("--vireo-checkpoint", default=None)
    parser.add_argument("--fcclip-config", default=None)
    parser.add_argument("--fcclip-checkpoint", default=None)
    parser.add_argument("--sed-config", default=None)
    parser.add_argument("--sed-checkpoint", default=None)
    parser.add_argument("--baseline-python", default=None)
    parser.add_argument(
        "--output-csv",
        default=str(RESULTS_DIR / "scenario_miou_scores.csv"),
    )
    parser.add_argument(
        "--output-plot",
        default=str(RESULTS_DIR / "scenario_miou_bar.png"),
    )
    return parser.parse_args()


def _pick_ckpt(arg: str | None, default_path) -> str | None:
    if arg:
        return arg
    p = Path(default_path)
    return str(p) if p.is_file() else None


def eval_conf(model, pairs, ncls):
    conf = np.zeros((ncls, ncls), dtype=np.int64)
    for img_path, gt_path, _ in pairs:
        gt = read_gt_label(gt_path)
        pred = resize_label_to_image(inference_label_map(model, img_path), gt.shape[0], gt.shape[1])
        conf += compute_confusion_matrix(pred, gt, ncls)
    return conf


def main():
    args = parse_args()
    np.random.seed(args.seed)
    scenarios = args.scenarios
    if len(scenarios) == 1 and scenarios[0].lower() == "all":
        scenarios = discover_acdc_scenarios(args.acdc_root, args.split)
        if not scenarios:
            raise RuntimeError(f"No scenarios found in {args.acdc_root}/rgb_anon for split={args.split}")
    pairs = resolve_pairs_from_acdc(args.acdc_root, [args.split], scenarios, max(args.num_samples * 3, args.num_samples))
    pairs = [pairs[i] for i in np.random.choice(len(pairs), args.num_samples, replace=False)]

    ncls = len(CITYSCAPES_PALETTE)
    seen = list(range(10))
    unseen = list(range(10, ncls))
    vireo_model = load_vireo_model(args.vireo_config, args.vireo_checkpoint, args.device)
    baseline_py = resolve_baseline_python(args.baseline_python)

    fc_ckpt = _pick_ckpt(args.fcclip_checkpoint, CV_PROJECT_ROOT / "checkpoints/fcclip_convnext_large.pth")
    if fc_ckpt and baseline_py:
        fc_model = CachedDetectron2Baseline(
            "fcclip",
            fc_ckpt,
            baseline_py,
            default_baseline_cache_dir("fcclip", fc_ckpt),
            device=args.device,
        )
        fc_model.prepare([p[0] for p in pairs])
    elif args.fcclip_config and args.fcclip_checkpoint:
        fc_model = load_mmseg_model(args.fcclip_config, args.fcclip_checkpoint, args.device)
    else:
        if fc_ckpt and not baseline_py:
            print("[warn] FC-CLIP weights present but detectron2 python missing.")
        else:
            print("[warn] FC-CLIP missing, using Vireo proxy.")
        fc_model = vireo_model

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
    elif args.sed_config and args.sed_checkpoint:
        sed_model = load_mmseg_model(args.sed_config, args.sed_checkpoint, args.device)
    else:
        if sed_ckpt and not baseline_py:
            print("[warn] SED weights present but detectron2 python missing.")
        else:
            print("[warn] SED missing, using Vireo proxy.")
        sed_model = vireo_model

    rows = []
    for name, model in [("FC-CLIP", fc_model), ("SED", sed_model), ("Our Model", vireo_model)]:
        iou = iou_from_confusion(eval_conf(model, pairs, ncls))
        if args.grouping == "seen_unseen":
            rows.append({"Model": name, "Category": "Seen", "mIoU": float(np.nanmean(iou[seen]))})
            rows.append({"Model": name, "Category": "Unseen", "mIoU": float(np.nanmean(iou[unseen]))})
        else:
            rows.append({"Model": name, "Category": "Overall", "mIoU": float(np.nanmean(iou))})

    df = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(df.to_string(index=False))

    plt.figure(figsize=(8, 5), dpi=300)
    categories = ["Seen", "Unseen"] if args.grouping == "seen_unseen" else ["Overall"]
    models = ["FC-CLIP", "SED", "Our Model"]
    width = 0.25 if len(categories) > 1 else 0.6
    x = np.arange(len(categories))
    for i, model_name in enumerate(models):
        vals = [
            float(df[(df["Model"] == model_name) & (df["Category"] == c)]["mIoU"].iloc[0])
            for c in categories
        ]
        bars = plt.bar(x + (i - 1) * width, vals, width=width, label=model_name)
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h + 0.003, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(x, categories)
    plt.ylabel("mIoU")
    title = "Scenario-based mIoU (Seen vs Unseen)" if args.grouping == "seen_unseen" else "Scenario-based mIoU (Overall)"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=300, bbox_inches="tight")
    print(f"[done] saved: {args.output_csv}")
    print(f"[done] saved: {args.output_plot}")


if __name__ == "__main__":
    main()
