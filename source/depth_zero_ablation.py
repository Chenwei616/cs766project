import argparse
from contextlib import contextmanager
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from presentation_common import (
    REPO_ROOT,
    RESULTS_DIR,
    colorize_cityscapes,
    discover_acdc_scenarios,
    inference_label_map,
    load_vireo_model,
    read_rgb,
    resolve_pairs_from_acdc,
    resize_label_to_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Zero depth ablation.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vireo-config", default=str(REPO_ROOT / "configs/dinov2_domain/vireo_dinov2_mask2former_512x512_bs1x4_citys.py"))
    parser.add_argument("--vireo-checkpoint", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--acdc-root", default=str(REPO_ROOT / "data/acdc"))
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["night", "rain", "fog", "snow"],
        help="Scenarios to search. Use 'all' to auto-discover.",
    )
    parser.add_argument("--search-pool", type=int, default=24, help="Candidate images to score for ablation differences.")
    parser.add_argument("--topk", type=int, default=3, help="Number of best-difference samples to visualize.")
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "depth_zero_ablation.png"),
    )
    return parser.parse_args()


@contextmanager
def depth_zero_ablation(model):
    vireo_mod = model.backbone.vireo
    original = vireo_mod.forward_delta_feat

    def patched(feats, tokens, depth_features, layers):
        return original(feats, tokens, torch.zeros_like(depth_features), layers)

    vireo_mod.forward_delta_feat = patched
    try:
        yield
    finally:
        vireo_mod.forward_delta_feat = original


def main():
    args = parse_args()
    model = load_vireo_model(args.vireo_config, args.vireo_checkpoint, args.device)
    candidates = []
    if args.image:
        candidates = [(args.image, "", "custom")]
    else:
        scenarios = args.scenarios
        if len(scenarios) == 1 and scenarios[0].lower() == "all":
            scenarios = discover_acdc_scenarios(args.acdc_root, args.split)
            if not scenarios:
                raise RuntimeError(f"No scenarios found in {args.acdc_root}/rgb_anon for split={args.split}")
        candidates = resolve_pairs_from_acdc(args.acdc_root, [args.split], scenarios, args.search_pool)
    if not candidates:
        raise RuntimeError("No candidate images found for depth ablation.")

    scored = []
    for img_path, _gt, scene in candidates:
        rgb = read_rgb(img_path)
        h, w = rgb.shape[:2]
        normal = resize_label_to_image(inference_label_map(model, img_path), h, w)
        with depth_zero_ablation(model):
            ablated = resize_label_to_image(inference_label_map(model, img_path), h, w)
        diff = (normal != ablated)
        diff_ratio = float(diff.mean())
        scored.append((diff_ratio, img_path, scene, rgb, normal, ablated, diff))

    scored.sort(key=lambda x: x[0], reverse=True)
    keep = scored[: max(1, min(args.topk, len(scored)))]

    fig, axes = plt.subplots(len(keep), 3, figsize=(15, 5 * len(keep)), dpi=300)
    if len(keep) == 1:
        axes = axes[None, :]

    for i, (diff_ratio, _img_path, scene, rgb, normal, ablated, diff) in enumerate(keep):
        edge = cv2.Canny((diff.astype(np.uint8) * 255), 30, 100)
        # Fill disagreement regions for stronger visibility and draw red edges.
        overlay = colorize_cityscapes(ablated).copy()
        overlay[diff] = (0.45 * overlay[diff] + 0.55 * np.array([255, 30, 30])).astype(np.uint8)
        overlay[edge > 0] = np.array([255, 0, 0], dtype=np.uint8)

        axes[i, 0].imshow(rgb); axes[i, 0].set_title("RGB Input"); axes[i, 0].axis("off")
        axes[i, 1].imshow(colorize_cityscapes(normal)); axes[i, 1].set_title("Normal Inference Mask"); axes[i, 1].axis("off")
        axes[i, 2].imshow(overlay); axes[i, 2].set_title("Depth-Ablated Mask + Diff Highlight"); axes[i, 2].axis("off")
        axes[i, 0].set_ylabel(f"{scene}\ndiff={diff_ratio:.2%}", fontsize=10)

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"[done] saved: {args.output}")


if __name__ == "__main__":
    main()
