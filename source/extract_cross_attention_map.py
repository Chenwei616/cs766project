import argparse
from pathlib import Path

import cv2
import numpy as np
from mmseg.apis import inference_model

from presentation_common import (
    REPO_ROOT,
    RESULTS_DIR,
    discover_acdc_scenarios,
    load_vireo_model,
    read_rgb,
    resolve_pairs_from_acdc,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-attention overlay.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vireo-config", default=str(REPO_ROOT / "configs/dinov2_domain/vireo_dinov2_mask2former_512x512_bs1x4_citys.py"))
    parser.add_argument("--vireo-checkpoint", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--acdc-root", default=str(REPO_ROOT / "data/acdc"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--scenarios", nargs="+", default=["night", "rain", "fog", "snow"])
    parser.add_argument("--hook-module", default="decode_head")
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--search-pool", type=int, default=24)
    parser.add_argument("--pick-best", action="store_true", help="Pick the image/query with the sharpest hotspot.")
    parser.add_argument("--alpha", type=float, default=0.50, help="Heatmap blending ratio.")
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "cross_attention_overlay.png"),
    )
    return parser.parse_args()


def get_module_by_path(model, path: str):
    cur = model
    for key in path.split("."):
        cur = cur[int(key)] if key.isdigit() else getattr(cur, key)
    return cur


def main():
    args = parse_args()
    model = load_vireo_model(args.vireo_config, args.vireo_checkpoint, args.device)
    if args.image:
        candidates = [args.image]
    else:
        scenarios = args.scenarios
        if len(scenarios) == 1 and scenarios[0].lower() == "all":
            scenarios = discover_acdc_scenarios(args.acdc_root, args.split)
            if not scenarios:
                raise RuntimeError(f"No scenarios found in {args.acdc_root}/rgb_anon for split={args.split}")
        pairs = resolve_pairs_from_acdc(args.acdc_root, [args.split], scenarios, args.search_pool)
        candidates = [p[0] for p in pairs]
    if not candidates:
        raise RuntimeError("No candidate image found for attention visualization.")

    store = {}
    handle = get_module_by_path(model, args.hook_module).register_forward_hook(
        lambda _m, _i, o: store.__setitem__("x", o)
    )
    try:
        scored = []
        for idx, image in enumerate(candidates):
            rgb = read_rgb(image)
            h, w = rgb.shape[:2]
            store.clear()
            _ = inference_model(model, image)
            out = store.get("x", None)
            if not isinstance(out, tuple) or len(out) < 2:
                continue
            cls_pred = out[0][-1][0].detach().cpu().numpy()  # Q x C
            mask_pred = out[1][-1][0].detach().cpu().numpy()  # Q x Hm x Wm
            q = int(np.clip(args.query_index, 0, mask_pred.shape[0] - 1))
            score = float(cls_pred[q].max())
            heat = mask_pred[q]
            # Robust contrast stretch for visible hotspot-like maps.
            lo, hi = np.percentile(heat, [2, 98])
            heat = np.clip((heat - lo) / (max(hi - lo, 1e-6)), 0.0, 1.0)
            heat = np.power(heat, 0.6)
            heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC)
            # "Peakiness" for selecting strongest sample/query map.
            peak = float(np.percentile(heat, 99) - np.percentile(heat, 60))
            scored.append((peak + 0.1 * score, idx, image, rgb, heat, q))
    finally:
        handle.remove()

    if not scored:
        raise RuntimeError("Failed to capture decoder outputs for attention visualization.")
    scored.sort(key=lambda x: x[0], reverse=True)
    if args.pick_best:
        _best_score, _idx, image, rgb, heat, q = scored[0]
    else:
        scored.sort(key=lambda x: x[1])
        _best_score, _idx, image, rgb, heat, q = scored[0]

    heat_u8 = (255 * heat).astype(np.uint8)
    jet = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    blend = cv2.addWeighted(base, 1.0 - args.alpha, jet, args.alpha, 0)
    cv2.putText(
        blend,
        f"query={q}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, blend)
    print(f"[info] source image: {image}")
    print(f"[done] saved: {args.output}")


if __name__ == "__main__":
    main()
