import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

try:
    _CONDA_BASE = subprocess.check_output(["conda", "info", "--base"], text=True).strip()
except Exception:  # noqa: BLE001
    _CONDA_BASE = ""

# Parent process imports `presentation_common` → `vireo`; ensure repo roots on path.
_CV_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_CV_ROOT))
from project_paths import get_vireo_root

_VIREO_ROOT = get_vireo_root()
for _p in (_VIREO_ROOT, _CV_ROOT):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from presentation_common import REPO_ROOT, RESULTS_DIR


def pick_least_busy_gpu() -> int:
    """Pick GPU index with lowest memory.used + utilization penalty."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    best_idx = 0
    best_score = 1e18
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        idx, mem_mib, util_pct = int(parts[0]), int(parts[1]), int(parts[2])
        score = mem_mib + util_pct * 100
        if score < best_score:
            best_score = score
            best_idx = idx
    return best_idx


def run(cmd, cwd: Path, env: Optional[Dict[str, str]] = None):
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd), env=env)


def main():
    cv_root = Path(__file__).resolve().parent
    os.chdir(cv_root)
    # Optional: use conda env `ov_baselines` for FC-CLIP / SED (see setup_baseline_env.sh).
    if not os.environ.get("OV_BASELINE_PYTHON"):
        for cand in filter(
            None,
            (
                Path(_CONDA_BASE) / "envs/ov_baselines/bin/python" if _CONDA_BASE else None,
                Path.home() / "miniconda3/envs/ov_baselines/bin/python",
            ),
        ):
            if cand.is_file():
                os.environ["OV_BASELINE_PYTHON"] = str(cand)
                print(f"[info] OV_BASELINE_PYTHON={cand} (auto)")
                break
    # Config uses relative paths (e.g. open_vocab/cityscapes.json); cwd must be Vireo repo.
    vireo_cwd = REPO_ROOT

    cfg = REPO_ROOT / "configs/dinov2_domain/vireo_dinov2_mask2former_512x512_bs1x4_citys.py"
    ckpt = (
        REPO_ROOT
        / "work_dirs/vireo_dinov2_mask2former_512x512_bs1x4_citys/iter_40000.pth"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    gpu = pick_least_busy_gpu()
    run_env = os.environ.copy()
    run_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    py_path = f"{REPO_ROOT}:{cv_root}"
    if run_env.get("PYTHONPATH"):
        py_path = f"{py_path}:{run_env['PYTHONPATH']}"
    run_env["PYTHONPATH"] = py_path
    print(f"[info] CUDA_VISIBLE_DEVICES={gpu} (auto-picked)")

    py = sys.executable
    model_base = [
        "--vireo-config",
        str(cfg),
        "--vireo-checkpoint",
        str(ckpt),
        "--device",
        "cuda:0",
    ]
    baseline_arg = []
    bp = os.environ.get("OV_BASELINE_PYTHON")
    if bp:
        baseline_arg = ["--baseline-python", bp]
    run(
        [
            py,
            str(cv_root / "qualitative_multimodel_grid.py"),
            "--num-samples",
            "4",
            "--scenarios",
            "all",
            "--output",
            str(RESULTS_DIR / "qualitative_1x5_grid.png"),
            *model_base,
            *baseline_arg,
        ],
        cwd=vireo_cwd,
        env=run_env,
    )
    run(
        [
            py,
            str(cv_root / "extract_cross_attention_map.py"),
            "--scenarios",
            "all",
            "--search-pool",
            "30",
            "--pick-best",
            "--output",
            str(RESULTS_DIR / "cross_attention_overlay.png"),
            *model_base,
        ],
        cwd=vireo_cwd,
        env=run_env,
    )
    run(
        [
            py,
            str(cv_root / "scenario_miou_eval.py"),
            "--num-samples",
            "50",
            "--scenarios",
            "all",
            "--grouping",
            "overall",
            "--output-csv",
            str(RESULTS_DIR / "scenario_miou_scores.csv"),
            "--output-plot",
            str(RESULTS_DIR / "scenario_miou_bar.png"),
            *model_base,
            *baseline_arg,
        ],
        cwd=vireo_cwd,
        env=run_env,
    )
    print(f"[done] outputs in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
