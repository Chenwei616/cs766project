#!/usr/bin/env python3
"""Download FC-CLIP and SED pretrained checkpoints (Google Drive) into cv_project/checkpoints/."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CV_PROJECT = Path(__file__).resolve().parent
CHECKPOINTS = CV_PROJECT / "checkpoints"

# Official repo links (ConvNeXt-L FC-CLIP row; SED (L) row from paper README tables).
FC_CLIP_CONVNEXT_L_ID = "1-91PIns86vyNaL3CzMmDD39zKGnPMtvj"
SED_CONVNEXT_L_ID = "1zAXE0QXy47n0cVn7j_2cSR85eqxdDGg8"


def _gdown_or_die(url: str, out: Path) -> None:
    try:
        import gdown  # noqa: WPS433
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown  # noqa: WPS433

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 1_000_000:
        print(f"[skip] already exists: {out}")
        return
    print(f"[download] {url} -> {out}")
    try:
        gdown.download(url, str(out), fuzzy=True)
    except TypeError:
        gdown.download(url, str(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fc-clip-only", action="store_true")
    p.add_argument("--sed-only", action="store_true")
    args = p.parse_args()

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    if not args.sed_only:
        _gdown_or_die(f"https://drive.google.com/uc?id={FC_CLIP_CONVNEXT_L_ID}", CHECKPOINTS / "fcclip_convnext_large.pth")
    if not args.fc_clip_only:
        _gdown_or_die(f"https://drive.google.com/uc?id={SED_CONVNEXT_L_ID}", CHECKPOINTS / "sed_convnext_l.pth")

    print(f"[done] checkpoints under {CHECKPOINTS}")


if __name__ == "__main__":
    main()
