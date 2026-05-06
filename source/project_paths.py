"""
Resolve locations of the Vireo training repo and optional dev layout.

Set **VIREO_REPO_ROOT** to the absolute path of the `Vireo` directory that contains
`configs/`, `vireo/`, `data/`, `checkpoints/`, `work_dirs/`, etc.

If unset, the resolver tries, in order:
1. Sibling path ``<parent of cv_project>/vireo_repro/Vireo`` (typical monorepo layout)
2. Legacy dev path (only if that directory exists on disk)
3. Fall back to (1) even if missing, so error messages point to the expected layout
"""

from __future__ import annotations

import os
from pathlib import Path

_CV_PROJECT = Path(__file__).resolve().parent


def get_vireo_root() -> Path:
    env = os.environ.get("VIREO_REPO_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()

    sibling = _CV_PROJECT.parent / "vireo_repro" / "Vireo"
    if sibling.is_dir():
        return sibling.resolve()

    legacy = Path("/data/chenwei/driving/DGSS/vireo_repro/Vireo")
    if legacy.is_dir():
        return legacy.resolve()

    return sibling.resolve()


# Re-export for a single import site in presentation_common
VIREO_ROOT: Path = get_vireo_root()
