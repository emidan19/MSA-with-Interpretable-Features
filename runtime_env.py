"""Runtime setup helpers shared by the MSA extraction scripts."""

from __future__ import annotations

import os
from pathlib import Path


def ensure_local_cache(root: str | os.PathLike[str] | None = None) -> Path:
    """Keep matplotlib/numba caches inside the project workspace."""
    cache_root = Path(root) if root is not None else Path.cwd() / ".cache"
    mappings = {
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "NUMBA_CACHE_DIR": cache_root / "numba",
        "XDG_CACHE_HOME": cache_root / "xdg",
    }
    for env_name, path in mappings.items():
        os.environ.setdefault(env_name, str(path))
        Path(os.environ[env_name]).mkdir(parents=True, exist_ok=True)
    return cache_root


ensure_local_cache()
