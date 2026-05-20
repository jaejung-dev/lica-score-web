from __future__ import annotations

import os
from pathlib import Path


HF_CACHE_ROOT = Path(os.environ.get("LICA_SCORE_CACHE_ROOT", "/mnt/local/hf-cache"))
TORCH_CACHE_ROOT = Path(os.environ.get("LICA_SCORE_TORCH_CACHE", "/mnt/local/torch-cache"))
PIP_CACHE_ROOT = Path(os.environ.get("LICA_SCORE_PIP_CACHE", "/mnt/local/pip-cache"))


def configure_external_caches() -> Path:
    """Route large model/download caches away from the small root disk."""
    cache_dirs = {
        "HF_HOME": HF_CACHE_ROOT,
        "HUGGINGFACE_HUB_CACHE": HF_CACHE_ROOT / "hub",
        "TRANSFORMERS_CACHE": HF_CACHE_ROOT / "transformers",
        "TORCH_HOME": TORCH_CACHE_ROOT,
        "XDG_CACHE_HOME": Path(os.environ.get("LICA_SCORE_XDG_CACHE", "/mnt/local/xdg-cache")),
        "PIP_CACHE_DIR": PIP_CACHE_ROOT,
    }
    for key, path in cache_dirs.items():
        os.environ.setdefault(key, str(path))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    return HF_CACHE_ROOT
