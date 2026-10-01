from __future__ import annotations

import json
import platform
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def save_run_metadata(path: str | Path, config: dict) -> None:
    payload = {"config": config, "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda}}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

