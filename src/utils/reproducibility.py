import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.utils.config import ROOT


def digest(value):
    content = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(content).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(1)


def metadata():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    packages = ["numpy", "pandas", "gymnasium", "torch", "stable-baselines3", "sb3-contrib"]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "python_version": platform.python_version(),
        "dependency_versions": {p: importlib.metadata.version(p) for p in packages},
    }
