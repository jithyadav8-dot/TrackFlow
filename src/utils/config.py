from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(name="simulator"):
    with (ROOT / "configs" / f"{name}.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dirs():
    for name in [
        "data/raw",
        "data/processed",
        "data/scenarios",
        "data/metadata",
        "models/checkpoints",
        "logs",
        "results/charts",
        "docs",
    ]:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
