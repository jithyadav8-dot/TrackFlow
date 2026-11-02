"""Exploratory one-factor reward study, validation only; never modifies model registry."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rl.train_ppo import train
from src.utils.config import ROOT
from src.utils.reproducibility import save_json

if __name__ == "__main__":
    studies = {
        "control": {},
        "completion_4": {"completion": 4.0},
        "waiting_010": {"waiting": 0.10},
        "hold_010": {"hold": 0.10},
        "unweighted": {"priority_weighting": False},
    }
    save_json(
        ROOT / "results/reward_study_plan.json",
        {
            "seed": 42,
            "requested_timesteps": 50000,
            "selection_split": "validation",
            "experiments": studies,
            "changes_to_final_test_models": "none",
            "caveat": "Short single-seed exploratory runs; unweighted is identical when all priorities are one.",
        },
    )
    rows = []
    for name, override in studies.items():
        path = ROOT / "models" / f"study_{name}" / "metadata.json"
        details = (
            json.loads(path.read_text())
            if path.exists()
            else train(seed=42, timesteps=50000, reward_override=override, label=f"study_{name}")
        )
        rows.append(
            {
                "variant": name,
                "seed": 42,
                "actual_timesteps": details["actual_timesteps"],
                **details["best_validation"],
            }
        )
        pd.DataFrame(rows).to_csv(ROOT / "results/reward_sensitivity.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
