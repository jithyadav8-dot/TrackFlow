import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.controllers import CONTROLLERS
from src.data.scenario_generator import load_split
from src.evaluation.experiments import run_episode
from src.rl.policy_adapter import PPOController
from src.utils.config import ROOT
from src.utils.reproducibility import save_json, seed_all

if __name__ == "__main__":
    seed_all(42)
    expected = pd.read_csv(ROOT / "results/episode_metrics.csv")
    controllers = {name: cls() for name, cls in CONTROLLERS.items()}
    registry = json.loads((ROOT / "models/registry.json").read_text())
    for seed, details in registry.items():
        controllers[f"PPO seed {seed}"] = PPOController(
            ROOT / details["checkpoint"].replace("\\", "/")
        )
    checked = 0
    for scenario in load_split("test")[:5]:
        for name, controller in controllers.items():
            metrics, _, _, _ = run_episode(scenario, controller)
            row = expected[
                (expected.scenario_id == scenario["scenario_id"]) & (expected.controller == name)
            ].iloc[0]
            for metric, value in metrics.items():
                if value is None:
                    assert pd.isna(row[metric]), metric
                else:
                    assert np.isclose(value, row[metric], rtol=0, atol=1e-10), (
                        name,
                        metric,
                        value,
                        row[metric],
                    )
            checked += 1
    save_json(
        ROOT / "results/reproducibility_check.json",
        {
            "verified_runs": checked,
            "scenarios": 5,
            "metrics_per_run": len(metrics),
            "absolute_tolerance": 1e-10,
            "status": "passed",
        },
    )
    print(f"Reproduced every aggregate metric for {checked} saved controller/scenario runs.")
