import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.controllers import CONTROLLERS
from src.data.scenario_generator import load_split
from src.simulator.metrics import calculate_metrics
from src.simulator.railway_section import RailwaySection
from src.utils.config import ROOT

if __name__ == "__main__":
    import pandas as pd

    rows = []
    for scenario in load_split("validation"):
        for name, factory in CONTROLLERS.items():
            section, controller = RailwaySection(scenario, record=False), factory()
            while not section.done:
                section.step(controller.action(section))
            rows.append(
                {
                    "controller": name,
                    "scenario_id": scenario["scenario_id"],
                    **calculate_metrics(section),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results/baseline_validation.csv", index=False)
    print(
        frame.groupby("controller")[["throughput", "mean_delay", "incomplete_trains"]]
        .mean()
        .to_string()
    )
