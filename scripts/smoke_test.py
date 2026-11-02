import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gymnasium.utils.env_checker import check_env

from src.controllers import CONTROLLERS
from src.data.scenario_generator import load_split
from src.simulator.environment import TrainTrafficEnv
from src.simulator.metrics import calculate_metrics
from src.simulator.railway_section import RailwaySection

if __name__ == "__main__":
    scenario = load_split("validation")[0]
    check_env(TrainTrafficEnv(scenario), skip_render_check=True)
    for name, factory in CONTROLLERS.items():
        section, controller = RailwaySection(scenario), factory()
        while not section.done:
            section.step(controller.action(section))
        metrics = calculate_metrics(section)
        assert metrics["safety_violations"] == metrics["invalid_actions"] == 0
        print(name, metrics["throughput"], metrics["mean_delay"])
    print("Smoke checks passed. Run python scripts/train.py --smoke for the 20,000-step PPO check.")
