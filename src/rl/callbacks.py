import csv

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.rl.evaluate import evaluate_scenarios


class ResearchCallback(BaseCallback):
    def __init__(self, directory, scenarios, interval=50000):
        super().__init__()
        self.directory = directory
        self.scenarios = scenarios
        self.interval = interval
        self.next_eval = interval
        self.best_cost = float("inf")
        self.episodes = []
        self.validations = []
        self.last_validation = -1

    def _on_step(self):
        for info in self.locals["infos"]:
            if "metrics" in info:
                row = {
                    "timesteps": self.num_timesteps,
                    "reward": info["episode"]["r"],
                    **info["metrics"],
                    **{f"reward_{k}": v for k, v in info["reward_totals"].items()},
                }
                self.episodes.append(row)
                for key in [
                    "mean_delay",
                    "throughput",
                    "utilization",
                    "incomplete_trains",
                    "avoidable_holds",
                ]:
                    value = row[key]
                    if value is not None:
                        self.logger.record_mean(f"operations/{key}", value)
                for key, value in info["reward_totals"].items():
                    self.logger.record_mean(f"rewards/{key}", value)
        if self.num_timesteps >= self.next_eval:
            self.validate()
            self.next_eval += self.interval
        return True

    def validate(self):
        results = evaluate_scenarios(self.model, self.scenarios)
        keys = [
            "selection_cost",
            "mean_delay",
            "incomplete_trains",
            "throughput",
            "avoidable_holds",
        ]
        row = {"timesteps": self.num_timesteps}
        for key in keys:
            values = [r[key] for r in results if r[key] is not None]
            row[key] = float(np.mean(values)) if values else None
        self.validations.append(row)
        if row["selection_cost"] < self.best_cost:
            self.best_cost = row["selection_cost"]
            self.model.save(self.directory / "best.zip")
        self.model.save(self.directory / "checkpoints" / f"step_{self.num_timesteps}.zip")
        self.last_validation = self.num_timesteps
        print(
            f"Validation step={self.num_timesteps} cost={row['selection_cost']:.2f} wait={row['mean_delay']} incomplete={row['incomplete_trains']:.2f}",
            flush=True,
        )
        self.flush()

    def flush(self):
        for name, records in [
            ("episodes.csv", self.episodes),
            ("validation.csv", self.validations),
        ]:
            if records:
                with (self.directory / name).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)

    def _on_training_end(self):
        if self.last_validation != self.num_timesteps:
            self.validate()
        self.flush()
