import copy

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.data.scenario_generator import KINDS, generate_scenario
from src.simulator.metrics import calculate_metrics
from src.simulator.railway_section import RailwaySection
from src.utils.config import ROOT, load_config


class TrainTrafficEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self, scenario=None, schedule=None, reward_config=None, render_mode=None, record=False
    ):
        super().__init__()
        self.fixed_scenario = copy.deepcopy(scenario)
        self.schedule = schedule
        if scenario is None and schedule is None:
            self.schedule = pd.read_csv(ROOT / "data/processed/train_schedule.csv")
        self.weights = dict(reward_config or load_config()["reward"])
        self.render_mode = render_mode
        self.record = record
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-1.0, 1.0, (14,), dtype=np.float32)
        self.section = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "scenario" in options:
            scenario = options["scenario"]
        elif self.fixed_scenario is not None:
            scenario = self.fixed_scenario
        else:
            scenario_seed = int(self.np_random.integers(100000, 2000000000))
            kind = KINDS[int(self.np_random.integers(len(KINDS)))]
            scenario = generate_scenario(self.schedule, kind, scenario_seed)
        self.section = RailwaySection(scenario, record=self.record)
        self.scenario_id = scenario.get("scenario_id", "manual")
        self.reward_totals = dict.fromkeys(
            ["completion", "waiting", "invalid", "hold", "backlog"], 0.0
        )
        self._ended = False
        return self.observation(), {"scenario_id": self.scenario_id}

    def action_masks(self):
        return np.asarray(self.section.masks(), dtype=bool)

    def observation(self):
        section = self.section
        n, duration = max(section.total, 1), section.cutoff
        heads = [queue[0] if queue else None for queue in section.queues]
        remaining = (
            section.occupied.actual_entry_time + section.occupied.travel_duration - section.time
            if section.occupied
            else max(0, section.clear_until - section.time)
        )
        next_arrivals = []
        for direction in (0, 1):
            # Timetable knowledge only: never expose assigned future ready_time or travel noise.
            pending = [
                t.scheduled_entry_time
                for t in section.trains
                if t.status == "NOT_READY" and t.direction == direction
            ]
            next_arrivals.append(max(0, min(pending) - section.time) / duration if pending else -1)
        features = [
            (duration - section.time) / duration,
            float(section.occupied is not None),
            remaining / duration,
            len(section.queues[0]) / n,
            len(section.queues[1]) / n,
            *[(section.time - t.ready_time) / duration if t else -1 for t in heads],
            *[t.priority / 2 if t else -1 for t in heads],
            *next_arrivals,
            section.last_direction,
            section.completed / n,
            (section.total - section.completed) / n,
        ]
        return np.clip(features, -1, 1).astype(np.float32)

    def step(self, action):
        if self._ended:
            raise RuntimeError("Reset required after termination/truncation")
        if self.section.done:
            components = dict.fromkeys(self.reward_totals, 0.0)
        else:
            data = self.section.step(int(action))
            waiting = (
                data["weighted_waiting"] if self.weights["priority_weighting"] else data["waiting"]
            )
            components = {
                "completion": self.weights["completion"] * data["completed"],
                "waiting": -self.weights["waiting"] * waiting,
                "invalid": -self.weights["invalid"] * data["invalid"],
                "hold": -self.weights["hold"] * data["avoidable_hold"],
                "backlog": 0.0,
            }
        terminated = self.section.completed == self.section.total
        truncated = self.section.time >= self.section.cutoff and not terminated
        if truncated:
            components["backlog"] = -self.weights["backlog"] * (
                self.section.total - self.section.completed
            )
        self._ended = terminated or truncated
        for key, value in components.items():
            self.reward_totals[key] += value
        info = {"reward_components": components}
        if self._ended:
            info.update(
                metrics=calculate_metrics(self.section),
                reward_totals=self.reward_totals.copy(),
                scenario_id=self.scenario_id,
            )
        return self.observation(), float(sum(components.values())), terminated, truncated, info

    def render(self):
        section = self.section
        return f"t={section.time} A={len(section.queues[0])} B={len(section.queues[1])} section={section.occupied.train_id if section.occupied else 'free'} completed={section.completed}"

    def close(self):
        pass
