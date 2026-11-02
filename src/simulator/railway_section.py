from collections import deque
from dataclasses import asdict

from src.simulator.events import Event
from src.simulator.safety import assert_invariants, authorize, legal_dispatch
from src.simulator.train import Train


class RailwaySection:
    """Decision at t; dispatch, accrue [t,t+1), exit at t+1, then admit ready queues."""

    def __init__(self, scenario, record=True):
        self.config = dict(scenario["simulation"])
        for name in [
            "section_capacity",
            "time_step_minutes",
            "control_horizon_minutes",
            "drain_horizon_minutes",
            "clearance_minutes",
        ]:
            if not isinstance(self.config[name], int):
                raise ValueError(f"{name} must be an integer")
        if self.config["section_capacity"] != 1 or self.config["time_step_minutes"] != 1:
            raise ValueError("Only single capacity and integer one-minute steps supported")
        self.horizon = self.config["control_horizon_minutes"]
        self.cutoff = self.horizon + self.config["drain_horizon_minutes"]
        self.clearance = self.config["clearance_minutes"]
        if self.horizon <= 0 or self.cutoff < self.horizon or self.clearance < 0:
            raise ValueError("Invalid simulation horizons or clearance")
        self.trains = sorted(
            [Train.from_record(r) for r in scenario["trains"]],
            key=lambda t: (t.ready_time, t.train_id),
        )
        if len({t.train_id for t in self.trains}) != len(self.trains):
            raise ValueError("Duplicate train IDs")
        self.total = len(self.trains)
        self.queues = [deque(), deque()]
        self.time = self.clear_until = self.completed = self.completed_control = 0
        self.occupied = None
        self.last_exit = None
        self.last_direction = -1
        self.arrival_index = 0
        self.invalid_actions = self.conflicting_attempts = self.avoidable_holds = 0
        self.busy_minutes = self.clearance_minutes = self.max_queue = 0
        self.queue_sums = [0, 0]
        self.accrued_wait = self.weighted_wait = 0
        self.action_counts = [0, 0, 0]
        self.events, self.frames = [], []
        self.record = record
        self._arrive()
        assert_invariants(self)
        self._frame()

    @property
    def done(self):
        return self.completed == self.total or self.time >= self.cutoff

    def _arrive(self):
        while (
            self.arrival_index < self.total
            and self.trains[self.arrival_index].ready_time <= self.time
        ):
            train = self.trains[self.arrival_index]
            train.transition("WAITING")
            self.queues[train.direction].append(train)
            self.arrival_index += 1

    def masks(self):
        return [True, legal_dispatch(self, 0), legal_dispatch(self, 1)]

    def _event(self, kind, train):
        if self.record:
            self.events.append(asdict(Event(self.time, kind, train.train_id, train.direction)))

    def _frame(self):
        if self.record:
            entered = [
                t.actual_entry_time - t.ready_time
                for t in self.trains
                if t.actual_entry_time is not None
            ]
            self.frames.append(
                {
                    "time": self.time,
                    "queue_a": len(self.queues[0]),
                    "queue_b": len(self.queues[1]),
                    "completed": self.completed,
                    "occupied": self.occupied.train_id if self.occupied else None,
                    "direction": self.occupied.direction if self.occupied else -1,
                    "entry": self.occupied.actual_entry_time if self.occupied else None,
                    "duration": self.occupied.travel_duration if self.occupied else None,
                    "clear_until": self.clear_until,
                    "mean_entered_wait": sum(entered) / len(entered) if entered else 0,
                }
            )

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is complete; reset before stepping again")
        legal = self.masks()
        invalid = not isinstance(action, int) or action not in (0, 1, 2) or not legal[action]
        avoidable = action == 0 and any(legal[1:])
        self.invalid_actions += int(invalid)
        self.avoidable_holds += int(avoidable)
        if isinstance(action, int) and action in (0, 1, 2):
            self.action_counts[action] += 1
        if invalid:
            self.conflicting_attempts += int(action in (1, 2) and self.occupied is not None)
        elif action in (1, 2) and authorize(self, action - 1):
            train = self.queues[action - 1].popleft()
            train.transition("IN_SECTION")
            train.actual_entry_time = self.time
            self.occupied = train
            self.last_direction = train.direction
            self._event("entry", train)
        waiting = sum(len(queue) for queue in self.queues)
        weighted = sum(t.priority for queue in self.queues for t in queue)
        self.accrued_wait += waiting
        self.weighted_wait += weighted
        if self.time < self.horizon:
            self.busy_minutes += int(self.occupied is not None)
            self.clearance_minutes += int(self.occupied is None and self.time < self.clear_until)
            self.queue_sums = [s + len(q) for s, q in zip(self.queue_sums, self.queues)]
            self.max_queue = max(self.max_queue, waiting)
        self.time += 1
        completed = 0
        if (
            self.occupied
            and self.time == self.occupied.actual_entry_time + self.occupied.travel_duration
        ):
            train = self.occupied
            train.actual_exit_time = self.time
            train.transition("COMPLETED")
            self.completed += 1
            self.completed_control += int(self.time <= self.horizon)
            self.last_exit = self.time
            self.clear_until = self.time + self.clearance
            self.occupied = None
            completed = 1
            self._event("exit", train)
        self._arrive()
        assert_invariants(self)
        self._frame()
        return {
            "completed": completed,
            "waiting": waiting,
            "weighted_waiting": weighted,
            "invalid": int(invalid),
            "avoidable_hold": int(avoidable),
        }
