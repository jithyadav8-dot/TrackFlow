import copy

import pytest

from src.simulator.metrics import calculate_metrics, capacity_bound
from src.simulator.railway_section import RailwaySection
from src.simulator.safety import SafetyError, assert_invariants
from src.simulator.train import Train
from src.utils.config import load_config


def scenario(rows=None, horizon=20, drain=20, clearance=2):
    config = copy.deepcopy(load_config()["simulation"])
    config.update(
        control_horizon_minutes=horizon, drain_horizon_minutes=drain, clearance_minutes=clearance
    )
    rows = rows if rows is not None else [("a", 0, 0, 3, 1), ("b", 1, 0, 3, 2)]
    return {
        "simulation": config,
        "trains": [
            dict(
                train_id=i,
                direction=d,
                scheduled_entry_time=r,
                ready_time=r,
                travel_duration=t,
                priority=p,
            )
            for i, d, r, t, p in rows
        ],
    }


def drain_section(section):
    while not section.done:
        masks = section.masks()
        section.step(1 if masks[1] else 2 if masks[2] else 0)


def test_timing_clearance_and_wait_cost():
    section = RailwaySection(scenario())
    assert section.step(1)["waiting"] == 1
    section.step(2)  # invalid opposing dispatch must not enter
    assert section.invalid_actions == 1
    section.step(0)
    assert section.trains[0].actual_exit_time == 3
    assert section.masks() == [True, False, False]
    section.step(0)
    section.step(0)
    assert section.time == 5 and section.masks()[2]
    drain_section(section)
    assert section.trains[1].actual_entry_time == 5
    assert section.trains[1].actual_exit_time == 8
    assert section.accrued_wait == 5
    metrics = calculate_metrics(section)
    assert metrics["mean_delay"] == 2.5
    assert metrics["utilization"] == 6 / 20
    assert metrics["throughput"] == 6


def test_ready_fifo_and_conservation():
    section = RailwaySection(scenario([("z", 0, 2, 1, 1), ("a", 0, 2, 1, 1), ("b", 1, 3, 1, 1)]))
    assert section.masks() == [True, False, False]
    section.step(1)
    section.step(0)
    assert [t.train_id for t in section.queues[0]] == ["a", "z"]
    drain_section(section)
    assert section.completed == 3
    assert_invariants(section)


def test_zero_trains_and_cutoff():
    empty = RailwaySection(scenario([]))
    assert empty.done
    assert calculate_metrics(empty)["mean_delay"] is None
    held = RailwaySection(scenario(horizon=1, drain=1))
    held.step(0)
    held.step(0)
    assert held.done and calculate_metrics(held)["incomplete_trains"] == 2
    with pytest.raises(RuntimeError):
        held.step(0)


def test_transition_and_invalid_input():
    train = Train("a", 0, 0, 0, 3)
    with pytest.raises(ValueError):
        train.transition("COMPLETED")
    with pytest.raises(ValueError):
        Train("x", 0, 3, 1, 3)
    with pytest.raises(ValueError):
        RailwaySection(scenario([("a", 0, 0, 2, 1), ("a", 1, 0, 2, 1)]))


def test_invariant_detects_tampering():
    section = RailwaySection(scenario())
    section.queues[0].append(section.queues[0][0])
    with pytest.raises(SafetyError):
        assert_invariants(section)


def test_invariant_detects_premature_timestamp():
    section = RailwaySection(scenario())
    section.trains[0].actual_entry_time = 0
    with pytest.raises(SafetyError):
        assert_invariants(section)


def test_fractional_config_rejected():
    data = scenario()
    data["simulation"]["clearance_minutes"] = 1.5
    with pytest.raises(ValueError):
        RailwaySection(data)


def test_horizon_boundary_and_capacity():
    section = RailwaySection(scenario(horizon=3, drain=10, clearance=0))
    drain_section(section)
    metrics = calculate_metrics(section)
    assert metrics["completed_control"] == 1 and metrics["drain_completions"] == 1
    assert metrics["utilization"] == 1
    assert section.trains[1].actual_entry_time == section.trains[0].actual_exit_time
    assert capacity_bound(section.trains, 6, 0) == 2


@pytest.mark.parametrize("seed", range(10))
def test_random_actions_never_break_safety(seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    section = RailwaySection(
        scenario([(str(i), i % 2, i, int(rng.integers(1, 6)), 1) for i in range(10)])
    )
    while not section.done:
        section.step(int(rng.integers(-1, 5)))
        assert_invariants(section)


def test_censored_metrics():
    section = RailwaySection(scenario(horizon=2, drain=0))
    section.step(1)
    section.step(0)
    metrics = calculate_metrics(section)
    assert metrics["mean_delay"] is None
    assert metrics["censored_wait_total"] == 2
    assert metrics["incomplete_trains"] == 2
