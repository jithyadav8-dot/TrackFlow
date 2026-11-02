from src.data.prepare_schedule import synthetic_schedule
from src.data.scenario_generator import generate_scenario
from src.simulator.railway_section import RailwaySection


def test_deterministic_and_independent():
    frame = synthetic_schedule()
    scenario = generate_scenario(frame, seed=100)
    assert scenario == generate_scenario(frame, seed=100)
    first, second = RailwaySection(scenario), RailwaySection(scenario)
    first.step(0)
    assert second.time == 0
    assert all("status" not in row for row in scenario["trains"])


def test_scenario_types():
    frame = synthetic_schedule()
    assert len(generate_scenario(frame, "peak")["trains"]) == 45
    imbalance = generate_scenario(frame, "imbalance")
    assert sum(t["direction"] == 0 for t in imbalance["trains"]) == 21
    assert len(generate_scenario(frame, "saturation")["trains"]) >= 30
    assert generate_scenario(frame, count=0)["trains"] == []
    assert generate_scenario(synthetic_schedule(count=0))["trains"] == []
