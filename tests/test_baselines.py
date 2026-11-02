from src.controllers import CONTROLLERS
from src.simulator.metrics import calculate_metrics
from src.simulator.railway_section import RailwaySection
from tests.test_simulator import scenario


def test_known_orders():
    for name, controller in CONTROLLERS.items():
        section = RailwaySection(scenario())
        first = controller().action(section)
        assert first == (2 if name == "Fixed Priority" else 1)
        section.step(first)
        assert controller().action(section) == 0


def test_fifo_priority_cannot_overtake():
    section = RailwaySection(scenario([("a", 0, 0, 2, 1), ("b", 0, 0, 2, 2), ("c", 1, 0, 2, 1)]))
    assert CONTROLLERS["Fixed Priority"]().action(section) == 1
    section.step(1)
    assert section.occupied.train_id == "a"


def test_fcfs_equals_longest_wait_and_all_safe():
    results = {}
    for name, controller in CONTROLLERS.items():
        section = RailwaySection(scenario())
        while not section.done:
            section.step(controller().action(section))
        results[name] = calculate_metrics(section)
        assert results[name]["invalid_actions"] == 0
        assert results[name]["incomplete_trains"] == 0
    assert results["FCFS"] == results["Longest Waiting"]
