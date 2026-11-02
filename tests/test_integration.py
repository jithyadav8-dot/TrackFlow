import pandas as pd

from src.controllers import FCFS
from src.data.scenario_generator import generate_scenario
from src.data.select_corridor import corridor_movements, select_corridor
from src.evaluation.experiments import run_episode
from src.evaluation.reporting import summarize
from src.utils.config import load_config
from src.visualization.charts import comparison_chart, timeline_chart
from tests.test_gtfs import fixture_tables


def test_data_to_evaluation_to_charts(tmp_path):
    config = dict(
        service_date="2026-10-28", rail_route_types=[2], min_travel_minutes=2, max_travel_minutes=30
    )
    schedule, _ = select_corridor(corridor_movements(fixture_tables(), config, {}))
    scenario = generate_scenario(schedule, seed=20001)
    metrics, trains, frames, _ = run_episode(scenario, FCFS(), record=True)
    assert metrics["incomplete_trains"] == 0
    frame = pd.DataFrame([dict(controller="FCFS", kind="normal", **metrics)])
    summary = summarize(frame, load_config("evaluation"))
    comparison_chart(summary, "throughput", "Test", "Trains/hour").write_html(
        tmp_path / "comparison.html"
    )
    timeline_chart(pd.DataFrame(trains)).write_html(tmp_path / "timeline.html")
    assert (tmp_path / "comparison.html").stat().st_size > 1000
    assert frames[-1]["completed"] == 4
