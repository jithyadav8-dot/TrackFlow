import pandas as pd
import pytest

from src.data.load_gtfs import active_services, load_gtfs, parse_time
from src.data.prepare_schedule import synthetic_schedule
from src.data.select_corridor import corridor_movements, select_corridor
from src.data.validate_gtfs import validate_schedule, validate_tables


@pytest.mark.parametrize("text,seconds", [("00:00:00", 0), ("25:15:30", 90930), ("8:00:00", 28800)])
def test_time(text, seconds):
    assert parse_time(text) == seconds


@pytest.mark.parametrize("text", ["", "12:60:00", "-1:20:00", "12:00:61", "12:10"])
def test_bad_time(text):
    with pytest.raises(ValueError):
        parse_time(text)


def fixture_tables():
    trips = pd.DataFrame(
        [{"trip_id": str(i), "route_id": "R", "service_id": "S"} for i in range(4)]
    )
    times = []
    for i in range(4):
        for seq, station in enumerate((["A1", "B"] if i % 2 == 0 else ["B", "A1"])):
            time = f"25:{i * 10 + seq * 5:02}:00"
            times.append(
                dict(
                    trip_id=str(i),
                    stop_id=station,
                    stop_sequence=str(seq),
                    arrival_time=time,
                    departure_time=time,
                )
            )
    return {
        "routes": pd.DataFrame([dict(route_id="R", route_type="2")]),
        "trips": trips,
        "stops": pd.DataFrame(
            [
                dict(stop_id="A1", parent_station="A"),
                dict(stop_id="A", parent_station=""),
                dict(stop_id="B", parent_station=""),
            ]
        ),
        "stop_times": pd.DataFrame(times),
        "calendar_dates": pd.DataFrame([dict(service_id="S", date="20261028", exception_type="1")]),
    }


def test_calendar_exceptions():
    tables = fixture_tables()
    assert active_services(tables, "2026-10-28") == {"S"}
    assert active_services(tables, "2026-10-29") == set()
    tables["calendar"] = pd.DataFrame(
        [dict(service_id="X", start_date="20260101", end_date="20261231", wednesday="1")]
    )
    tables["calendar_dates"] = pd.DataFrame(
        [dict(service_id="X", date="20261028", exception_type="2")]
    )
    assert active_services(tables, "2026-10-28") == set()


def test_joins_and_selection():
    tables = fixture_tables()
    report = validate_tables(tables)
    assert "shapes" in report["missing_optional"]
    config = dict(
        service_date="2026-10-28", rail_route_types=[2], min_travel_minutes=2, max_travel_minutes=30
    )
    moves = corridor_movements(tables, config, report)
    frame, info = select_corridor(moves)
    assert info["station_pair"] == ["A", "B"]
    assert len(frame) == 4
    assert frame.nominal_travel_min.tolist() == [5] * 4
    assert frame.scheduled_entry_min.tolist() == [0, 10, 20, 30]
    pd.testing.assert_frame_equal(frame, select_corridor(moves.sample(frac=1, random_state=2))[0])
    validate_schedule(frame)


def test_synthetic_schema():
    frame = validate_schedule(synthetic_schedule())
    assert len(frame) == 30
    assert frame.source_type.eq("synthetic").all()
    pd.testing.assert_frame_equal(frame, synthetic_schedule())


def test_zip_optional(tmp_path):
    from zipfile import ZipFile

    path = tmp_path / "feed.zip"
    with ZipFile(path, "w") as archive:
        for key, frame in fixture_tables().items():
            archive.writestr(key + ".txt", frame.to_csv(index=False))
    assert "shapes" not in load_gtfs(path)


def test_frequency_expansion():
    tables = fixture_tables()
    tables["frequencies"] = pd.DataFrame(
        [
            dict(
                trip_id="0",
                start_time="25:00:00",
                end_time="25:30:00",
                headway_secs="600",
                exact_times="0",
            )
        ]
    )
    report = {}
    config = dict(
        service_date="2026-10-28", rail_route_types=[2], min_travel_minutes=2, max_travel_minutes=30
    )
    result = corridor_movements(tables, config, report)
    assert len(result) == 6
    assert result.source_type.eq("gtfs_frequency_synthetic").sum() == 3


def test_network_fallback_is_explicit(tmp_path, monkeypatch):
    import requests

    import src.data.prepare_schedule as pipeline

    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("offline test")

    for name in ["data/processed", "data/metadata"]:
        (tmp_path / name).mkdir(parents=True)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "download_gtfs", unavailable)
    frame, info = pipeline.prepare()
    assert info["source_type"] == "synthetic"
    assert "offline test" in info["fallback_reason"]
    assert frame.source_type.eq("synthetic").all()
    assert (tmp_path / "data/processed/train_schedule.parquet").exists()
    with pytest.raises(requests.ConnectionError):
        pipeline.prepare(strict=True)
