import logging
from zipfile import BadZipFile

import numpy as np
import pandas as pd
import requests

from src.data.download_gtfs import download_gtfs
from src.data.load_gtfs import load_gtfs
from src.data.select_corridor import corridor_movements, select_corridor
from src.data.validate_gtfs import SCHEDULE_COLUMNS, validate_schedule, validate_tables
from src.utils.config import ROOT, ensure_dirs, load_config
from src.utils.reproducibility import digest, save_json


def synthetic_schedule(seed=42, count=30, horizon=240):
    rng = np.random.default_rng(seed)
    arrivals = np.sort(rng.integers(0, horizon, count))
    rows = []
    for i, arrival in enumerate(arrivals):
        direction = i % 2
        rows.append(
            [
                f"S{i:04}",
                f"synthetic-{i}",
                "synthetic",
                "synthetic",
                "A" if direction == 0 else "B",
                "B" if direction == 0 else "A",
                direction,
                int(arrival),
                8,
                2 if rng.random() < 0.2 else 1,
                "synthetic",
            ]
        )
    return pd.DataFrame(rows, columns=SCHEDULE_COLUMNS).astype(
        {
            "direction": int,
            "scheduled_entry_min": int,
            "nominal_travel_min": int,
            "priority": int,
        }
    )


def prepare(synthetic=False, archive=None, date=None, strict=False):
    ensure_dirs()
    config = load_config("data")
    if date:
        config["service_date"] = date
    report = {}
    provenance = {}
    try:
        if synthetic:
            raise ValueError("Synthetic input explicitly requested")
        archive_path = archive or ROOT / "data/raw/MBTA_GTFS.zip"
        if archive is None:
            provenance = download_gtfs(
                config["url"], archive_path, ROOT / "data/metadata/download.json"
            )
        else:
            from pathlib import Path

            provenance = {
                "source_type": "user_supplied_gtfs",
                "archive": str(archive),
                "sha256": digest(Path(archive).read_bytes()),
            }
        tables = load_gtfs(archive_path)
        report = validate_tables(tables)
        movements = corridor_movements(tables, config, report)
        frame, corridor = select_corridor(movements, config["window_minutes"])
        provenance.update(corridor)
        provenance["service_date"] = config["service_date"]
    except (OSError, ValueError, BadZipFile, requests.RequestException) as error:
        # CLI strict mode surfaces feed/data defects; default fallback records the exact cause.
        if strict and not synthetic:
            raise
        logging.warning("Using explicitly labelled synthetic timetable: %s", error)
        frame = synthetic_schedule(config["fallback_seed"])
        provenance = {
            "source_type": "synthetic",
            "fallback_reason": str(error),
            "generation_seed": config["fallback_seed"],
            "station_pair": ["A", "B"],
        }
    validate_schedule(frame)
    target = ROOT / "data/processed/train_schedule.csv"
    frame.to_csv(target, index=False)
    frame.to_parquet(target.with_suffix(".parquet"), index=False)
    provenance.update(
        {
            "processed_sha256": digest(target.read_bytes()),
            "train_count": len(frame),
            "priority_assumption": "GTFS: all local (1); synthetic: seeded 20% express (2)",
            "infrastructure": "Hypothetical single-track section; NOT verified MBTA infrastructure",
            "time_rounding": "Service seconds and positive traversal durations rounded up to minutes",
        }
    )
    save_json(ROOT / "data/metadata/dataset.json", provenance)
    save_json(ROOT / "data/metadata/quality.json", report)
    return frame, provenance
