import copy
import json
import math

import numpy as np
import pandas as pd

from src.data.validate_gtfs import validate_schedule
from src.utils.config import ROOT, ensure_dirs, load_config
from src.utils.reproducibility import digest, save_json

KINDS = ("normal", "peak", "disruption", "imbalance", "saturation")


def generate_scenario(schedule, kind="normal", seed=42, count=None, delay_scale=1.0, config=None):
    validate_schedule(schedule)
    if kind not in KINDS:
        raise ValueError(f"Unknown scenario kind: {kind}")
    if count is not None and count < 0 or delay_scale < 0:
        raise ValueError("Negative count or delay scale")
    config = copy.deepcopy(config or load_config("simulator")["simulation"])
    rng = np.random.default_rng(seed)
    base = schedule.sort_values(["scheduled_entry_min", "train_id"]).to_dict("records")
    n = len(base) if count is None else count
    if kind == "peak" and count is None:
        n = math.ceil(1.5 * len(base))
    if kind == "saturation" and count is None and base:
        capacity = config["control_horizon_minutes"] / (
            np.mean([r["nominal_travel_min"] for r in base]) + config["clearance_minutes"]
        )
        n = max(len(base), math.ceil(capacity * 1.4))
    if not base and n:
        raise ValueError("Cannot intensify an empty base timetable")
    records = []
    # Keep all original arrivals when no intensification is requested. Added services
    # use a seeded timetable sample with bounded +/-3 minute offsets, explicitly synthetic.
    for i in range(n):
        index = i if i < len(base) else int(rng.integers(len(base)))
        row = base[index]
        arrival = int(row["scheduled_entry_min"])
        modified = i >= len(base)
        if modified:
            arrival = int(
                np.clip(arrival + rng.integers(-3, 4), 0, config["control_horizon_minutes"] - 1)
            )
        if kind == "saturation":
            arrival = int(i * config["control_horizon_minutes"] / max(n, 1))
            modified = True
        direction = int(row["direction"])
        if kind == "imbalance":
            direction = 0 if i < round(0.7 * n) else 1
            modified = True
        records.append(
            {
                "train_id": f"{row['train_id']}#{i}",
                "direction": direction,
                "scheduled_entry_time": arrival,
                "priority": int(row["priority"]),
                "nominal_travel_min": int(row["nominal_travel_min"]),
                "source_trip_id": row["source_trip_id"],
                "source_type": row["source_type"],
                "synthetic_schedule_transform": modified,
            }
        )
    if kind == "imbalance":
        directions = [row["direction"] for row in records]
        rng.shuffle(directions)
        for row, direction in zip(records, directions):
            row["direction"] = direction
    probability = 0.30 if kind == "disruption" else config["random_delay"]["probability"]
    maximum = (
        15
        if kind == "disruption"
        else 3
        if kind == "normal"
        else config["random_delay"]["max_minutes"]
    )
    for row in records:
        delay = (
            int(round(rng.integers(0, maximum + 1) * delay_scale))
            if rng.random() < probability
            else 0
        )
        variation = int(
            rng.integers(
                -config["travel_variation_minutes"], config["travel_variation_minutes"] + 1
            )
        )
        row["ready_time"] = row["scheduled_entry_time"] + delay
        row["travel_duration"] = max(1, row["nominal_travel_min"] + variation)
    records.sort(key=lambda r: (r["ready_time"], r["train_id"]))
    return {
        "scenario_id": f"{kind}-{seed}",
        "scenario_seed": int(seed),
        "kind": kind,
        "simulation": config,
        "schedule_sha256": digest(schedule.to_csv(index=False).encode()),
        "assumptions": {
            "bottleneck": "hypothetical single track",
            "delays": "seeded synthetic",
            "duration_variation": "seeded synthetic around timetable duration",
            "intensification": "seeded sampling with +/-3 minute jitter",
            "delay_scale": delay_scale,
        },
        "trains": records,
    }


def save_splits():
    ensure_dirs()
    settings = load_config("scenarios")
    schedule = pd.read_csv(ROOT / "data/processed/train_schedule.csv")
    manifest = {}
    for split in ["validation", "test"]:
        directory = ROOT / "data/scenarios" / split
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(settings[split]["count"]):
            scenario = generate_scenario(
                schedule, KINDS[i % len(KINDS)], settings[split]["seed_start"] + i
            )
            name = scenario["scenario_id"] + ".json"
            save_json(directory / name, scenario)
            paths.append({"file": name, "sha256": digest(scenario)})
        manifest[split] = paths
    save_json(ROOT / "data/scenarios/manifest.json", manifest)
    return manifest


def load_split(split):
    manifest = json.loads((ROOT / "data/scenarios/manifest.json").read_text())
    result = []
    for entry in manifest[split]:
        scenario = json.loads((ROOT / "data/scenarios" / split / entry["file"]).read_text())
        if digest(scenario) != entry["sha256"]:
            raise ValueError(f"Scenario checksum mismatch: {entry['file']}")
        result.append(scenario)
    return result
