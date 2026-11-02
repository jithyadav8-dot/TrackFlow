import math
from collections import Counter

import pandas as pd

from src.data.load_gtfs import active_services, parse_time


def corridor_movements(tables, config, report):
    routes = tables["routes"]
    types = routes.route_type.apply(lambda x: int(x) if x.isdigit() else -1)
    rail = routes[
        types.isin(config["rail_route_types"])
        | routes.route_id.str.startswith(config.get("route_id_prefix", "CR-"))
    ]
    trips = tables["trips"]
    trips = trips[trips.service_id.isin(active_services(tables, config["service_date"]))]
    trips = trips.merge(rail[["route_id"]], on="route_id", validate="many_to_one")
    stop_times = tables["stop_times"]
    stop_times = stop_times[stop_times.trip_id.isin(trips.trip_id)].copy()
    known_stops = set(tables["stops"].stop_id)
    unknown = ~stop_times.stop_id.isin(known_stops)
    report["unknown_stop_records"] = int(unknown.sum())
    stop_times = stop_times[~unknown]
    parents = {}
    for row in tables["stops"].to_dict("records"):
        parents[row["stop_id"]] = row.get("parent_station") or row["stop_id"]
    route_map = dict(zip(trips.trip_id, trips.route_id))
    frequencies = tables.get("frequencies", pd.DataFrame())
    freq_groups = (
        {key: value for key, value in frequencies.groupby("trip_id")} if len(frequencies) else {}
    )
    report["frequency_trip_count"] = len(freq_groups)
    report["frequency_policy"] = (
        "Expand headways; exact_times=0 departures are synthetic realizations"
    )
    rejected = Counter()
    records = []
    for trip_id, group in stop_times.groupby("trip_id", sort=True):
        try:
            group = group.assign(order=pd.to_numeric(group.stop_sequence)).sort_values("order")
            if group.order.duplicated().any():
                raise ValueError("duplicate stop sequence")
            stops = []
            for row in group.itertuples():
                station = parents[row.stop_id]
                arrival, departure = parse_time(row.arrival_time), parse_time(row.departure_time)
                if departure < arrival or (stops and arrival < stops[-1][2]):
                    raise ValueError("nonmonotonic trip")
                if stops and station == stops[-1][0]:
                    stops[-1] = (station, stops[-1][1], departure)
                else:
                    stops.append((station, arrival, departure))
            offsets = [(0, trip_id, "gtfs_derived")]
            if trip_id in freq_groups:
                offsets = []
                for row in freq_groups[trip_id].to_dict("records"):
                    headway = int(row["headway_secs"])
                    if headway <= 0:
                        raise ValueError("nonpositive frequency headway")
                    for start in range(
                        parse_time(row["start_time"]), parse_time(row["end_time"]), headway
                    ):
                        kind = (
                            "gtfs_frequency_derived"
                            if row.get("exact_times") == "1"
                            else "gtfs_frequency_synthetic"
                        )
                        offsets.append((start - stops[0][2], f"{trip_id}@{start}", kind))
            for a, b in zip(stops, stops[1:]):
                duration = (b[1] - a[2]) / 60
                if not config["min_travel_minutes"] <= duration <= config["max_travel_minutes"]:
                    rejected["section_duration"] += 1
                    continue
                pair = tuple(sorted([a[0], b[0]]))
                for offset, movement_id, kind in offsets:
                    records.append(
                        {
                            "pair": pair,
                            "train_id": movement_id,
                            "source_trip_id": trip_id,
                            "source_route_id": route_map[trip_id],
                            "service_date": config["service_date"],
                            "origin_stop_id": a[0],
                            "destination_stop_id": b[0],
                            "direction": int(a[0] != pair[0]),
                            "scheduled_entry_min": math.ceil((a[2] + offset) / 60),
                            "nominal_travel_min": math.ceil(duration),
                            "priority": 1,
                            "source_type": kind,
                        }
                    )
        except (ValueError, IndexError):
            rejected["invalid_trip"] += 1
    report["rejected"] = dict(rejected)
    return pd.DataFrame(records)


def select_corridor(movements, window_minutes=240, clearance=2):
    """Lexicographic score: count, directional balance, potential conflicts, concentration."""
    candidates = []
    if movements.empty:
        raise ValueError("No usable active rail movements on selected service date")
    for pair, group in movements.groupby("pair", sort=True):
        group = group.sort_values(["scheduled_entry_min", "train_id"])
        # Four-hour window keeps timetable spacing, rather than compressing an entire day.
        for start in sorted(group.scheduled_entry_min.unique()):
            part = group[
                (group.scheduled_entry_min >= start)
                & (group.scheduled_entry_min < start + window_minutes)
            ].copy()
            counts = part.direction.value_counts()
            if len(counts) != 2 or len(part) < 4 or part.train_id.duplicated().any():
                continue
            rows = part.to_dict("records")
            conflicts = sum(
                b["scheduled_entry_min"]
                < a["scheduled_entry_min"] + a["nominal_travel_min"] + clearance
                for i, a in enumerate(rows)
                for b in rows[i + 1 :]
            )
            span = int(part.scheduled_entry_min.max() - start)
            score = (len(part), int(counts.min()), conflicts, -span)
            candidates.append((score, pair, int(start), part))
    if not candidates:
        raise ValueError("No bidirectional corridor with >=4 movements in a control window")
    candidates.sort(key=lambda x: (tuple(-v for v in x[0]), x[1], x[2]))
    score, pair, start, selected = candidates[0]
    selected.scheduled_entry_min -= start
    selected = selected.drop(columns="pair").reset_index(drop=True)
    ranking = [
        {"pair": list(p), "start_minute": s, "score": list(sc)} for sc, p, s, _ in candidates[:20]
    ]
    return selected, {
        "station_pair": list(pair),
        "window_start_minute": start,
        "score": list(score),
        "candidate_windows": len(candidates),
        "ranking": ranking,
    }
