import numpy as np

SCHEDULE_COLUMNS = [
    "train_id",
    "source_trip_id",
    "source_route_id",
    "service_date",
    "origin_stop_id",
    "destination_stop_id",
    "direction",
    "scheduled_entry_min",
    "nominal_travel_min",
    "priority",
    "source_type",
]


def validate_tables(tables):
    columns = {
        "routes": ["route_id", "route_type"],
        "trips": ["trip_id", "route_id", "service_id"],
        "stops": ["stop_id"],
        "stop_times": ["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"],
        "calendar": [
            "service_id",
            "start_date",
            "end_date",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ],
        "calendar_dates": ["service_id", "date", "exception_type"],
        "frequencies": ["trip_id", "start_time", "end_time", "headway_secs"],
    }
    for name, required in columns.items():
        if name in tables and set(required) - set(tables[name].columns):
            raise ValueError(f"Missing columns in {name}: {set(required) - set(tables[name])}")
    for table, key in [("routes", "route_id"), ("trips", "trip_id"), ("stops", "stop_id")]:
        if tables[table][key].duplicated().any():
            raise ValueError(f"Duplicate primary keys in {table}")
    return {
        "rows": {name: len(frame) for name, frame in tables.items()},
        "missing_optional": [
            name
            for name in ["shapes", "frequencies", "calendar", "calendar_dates"]
            if name not in tables
        ],
    }


def validate_schedule(frame):
    if set(SCHEDULE_COLUMNS) - set(frame):
        raise ValueError("Incomplete processed schedule schema")
    if frame.train_id.duplicated().any() or frame.train_id.isna().any():
        raise ValueError("Train IDs must be unique and non-null")
    for column in ["scheduled_entry_min", "nominal_travel_min", "priority", "direction"]:
        if not np.isfinite(frame[column]).all() or (frame[column] % 1 != 0).any():
            raise ValueError(f"{column} must contain finite integer values")
    if not frame.direction.isin([0, 1]).all():
        raise ValueError("Directions must be 0 or 1")
    if (frame.scheduled_entry_min < 0).any() or (frame.nominal_travel_min < 1).any():
        raise ValueError("Invalid schedule times")
    if (frame.priority < 1).any():
        raise ValueError("Invalid priority")
    return frame
