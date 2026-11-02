from datetime import datetime
from zipfile import ZipFile

import pandas as pd

REQUIRED = ("routes", "trips", "stop_times", "stops")
OPTIONAL = ("calendar", "calendar_dates", "shapes", "frequencies", "feed_info", "agency")


def parse_time(value):
    """GTFS service-day seconds; hours may exceed 23. Never wrap at midnight."""
    pieces = str(value).split(":")
    if len(pieces) != 3 or any(not p.isdigit() for p in pieces):
        raise ValueError(f"Invalid GTFS time: {value!r}")
    hours, minutes, seconds = map(int, pieces)
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"Invalid GTFS time: {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def load_gtfs(path):
    tables = {}
    with ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist()) > 2_000_000_000:
            raise ValueError("Uncompressed GTFS exceeds 2 GB limit")
        members = {name.rsplit("/", 1)[-1]: name for name in archive.namelist()}
        for name in REQUIRED + OPTIONAL:
            if f"{name}.txt" in members:
                with archive.open(members[f"{name}.txt"]) as handle:
                    # Shapes are validated as a table, but not treated as infrastructure.
                    tables[name] = pd.read_csv(handle, dtype=str, keep_default_na=False)
            elif name in REQUIRED:
                raise ValueError(f"Missing required GTFS file: {name}.txt")
    if "calendar" not in tables and "calendar_dates" not in tables:
        raise ValueError("GTFS requires calendar.txt or calendar_dates.txt")
    return tables


def active_services(tables, service_date):
    day = datetime.strptime(service_date, "%Y-%m-%d")
    stamp = day.strftime("%Y%m%d")
    weekday = day.strftime("%A").lower()
    active = set()
    if "calendar" in tables:
        calendar = tables["calendar"]
        selected = calendar[
            (calendar.start_date <= stamp)
            & (calendar.end_date >= stamp)
            & (calendar[weekday] == "1")
        ]
        active.update(selected.service_id)
    if "calendar_dates" in tables:
        for row in tables["calendar_dates"].itertuples():
            if row.date == stamp:
                if row.exception_type == "1":
                    active.add(row.service_id)
                elif row.exception_type == "2":
                    active.discard(row.service_id)
    return active
