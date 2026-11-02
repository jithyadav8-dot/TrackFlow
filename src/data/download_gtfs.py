from datetime import datetime, timezone
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import requests

from src.utils.reproducibility import digest, save_json


def download_gtfs(url, destination, metadata_path):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")
    try:
        with requests.get(url, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            size = 0
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    if size > 300 * 1024 * 1024:
                        raise ValueError("GTFS download exceeds 300 MiB limit")
                    handle.write(chunk)
        with ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise BadZipFile("GTFS CRC validation failed")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    record = {
        "source_url": url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "sha256": digest(destination.read_bytes()),
        "attribution": "Massachusetts Bay Transportation Authority (MBTA)",
        "source_type": "public_gtfs_timetable",
    }
    save_json(metadata_path, record)
    return record
