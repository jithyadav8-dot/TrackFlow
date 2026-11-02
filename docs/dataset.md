# Dataset and provenance

Primary source: [official MBTA GTFS ZIP](https://cdn.mbta.com/MBTA_GTFS.zip).
The [Mobility Database](https://mobilitydatabase.org/) is an alternative feed directory.
The pipeline follows the [GTFS Schedule reference](https://gtfs.org/documentation/schedule/reference/).

`prepare_data.py` downloads to a temporary file, checks ZIP integrity, limits archive
size and records URL, UTC retrieval timestamp, SHA-256 and attribution. It reads tables
directly without extracting archive paths. Required file/column and key checks fail
clearly. `--strict` disables the automatic documented synthetic fallback. `--archive`
reprocesses a saved feed offline and `--date YYYY-MM-DD` chooses the service date.
No automatic substitution of a different GTFS date occurs.

Rail types are configurable, with an MBTA `CR-` route-ID override. Calendar weekday
flags and inclusive date intervals are applied, then calendar_dates adds/removes
services. Stops map to parent stations; adjacent repeated parents are collapsed.
Trips with missing/invalid times, duplicate sequences or nonmonotonic times are
rejected and counted. Unknown stop references and out-of-range section durations
are reported. Blank intermediate times are rejected rather than silently interpolated.
Times after 24:00:00 remain on their original service day.

Adjacent station pairs must have both directions and >=4 movements in a four-hour
window. Candidate score is lexicographic: movement count, smaller directional count,
potential section-occupation/clearance overlaps, then negative arrival span.
Station IDs and window start break ties. The selected window preserves original
arrival offsets and individual timetable traversal durations. The top 20 candidate
windows and scoring details are saved in `data/metadata/dataset.json`.

Frequency trips are expanded with start-inclusive/end-exclusive headways.
`exact_times=0` expansion is explicitly marked as a synthetic realization of GTFS
headways, not observed train departure times. Shapes are loaded and counted when
present, but cannot establish track count or signaling infrastructure.

Processed CSV and Parquet contain source trip, route, date and station IDs. All
GTFS trains receive priority 1 because GTFS does not supply dispatch priority.
Synthetic fallback trains receive a seeded 20% probability of priority 2. The single
track, clearance, priorities, perturbations, peak additions, imbalance and saturation
experiments are modeling assumptions. They are never claimed as MBTA observations.

Scenario generation retains the normal timetable. Peak demand adds 50% train objects
using seeded sampling and +/-3 minute arrival jitter. Imbalance assigns exactly a
rounded 70/30 directional split and shuffles it across arrivals. Saturation spreads
demand evenly across the horizon with at least 1.4 times nominal service capacity.
Every schedule transform has a train-level flag. Synthetic disturbances and traversal
durations are fixed before any controller runs. SHA-256-checked manifests freeze
30 validation and 100 test scenarios with disjoint seeds. Training draws only from
the separate seed range starting at 100000.

The experiments reuse one selected timetable date. They test unseen synthetic
perturbations of that timetable, not independent real-world operating conditions.
