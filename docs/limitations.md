# Scientific and technical limits

- Published GTFS is a timetable, not observed operations or railway infrastructure.
  Neither station-pair selection nor shapes establish a real single-track section.
- This experiment imposes a hypothetical single-track bottleneck with two FIFO queues,
  one train in flight, integer-minute traversal and a constant clearance interval.
  It does not model braking, signal aspects, blocks, switching, platforms, train length,
  route locking, timetable recovery or passenger capacity. It is not deployment-ready.
- Normal schedules use one GTFS date/window. Unseen test seeds are synthetic perturbations,
  not independent operational days. Peak, imbalance and saturation alter that timetable.
- All GTFS train priorities are set to one. Consequently priority and FCFS coincide on
  these experiments; synthetic schedules and hand-built tests exercise unequal priority.
- Longest-waiting-first and FCFS are mathematically equivalent with readiness-based
  FIFO and identical tie-breaking. They are retained as a useful correctness check.
- Single-section decisions are limited. A deterministic work-conserving directional
  preference may rival RL. There is no assumption that RL must win on throughput or delay.
- The 14-feature policy does not observe head travel durations, full queues or realized
  future delays. That partial observability limits scheduling decisions. HOLD may be
  wasteful. Action counts and avoidable holds are exported for diagnosis.
- Completed-train mean delay alone can favor policies that leave difficult trains behind.
  Backlog, censored waits and accrued waiting must be considered jointly. Validation uses
  an explicit wait-plus-backlog objective rather than this potentially biased mean.
- Confidence intervals are scenario bootstraps conditional on one timetable and a small
  set of model seeds. Multiple comparison adjustments and independent service-date tests
  would be needed for stronger claims.
- Python 3.12 is used in this environment because 3.11 was unavailable. Package metadata
  permits 3.11 and 3.12, but only the installed runtime was executed here. The lockfile
  records resolved versions for this Windows/Python 3.12 run; re-resolve requirements.txt
  on other platforms if a transitive wheel is unavailable. CPU results are reproducible
  under the saved stack; CUDA and different numeric libraries can vary.
- The feed ZIP is excluded from Git by default because of size. Keep the archive and
  checksum locally for exact reprocessing. Checked-in processed data and fixed scenarios
  support offline reruns without relying on the source URL staying available.

Future extensions: multiple block sections with per-block safety invariants; passing
loops with finite holding capacity and route reservation; coordinated or multi-agent
policies with a centralized conflict arbiter; real-time GTFS updates with clear observed
vs forecast timestamps; validated signaling/braking models; network-wide objectives
including missed connections, passenger delay and dispatch fairness. These require new
validation scenarios and must preserve the working single-section reference system.
