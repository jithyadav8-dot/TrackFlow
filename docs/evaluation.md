# Evaluation protocol

30 fixed validation scenarios select checkpoints. 100 separate test scenarios
(20 per kind) measure final behavior. Scenario seeds and all random arrival/travel
realizations are saved before evaluation and checked against a SHA-256 manifest.
Every controller receives a fresh copy of the same scenario. An input hash is checked
before and after each episode; traces include scenario, controller and checkpoint.

Throughput = completions at or before minute 240 / 4 hours. Drain completions are
excluded. Utilization = physically occupied minutes in [0,240) / 240, excluding
clearance. Queue means use the same interval and include remaining idle time after
early termination. Queue maxima measure waiting queues after each dispatch decision.

Mean and P95 waiting delay use trains completed by the final cutoff, including drain
completions. Waiting = actual entry - ready time. Schedule-relative delay = actual
entry - scheduled entry. No-completion means are undefined (null), not fabricated zeros.
Also report unfinished counts, not-yet-entered censored waits, all accrued waiting,
main-horizon backlog and drain output. Max wait includes entered and censored waits.
Selection cost includes all waiting plus 360 times incomplete fraction.

For each scenario, throughput and utilization gains are `(RL-baseline)/baseline`.
Delay/backlog reductions are `(baseline-RL)/baseline`. Percentages with zero denominators
are undefined. Paired differences always use `RL - baseline`, so negative differences
favor RL for delay. A fixed-seed 2000-resample percentile bootstrap gives 95% intervals
over scenarios, not time steps. These intervals measure this synthetic scenario mixture;
they do not establish real-world causal performance or correct multiple comparisons.

Each independently trained PPO seed is reported separately. A seed-mean summary
first averages models within a scenario, then bootstraps the 100 scenario means.
It does not misrepresent 300 correlated scenario/model runs as independent scenarios.
Scenario-only intervals are conditional on the trained seeds and base timetable;
they do not fully characterize training-population uncertainty with only three seeds.

The service bound sorts assigned train durations, then finds the largest k with
`sum(k shortest durations) + (k-1)*clearance <= horizon`. Ignoring actual arrival times
makes this an optimistic but valid finite-horizon upper bound. Every run checks it.
For constant 8-minute travel and 2-minute clearance, the sustained rate is 6 trains/hour.
Different GTFS durations change capacity: results must not be compared to that example
as if all trains took eight minutes.

`results/episode_metrics.csv` and `train_metrics.csv` retain individual runs.
`comparison_summary.csv` includes overall and per-kind bootstrap intervals.
`paired_differences.csv` and `paired_summary.csv` retain scenario pairs and their
intervals. The first five test scenarios have full queue frames and entry/exit events
for charting, one from each kind. Experiment JSON stores software/config/dataset/model
hashes, code hashes and every run's metrics. A null git commit means the repository
had no commit; source hashes still identify the executed files.
