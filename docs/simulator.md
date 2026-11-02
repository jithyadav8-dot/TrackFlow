# Simulator contract

This is an experimental capacity-one section, not a railway interlocking or signaling model.
Both directions share one occupied-train slot. Each direction has a FIFO ordered by
`(ready_time, train_id)`. Priority never permits overtaking within a directional queue.

At decision time `t`, arrivals with readiness <= t have joined the queues and exits at
t have already been processed. Dispatch must pass the independent safety gate.
After dispatch, queued trains accrue one minute of waiting over `[t, t+1)`; an occupying
train accrues one occupied minute. Time advances; any exit due at t+1 is processed,
then new arrivals join. Exit at x blocks entry until `x + clearance`. Zero clearance
permits an entry at x on the following decision, without overlapping occupation.

Only whole integer minutes are supported. GTFS seconds and positive durations are
rounded upward separately; this discretization is explicitly an approximation.
GTFS traversal times remain the nominal durations; independently seeded +/-2 minute
synthetic variation is applied once per train and then stored in its scenario.

The operational horizon is 240 minutes and the episode limit is 360. Every controller
continues its own policy through the drain period. Completion at exactly minute 240
counts in throughput; later completion counts only as drain output. Early termination
leaves the remainder of the horizon idle for time-average metrics. Trains arriving
after the cutoff remain NOT_READY and count as incomplete, with zero incurred wait.

Invariants are checked at initialization and every transition: unique IDs and queue
membership, FIFO, readiness, train conservation, lifecycle/occupancy consistency,
one occupying train, and exact traversal duration. A failed invariant aborts the run;
the evaluator never turns a failing run into a successful zero-violation result.
Invalid action requests are counted and safely held. They are distinct from actual
physical safety violations. Tests inject illegal requests and corrupt state.

Waiting is accumulated once per queued train per interval, after dispatch. Reward
components are completion, per-step waiting, invalid requests, avoidable HOLD, and
cutoff backlog. Waiting is measured from readiness; schedule-relative delay includes
the external arrival disturbance. A completed-train mean can be selection-biased when
trains remain incomplete: always inspect accrued wait and censored backlog alongside it.
