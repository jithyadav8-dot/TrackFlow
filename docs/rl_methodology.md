# Reinforcement learning protocol

MaskablePPO uses separate policy/value MLP branches, each with hidden layers
128,128 and tanh activations. The configuration includes learning rate .0003,
gamma .99, GAE .95, rollout 1024 per environment, batch 256, ten epochs,
clip .2, entropy .01, value coefficient .5 and max gradient norm .5. Four
DummyVecEnv environments execute on CPU with PyTorch restricted to one thread.
CUDA can be enabled in YAML, but numerical reproducibility is not guaranteed there.

HOLD is always legal. Each directional action additionally requires a ready queue,
an unoccupied section and completed clearance. Predictions explicitly pass masks;
training discovers `action_masks()` on the environment. Physical constraints are
also enforced by the independent simulator gate. The model never bypasses it.
Reload smoke evaluation uses SB3-Contrib's masked evaluation utility as required
by the [MaskablePPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).

The 14-vector is float32 clipped to [-1,1]. N=max(train count,1); D=episode limit.

|Index|Feature|Normalization / sentinel|
|---:|---|---|
|0|Episode time remaining|(D-t)/D|
|1|Physical occupancy|0 or 1|
|2|Remaining traversal or clearance|minutes/D|
|3,4|Directional queues|length/N|
|5,6|Head wait|(t-ready)/D; -1 for empty|
|7,8|Head priority|priority/2; -1 for empty|
|9,10|Next timetable arrival|max(scheduled-t,0)/D; -1 if none pending|
|11|Last direction|-1 none, 0 A to B, 1 B to A|
|12|Completed fraction|completed/N|
|13|Uncompleted fraction|(N-completed)/N, using actual total in numerator|

Future actual ready times, delay realizations and travel noise are not observed.
Only timetable times of not-yet-ready trains are used for next-arrival features.
Thus delayed trains whose scheduled time passed show zero until they arrive.
The current head's waiting time is known. The policy does not see head traversal
duration: this partial observability can limit scheduling of unequal-duration trains.

Reward: +2 per completion, -.05 per priority-weighted waiting train-minute,
-.10 per invalid request, -.02 per avoidable HOLD. At truncation only, -2 per
uncompleted train is a separately logged finite backlog penalty. There is no
double counting of cumulative waits. All five terms are saved per episode.

Episodes end when all trains finish, or truncate at the 360-minute limit.
The same learned policy controls the drain. A fixed validation set chooses
checkpoints by `mean(accrued_wait/total + 360*incomplete/total)` across scenarios,
with max(total,1) for empty cases. This includes censored waiting rather than
rewarding abandonment. Selection never reads the held-out test set.

Smoke training requests 20000 steps. Full training requests 500000 steps per seed;
SB3 rounds to a complete rollout, so metadata records actual steps. The supported
seeds are 42,123,2026; only completed runs in the registry may be reported as trained.
Losses/entropy/KL are in CSV and TensorBoard; operational metrics and reward components
are in episodes.csv. Validation scores, checkpoints, hashes and software versions
are saved with each run. CLI reward flags support controlled validation-only studies.

## Executed exploratory reward study

Five seed-42 runs each completed 53248 transitions (50000 requested, rounded to
rollouts): default control, completion reward 4, waiting coefficient .10, HOLD
coefficient .10 and disabled priority weighting. The fixed validation cost was
103.5739 for control, waiting, HOLD and unweighted; completion reward 4 reached
103.5303. These short-run differences are small and do not justify a claim of
superiority. Disabling priority weighting is mathematically neutral for this
GTFS-derived dataset because every train has priority one. All variants learned
zero avoidable HOLD on validation at their selected checkpoints. The study used
validation only and did not replace the registered final-test models. Detailed
outputs and its prespecified plan are under `results/reward_sensitivity.csv` and
`results/reward_study_plan.json`.
