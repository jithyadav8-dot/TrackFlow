# Architecture

```text
Public GTFS ZIP → validated tables → ranked corridor → CSV / Parquet schedule
                                                        ↓
                                fixed JSON scenarios + SHA-256 manifest
                                                        ↓
 FCFS / Priority / Longest Waiting / Work Conserving / MaskablePPO
                                                        ↓
                         Gymnasium adapter → independent safety gate
                                                        ↓
                           capacity-one discrete-time railway simulator
                                                        ↓
                  episode metrics + train records + events + queue frames
                                                        ↓
                      paired statistics → measured charts → Streamlit
```

Data ingestion has no dependency on RL. The simulator accepts fully realized scenario
records and never draws random disturbances internally. Controllers cannot bypass
the simulator gate. Gymnasium adds observations and reward to that same transition
system. Training uses dynamic scenarios from an isolated seed range, evaluates only
the frozen validation set and saves checkpoints with full metadata. Evaluation
reloads immutable test cases and creates independent train objects for each controller.

The dashboard reads artifacts and can simulate a user-selected scenario. Simulation
playback is a recorded event/frame sequence, advanced by Streamlit fragments. Pause,
resume, seek and reset operate on session state without retraining the model or
changing the underlying scenario. Model loading is explicit and baseline substitution
is prohibited. Missing input, model and results states show executable next steps.

Code lives in `src/data`, `src/simulator`, `src/controllers`, `src/rl`,
`src/evaluation`, `src/visualization`, and `src/utils`. CLI scripts bootstrap the
project root, so they work without installing the package as editable. All project
data paths derive from the source location, not the shell's current directory.
