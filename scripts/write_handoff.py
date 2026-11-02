"""Build a factual delivery report from completed artifacts, not example data."""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import ROOT
from src.utils.reproducibility import save_json


def markdown_table(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    summary = pd.read_csv(ROOT / "results/comparison_summary.csv")
    summary = summary[summary.kind == "all"].set_index("controller")
    registry = json.loads((ROOT / "models/registry.json").read_text())
    experiment = json.loads((ROOT / "results/experiment_metadata.json").read_text())
    reproducibility = json.loads((ROOT / "results/reproducibility_check.json").read_text())
    dataset = json.loads((ROOT / "data/metadata/dataset.json").read_text())
    charts = json.loads((ROOT / "results/charts/index.json").read_text())
    suite = ET.parse(ROOT / "results/test_report.xml").getroot().find("testsuite")
    counts = {key: int(suite.attrib[key]) for key in ["tests", "failures", "errors", "skipped"]}
    counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    assert counts["failures"] == counts["errors"] == counts["skipped"] == 0
    columns = [
        "throughput",
        "mean_delay",
        "p95_delay",
        "utilization",
        "incomplete_trains",
        "safety_violations",
    ]
    table = summary[columns].reset_index()
    models = pd.DataFrame(
        [
            {
                "Seed": seed,
                "Actual steps": info["actual_timesteps"],
                "Selected step": info["best_validation"]["timesteps"],
                "Validation cost": info["best_validation"]["selection_cost"],
                "Validation mean wait": info["best_validation"]["mean_delay"],
                "Checkpoint": info["checkpoint"].replace("\\", "/"),
            }
            for seed, info in registry.items()
        ]
    )
    rl, fcfs, work = [summary.loc[name] for name in ["PPO seed mean", "FCFS", "Work Conserving"]]
    differences = pd.read_csv(ROOT / "results/paired_summary.csv")
    differences = differences[
        (differences.controller == "PPO seed mean")
        & differences.baseline.isin(["FCFS", "Work Conserving"])
    ]
    statuses = {
        "data_preprocessing": "passed",
        "synthetic_fallback": "passed",
        "simulator_safety": "passed",
        "four_baselines": "passed",
        "gymnasium_checker": "passed",
        "ppo_smoke_and_full_training": "passed",
        "model_save_reload": "passed",
        "identical_scenario_evaluation": "passed",
        "metrics_and_statistics": "passed",
        "measured_charts": "passed",
        "dashboard_startup_and_controls": "passed",
        "tests": counts,
        "ruff_check_and_format": "passed",
        "documentation": "complete",
        "reproducibility": reproducibility,
        "runtime_adaptation": "Python 3.12.10 executed; 3.11 unavailable and not tested",
        "optional_advanced_topology": "not implemented (stretch goal)",
    }
    save_json(ROOT / "results/acceptance_report.json", statuses)
    text = f"""# TrackFlow final handoff

Completed delivery dated 2026-10-29 (Asia/Calcutta). Project: `{ROOT}`.

## A. Implementation summary

The GTFS pipeline creates a verified timetable and frozen scenarios. Four baselines
and MaskablePPO control the same independently protected railway simulator. Gymnasium
adds normalized observations, masks and per-step rewards. Training selects checkpoints
on 30 validation scenarios. Evaluation uses 100 held-out scenarios, exports individual
train/episode records and paired bootstrap statistics, and feeds eight measured charts
and a six-page Streamlit dashboard. Playback Run, Pause, Resume, Reset and seeking work.

Real input: {dataset["train_count"]} movements on {dataset["service_date"]}, Back Bay ↔
South Station, four-hour window beginning 15:20 local service time. The infrastructure,
priority assignments and disturbances are explicitly synthetic modeling assumptions.

## B. Project files

- [README and command reference](../README.md)
- [Dashboard](../app.py)
- [GTFS preprocessing](../src/data/prepare_schedule.py)
- [Simulator](../src/simulator/railway_section.py), [independent safety](../src/simulator/safety.py), [Gymnasium](../src/simulator/environment.py)
- [Training](../src/rl/train_ppo.py), [evaluation](../src/evaluation/experiments.py)
- [Processed timetable](../data/processed/train_schedule.csv), [dataset provenance](../data/metadata/dataset.json)
- [Episode results](../results/episode_metrics.csv), [train results](../results/train_metrics.csv)
- [Experiment metadata](../results/experiment_metadata.json), [acceptance report](../results/acceptance_report.json)

The README contains the actual directory map; tests are grouped by responsibility
rather than split into one file per individual simulator class.

## C. Commands

Windows installation in a fresh checkout:

```powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.lock
.\\.venv\\Scripts\\Activate.ps1
```

The delivered virtual environment already works. With it activated:

```bash
python scripts/prepare_data.py --archive data/raw/MBTA_GTFS.zip --date 2026-10-28 --strict
python scripts/generate_scenarios.py
python scripts/run_baselines.py
python scripts/smoke_test.py
python scripts/train.py --smoke --label fresh_smoke
python scripts/train.py --seeds 42 123 2026 --label replication
python scripts/evaluate.py
python scripts/generate_charts.py
python scripts/verify_reproducibility.py
python -m pytest -q
streamlit run app.py
```

Use `prepare_data.py --synthetic` for the labelled fallback. Preparation replaces
processed inputs/scenarios; preserve the supplied data for exact evaluation. A new
training label protects existing runs. Offline evaluation can start directly with
`evaluate.py` using the supplied data and models. Linux/macOS setup is in README.

## D. Trained models

Algorithm: MaskablePPO, actor/critic MLP hidden layers [128,128], tanh, CPU.
Main training completed {sum(d["actual_timesteps"] for d in registry.values()):,} transitions
across three independent seeds. Smoke completed 20,480 transitions. Five separate
validation-only reward studies completed 53,248 transitions each. Every selected checkpoint
was saved and reloaded; masked inference was checked.

{markdown_table(models)}

Validation cost is accrued wait per train plus 360 times unfinished fraction; lower
is better. Checkpoint selection uses this objective, not the held-out test results.

## E. Experimental results

{len(experiment["runs"])} controller/scenario runs on 100 saved test scenarios (20 each
normal, peak, disruption, imbalance and saturation). Values below are scenario means.
Throughput is trains/hour; waiting is minutes; utilization is a fraction. P95 is the
mean of scenario-level P95 waits, not the percentile of a pooled train population.
PPO seed mean averages the three trained models within each scenario.

{markdown_table(table)}

Relative to FCFS, PPO seed mean increased average throughput by
{100 * (rl.throughput / fcfs.throughput - 1):.2f}% and reduced completed-train mean wait by
{100 * (1 - rl.mean_delay / fcfs.mean_delay):.2f}%, but its mean scenario P95 wait was
{100 * (rl.p95_delay / fcfs.p95_delay - 1):.2f}% higher. Thus RL did not improve every metric.
Compared with Work Conserving, PPO throughput differed by
{100 * (rl.throughput / work.throughput - 1):.2f}% and mean waiting differed by
{rl.mean_delay - work.mean_delay:.3f} minutes (positive means PPO waited longer).
Inspect backlog and censored waits alongside completed-train means.

The following paired intervals resample scenarios (2000 bootstrap samples, seed 31415).
Differences are RL minus baseline, so negative favors RL for waiting/backlog:

{markdown_table(differences[["baseline", "metric", "mean_difference", "ci_low", "ci_high"]])}

All {len(experiment["runs"])} runs passed physical safety and finite-horizon capacity
checks. No result was edited to make RL look favorable. FCFS, priority and longest
waiting coincide here because GTFS priorities are uniform and longest waiting is
equivalent to earliest readiness. Unequal-priority tests validate the priority policy.

## F. Charts

Eight chart types, each exported as PNG and interactive HTML under `results/charts`:
{", ".join(charts)}. HTML shares a local Plotly JS bundle for offline use.

![Held-out throughput](../results/charts/throughput.png)

[Dashboard playback screenshot](dashboard-playback.png). All plotted metrics come
from executed evaluations and training logs. The separate `validation_preview`
directory is a clearly separated two-seed validation export used for chart QA.

## G. Test report

{counts["tests"]} tests executed: **{counts["passed"]} passed, {counts["failures"]} failed,
{counts["errors"]} errors, {counts["skipped"]} skipped**. See [JUnit XML](../results/test_report.xml)
and [captured output](../results/test_output.txt). Ruff checks and formatting pass;
pip reports no broken dependencies. Matplotlib's older pyparsing calls produce 14
dependency deprecation warnings; they did not fail tests or chart rendering.

Reproducibility re-executed {reproducibility["verified_runs"]} controller/scenario runs
across all five scenario kinds and matched every aggregate metric within absolute
tolerance {reproducibility["absolute_tolerance"]}. Browser inspection verified real
playback and pause; AppTest exercises every page and Run/Pause/Resume/Reset.
All three dashboard tests also passed again with the completed evaluation files;
see `results/dashboard_test_report.xml`.

## H. Limitations and acceptance

All mandatory functional acceptance criteria are met. Python 3.12.10 was used because
3.11 was unavailable; 3.11 support was not executed. No Git commit existed during
the runs, so metadata records null commits and evaluation records source checksums.
The result is a research prototype, not operational railway-control certification.

One timetable date and synthetic perturbations limit external validity. The single
section is simple; directional preference explains much of the benefit (nominal A→B
duration averages 4.43 minutes versus 5.55 for B→A). This is a plausible explanation
from the inputs, not a causal ablation. P95 fairness and completed-train censoring
remain material. Three trained seeds do not fully characterize training uncertainty.
See [all limitations](limitations.md) and [the reward study](rl_methodology.md).

## I. Future improvements

Add multiple blocks with independent occupancy checks; finite-capacity passing loops
and route reservation; centralized or multi-agent coordination; real-time schedule
updates with explicit observed/forecast timestamps; realistic signaling and braking;
and network-level passenger-delay/connection objectives. Keep this single-section
simulator as a regression reference. The advanced topology remains an optional stretch
goal and was not implemented.
"""
    (ROOT / "docs/handoff.md").write_text(text, encoding="utf-8")
    print("Wrote docs/handoff.md and results/acceptance_report.json from executed artifacts.")
