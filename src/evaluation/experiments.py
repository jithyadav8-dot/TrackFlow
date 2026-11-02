import json
from pathlib import Path

import pandas as pd

from src.controllers import CONTROLLERS
from src.data.scenario_generator import load_split
from src.evaluation.reporting import summarize
from src.evaluation.statistics import comparisons
from src.rl.policy_adapter import PPOController
from src.simulator.environment import TrainTrafficEnv
from src.simulator.metrics import calculate_metrics, train_metrics
from src.utils.config import ROOT, ensure_dirs, load_config
from src.utils.reproducibility import digest, metadata, save_json, seed_all


def run_episode(scenario, controller, record=False):
    before = digest(scenario)
    env = TrainTrafficEnv(scenario, record=record)
    env.reset()
    while not env.section.done:
        action = (
            controller.action(env)
            if isinstance(controller, PPOController)
            else controller.action(env.section)
        )
        env.step(action)
    metrics = calculate_metrics(env.section)
    if digest(scenario) != before:
        raise AssertionError("Controller mutated scenario definition")
    return metrics, train_metrics(env.section), env.section.frames, env.section.events


def evaluate(split="test", baselines_only=False, checkpoint=None, output=None):
    ensure_dirs()
    seed_all(42)
    output = Path(output) if output else ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    scenarios = load_split(split)
    controllers = {name: factory() for name, factory in CONTROLLERS.items()}
    checkpoints, seeds, model_details = {}, {}, {}
    if not baselines_only:
        registry_path = ROOT / "models/registry.json"
        if checkpoint:
            controllers["PPO"] = PPOController(checkpoint)
            checkpoints["PPO"] = str(checkpoint)
        elif registry_path.exists():
            registry = json.loads(registry_path.read_text())
            for seed, info in registry.items():
                name = f"PPO seed {seed}"
                path = ROOT / info["checkpoint"].replace("\\", "/")
                if digest(path.read_bytes()) != info["checkpoint_sha256"]:
                    raise ValueError("Model checksum mismatch")
                if (
                    digest((ROOT / "data/processed/train_schedule.csv").read_bytes())
                    != info["dataset_checksum"]
                ):
                    raise ValueError(
                        "Current dataset differs from training; regenerate/train intentionally"
                    )
                controllers[name] = PPOController(path)
                checkpoints[name] = str(path.relative_to(ROOT).as_posix())
                seeds[name] = int(seed)
                model_details[name] = info
        else:
            raise FileNotFoundError(
                "No trained-model registry. Train first or explicitly use --baselines-only."
            )
    rows, trains, frames, events, runs = [], [], [], [], []
    config = load_config("evaluation")
    dataset = json.loads((ROOT / "data/metadata/dataset.json").read_text())
    meta = metadata()
    for i, scenario in enumerate(scenarios):
        for name, controller in controllers.items():
            result, train_rows, frame_rows, event_rows = run_episode(
                scenario, controller, record=(i < 5)
            )
            identity = {
                "scenario_id": scenario["scenario_id"],
                "scenario_seed": scenario["scenario_seed"],
                "kind": scenario["kind"],
                "controller": name,
                "training_seed": seeds.get(name),
                "model_checkpoint": checkpoints.get(name),
                "scenario_checksum": digest(scenario),
            }
            rows.append({**identity, **result})
            trains.extend({**identity, **row} for row in train_rows)
            frames.extend({**identity, **row} for row in frame_rows)
            events.extend({**identity, **row} for row in event_rows)
            runs.append(
                {
                    **identity,
                    "evaluation_metrics": result,
                    "configuration_checksum": digest(scenario["simulation"]),
                }
            )
        if (i + 1) % 20 == 0:
            print(f"Evaluated {i + 1}/{len(scenarios)} {split} scenarios", flush=True)
    episodes = pd.DataFrame(rows)
    summary_input = episodes.copy()
    rl = episodes[episodes.controller.str.startswith("PPO")]
    # Across-seed mean per scenario: bootstrap scenarios, never treat 3 seeds x 100
    # scenarios as 300 independent test conditions.
    if rl.controller.nunique() > 1:
        aggregated = rl.groupby(["scenario_id", "kind"], as_index=False).mean(numeric_only=True)
        aggregated["controller"] = "PPO seed mean"
        summary_input = pd.concat([summary_input, aggregated], ignore_index=True)
    summary = summarize(summary_input, config)
    differences, intervals = comparisons(summary_input, config)
    for name, frame in [
        ("episode_metrics", episodes),
        ("train_metrics", pd.DataFrame(trains)),
        ("queue_frames", pd.DataFrame(frames)),
        ("events", pd.DataFrame(events)),
        ("comparison_summary", summary),
        ("paired_differences", differences),
        ("paired_summary", intervals),
    ]:
        frame.to_csv(output / f"{name}.csv", index=False)
    experiment_id = digest(
        {
            "scenarios": [digest(s) for s in scenarios],
            "models": checkpoints,
            "timestamp": meta["timestamp"],
        }
    )[:16]
    source_hashes = {
        p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in (ROOT / "src").rglob("*.py")
    }
    save_json(
        output / "experiment_metadata.json",
        {
            **meta,
            "experiment_id": experiment_id,
            "split": split,
            "dataset_source": dataset,
            "dataset_checksum": dataset["processed_sha256"],
            "configuration": config,
            "configuration_checksum": digest(config),
            "model_details": model_details,
            "source_checksums": source_hashes,
            "runs": runs,
            "inference": "deterministic, explicit action masks",
            "trace_scenarios": [s["scenario_id"] for s in scenarios[:5]],
            "bootstrap_unit": "scenario; seed-mean rows average model seeds within scenario first",
        },
    )
    print(
        summary[summary.kind == "all"][
            [
                "controller",
                "throughput",
                "mean_delay",
                "p95_delay",
                "utilization",
                "incomplete_trains",
                "safety_violations",
            ]
        ].to_string(index=False)
    )
    return episodes, summary
