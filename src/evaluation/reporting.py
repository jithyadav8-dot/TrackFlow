import pandas as pd

from src.evaluation.statistics import bootstrap_mean

METRICS = [
    "throughput",
    "mean_delay",
    "p95_delay",
    "utilization",
    "incomplete_trains",
    "safety_violations",
    "max_queue",
    "average_queue",
    "backlog_control",
    "drain_completions",
    "censored_wait_total",
    "selection_cost",
    "avoidable_holds",
]


def summarize(episodes, config):
    records = []
    for kind in ["all", *sorted(episodes.kind.unique())]:
        filtered = episodes if kind == "all" else episodes[episodes.kind == kind]
        for controller, group in filtered.groupby("controller", sort=True):
            row = {"controller": controller, "kind": kind, "scenario_count": len(group)}
            for metric in METRICS:
                mean, low, high = bootstrap_mean(
                    group[metric],
                    config["bootstrap_seed"],
                    config["bootstrap_samples"],
                    config["confidence"],
                )
                row.update({metric: mean, f"{metric}_ci_low": low, f"{metric}_ci_high": high})
            records.append(row)
    return pd.DataFrame(records)
