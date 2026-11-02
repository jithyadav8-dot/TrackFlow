import numpy as np
import pandas as pd


def percentage_improvement(candidate, baseline, higher=True):
    if not np.isfinite(candidate) or not np.isfinite(baseline) or baseline == 0:
        return None
    return float(100 * (candidate - baseline) / baseline * (1 if higher else -1))


def bootstrap_mean(values, seed=31415, samples=2000, confidence=0.95):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None, None, None
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return float(values.mean()), float(low), float(high)


def comparisons(episodes, config):
    records, intervals = [], []
    baselines = [name for name in episodes.controller.unique() if not name.startswith("PPO")]
    candidates = [name for name in episodes.controller.unique() if name.startswith("PPO")]
    for name in candidates:
        candidate = episodes[episodes.controller == name]
        for baseline in baselines:
            paired = candidate.merge(
                episodes[episodes.controller == baseline],
                on="scenario_id",
                suffixes=("_rl", "_base"),
                validate="one_to_one",
            )
            for metric, higher in [
                ("throughput", True),
                ("mean_delay", False),
                ("p95_delay", False),
                ("incomplete_trains", False),
                ("selection_cost", False),
                ("utilization", True),
            ]:
                differences = []
                for row in paired.to_dict("records"):
                    rl, base = row[f"{metric}_rl"], row[f"{metric}_base"]
                    difference = rl - base
                    differences.append(difference)
                    records.append(
                        {
                            "controller": name,
                            "baseline": baseline,
                            "scenario_id": row["scenario_id"],
                            "kind": row["kind_rl"],
                            "metric": metric,
                            "difference_rl_minus_baseline": difference,
                            "improvement_percent": percentage_improvement(rl, base, higher),
                        }
                    )
                mean, low, high = bootstrap_mean(
                    differences,
                    config["bootstrap_seed"],
                    config["bootstrap_samples"],
                    config["confidence"],
                )
                intervals.append(
                    {
                        "controller": name,
                        "baseline": baseline,
                        "metric": metric,
                        "mean_difference": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "paired_scenarios": len(paired),
                    }
                )
    return pd.DataFrame(records), pd.DataFrame(intervals)
