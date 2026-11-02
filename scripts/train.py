import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rl.train_ppo import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--label")
    parser.add_argument("--waiting-weight", type=float)
    parser.add_argument("--completion-weight", type=float)
    parser.add_argument("--hold-weight", type=float)
    parser.add_argument("--unweighted", action="store_true")
    args = parser.parse_args()
    override = {
        key: value
        for key, value in [
            ("waiting", args.waiting_weight),
            ("completion", args.completion_weight),
            ("hold", args.hold_weight),
        ]
        if value is not None
    }
    if args.unweighted:
        override["priority_weighting"] = False
    for seed in args.seeds or [args.seed]:
        label = f"{args.label}_seed_{seed}" if args.label and args.seeds else args.label
        train(seed, args.timesteps, args.smoke, override or None, label)
