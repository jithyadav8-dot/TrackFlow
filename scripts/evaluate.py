import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.experiments import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    args = parser.parse_args()
    evaluate(args.split, args.baselines_only, args.checkpoint, args.output)
