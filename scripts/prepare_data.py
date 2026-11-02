import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.prepare_schedule import prepare
from src.utils.logging_utils import configure_logging

if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--archive")
    parser.add_argument("--date")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    _, info = prepare(args.synthetic, args.archive, args.date, args.strict)
    print(info)
