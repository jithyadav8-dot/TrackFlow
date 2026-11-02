import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.scenario_generator import save_splits

if __name__ == "__main__":
    print({split: len(rows) for split, rows in save_splits().items()})
