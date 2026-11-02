import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.visualization.charts import export_charts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results")
    args = parser.parse_args()
    export_charts(args.results)
