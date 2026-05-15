from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--area-id", action="append", default=[])
    args = parser.parse_args()
    areas = pd.read_csv(ROOT / "configs" / "pilot_areas.csv")
    if args.area_id:
        areas = areas[areas["area_id"].isin(args.area_id)]
    for area_id in areas["area_id"]:
        full = ROOT / "data_processed" / f"{area_id}-facility_full_indicators.parquet"
        if args.only_missing and full.exists():
            print(f"skip complete {area_id}")
            continue
        cmd = [sys.executable, "scripts/15_run_pilot_city.py", area_id]
        if args.force:
            cmd.append("--force")
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

