from __future__ import annotations

import importlib.util
import sys

REQUIRED = [
    "duckdb",
    "geopandas",
    "pandas",
    "numpy",
    "pyarrow",
    "rasterio",
    "xarray",
    "shapely",
    "matplotlib",
]


def main() -> None:
    print(f"Python: {sys.version}")
    missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
    if missing:
        print("Missing packages:")
        for name in missing:
            print(f"  - {name}")
        raise SystemExit(1)
    print("Environment check passed.")


if __name__ == "__main__":
    main()

