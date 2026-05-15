from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(args: list[str], force: bool = False, output: str | Path | None = None) -> None:
    if output and Path(output).exists() and not force:
        print(f"skip existing {output}")
        return
    print("run:", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def worldpop_url(iso3: str) -> str:
    code = iso3.lower()
    return (
        "https://data.worldpop.org/GIS/Population/Global_2021_2022_1km_UNadj/"
        f"constrained/2022/{iso3}/{code}_ppp_2022_1km_UNadj_constrained.tif"
    )


def download_worldpop(iso3: str, force: bool = False) -> Path:
    out = ROOT / "data_raw" / "worldpop" / f"{iso3}_ppp_2022_1km_UNadj_constrained.tif"
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    url = worldpop_url(iso3)
    print(f"download WorldPop {iso3}: {url}")
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with out.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return out


def load_area(area_id: str) -> pd.Series:
    areas = pd.read_csv(ROOT / "configs" / "pilot_areas.csv")
    match = areas[areas["area_id"] == area_id]
    if match.empty:
        known = ", ".join(areas["area_id"])
        raise ValueError(f"Unknown area_id={area_id}. Known: {known}")
    return match.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("area_id")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-overture", action="store_true")
    parser.add_argument("--heat-step", type=float, default=0.25)
    parser.add_argument("--heat-threshold-c", type=float, default=35)
    parser.add_argument("--flood-size", type=int, default=2048)
    args = parser.parse_args()

    area = load_area(args.area_id)
    area_id = area["area_id"]
    iso3 = area["country_iso3"]
    bbox = [
        str(area["bbox_west"]),
        str(area["bbox_south"]),
        str(area["bbox_east"]),
        str(area["bbox_north"]),
    ]
    pop = download_worldpop(iso3, force=False)

    raw_overture = ROOT / "data_raw" / "overture" / f"{area_id}-facilities.parquet"
    registry = ROOT / "data_processed" / f"{area_id}-facility_registry.parquet"
    service = ROOT / "data_processed" / f"{area_id}-facility_service_population.parquet"
    heat = ROOT / "data_processed" / f"{area_id}-facility_heat_population.parquet"
    heat_csv = ROOT / "data_processed" / f"{area_id}-heat_grid_2025.csv"
    flood = ROOT / "data_processed" / f"{area_id}-facility_heat_pop_flood.parquet"
    flood_png = ROOT / "data_processed" / f"{area_id}-floodhazard100y-wms.png"
    aqueduct = ROOT / "data_raw" / "aqueduct" / f"{area_id}_aqueduct_baseline.geojson"
    water = ROOT / "data_processed" / f"{area_id}-facility_heat_pop_flood_water.parquet"
    full = ROOT / "data_processed" / f"{area_id}-facility_full_indicators.parquet"

    if not args.skip_overture:
        run(
            [
                PYTHON,
                "scripts/01_extract_overture_facilities.py",
                "--bbox",
                *bbox,
                "--out",
                str(raw_overture),
            ],
            args.force,
            raw_overture,
        )
    run(
        [
            PYTHON,
            "scripts/03_build_facility_registry.py",
            "--overture",
            str(raw_overture),
            "--out",
            str(registry),
        ],
        args.force,
        registry,
    )
    run(
        [
            PYTHON,
            "scripts/05_compute_service_population.py",
            "--facilities",
            str(registry),
            "--population-raster",
            str(pop),
            "--out",
            str(service),
            "--buffers-km",
            "1",
            "5",
            "10",
        ],
        args.force,
        service,
    )
    run(
        [
            PYTHON,
            "scripts/08_fetch_openmeteo_heat.py",
            "--bbox",
            *bbox,
            "--facilities",
            str(service),
            "--heat-csv",
            str(heat_csv),
            "--out",
            str(heat),
            "--step",
            str(args.heat_step),
            "--threshold-c",
            str(args.heat_threshold_c),
        ],
        args.force,
        heat,
    )
    run(
        [
            PYTHON,
            "scripts/09_assign_wms_flood.py",
            "--bbox",
            *bbox,
            "--facilities",
            str(heat),
            "--png",
            str(flood_png),
            "--out",
            str(flood),
            "--width",
            str(args.flood_size),
            "--height",
            str(args.flood_size),
        ],
        args.force,
        flood,
    )
    run(
        [
            PYTHON,
            "scripts/10_assign_aqueduct_water.py",
            "--bbox",
            *bbox,
            "--facilities",
            str(flood),
            "--aqueduct-geojson",
            str(aqueduct),
            "--out",
            str(water),
        ],
        args.force,
        water,
    )
    run(
        [
            PYTHON,
            "scripts/11_sample_grdi.py",
            "--facilities",
            str(water),
            "--out",
            str(full),
        ],
        args.force,
        full,
    )
    print(f"completed {area_id}: {full}")


if __name__ == "__main__":
    main()

