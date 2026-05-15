from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

from cnrisk.config import load_project_config
from cnrisk.indicators import aggregate_exposure, binary_hazard_flags, compute_escri
from cnrisk.paths import DATA_PROCESSED, MANUSCRIPT, ensure_dirs


def compute_indices(in_path: str, out_path: str, config_path: str) -> None:
    ensure_dirs()
    config = load_project_config(config_path)
    thresholds = config["hazard_thresholds"]
    gdf = gpd.read_parquet(in_path)
    gdf = binary_hazard_flags(
        gdf,
        utci_threshold=thresholds["utci_strong_heat_c"],
        flood_threshold=thresholds["flood_depth_m"],
        water_threshold=thresholds["aqueduct_high_water_stress"],
    )
    gdf = compute_escri(gdf)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out_path, index=False)

    tables_dir = MANUSCRIPT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    aggregate_exposure(gdf, ["facility_type"]).to_csv(tables_dir / "table_facility_type_exposure.csv", index=False)
    if "country_iso3" in gdf.columns:
        aggregate_exposure(gdf, ["country_iso3", "facility_type"]).to_csv(
            tables_dir / "table_country_exposure.csv", index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DATA_PROCESSED / "facility_service_population.parquet"))
    parser.add_argument("--out", default=str(DATA_PROCESSED / "facility_indices.parquet"))
    parser.add_argument("--config", default="configs/project.json")
    args = parser.parse_args()
    compute_indices(args.input, args.out, args.config)


if __name__ == "__main__":
    main()

