from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from cnrisk.config import facility_category_terms, load_project_config
from cnrisk.facilities import add_facility_type, source_confidence, write_coverage_audit
from cnrisk.paths import DATA_PROCESSED, ensure_dirs


def build_registry(overture_path: str, out_path: str, config_path: str) -> None:
    ensure_dirs()
    config = load_project_config(config_path)
    df = pd.read_parquet(overture_path)
    df = add_facility_type(
        df,
        school_terms=facility_category_terms(config, "school"),
        health_terms=facility_category_terms(config, "health"),
    )
    df["source_confidence"] = df.apply(source_confidence, axis=1)
    if "geometry" in df.columns:
        try:
            gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkb(df["geometry"]), crs="EPSG:4326")
        except Exception:
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    else:
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    keep_cols = [
        col
        for col in [
            "id",
            "name",
            "facility_type",
            "category_text",
            "confidence",
            "source_confidence",
            "lon",
            "lat",
            "geometry",
        ]
        if col in gdf.columns
    ]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    gdf[keep_cols].to_parquet(out_path, index=False)
    write_coverage_audit(gdf, DATA_PROCESSED / "facility_coverage_audit.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overture", required=True)
    parser.add_argument("--out", default=str(DATA_PROCESSED / "facility_registry.parquet"))
    parser.add_argument("--config", default="configs/project.json")
    args = parser.parse_args()
    build_registry(args.overture, args.out, args.config)


if __name__ == "__main__":
    main()

