from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio

from cnrisk.paths import DATA_PROCESSED, ensure_dirs


def sample_raster(points: gpd.GeoDataFrame, raster_path: str, out_col: str) -> pd.Series:
    with rasterio.open(raster_path) as src:
        pts = points.to_crs(src.crs)
        coords = [(geom.x, geom.y) for geom in pts.geometry]
        values = [v[0] for v in src.sample(coords)]
        nodata = src.nodata
    series = pd.Series(values, index=points.index, name=out_col)
    if nodata is not None:
        series = series.mask(series == nodata)
    return series


def overlay(facilities_path: str, out_path: str, raster_args: list[str]) -> None:
    ensure_dirs()
    gdf = gpd.read_parquet(facilities_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    for item in raster_args:
        col, path = item.split("=", 1)
        gdf[col] = sample_raster(gdf, path, col)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facilities", default=str(DATA_PROCESSED / "facility_registry.parquet"))
    parser.add_argument("--out", default=str(DATA_PROCESSED / "facility_hazard_overlay.parquet"))
    parser.add_argument(
        "--raster",
        action="append",
        default=[],
        help="Column and raster path, e.g. flood_depth_m=data_raw/jrc_flood/flood_depth_100y.tif",
    )
    args = parser.parse_args()
    overlay(args.facilities, args.out, args.raster)


if __name__ == "__main__":
    main()

