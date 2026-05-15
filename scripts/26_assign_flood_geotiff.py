from __future__ import annotations

"""Sample a flood-depth GeoTIFF (e.g. JRC/GloFAS 100-yr mosaics) at facility points.

Place a single-band depth raster (meters or a coded depth) at the path expected in
configs/data_manifest.csv, or pass --raster explicitly.

This complements scripts/09_assign_wms_flood.py (GloFAS WMS PNG). For paper-grade
workflows, prefer documenting both: WMS for cross-check, GeoTIFF for exact depths.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


def sample_raster(points: gpd.GeoDataFrame, raster_path: str, band: int = 1) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        pts = points.to_crs(src.crs)
        coords = [(float(geom.x), float(geom.y)) for geom in pts.geometry]
        samples = np.array([v[0] for v in src.sample(coords)], dtype=float)
        nodata = src.nodata
    if nodata is not None:
        samples = np.where(samples == float(nodata), np.nan, samples)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facilities", required=True, help="Input parquet from heat or WMS flood step.")
    parser.add_argument("--raster", required=True, help="Path to flood depth GeoTIFF.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--depth-exposure-min", type=float, default=0.0, help="Values > this count as flooded.")
    parser.add_argument(
        "--write-alternate-exposed-col",
        action="store_true",
        help="If set, write flood100y_raster_exposed instead of overwriting flood100y_exposed.",
    )
    args = parser.parse_args()

    raster_path = Path(args.raster)
    if not raster_path.is_file():
        raise FileNotFoundError(
            f"Flood raster not found: {raster_path}\n"
            "Download a JRC/CEMS-GloFAS (or other) 100-yr hazard GeoTIFF and place it here, "
            "or run with --flood-backend wms only. See docs/26_NATURE_FAMILY_DATA_V2_RUNBOOK.md."
        )

    gdf = gpd.read_parquet(args.facilities)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    depth = sample_raster(gdf, str(raster_path))
    gdf["flood_depth_m_raster"] = depth
    exposed = np.where(np.isfinite(depth), depth > args.depth_exposure_min, False)
    if args.write_alternate_exposed_col:
        gdf["flood100y_raster_exposed"] = exposed
    else:
        gdf["flood100y_exposed"] = exposed
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()
