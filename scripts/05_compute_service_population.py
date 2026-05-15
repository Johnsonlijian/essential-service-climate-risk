from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.windows import from_bounds
from scipy.spatial import cKDTree

from cnrisk.paths import DATA_PROCESSED, ensure_dirs


def raster_population_points(gdf: gpd.GeoDataFrame, pop_raster: str, max_buffer_km: float):
    with rasterio.open(pop_raster) as src:
        bounds = gdf.to_crs(src.crs).total_bounds
        # A generous degree margin for fast pilot clipping; exact distance is applied after projection.
        margin = max(0.25, max_buffer_km / 80)
        window = from_bounds(
            bounds[0] - margin,
            bounds[1] - margin,
            bounds[2] + margin,
            bounds[3] + margin,
            transform=src.transform,
        ).round_offsets().round_lengths()
        arr = src.read(1, window=window, masked=True)
        transform = src.window_transform(window)
        rows, cols = np.where(~arr.mask & np.isfinite(arr.filled(np.nan)) & (arr.filled(0) > 0))
        if len(rows) == 0:
            return np.empty((0, 2)), np.empty((0,))
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
        to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        lon, lat = to_wgs84.transform(xs, ys)
        to_equal_area = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
        x_m, y_m = to_equal_area.transform(lon, lat)
        values = arr[rows, cols].filled(0).astype(float)
    return np.column_stack([x_m, y_m]), values


def service_population_sums(gdf: gpd.GeoDataFrame, pop_raster: str, buffers_km: list[float]) -> dict[float, list[float]]:
    points_xy, values = raster_population_points(gdf, pop_raster, max(buffers_km))
    if len(values) == 0:
        return {km: [0.0] * len(gdf) for km in buffers_km}
    tree = cKDTree(points_xy)
    facilities = gdf.to_crs("EPSG:6933")
    facility_xy = np.column_stack([facilities.geometry.x.to_numpy(), facilities.geometry.y.to_numpy()])
    output: dict[float, list[float]] = {}
    for km in buffers_km:
        radius = km * 1000
        output[km] = [float(values[idx].sum()) for idx in tree.query_ball_point(facility_xy, radius)]
    return output


def compute(facilities_path: str, pop_raster: str, out_path: str, buffers_km: list[float]) -> None:
    ensure_dirs()
    gdf = gpd.read_parquet(facilities_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    sums = service_population_sums(gdf, pop_raster, buffers_km)
    for km, values in sums.items():
        label = str(km).replace(".", "p")
        gdf[f"service_pop_{label}km"] = values
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facilities", default=str(DATA_PROCESSED / "facility_hazard_overlay.parquet"))
    parser.add_argument("--population-raster", required=True)
    parser.add_argument("--out", default=str(DATA_PROCESSED / "facility_service_population.parquet"))
    parser.add_argument("--buffers-km", nargs="+", type=float, default=[1, 5, 10])
    args = parser.parse_args()
    compute(args.facilities, args.population_raster, args.out, args.buffers_km)


if __name__ == "__main__":
    main()
