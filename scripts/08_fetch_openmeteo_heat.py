from __future__ import annotations

import argparse
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree


OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def make_grid(bbox: list[float], step: float) -> pd.DataFrame:
    west, south, east, north = bbox
    lons = np.arange(west, east + step / 2, step)
    lats = np.arange(south, north + step / 2, step)
    rows = []
    for lat in lats:
        for lon in lons:
            rows.append({"lon": round(float(lon), 5), "lat": round(float(lat), 5)})
    return pd.DataFrame(rows)


def fetch_daily_temperature(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_mean",
        "timezone": "UTC",
    }
    response = requests.get(OPENMETEO_ARCHIVE, params=params, timeout=60)
    response.raise_for_status()
    daily = response.json()["daily"]
    df = pd.DataFrame(daily)
    df["lat"] = lat
    df["lon"] = lon
    return df


def build_heat_grid(
    bbox: list[float],
    out_csv: str,
    step: float = 0.25,
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    threshold_c: float = 35,
) -> pd.DataFrame:
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(bbox, step)
    summaries = []
    for row in grid.itertuples(index=False):
        daily = fetch_daily_temperature(row.lat, row.lon, start_date, end_date)
        summaries.append(
            {
                "lon": row.lon,
                "lat": row.lat,
                "year": start_date[:4],
                "heat_threshold_c": threshold_c,
                "tasmax_hot_days": int((daily["temperature_2m_max"] >= threshold_c).sum()),
                "tasmax_mean_c": float(daily["temperature_2m_max"].mean()),
                "tasmean_mean_c": float(daily["temperature_2m_mean"].mean()),
                "tasmax_p95_c": float(daily["temperature_2m_max"].quantile(0.95)),
            }
        )
        time.sleep(0.05)
    out = pd.DataFrame(summaries)
    out.to_csv(out_csv, index=False)
    return out


def assign_heat_to_facilities(facilities_path: str, heat_csv: str, out_path: str) -> None:
    gdf = gpd.read_parquet(facilities_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    heat = pd.read_csv(heat_csv)
    tree = cKDTree(heat[["lon", "lat"]].to_numpy())
    distances, idx = tree.query(np.column_stack([gdf.geometry.x, gdf.geometry.y]), k=1)
    for col in ["tasmax_hot_days", "tasmax_mean_c", "tasmean_mean_c", "tasmax_p95_c"]:
        gdf[col] = heat.iloc[idx][col].to_numpy()
    gdf["heat_grid_distance_deg"] = distances
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--facilities", required=True)
    parser.add_argument("--heat-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--threshold-c", type=float, default=35)
    args = parser.parse_args()
    build_heat_grid(args.bbox, args.heat_csv, args.step, args.start_date, args.end_date, args.threshold_c)
    assign_heat_to_facilities(args.facilities, args.heat_csv, args.out)


if __name__ == "__main__":
    main()

