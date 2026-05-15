from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from scipy.spatial import cKDTree
from shapely.geometry import Point


ROOT = Path(__file__).resolve().parents[1]


def city_inputs() -> dict[str, dict]:
    areas = pd.read_csv(ROOT / "configs" / "pilot_areas.csv")
    out: dict[str, dict] = {}
    for _, row in areas.iterrows():
        area_id = row["area_id"]
        iso3 = row["country_iso3"]
        out[area_id] = {
            "city": row["name"],
            "bbox": [row["bbox_west"], row["bbox_south"], row["bbox_east"], row["bbox_north"]],
            "population_raster": f"data_raw/worldpop/{iso3}_ppp_2022_1km_UNadj_constrained.tif",
            "heat_csv": f"data_processed/{area_id}-heat_grid_2025.csv",
            "flood_png": f"data_processed/{area_id}-floodhazard100y-wms.png",
            "aqueduct": f"data_raw/aqueduct/{area_id}_aqueduct_baseline.geojson",
        }
    return out


def population_candidates(population_raster: str, bbox: list[float]) -> pd.DataFrame:
    west, south, east, north = bbox
    with rasterio.open(population_raster) as src:
        window = from_bounds(west, south, east, north, transform=src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=window, masked=True)
        transform = src.window_transform(window)
        rows, cols = np.where(~arr.mask & np.isfinite(arr.filled(np.nan)) & (arr.filled(0) > 0))
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
        values = arr[rows, cols].filled(0).astype(float)
    return pd.DataFrame({"lon": xs, "lat": ys, "population": values})


def assign_heat(df: pd.DataFrame, heat_csv: str) -> pd.Series:
    heat = pd.read_csv(heat_csv)
    tree = cKDTree(heat[["lon", "lat"]].to_numpy())
    _, idx = tree.query(df[["lon", "lat"]].to_numpy(), k=1)
    return pd.Series(heat.iloc[idx]["tasmax_hot_days"].to_numpy(), index=df.index)


def assign_flood(df: pd.DataFrame, flood_png: str, bbox: list[float]) -> pd.Series:
    from PIL import Image

    west, south, east, north = bbox
    arr = np.asarray(Image.open(flood_png).convert("RGBA"))
    height, width = arr.shape[:2]
    x = ((df["lon"] - west) / (east - west) * (width - 1)).round().astype(int)
    y = ((north - df["lat"]) / (north - south) * (height - 1)).round().astype(int)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    alpha = np.zeros(len(df), dtype=np.uint8)
    alpha[valid.to_numpy()] = arr[y[valid], x[valid], 3]
    return pd.Series(alpha > 0, index=df.index)


def assign_water(df: pd.DataFrame, aqueduct_geojson: str) -> pd.Series:
    pts = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])], crs="EPSG:4326")
    water = gpd.read_file(aqueduct_geojson).to_crs(pts.crs)
    joined = gpd.sjoin(pts, water[["bws_cat", "geometry"]], how="left", predicate="within")
    return joined["bws_cat"].fillna(0).reset_index(drop=True)


def simulate_city(
    city_key: str,
    meta: dict,
    observed: gpd.GeoDataFrame,
    reps: int,
    max_n: int,
    heat_days_threshold: float,
    seed: int,
) -> pd.DataFrame:
    candidates = population_candidates(meta["population_raster"], meta["bbox"])
    candidates["heat_exposed"] = assign_heat(candidates, meta["heat_csv"]) >= heat_days_threshold
    candidates["flood_exposed"] = assign_flood(candidates, meta["flood_png"], meta["bbox"])
    candidates["water_stress_exposed"] = assign_water(candidates, meta["aqueduct"]) >= 3
    candidates["compound_exposed"] = (
        candidates[["heat_exposed", "flood_exposed", "water_stress_exposed"]].sum(axis=1) >= 2
    )
    weights = candidates["population"].to_numpy(dtype=float)
    weights = weights / weights.sum()
    rng = np.random.default_rng(seed)
    rows = []
    for facility_type, obs_group in observed[observed["city_key"] == city_key].groupby("facility_type"):
        n = min(len(obs_group), max_n)
        obs_compound = float(obs_group["compound_exposed"].mean())
        obs_heat = float(obs_group["heat_exposed"].mean())
        obs_flood = float(obs_group["flood_exposed"].mean())
        obs_water = float(obs_group["water_stress_exposed"].mean())
        sims = []
        for rep in range(reps):
            idx = rng.choice(len(candidates), size=n, replace=True, p=weights)
            sims.append(float(candidates.iloc[idx]["compound_exposed"].mean()))
        sims_arr = np.asarray(sims)
        rows.append(
            {
                "city": meta["city"],
                "facility_type": facility_type,
                "n_observed": len(obs_group),
                "n_simulated_per_rep": n,
                "observed_heat_share": obs_heat,
                "observed_flood_share": obs_flood,
                "observed_water_share": obs_water,
                "observed_compound_share": obs_compound,
                "counterfactual_compound_mean": float(sims_arr.mean()),
                "counterfactual_compound_p05": float(np.quantile(sims_arr, 0.05)),
                "counterfactual_compound_p95": float(np.quantile(sims_arr, 0.95)),
                "observed_minus_counterfactual": float(obs_compound - sims_arr.mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", default="data_processed/pilot_facility_indices.parquet")
    parser.add_argument("--out", default="manuscript/tables/table5_population_weighted_counterfactual.csv")
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--max-n", type=int, default=5000)
    parser.add_argument("--heat-hot-days-threshold", type=float, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    observed = gpd.read_parquet(args.indices)
    output = pd.concat(
        [
            simulate_city(city, meta, observed, args.reps, args.max_n, args.heat_hot_days_threshold, args.seed + i)
            for i, (city, meta) in enumerate(city_inputs().items())
            if (observed["city_key"] == city).any()
        ],
        ignore_index=True,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
