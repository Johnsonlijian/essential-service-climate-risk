from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def minmax(values: pd.Series) -> pd.Series:
    x = values.astype(float)
    lo = np.nanpercentile(x, 1)
    hi = np.nanpercentile(x, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.0, index=values.index)
    return ((x.clip(lo, hi) - lo) / (hi - lo)).fillna(0)


def load_metadata() -> dict[str, dict[str, str]]:
    areas = pd.read_csv(ROOT / "configs" / "pilot_areas.csv")
    countries = pd.read_csv(ROOT / "configs" / "country_metadata.csv")
    merged = areas.merge(countries, on="country_iso3", how="left")
    return {
        row["area_id"]: {
            "city": row["name"],
            "country_iso3": row["country_iso3"],
            "income_group": row.get("income_group", "Unknown"),
        }
        for _, row in merged.iterrows()
    }


def load_city(path: str, city_key: str, metadata: dict[str, dict[str, str]]) -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(path)
    for key, value in metadata[city_key].items():
        gdf[key] = value
    gdf["city_key"] = city_key
    return gdf


def compute_indices(gdf: gpd.GeoDataFrame, heat_hot_days_threshold: float = 20) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out["heat_exposed"] = out["tasmax_hot_days"].fillna(0) >= heat_hot_days_threshold
    out["flood_exposed"] = out["flood100y_exposed"].fillna(False).astype(bool)
    out["water_stress_exposed"] = out["aqueduct_bws_cat"].fillna(0) >= 3
    out["n_hazards"] = out[["heat_exposed", "flood_exposed", "water_stress_exposed"]].sum(axis=1)
    out["compound_exposed"] = out["n_hazards"] >= 2
    out["grdi_tercile"] = pd.qcut(out["grdi"], q=3, labels=["low", "middle", "high"], duplicates="drop")
    service_col = "service_pop_5p0km"
    heat_score = minmax(out["tasmax_hot_days"])
    flood_score = out["flood_exposed"].astype(float)
    water_score = minmax(out["aqueduct_bws_score"].fillna(out["aqueduct_bws_cat"]))
    service_score = minmax(np.log1p(out[service_col].fillna(0)))
    vulnerability_score = minmax(out["grdi"])
    confidence = out["source_confidence"].fillna(0.5).astype(float).clip(0, 1)
    out["hazard_score"] = (heat_score + flood_score + water_score) / 3
    out["service_score"] = service_score
    out["vulnerability_score"] = vulnerability_score
    out["escri"] = out["hazard_score"] * (0.5 + 0.5 * service_score) * (0.5 + 0.5 * vulnerability_score) * confidence
    return out


def aggregate_tables(gdf: gpd.GeoDataFrame, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "n_facilities": ("id", "count"),
        "heat_share": ("heat_exposed", "mean"),
        "flood_share": ("flood_exposed", "mean"),
        "water_stress_share": ("water_stress_exposed", "mean"),
        "compound_share": ("compound_exposed", "mean"),
        "median_service_pop_5km": ("service_pop_5p0km", "median"),
        "mean_escri": ("escri", "mean"),
        "median_grdi": ("grdi", "median"),
    }
    (
        gdf.groupby(["city", "country_iso3", "income_group", "facility_type"], dropna=False)
        .agg(**metrics)
        .reset_index()
        .to_csv(out / "table1_pilot_facility_exposure.csv", index=False)
    )
    (
        gdf.groupby(["city", "grdi_tercile", "facility_type"], dropna=False)
        .agg(**metrics)
        .reset_index()
        .to_csv(out / "table2_pilot_grdi_inequality.csv", index=False)
    )
    (
        gdf.groupby(["city"], dropna=False)
        .agg(**metrics)
        .reset_index()
        .sort_values("mean_escri", ascending=False)
        .to_csv(out / "table3_pilot_city_ranking.csv", index=False)
    )
    (
        gdf.groupby(["city", "facility_type"], dropna=False)
        .agg(n=("id", "count"), median_source_confidence=("source_confidence", "median"))
        .reset_index()
        .to_csv(out / "table4_pilot_source_audit.csv", index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-file", action="append", default=[])
    parser.add_argument("--out", default="data_processed/pilot_facility_indices.parquet")
    parser.add_argument("--table-dir", default="manuscript/tables")
    parser.add_argument("--heat-hot-days-threshold", type=float, default=20)
    args = parser.parse_args()
    metadata = load_metadata()
    city_files = args.city_file or [
        f"{area_id}=data_processed/{area_id}-facility_full_indicators.parquet"
        for area_id in metadata
        if (ROOT / "data_processed" / f"{area_id}-facility_full_indicators.parquet").exists()
    ]
    frames = []
    for item in city_files:
        city_key, path = item.split("=", 1)
        frames.append(load_city(path, city_key, metadata))
    gdf = pd.concat(frames, ignore_index=True)
    out = compute_indices(gdf, args.heat_hot_days_threshold)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    aggregate_tables(out, args.table_dir)


if __name__ == "__main__":
    main()
