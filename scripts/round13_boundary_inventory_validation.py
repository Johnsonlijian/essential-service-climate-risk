#!/usr/bin/env python3
"""Round 13 boundary and facility-inventory validation increments.

The script adds two high-priority top-journal checks:

1. Boundary/window sensitivity for the 30-city facility analysis.
2. Facility inventory quality diagnostics using source confidence, category scope,
   OSM/Overture count ratios where available, and distance-based duplicate proxies.

The boundary checks use reproducible geometry perturbations that can be computed
from existing project data. If OSM/Nominatim boundary polygons are reachable, the
script also adds an official-place-polygon check and clearly labels it as such.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point, shape
from sklearn.neighbors import BallTree


ROOT = Path(__file__).resolve().parents[3]
AI = ROOT / "ai_autoboost"
OUT = AI / "outputs" / "round13_boundary_inventory_validation"
DOCS = AI / "docs"
TABLE_DIR = ROOT / "manuscript" / "tables"
FIG_DIR = ROOT / "manuscript" / "figures"
DATA = ROOT / "data_processed" / "pilot_facility_indices.parquet"
AREAS = ROOT / "configs" / "pilot_areas.csv"
OSM_CACHE = OUT / "osm_boundary_cache"


CORE_TERMS = {
    "school": ["school", "primary_school", "secondary_school", "elementary_school", "middle_school", "high_school"],
    "health": ["hospital", "clinic", "doctor", "medical_center", "healthcare"],
}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def fetch_osm_boundary(city: str, country_hint: str, area_id: str) -> dict[str, Any] | None:
    OSM_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = OSM_CACHE / f"{area_id}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            return None
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": f"{city}, {country_hint}",
        "format": "json",
        "polygon_geojson": 1,
        "limit": 1,
        "addressdetails": 1,
    }
    headers = {"User-Agent": "IMUT-essential-service-risk-boundary-audit/0.2 (research reproducibility)"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=25)
        response.raise_for_status()
        data = response.json()
        if not data:
            return None
        item = data[0]
        if "geojson" not in item:
            return None
        cache_file.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(1.1)
        return item
    except Exception:
        return None


def boundary_masks(df: pd.DataFrame, area: pd.Series, osm_item: dict[str, Any] | None) -> dict[str, pd.Series]:
    lon = df["lon"]
    lat = df["lat"]
    west, east = float(area["bbox_west"]), float(area["bbox_east"])
    south, north = float(area["bbox_south"]), float(area["bbox_north"])
    width = east - west
    height = north - south
    masks: dict[str, pd.Series] = {
        "bbox_main": pd.Series(True, index=df.index),
        "bbox_inner_90": (lon >= west + 0.05 * width)
        & (lon <= east - 0.05 * width)
        & (lat >= south + 0.05 * height)
        & (lat <= north - 0.05 * height),
        "bbox_inner_80": (lon >= west + 0.10 * width)
        & (lon <= east - 0.10 * width)
        & (lat >= south + 0.10 * height)
        & (lat <= north - 0.10 * height),
    }
    if len(df) >= 20:
        q_lon = lon.quantile([0.025, 0.975])
        q_lat = lat.quantile([0.025, 0.975])
        masks["facility_quantile_95"] = (
            (lon >= q_lon.loc[0.025])
            & (lon <= q_lon.loc[0.975])
            & (lat >= q_lat.loc[0.025])
            & (lat <= q_lat.loc[0.975])
        )
    if osm_item and "geojson" in osm_item:
        try:
            geom = shape(osm_item["geojson"])
            masks["osm_nominatim_polygon"] = pd.Series(
                [geom.contains(Point(x, y)) or geom.touches(Point(x, y)) for x, y in zip(lon, lat)],
                index=df.index,
            )
        except Exception:
            pass
    return masks


def summarize_subset(sub: pd.DataFrame, city: str, scope: str, total_n: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "city": city,
        "boundary_scope": scope,
        "n_facilities": int(len(sub)),
        "retention_share": float(len(sub) / total_n) if total_n else np.nan,
    }
    for col in ["heat_exposed", "flood_exposed", "water_stress_exposed", "compound_exposed"]:
        row[col.replace("_exposed", "_share")] = float(sub[col].mean()) if len(sub) else np.nan
    row["mean_escri"] = float(sub["escri"].mean()) if len(sub) else np.nan
    row["median_service_pop_5km"] = float(sub["service_pop_5p0km"].median()) if len(sub) else np.nan
    return row


def boundary_sensitivity(facilities: pd.DataFrame, areas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    osm_status: list[dict[str, Any]] = []
    for _, area in areas.iterrows():
        city = str(area["name"])
        area_id = str(area["area_id"])
        city_df = facilities[facilities["city_key"] == area_id].copy()
        if city_df.empty:
            city_df = facilities[facilities["city"] == city].copy()
        country_hint = str(city_df["name_0"].mode().iloc[0]) if "name_0" in city_df and not city_df["name_0"].dropna().empty else str(area["country_iso3"])
        osm_item = fetch_osm_boundary(city, country_hint, area_id)
        osm_status.append(
            {
                "area_id": area_id,
                "city": city,
                "osm_boundary_available": bool(osm_item and "geojson" in osm_item),
                "osm_display_name": osm_item.get("display_name", "") if osm_item else "",
                "osm_class": osm_item.get("class", "") if osm_item else "",
                "osm_type": osm_item.get("type", "") if osm_item else "",
            }
        )
        masks = boundary_masks(city_df, area, osm_item)
        for scope, mask in masks.items():
            sub = city_df[mask].copy()
            rows.append(summarize_subset(sub, city, scope, len(city_df)))
    detail = pd.DataFrame(rows)

    base = detail[detail["boundary_scope"] == "bbox_main"].set_index("city")
    shifts = []
    for _, row in detail.iterrows():
        b = base.loc[row["city"]]
        shifts.append(
            {
                **row.to_dict(),
                "compound_share_shift_vs_bbox": row["compound_share"] - b["compound_share"],
                "mean_escri_shift_vs_bbox": row["mean_escri"] - b["mean_escri"],
                "absolute_compound_share_shift_vs_bbox": abs(row["compound_share"] - b["compound_share"]),
                "absolute_mean_escri_shift_vs_bbox": abs(row["mean_escri"] - b["mean_escri"]),
            }
        )
    shifted = pd.DataFrame(shifts)
    osm_df = pd.DataFrame(osm_status)
    return shifted, osm_df


def boundary_rank_summary(boundary: pd.DataFrame) -> pd.DataFrame:
    base = boundary[boundary["boundary_scope"] == "bbox_main"].set_index("city")
    base_rank_escri = base["mean_escri"].rank(ascending=False, method="min")
    base_rank_compound = base["compound_share"].rank(ascending=False, method="min")
    rows = []
    for scope, sub in boundary.groupby("boundary_scope"):
        work = sub.set_index("city")
        common = base.index.intersection(work.index)
        rows.append(
            {
                "boundary_scope": scope,
                "n_cities": int(len(common)),
                "median_retention_share": float(work.loc[common, "retention_share"].median()),
                "spearman_escri_rank_vs_bbox": float(base_rank_escri.loc[common].corr(work.loc[common, "mean_escri"].rank(ascending=False, method="min"), method="spearman")),
                "spearman_compound_rank_vs_bbox": float(base_rank_compound.loc[common].corr(work.loc[common, "compound_share"].rank(ascending=False, method="min"), method="spearman")),
                "median_abs_compound_shift": float(work.loc[common, "absolute_compound_share_shift_vs_bbox"].median()),
                "max_abs_compound_shift": float(work.loc[common, "absolute_compound_share_shift_vs_bbox"].max()),
                "median_abs_escri_shift": float(work.loc[common, "absolute_mean_escri_shift_vs_bbox"].median()),
                "max_abs_escri_shift": float(work.loc[common, "absolute_mean_escri_shift_vs_bbox"].max()),
            }
        )
    return pd.DataFrame(rows)


def nearest_neighbor_stats(group: pd.DataFrame) -> dict[str, float]:
    if len(group) < 2:
        return {
            "nn_distance_m_median": np.nan,
            "share_nn_within_20m": np.nan,
            "share_nn_within_50m": np.nan,
            "share_nn_within_100m": np.nan,
        }
    coords_rad = np.deg2rad(group[["lat", "lon"]].to_numpy(dtype=float))
    tree = BallTree(coords_rad, metric="haversine")
    dist, _ = tree.query(coords_rad, k=2)
    nn_m = dist[:, 1] * 6371000.0
    return {
        "nn_distance_m_median": float(np.median(nn_m)),
        "share_nn_within_20m": float((nn_m <= 20).mean()),
        "share_nn_within_50m": float((nn_m <= 50).mean()),
        "share_nn_within_100m": float((nn_m <= 100).mean()),
    }


def is_core_category(row: pd.Series) -> bool:
    text = str(row.get("category_text", "")).lower()
    terms = CORE_TERMS.get(str(row["facility_type"]), [])
    return any(term in text for term in terms)


def inventory_quality(facilities: pd.DataFrame) -> pd.DataFrame:
    osm = pd.read_csv(TABLE_DIR / "table8_osm_overture_facility_audit.csv")
    rows: list[dict[str, Any]] = []
    facilities = facilities.copy()
    facilities["is_core_category_r13"] = facilities.apply(is_core_category, axis=1)
    facilities["source_confidence"] = pd.to_numeric(facilities["source_confidence"], errors="coerce").fillna(0.5)
    for (city, facility_type), group in facilities.groupby(["city", "facility_type"]):
        nn = nearest_neighbor_stats(group)
        osm_row = osm[osm["city"].eq(city)]
        if facility_type == "school":
            osm_count = float(osm_row["osm_school_core"].iloc[0]) if not osm_row.empty else np.nan
            overture_count = float(osm_row["overture_school"].iloc[0]) if not osm_row.empty else np.nan
            ratio = float(osm_row["osm_to_overture_school_ratio"].iloc[0]) if not osm_row.empty else np.nan
        else:
            osm_count = float(osm_row["osm_health_core"].iloc[0]) if not osm_row.empty else np.nan
            overture_count = float(osm_row["overture_health"].iloc[0]) if not osm_row.empty else np.nan
            ratio = float(osm_row["osm_to_overture_health_ratio"].iloc[0]) if not osm_row.empty else np.nan
        rows.append(
            {
                "city": city,
                "facility_type": facility_type,
                "n_records": int(len(group)),
                "median_source_confidence": float(group["source_confidence"].median()),
                "high_confidence_share_ge_0p75": float((group["source_confidence"] >= 0.75).mean()),
                "core_category_share": float(group["is_core_category_r13"].mean()),
                "osm_core_count": osm_count,
                "overture_count_from_osm_audit": overture_count,
                "osm_to_overture_core_ratio": ratio,
                **nn,
            }
        )
    return pd.DataFrame(rows)


def write_figures(boundary_summary: pd.DataFrame, inventory: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plot = boundary_summary[boundary_summary["boundary_scope"] != "bbox_main"].copy()
    plt.figure(figsize=(8, 5), dpi=300)
    y = np.arange(len(plot))
    plt.barh(y - 0.18, plot["spearman_escri_rank_vs_bbox"], height=0.35, label="ESCRI rank", color="#4daf4a")
    plt.barh(y + 0.18, plot["spearman_compound_rank_vs_bbox"], height=0.35, label="Compound rank", color="#377eb8")
    plt.axvline(0.8, color="#d95f02", linestyle="--", linewidth=1, label="0.8 gate")
    plt.yticks(y, plot["boundary_scope"])
    plt.xlabel("Spearman rank correlation vs bbox")
    plt.title("Boundary/window sensitivity of city rankings")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure9_boundary_window_sensitivity.png", dpi=300)
    plt.close()

    inv_city = inventory.groupby("city", as_index=False).agg(
        high_confidence_share_ge_0p75=("high_confidence_share_ge_0p75", "mean"),
        share_nn_within_50m=("share_nn_within_50m", "mean"),
    )
    inv_city = inv_city.sort_values("high_confidence_share_ge_0p75")
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(
        inv_city["high_confidence_share_ge_0p75"],
        inv_city["share_nn_within_50m"],
        s=42,
        color="#984ea3",
        alpha=0.85,
    )
    for _, r in inv_city.iterrows():
        if r["share_nn_within_50m"] >= inv_city["share_nn_within_50m"].quantile(0.85) or r["high_confidence_share_ge_0p75"] <= inv_city["high_confidence_share_ge_0p75"].quantile(0.15):
            plt.text(r["high_confidence_share_ge_0p75"], r["share_nn_within_50m"], str(r["city"]), fontsize=7)
    plt.xlabel("Mean high-confidence record share")
    plt.ylabel("Mean nearest-neighbor <=50 m share")
    plt.title("Facility inventory quality diagnostics")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure10_inventory_quality_diagnostics.png", dpi=300)
    plt.close()


def write_report(boundary_summary: pd.DataFrame, osm_status: pd.DataFrame, inventory: pd.DataFrame) -> None:
    comparable = boundary_summary[boundary_summary["boundary_scope"] != "bbox_main"]
    passed = int(((comparable["spearman_escri_rank_vs_bbox"] >= 0.8) & (comparable["spearman_compound_rank_vs_bbox"] >= 0.8)).sum())
    osm_n = int(osm_status["osm_boundary_available"].sum())
    osm_ratios = inventory["osm_to_overture_core_ratio"].dropna()
    median_ratio = float(osm_ratios.median()) if len(osm_ratios) else np.nan
    report = f"""# ROUND13_BOUNDARY_INVENTORY_VALIDATION_REPORT

Generated from `ai_autoboost/code/round13_boundary_inventory_validation/round13_boundary_inventory_validation.py`.

## Completed real calculations

1. Boundary/window sensitivity using `bbox_main`, `bbox_inner_90`, `bbox_inner_80`, `facility_quantile_95`, and OSM/Nominatim polygons where available.
2. Facility inventory quality diagnostics using source confidence, category scope, OSM/Overture count ratios where available, and nearest-neighbor duplicate proxies.

## Key results

- Boundary scopes passing both ESCRI-rank and compound-rank Spearman >= 0.8 gates: {passed} of {len(comparable)} non-main scopes.
- OSM/Nominatim polygons retrieved: {osm_n} of {len(osm_status)} cities.
- Median OSM-to-Overture core ratio among available city-facility checks: {median_ratio:.3f}.
- Median high-confidence share across city-facility groups: {inventory['high_confidence_share_ge_0p75'].median():.3f}.
- Median nearest-neighbor <=50 m duplicate-proxy share across city-facility groups: {inventory['share_nn_within_50m'].median():.3f}.

## Interpretation

The boundary checks reduce but do not eliminate the bounding-box concern. If the inner-window and OSM-polygon rank correlations remain high, the main rankings are not solely an artifact of the broad city windows. If selected cities show large retention or exposure shifts, those cities should be explicitly named as boundary-sensitive and prioritized for local boundary validation.

The inventory diagnostics reduce the facility-record concern by quantifying confidence, category scope and duplicate-proxy behavior. They do not turn Overture records into an official registry and should be reported as source-quality diagnostics, not definitive precision/recall against gold-standard registries.
"""
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "ROUND13_BOUNDARY_INVENTORY_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    facilities = pd.read_parquet(DATA)
    areas = pd.read_csv(AREAS)

    boundary, osm_status = boundary_sensitivity(facilities, areas)
    boundary_summary = boundary_rank_summary(boundary)
    inventory = inventory_quality(facilities)

    boundary.to_csv(OUT / "boundary_window_sensitivity_detail.csv", index=False)
    boundary_summary.to_csv(OUT / "boundary_window_sensitivity_summary.csv", index=False)
    osm_status.to_csv(OUT / "osm_boundary_fetch_status.csv", index=False)
    inventory.to_csv(OUT / "facility_inventory_quality.csv", index=False)

    boundary_summary.to_csv(TABLE_DIR / "table23_boundary_window_sensitivity.csv", index=False)
    inventory.to_csv(TABLE_DIR / "table24_facility_inventory_quality.csv", index=False)
    osm_status.to_csv(TABLE_DIR / "table25_osm_boundary_fetch_status.csv", index=False)

    write_figures(boundary_summary, inventory)
    write_report(boundary_summary, osm_status, inventory)

    comparable = boundary_summary[boundary_summary["boundary_scope"] != "bbox_main"]
    summary = {
        "facility_records": int(len(facilities)),
        "boundary_scopes": boundary_summary["boundary_scope"].tolist(),
        "non_main_boundary_scopes_passing_rank_gate": int(((comparable["spearman_escri_rank_vs_bbox"] >= 0.8) & (comparable["spearman_compound_rank_vs_bbox"] >= 0.8)).sum()),
        "osm_boundaries_available": int(osm_status["osm_boundary_available"].sum()),
        "median_high_confidence_share": float(inventory["high_confidence_share_ge_0p75"].median()),
        "median_nn_within_50m_share": float(inventory["share_nn_within_50m"].median()),
        "outputs": [
            "table23_boundary_window_sensitivity.csv",
            "table24_facility_inventory_quality.csv",
            "table25_osm_boundary_fetch_status.csv",
            "figure9_boundary_window_sensitivity.png",
            "figure10_inventory_quality_diagnostics.png",
        ],
    }
    (OUT / "round13_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
