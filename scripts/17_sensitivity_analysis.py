from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd


TABLE_DIR = Path("manuscript/tables")


SCHOOL_CORE = {"school", "kindergarten", "college", "university"}
HEALTH_CORE = {"hospital", "clinic", "doctor", "doctors", "medical_center", "urgent_care"}
HEALTH_AUX = {"pharmacy", "dentist", "laboratory", "physiotherapist", "nursing_home"}


def facility_scope(category_text: str, facility_type: str) -> str:
    tokens = set(str(category_text).replace(",", " ").split())
    if facility_type == "school":
        return "core" if tokens.intersection(SCHOOL_CORE) else "broad"
    if facility_type == "health":
        if tokens.intersection(HEALTH_CORE):
            return "core"
        if tokens.intersection(HEALTH_AUX):
            return "auxiliary"
    return "broad"


def exposure_table(df: pd.DataFrame, heat_days_threshold: int, water_cat_threshold: int, subset_label: str) -> pd.DataFrame:
    x = df.copy()
    x["heat_exposed_sens"] = x["tasmax_hot_days"].fillna(0) >= heat_days_threshold
    x["flood_exposed_sens"] = x["flood100y_exposed"].fillna(False).astype(bool)
    x["water_stress_exposed_sens"] = x["aqueduct_bws_cat"].fillna(0) >= water_cat_threshold
    x["compound_exposed_sens"] = (
        x[["heat_exposed_sens", "flood_exposed_sens", "water_stress_exposed_sens"]].sum(axis=1) >= 2
    )
    out = (
        x.groupby(["city", "facility_type"], dropna=False)
        .agg(
            n_facilities=("id", "count"),
            heat_share=("heat_exposed_sens", "mean"),
            flood_share=("flood_exposed_sens", "mean"),
            water_share=("water_stress_exposed_sens", "mean"),
            compound_share=("compound_exposed_sens", "mean"),
        )
        .reset_index()
    )
    out["heat_days_threshold"] = heat_days_threshold
    out["water_cat_threshold"] = water_cat_threshold
    out["subset"] = subset_label
    return out


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_parquet("data_processed/pilot_facility_indices.parquet")
    gdf["facility_scope"] = [
        facility_scope(cat, ftype) for cat, ftype in zip(gdf["category_text"], gdf["facility_type"], strict=True)
    ]
    rows = []
    for heat_threshold in [1, 10, 20, 30, 40]:
        for water_threshold in [3, 4, 5]:
            rows.append(exposure_table(gdf, heat_threshold, water_threshold, "all_classes"))
            core = gdf[(gdf["facility_type"] == "school") | (gdf["facility_scope"] == "core")]
            rows.append(exposure_table(core, heat_threshold, water_threshold, "school_all_health_core"))
            high_conf = gdf[gdf["source_confidence"].fillna(0) >= 0.6]
            rows.append(exposure_table(high_conf, heat_threshold, water_threshold, "source_confidence_ge_0p6"))
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(TABLE_DIR / "table6_threshold_and_scope_sensitivity.csv", index=False)

    scope_summary = (
        gdf.groupby(["city", "facility_type", "facility_scope"], dropna=False)
        .agg(n=("id", "count"), median_confidence=("source_confidence", "median"))
        .reset_index()
    )
    scope_summary.to_csv(TABLE_DIR / "table7_facility_scope_audit.csv", index=False)


if __name__ == "__main__":
    main()

