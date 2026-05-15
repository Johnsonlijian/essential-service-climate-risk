from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd


OSM_CORE_SCHOOL = {"school", "kindergarten", "college", "university"}
OSM_CORE_HEALTH = {"hospital", "clinic", "doctors"}


def osm_counts(area_id: str) -> dict[str, int] | None:
    path = Path("data_processed") / f"{area_id}_osm_facility_category_counts.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    counts = dict(zip(df["Name"], df["Count"], strict=False))
    school = sum(int(counts.get(k, 0)) for k in OSM_CORE_SCHOOL)
    health = sum(int(counts.get(k, 0)) for k in OSM_CORE_HEALTH)
    return {"osm_school_core": school, "osm_health_core": health}


def main() -> None:
    rows = []
    areas = pd.read_csv("configs/pilot_areas.csv")
    for _, area in areas.iterrows():
        area_id = area["area_id"]
        reg = Path("data_processed") / f"{area_id}-facility_registry.parquet"
        if not reg.exists():
            continue
        gdf = gpd.read_parquet(reg)
        overture = gdf.groupby("facility_type").size().to_dict()
        osm = osm_counts(area_id) or {"osm_school_core": None, "osm_health_core": None}
        rows.append(
            {
                "area_id": area_id,
                "city": area["name"],
                "country_iso3": area["country_iso3"],
                "overture_school": int(overture.get("school", 0)),
                "overture_health": int(overture.get("health", 0)),
                **osm,
            }
        )
    out = pd.DataFrame(rows)
    out["osm_to_overture_school_ratio"] = out["osm_school_core"] / out["overture_school"]
    out["osm_to_overture_health_ratio"] = out["osm_health_core"] / out["overture_health"]
    Path("manuscript/tables").mkdir(parents=True, exist_ok=True)
    out.to_csv("manuscript/tables/table8_osm_overture_facility_audit.csv", index=False)


if __name__ == "__main__":
    main()

