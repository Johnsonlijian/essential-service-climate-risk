from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAS = Path(r"\\IMUTBIM_NAS\Nature_cities")


BUILDING_ALIASES = {
    "new_york": "New_York",
    "new_york_city": "New_York",
    "los_angeles": "Los_Angeles",
    "mexico_city": "Mexico_City",
    "sao_paulo": "Sao_Paulo",
    "ho_chi_minh": "Ho_Chi_Minh_City",
    "ho_chi_minh_city": "Ho_Chi_Minh_City",
}


def city_file_stem(area_id: str, name: str) -> str:
    key = area_id.lower()
    if key in BUILDING_ALIASES:
        return BUILDING_ALIASES[key]
    normalized_name = name.replace(" ", "_").replace("-", "_")
    return BUILDING_ALIASES.get(normalized_name.lower(), normalized_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nas-root", default=str(DEFAULT_NAS))
    parser.add_argument("--out", default=str(ROOT / "data_processed" / "shared_nas_coverage_audit.csv"))
    args = parser.parse_args()

    nas_root = Path(args.nas_root)
    population_dir = nas_root / "0-Oridata" / "4-population" / "data" / "raw" / "population"
    buildings_dir = nas_root / "0-Oridata" / "6-buildings-overture" / "data" / "raw" / "buildings"

    areas = pd.read_csv(ROOT / "configs" / "pilot_areas.csv")
    rows = []
    for row in areas.itertuples(index=False):
        iso3 = str(row.country_iso3).lower()
        pop_path = population_dir / f"{iso3}_pop_2020.tif"
        building_stem = city_file_stem(str(row.area_id), str(row.name))
        building_path = buildings_dir / f"{building_stem}_buildings.geojson"
        rows.append(
            {
                "area_id": row.area_id,
                "city": row.name,
                "country_iso3": row.country_iso3,
                "nas_worldpop_2020_available": pop_path.exists(),
                "nas_worldpop_2020_path": str(pop_path) if pop_path.exists() else "",
                "nas_overture_buildings_available": building_path.exists(),
                "nas_overture_buildings_path": str(building_path) if building_path.exists() else "",
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(rows)
    audit.to_csv(out, index=False)

    summary = {
        "cities": len(audit),
        "worldpop_2020_available": int(audit["nas_worldpop_2020_available"].sum()),
        "overture_buildings_available": int(audit["nas_overture_buildings_available"].sum()),
    }
    print(summary)
    print(out)


if __name__ == "__main__":
    main()
