from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_service_module():
    path = ROOT / "scripts" / "05_compute_service_population.py"
    spec = importlib.util.spec_from_file_location("service_population", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize_city(area_id: str, city: str, current_path: Path, sensitivity_path: Path) -> dict[str, object]:
    current = pd.read_parquet(current_path, columns=["id", "service_pop_5p0km"])
    sensitivity = pd.read_parquet(sensitivity_path, columns=["id", "service_pop_5p0km"])
    merged = current.merge(sensitivity, on="id", suffixes=("_worldpop2022", "_worldpop2020"))
    current_median = float(merged["service_pop_5p0km_worldpop2022"].median())
    sensitivity_median = float(merged["service_pop_5p0km_worldpop2020"].median())
    ratio = sensitivity_median / current_median if current_median else None
    corr = merged[["service_pop_5p0km_worldpop2022", "service_pop_5p0km_worldpop2020"]].corr().iloc[0, 1]
    return {
        "area_id": area_id,
        "city": city,
        "status": "compared",
        "error": "",
        "n_facilities_compared": len(merged),
        "median_service_pop_5km_worldpop2022": current_median,
        "median_service_pop_5km_worldpop2020": sensitivity_median,
        "median_ratio_2020_to_2022": ratio,
        "pearson_corr_facility_service_pop": float(corr) if pd.notna(corr) else None,
    }


def safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", default=str(ROOT / "data_processed" / "shared_nas_coverage_audit_cached.csv"))
    parser.add_argument("--out-dir", default=str(ROOT / "data_processed" / "worldpop2020_sensitivity"))
    parser.add_argument("--table-out", default=str(ROOT / "manuscript" / "tables" / "table9_worldpop2020_service_sensitivity.csv"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    service_module = load_service_module()
    coverage = pd.read_csv(args.coverage)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in coverage.itertuples(index=False):
        if not bool(row.nas_worldpop_2020_available):
            continue
        area_id = str(row.area_id)
        registry = ROOT / "data_processed" / f"{area_id}-facility_registry.parquet"
        current = ROOT / "data_processed" / f"{area_id}-facility_service_population.parquet"
        local_path = getattr(row, "local_worldpop_2020_path", "")
        pop_raster = Path(str(local_path)) if str(local_path) and str(local_path) != "nan" else Path(str(row.nas_worldpop_2020_path))
        sensitivity = out_dir / f"{area_id}-facility_service_population_worldpop2020.parquet"
        if not safe_exists(registry) or not safe_exists(current) or not safe_exists(pop_raster):
            rows.append(
                {
                    "area_id": area_id,
                    "city": row.city,
                    "status": "missing_required_input",
                    "error": "",
                    "n_facilities_compared": 0,
                    "median_service_pop_5km_worldpop2022": None,
                    "median_service_pop_5km_worldpop2020": None,
                    "median_ratio_2020_to_2022": None,
                    "pearson_corr_facility_service_pop": None,
                }
            )
            continue
        try:
            if args.force or not sensitivity.exists():
                service_module.compute(str(registry), str(pop_raster), str(sensitivity), [5.0])
            rows.append(summarize_city(area_id, str(row.city), current, sensitivity))
        except Exception as exc:
            rows.append(
                {
                    "area_id": area_id,
                    "city": row.city,
                    "status": "failed",
                    "error": str(exc),
                    "n_facilities_compared": 0,
                    "median_service_pop_5km_worldpop2022": None,
                    "median_service_pop_5km_worldpop2020": None,
                    "median_ratio_2020_to_2022": None,
                    "pearson_corr_facility_service_pop": None,
                }
            )

    table = pd.DataFrame(rows).sort_values("city")
    table_path = Path(args.table_out)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)

    summary_path = out_dir / "worldpop2020_sensitivity_summary.txt"
    compared_table = table[table["status"] == "compared"]
    compared = len(compared_table)
    failed = int((table["status"] == "failed").sum()) if not table.empty else 0
    median_ratio = compared_table["median_ratio_2020_to_2022"].median() if compared else None
    median_corr = compared_table["pearson_corr_facility_service_pop"].median() if compared else None
    summary_path.write_text(
        "\n".join(
            [
                "WorldPop 2020 service-population sensitivity",
                f"cities_compared={compared}",
                f"cities_failed={failed}",
                f"median_city_ratio_2020_to_2022={median_ratio}",
                f"median_city_facility_level_correlation={median_corr}",
                f"table={table_path}",
            ]
        ),
        encoding="utf-8",
    )
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
