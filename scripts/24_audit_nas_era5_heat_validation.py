"""Audit reusable NAS ERA5 heat files for the climate-risk paper.

The shared Nature_cities archive stores daily ERA5 files with a ``.nc``
extension, but each file is a small ZIP container that contains ``data_0.nc``.
This script reads that wrapper format directly, summarizes the available
variables and heat-threshold diagnostics, and writes manuscript-ready tables.
"""

from __future__ import annotations

import csv
import math
import re
import zipfile
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAS_ERA5_ROOT = Path(r"\\IMUTBIM_NAS\Nature_cities\0-Oridata\3-climate\data\raw\era5")

PILOT_AREAS = PROJECT_ROOT / "configs" / "pilot_areas.csv"
PILOT_EXPOSURE = PROJECT_ROOT / "manuscript" / "tables" / "table1_pilot_facility_exposure.csv"

OUT_PROCESSED = PROJECT_ROOT / "data_processed" / "nas_era5_heat_validation.csv"
OUT_TABLE = PROJECT_ROOT / "manuscript" / "tables" / "table10_nas_era5_heat_validation.csv"

NAS_TO_PILOT_AREA = {
    "Delhi": "delhi",
    "Lagos": "lagos",
    "London": "london",
    "New_York": "new_york",
    "Tokyo": "tokyo",
    # Present in the NAS archive but outside the current 30-city pilot.
    "Dubai": None,
    "Marrakech": None,
    "Singapore": None,
    "Sydney": None,
}


def _read_zip_wrapped_netcdf(path: Path) -> Dataset:
    """Return an in-memory netCDF4 Dataset from a ZIP-wrapped .nc file."""
    with zipfile.ZipFile(path) as zf:
        names = [name for name in zf.namelist() if name.lower().endswith(".nc")]
        if not names:
            raise ValueError("ZIP wrapper contains no .nc member")
        payload = zf.read(names[0])
    return Dataset("inmemory.nc", memory=payload)


def _date_from_values(ds: Dataset) -> tuple[str | None, str | None]:
    if "valid_time" not in ds.variables:
        return None, None
    time_var = ds.variables["valid_time"]
    values = time_var[:]
    if len(values) == 0:
        return None, None
    try:
        dates = num2date(values, units=time_var.units, calendar=getattr(time_var, "calendar", "standard"))
        return str(dates[0])[:19], str(dates[-1])[:19]
    except Exception:
        return None, None


def _extract_date_from_filename(path: Path) -> str | None:
    match = re.search(r"(\d{8})", path.name)
    if not match:
        return None
    raw = match.group(1)
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"


def _finite_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _as_float_array(values: object) -> np.ndarray:
    """Convert netCDF arrays, including masked arrays, to float arrays with NaN."""
    return np.ma.filled(np.ma.asarray(values, dtype="float64"), np.nan)


def _load_pilot_context() -> dict[str, dict[str, object]]:
    areas = pd.read_csv(PILOT_AREAS)
    exposure = pd.read_csv(PILOT_EXPOSURE)
    exposure_summary = (
        exposure.groupby("city", as_index=False)
        .agg(
            pilot_heat_share_mean=("heat_share", "mean"),
            pilot_heat_share_min=("heat_share", "min"),
            pilot_heat_share_max=("heat_share", "max"),
        )
        .set_index("city")
        .to_dict(orient="index")
    )
    context: dict[str, dict[str, object]] = {}
    for row in areas.to_dict(orient="records"):
        context[row["area_id"]] = {
            "pilot_city": row["name"],
            "country_iso3": row["country_iso3"],
            **exposure_summary.get(row["name"], {}),
        }
    return context


def _classify_alignment(pilot_heat_share: float | None, hot_days_domain: int | None) -> str:
    if pilot_heat_share is None or hot_days_domain is None:
        return "not_applicable"
    if pilot_heat_share >= 0.5 and hot_days_domain > 0:
        return "directionally_consistent_hot"
    if pilot_heat_share <= 0.05 and hot_days_domain == 0:
        return "directionally_consistent_low_heat"
    return "needs_interpretation_year_or_threshold_mismatch"


def audit_city(nas_city: str, pilot_context: dict[str, dict[str, object]]) -> dict[str, object]:
    folder = NAS_ERA5_ROOT / nas_city
    files = sorted(folder.glob("*.nc"))
    pilot_area_id = NAS_TO_PILOT_AREA.get(nas_city)
    pilot = pilot_context.get(pilot_area_id or "", {})

    variables: set[str] = set()
    first_time: str | None = None
    last_time: str | None = None
    first_file_date: str | None = None
    last_file_date: str | None = None
    readable = 0
    unreadable = 0
    errors: list[str] = []
    domain_max_c_values: list[float] = []
    area_mean_daily_max_c_values: list[float] = []
    dewpoint_domain_max_c_values: list[float] = []
    ssrd_domain_max_values: list[float] = []

    for path in files:
        file_date = _extract_date_from_filename(path)
        if file_date is not None:
            first_file_date = min(first_file_date, file_date) if first_file_date else file_date
            last_file_date = max(last_file_date, file_date) if last_file_date else file_date
        try:
            ds = _read_zip_wrapped_netcdf(path)
            try:
                readable += 1
                for name in ds.variables:
                    variables.add(name)
                start, end = _date_from_values(ds)
                if start is not None:
                    first_time = min(first_time, start) if first_time else start
                if end is not None:
                    last_time = max(last_time, end) if last_time else end

                if "t2m" in ds.variables:
                    t2m_c = _as_float_array(ds.variables["t2m"][:]) - 273.15
                    if t2m_c.size and np.isfinite(t2m_c).any():
                        domain_max_c_values.append(float(np.nanmax(t2m_c)))
                        valid_cells = np.isfinite(t2m_c).any(axis=0)
                        if np.any(valid_cells):
                            daily_cell_max = np.nanmax(t2m_c[:, valid_cells], axis=0)
                            area_mean_daily_max_c_values.append(float(np.nanmean(daily_cell_max)))

                if "d2m" in ds.variables:
                    d2m_c = _as_float_array(ds.variables["d2m"][:]) - 273.15
                    if d2m_c.size and np.isfinite(d2m_c).any():
                        dewpoint_domain_max_c_values.append(float(np.nanmax(d2m_c)))

                if "ssrd" in ds.variables:
                    ssrd = _as_float_array(ds.variables["ssrd"][:])
                    if ssrd.size and np.isfinite(ssrd).any():
                        ssrd_domain_max_values.append(float(np.nanmax(ssrd)))
            finally:
                ds.close()
        except Exception as exc:  # keep auditing even if individual files fail
            unreadable += 1
            if len(errors) < 5:
                errors.append(f"{path.name}: {type(exc).__name__}: {exc}")

    hot_days_domain = sum(value >= 35.0 for value in domain_max_c_values) if domain_max_c_values else None
    hot_days_area_mean = (
        sum(value >= 35.0 for value in area_mean_daily_max_c_values)
        if area_mean_daily_max_c_values
        else None
    )
    pilot_heat_share = _finite_float(pilot.get("pilot_heat_share_mean"))

    return {
        "nas_city": nas_city,
        "in_30_city_pilot": pilot_area_id is not None,
        "pilot_area_id": pilot_area_id,
        "pilot_city": pilot.get("pilot_city"),
        "country_iso3": pilot.get("country_iso3"),
        "nas_files_total": len(files),
        "readable_files": readable,
        "unreadable_files": unreadable,
        "file_date_min": first_file_date,
        "file_date_max": last_file_date,
        "valid_time_min": first_time,
        "valid_time_max": last_time,
        "variables": ";".join(sorted(variables)),
        "has_t2m": "t2m" in variables,
        "has_d2m": "d2m" in variables,
        "has_ssrd": "ssrd" in variables,
        "era5_domain_max_t2m_c": _finite_float(max(domain_max_c_values) if domain_max_c_values else None),
        "era5_mean_area_daily_max_t2m_c": _finite_float(
            mean(area_mean_daily_max_c_values) if area_mean_daily_max_c_values else None
        ),
        "era5_hot_days_domain_max_gt35c": hot_days_domain,
        "era5_hot_days_area_mean_gt35c": hot_days_area_mean,
        "era5_domain_max_dewpoint_c": _finite_float(
            max(dewpoint_domain_max_c_values) if dewpoint_domain_max_c_values else None
        ),
        "era5_domain_max_ssrd_j_m2": _finite_float(
            max(ssrd_domain_max_values) if ssrd_domain_max_values else None
        ),
        "pilot_heat_share_mean_2025": pilot_heat_share,
        "pilot_heat_share_min_2025": _finite_float(pilot.get("pilot_heat_share_min")),
        "pilot_heat_share_max_2025": _finite_float(pilot.get("pilot_heat_share_max")),
        "directional_alignment": _classify_alignment(pilot_heat_share, hot_days_domain),
        "read_error_examples": " | ".join(errors),
        "source_note": "NAS ERA5 ZIP-wrapped NetCDF; diagnostic validation layer, not main manuscript heat replacement",
    }


def main() -> None:
    if not NAS_ERA5_ROOT.exists():
        raise FileNotFoundError(f"NAS ERA5 root not found: {NAS_ERA5_ROOT}")

    pilot_context = _load_pilot_context()
    nas_cities = sorted(path.name for path in NAS_ERA5_ROOT.iterdir() if path.is_dir())
    rows = [audit_city(city, pilot_context) for city in nas_cities]
    df = pd.DataFrame(rows)

    OUT_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)

    # Preserve the full machine-readable audit.
    df.to_csv(OUT_PROCESSED, index=False, quoting=csv.QUOTE_MINIMAL)

    # Manuscript table: concise, rounded, and sorted with pilot overlap first.
    table_cols = [
        "nas_city",
        "in_30_city_pilot",
        "pilot_city",
        "nas_files_total",
        "readable_files",
        "file_date_min",
        "file_date_max",
        "variables",
        "era5_domain_max_t2m_c",
        "era5_mean_area_daily_max_t2m_c",
        "era5_hot_days_domain_max_gt35c",
        "era5_hot_days_area_mean_gt35c",
        "pilot_heat_share_mean_2025",
        "directional_alignment",
    ]
    table = df[table_cols].copy()
    for col in [
        "era5_domain_max_t2m_c",
        "era5_mean_area_daily_max_t2m_c",
        "pilot_heat_share_mean_2025",
    ]:
        table[col] = table[col].map(lambda value: round(value, 3) if pd.notna(value) else value)
    table = table.sort_values(["in_30_city_pilot", "nas_city"], ascending=[False, True])
    table.to_csv(OUT_TABLE, index=False, quoting=csv.QUOTE_MINIMAL)

    overlap = int(df["in_30_city_pilot"].sum())
    readable_overlap = int(df.loc[df["in_30_city_pilot"], "readable_files"].sum())
    total_overlap = int(df.loc[df["in_30_city_pilot"], "nas_files_total"].sum())
    print(f"Wrote {OUT_PROCESSED}")
    print(f"Wrote {OUT_TABLE}")
    print(
        f"NAS ERA5 folders: {len(df)}; pilot overlaps: {overlap}; "
        f"overlap readable files: {readable_overlap}/{total_overlap}"
    )


if __name__ == "__main__":
    main()
