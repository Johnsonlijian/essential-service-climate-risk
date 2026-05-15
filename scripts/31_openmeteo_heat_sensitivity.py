from __future__ import annotations

"""Compute local-percentile and humid-heat sensitivity with Open-Meteo Archive.

For each 0.25-degree grid point in each pilot bbox, the script uses 2020-2024
daily maxima as the baseline and counts 2025 days above local p90/p95 thresholds.
It also computes apparent-temperature hot days as a humid-heat proxy.

Outputs:
- data_processed/heat_sensitivity_grids/{area_id}_{provider}_heat_sensitivity.csv
- data_processed/{area_id}-facility_heat_sensitivity_{provider}.parquet
- manuscript/tables/table13_{provider}_heat_local_humid_sensitivity.csv
"""

import argparse
from collections.abc import Iterable
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
NASA_POWER_DAILY = "https://power.larc.nasa.gov/api/temporal/daily/point"


def make_grid(row: pd.Series, step: float) -> pd.DataFrame:
    lons = np.arange(float(row["bbox_west"]), float(row["bbox_east"]) + step / 2, step)
    lats = np.arange(float(row["bbox_south"]), float(row["bbox_north"]) + step / 2, step)
    return pd.DataFrame(
        [{"lon": round(float(lon), 5), "lat": round(float(lat), 5)} for lat in lats for lon in lons]
    )


def provider_suffix(provider: str) -> str:
    return "openmeteo" if provider == "openmeteo" else provider.replace("-", "_")


def _request_openmeteo(params: dict[str, str | float], retries: int = 9) -> object:
    headers = {"User-Agent": "essential-service-climate-risk-research/0.1"}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(OPENMETEO_ARCHIVE, params=params, headers=headers, timeout=180)
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else min(600.0, 60.0 * (attempt + 1))
                print(f"Open-Meteo 429; waiting {wait:.0f}s before retry")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # pragma: no cover - defensive network retry
            last_error = exc
            time.sleep(min(120.0, 10.0 * (attempt + 1)))
    raise RuntimeError(f"Open-Meteo request failed: {last_error}")


def fetch_daily(lat: float, lon: float, start: str, end: str, retries: int = 9) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_max,temperature_2m_mean,apparent_temperature_max",
        "timezone": "UTC",
    }
    data = _request_openmeteo(params, retries=retries)
    daily = pd.DataFrame(data["daily"])
    daily["time"] = pd.to_datetime(daily["time"])
    return daily


def _chunks(items: list[tuple[float, float]], batch_size: int) -> Iterable[list[tuple[float, float]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def fetch_daily_batch(
    points: list[tuple[float, float]],
    start: str,
    end: str,
    retries: int = 9,
) -> list[pd.DataFrame]:
    """Fetch one Open-Meteo response for multiple coordinate pairs.

    Open-Meteo returns a list for multi-coordinate calls and a dict for a
    single-coordinate call. This wrapper normalizes both shapes while keeping
    result order aligned with the input coordinate order.
    """
    if len(points) == 1:
        lat, lon = points[0]
        return [fetch_daily(lat, lon, start, end, retries=retries)]
    params = {
        "latitude": ",".join(f"{lat:.5f}" for lat, _ in points),
        "longitude": ",".join(f"{lon:.5f}" for _, lon in points),
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_max,temperature_2m_mean,apparent_temperature_max",
        "timezone": "UTC",
    }
    data = _request_openmeteo(params, retries=retries)
    responses = data if isinstance(data, list) else [data]
    if len(responses) != len(points):
        raise RuntimeError(
            f"Open-Meteo returned {len(responses)} locations for {len(points)} requested points"
        )
    out: list[pd.DataFrame] = []
    for response in responses:
        daily = pd.DataFrame(response["daily"])
        daily["time"] = pd.to_datetime(daily["time"])
        out.append(daily)
    return out


def _heat_index_c(t_c: pd.Series, rh: pd.Series) -> pd.Series:
    """NOAA-style heat-index proxy from air temperature and relative humidity.

    NASA POWER supplies daily Tmax and daily mean RH2M, not hourly humidity at
    Tmax. The proxy is therefore used as a sensitivity metric rather than as a
    replacement for a full UTCI/WBGT calculation.
    """
    t_f = t_c.astype(float) * 9.0 / 5.0 + 32.0
    r = rh.astype(float).clip(0, 100)
    hi_f = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * r
        - 0.22475541 * t_f * r
        - 0.00683783 * t_f**2
        - 0.05481717 * r**2
        + 0.00122874 * t_f**2 * r
        + 0.00085282 * t_f * r**2
        - 0.00000199 * t_f**2 * r**2
    )
    hi_c = (hi_f - 32.0) * 5.0 / 9.0
    return pd.Series(np.where(t_c.astype(float) >= 26.7, hi_c, t_c.astype(float)), index=t_c.index)


def fetch_nasa_power_daily(lat: float, lon: float, start: str, end: str, retries: int = 5) -> pd.DataFrame:
    params = {
        "parameters": "T2M_MAX,T2M,RH2M",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start.replace("-", ""),
        "end": end.replace("-", ""),
        "format": "JSON",
        "time-standard": "UTC",
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(NASA_POWER_DAILY, params=params, timeout=120)
            r.raise_for_status()
            raw = r.json()["properties"]["parameter"]
            out = pd.DataFrame(
                {
                    "time": pd.to_datetime(pd.Series(raw["T2M_MAX"].keys()), format="%Y%m%d"),
                    "temperature_2m_max": pd.Series(raw["T2M_MAX"].values(), dtype=float).to_numpy(),
                    "temperature_2m_mean": pd.Series(raw["T2M"].values(), dtype=float).to_numpy(),
                    "rh2m_mean_pct": pd.Series(raw["RH2M"].values(), dtype=float).to_numpy(),
                }
            )
            for col in ["temperature_2m_max", "temperature_2m_mean", "rh2m_mean_pct"]:
                out[col] = out[col].where(out[col] > -900)
            out["apparent_temperature_max"] = _heat_index_c(out["temperature_2m_max"], out["rh2m_mean_pct"])
            return out
        except Exception as exc:  # pragma: no cover - defensive network retry
            last_error = exc
            time.sleep(min(120.0, 10.0 * (attempt + 1)))
    raise RuntimeError(f"NASA POWER request failed for {lat},{lon}: {last_error}")


def fetch_nasa_power_batch(points: list[tuple[float, float]], start: str, end: str) -> list[pd.DataFrame]:
    return [fetch_nasa_power_daily(lat, lon, start, end) for lat, lon in points]


def summarize_cell(
    daily: pd.DataFrame,
    baseline_end_year: int,
    target_year: int,
    absolute_threshold_c: float,
) -> dict[str, float | int]:
    baseline = daily[daily["time"].dt.year <= baseline_end_year]
    target = daily[daily["time"].dt.year == target_year]
    tbase = baseline["temperature_2m_max"].astype(float)
    abase = baseline["apparent_temperature_max"].astype(float)
    ttgt = target["temperature_2m_max"].astype(float)
    atgt = target["apparent_temperature_max"].astype(float)
    t90 = float(tbase.quantile(0.90))
    t95 = float(tbase.quantile(0.95))
    a90 = float(abase.quantile(0.90))
    a95 = float(abase.quantile(0.95))
    return {
        "tmax_abs35_days_2025": int((ttgt >= absolute_threshold_c).sum()),
        "tmax_baseline_p90_c": t90,
        "tmax_baseline_p95_c": t95,
        "tmax_local_p90_days_2025": int((ttgt >= t90).sum()),
        "tmax_local_p95_days_2025": int((ttgt >= t95).sum()),
        "apparent_abs35_days_2025": int((atgt >= absolute_threshold_c).sum()),
        "apparent_baseline_p90_c": a90,
        "apparent_baseline_p95_c": a95,
        "apparent_local_p90_days_2025": int((atgt >= a90).sum()),
        "apparent_local_p95_days_2025": int((atgt >= a95).sum()),
        "apparent_minus_tmax_abs35_days": int((atgt >= absolute_threshold_c).sum())
        - int((ttgt >= absolute_threshold_c).sum()),
        "tmax_2025_mean_c": float(ttgt.mean()),
        "apparent_2025_mean_c": float(atgt.mean()),
    }


def build_grid(
    area: pd.Series,
    out_csv: Path,
    step: float,
    baseline_start: str,
    target_end: str,
    baseline_end_year: int,
    target_year: int,
    absolute_threshold_c: float,
    force: bool,
    sleep_s: float,
    batch_size: int,
    provider: str,
) -> pd.DataFrame:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    expected = make_grid(area, step)
    cached = pd.DataFrame()
    if out_csv.exists() and not force:
        cached = pd.read_csv(out_csv)
        if "heat_sensitivity_source" not in cached.columns:
            cached["heat_sensitivity_source"] = "Open-Meteo Archive"
        if "humid_heat_proxy_method" not in cached.columns:
            cached["humid_heat_proxy_method"] = "apparent_temperature_max_from_Open-Meteo"
        cached_points = set(zip(cached["lat"].round(5), cached["lon"].round(5)))
        expected_points = set(zip(expected["lat"].round(5), expected["lon"].round(5)))
        if expected_points.issubset(cached_points):
            cached.to_csv(out_csv, index=False)
            return cached
        print(f"resume incomplete heat grid {out_csv.name}: {len(cached_points)}/{len(expected_points)} points")

    rows: list[dict[str, float | int]] = cached.to_dict("records") if not cached.empty else []
    done_points = set(zip(cached.get("lat", pd.Series(dtype=float)).round(5), cached.get("lon", pd.Series(dtype=float)).round(5)))
    missing = [
        (float(point.lat), float(point.lon))
        for point in expected.itertuples(index=False)
        if (round(float(point.lat), 5), round(float(point.lon), 5)) not in done_points
    ]
    for batch in _chunks(missing, max(1, batch_size)):
        if provider == "openmeteo":
            daily_frames = fetch_daily_batch(batch, baseline_start, target_end)
            source = "Open-Meteo Archive"
            method = "apparent_temperature_max_from_Open-Meteo"
        elif provider == "nasa-power":
            daily_frames = fetch_nasa_power_batch(batch, baseline_start, target_end)
            source = "NASA POWER"
            method = "NOAA_heat_index_from_POWER_T2M_MAX_and_RH2M"
        else:
            raise ValueError(f"unknown heat sensitivity provider: {provider}")
        for (lat, lon), daily in zip(batch, daily_frames, strict=True):
            summary = summarize_cell(daily, baseline_end_year, target_year, absolute_threshold_c)
            summary.update({"lon": round(lon, 5), "lat": round(lat, 5)})
            summary["heat_sensitivity_source"] = source
            summary["humid_heat_proxy_method"] = method
            rows.append(summary)
        out = pd.DataFrame(rows).drop_duplicates(["lat", "lon"], keep="last")
        out = out.sort_values(["lat", "lon"]).reset_index(drop=True)
        out.to_csv(out_csv, index=False)
        print(f"  cached {out_csv.name}: {len(out)}/{len(expected)} points")
        time.sleep(sleep_s)
    out = pd.DataFrame(rows).drop_duplicates(["lat", "lon"], keep="last")
    out = out.sort_values(["lat", "lon"]).reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    return out


def assign_to_facilities(area_id: str, grid: pd.DataFrame, force: bool, suffix: str) -> gpd.GeoDataFrame | None:
    in_path = ROOT / "data_processed" / f"{area_id}-facility_full_indicators.parquet"
    out_path = ROOT / "data_processed" / f"{area_id}-facility_heat_sensitivity_{suffix}.parquet"
    if out_path.exists() and not force:
        return gpd.read_parquet(out_path)
    if not in_path.exists():
        print(f"skip missing {in_path}")
        return None
    gdf = gpd.read_parquet(in_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    tree = cKDTree(grid[["lon", "lat"]].to_numpy())
    dist, idx = tree.query(np.column_stack([gdf.geometry.x, gdf.geometry.y]), k=1)
    for col in grid.columns:
        if col not in {"lon", "lat"}:
            gdf[col] = grid.iloc[idx][col].to_numpy()
    gdf["heat_sensitivity_grid_distance_deg"] = dist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out_path, index=False)
    return gdf


def summarize_facilities(gdf: gpd.GeoDataFrame, area: pd.Series, countries: pd.DataFrame, hot_days_threshold: int) -> pd.DataFrame:
    country = countries[countries["country_iso3"] == area["country_iso3"]]
    income = country["income_group"].iloc[0] if not country.empty else "Unknown"
    x = gdf.copy()
    tests = {
        "tmax_abs35_share_20d": "tmax_abs35_days_2025",
        "tmax_local_p90_share_20d": "tmax_local_p90_days_2025",
        "tmax_local_p95_share_20d": "tmax_local_p95_days_2025",
        "apparent_abs35_share_20d": "apparent_abs35_days_2025",
        "apparent_local_p90_share_20d": "apparent_local_p90_days_2025",
        "apparent_local_p95_share_20d": "apparent_local_p95_days_2025",
    }
    for out_col, days_col in tests.items():
        x[out_col] = x[days_col].fillna(0) >= hot_days_threshold
    source = (
        ";".join(sorted(x.get("heat_sensitivity_source", pd.Series(["Open-Meteo Archive"])).dropna().astype(str).unique()))
        or "Unknown"
    )
    proxy_method = (
        ";".join(
            sorted(
                x.get(
                    "humid_heat_proxy_method",
                    pd.Series(["apparent_temperature_max_from_Open-Meteo"]),
                )
                .dropna()
                .astype(str)
                .unique()
            )
        )
        or "Unknown"
    )
    grouped = (
        x.groupby("facility_type", dropna=False)
        .agg(
            n_facilities=("id", "count"),
            tmax_abs35_share_20d=("tmax_abs35_share_20d", "mean"),
            tmax_local_p90_share_20d=("tmax_local_p90_share_20d", "mean"),
            tmax_local_p95_share_20d=("tmax_local_p95_share_20d", "mean"),
            apparent_abs35_share_20d=("apparent_abs35_share_20d", "mean"),
            apparent_local_p90_share_20d=("apparent_local_p90_share_20d", "mean"),
            apparent_local_p95_share_20d=("apparent_local_p95_share_20d", "mean"),
            mean_tmax_baseline_p95_c=("tmax_baseline_p95_c", "mean"),
            mean_apparent_baseline_p95_c=("apparent_baseline_p95_c", "mean"),
            mean_apparent_minus_tmax_abs35_days=("apparent_minus_tmax_abs35_days", "mean"),
        )
        .reset_index()
    )
    grouped.insert(0, "city", area["name"])
    grouped.insert(1, "country_iso3", area["country_iso3"])
    grouped.insert(2, "income_group", income)
    grouped.insert(3, "area_id", area["area_id"])
    grouped.insert(4, "heat_sensitivity_source", source)
    grouped.insert(5, "humid_heat_proxy_method", proxy_method)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--area-id", action="append", default=[])
    parser.add_argument("--first-n", type=int, default=30)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--baseline-start", default="2020-01-01")
    parser.add_argument("--target-end", default="2025-12-31")
    parser.add_argument("--baseline-end-year", type=int, default=2024)
    parser.add_argument("--target-year", type=int, default=2025)
    parser.add_argument("--absolute-threshold-c", type=float, default=35)
    parser.add_argument("--hot-days-threshold", type=int, default=20)
    parser.add_argument("--sleep-s", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--provider", choices=["openmeteo", "nasa-power"], default="openmeteo")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    areas = pd.read_csv(ROOT / "configs" / "pilot_areas.csv")
    areas = areas.iloc[: args.first_n].copy() if args.first_n else areas
    if args.area_id:
        areas = areas[areas["area_id"].isin(args.area_id)].copy()
    countries = pd.read_csv(ROOT / "configs" / "country_metadata.csv")

    rows: list[pd.DataFrame] = []
    for _, area in areas.iterrows():
        aid = str(area["area_id"])
        suffix = provider_suffix(args.provider)
        print(f"compute heat sensitivity for {aid}")
        grid_csv = ROOT / "data_processed" / "heat_sensitivity_grids" / f"{aid}_{suffix}_heat_sensitivity.csv"
        grid = build_grid(
            area,
            grid_csv,
            args.step,
            args.baseline_start,
            args.target_end,
            args.baseline_end_year,
            args.target_year,
            args.absolute_threshold_c,
            args.force,
            args.sleep_s,
            args.batch_size,
            args.provider,
        )
        gdf = assign_to_facilities(aid, grid, args.force, suffix)
        if gdf is not None:
            rows.append(summarize_facilities(gdf, area, countries, args.hot_days_threshold))

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    table_dir = ROOT / "manuscript" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    suffix = provider_suffix(args.provider)
    out.to_csv(table_dir / f"table13_{suffix}_heat_local_humid_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
