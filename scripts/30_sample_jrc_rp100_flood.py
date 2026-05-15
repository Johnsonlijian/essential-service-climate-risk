from __future__ import annotations

"""Sample original JRC/CEMS-GloFAS RP100 flood-depth GeoTIFF tiles.

This replaces the pilot WMS PNG flood mask with a quantitative raster-derived
facility exposure layer, while preserving WMS values for validation.

Outputs:
- data_processed/{area_id}-facility_full_indicators_jrc_rp100.parquet
- data_processed/pilot_facility_indices_jrc_rp100.parquet
- manuscript/tables/table11_jrc_rp100_flood_validation.csv
- manuscript/tables/table12_jrc_rp100_facility_exposure.csv
"""

import argparse
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[1]
FTP_BASE = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard"
TILE_EXTENTS_URL = f"{FTP_BASE}/tile_extents.geojson"


def tile_url(rp_folder: str, tile_id: int, name: str) -> str:
    return f"{FTP_BASE}/{rp_folder}/ID{tile_id}_{name}_{rp_folder}_depth.tif"


def load_areas(path: Path, first_n: int | None) -> pd.DataFrame:
    areas = pd.read_csv(path)
    return areas.iloc[:first_n].copy() if first_n else areas


def load_tiles(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        import requests

        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(TILE_EXTENTS_URL, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    tiles = gpd.read_file(path)
    if tiles.crs is None:
        tiles = tiles.set_crs("EPSG:4326")
    return tiles


def area_bounds(row: pd.Series, pad: float = 0.002) -> tuple[float, float, float, float]:
    return (
        float(row["bbox_west"]) - pad,
        float(row["bbox_south"]) - pad,
        float(row["bbox_east"]) + pad,
        float(row["bbox_north"]) + pad,
    )


def sample_tile_window(
    src: rasterio.io.DatasetReader,
    xs: np.ndarray,
    ys: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    west, south, east, north = bounds
    window = from_bounds(west, south, east, north, transform=src.transform)
    window = window.round_offsets().round_lengths()
    arr = src.read(1, window=window, masked=False)
    transform = src.window_transform(window)
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    out = np.full(len(xs), np.nan, dtype=float)
    ok = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
    if ok.any():
        vals = arr[rows[ok], cols[ok]].astype(float)
        if src.nodata is not None:
            vals = np.where(vals == float(src.nodata), np.nan, vals)
        out[ok] = vals
    return out


def sample_area_depth(
    gdf: gpd.GeoDataFrame,
    area_row: pd.Series,
    tiles: gpd.GeoDataFrame,
    rp_folder: str,
) -> tuple[np.ndarray, str]:
    bounds = area_bounds(area_row)
    geom = box(*bounds)
    hit = tiles[tiles.intersects(geom)].copy()
    if hit.empty:
        return np.full(len(gdf), np.nan, dtype=float), ""

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    pts = gdf.to_crs("EPSG:4326")
    xs = pts.geometry.x.to_numpy(dtype=float)
    ys = pts.geometry.y.to_numpy(dtype=float)
    best = np.full(len(gdf), np.nan, dtype=float)
    used: list[str] = []

    for _, tile in hit.iterrows():
        url = tile_url(rp_folder, int(tile["id"]), str(tile["name"]))
        used.append(url)
        with rasterio.open("/vsicurl/" + url) as src:
            tb = src.bounds
            in_tile = (xs >= tb.left) & (xs <= tb.right) & (ys >= tb.bottom) & (ys <= tb.top)
            if not in_tile.any():
                continue
            sample = sample_tile_window(src, xs[in_tile], ys[in_tile], bounds)
            old = best[in_tile]
            merged = np.where(np.isfinite(old), np.fmax(old, sample), sample)
            best[in_tile] = merged
    return best, ";".join(used)


def add_metadata(gdf: gpd.GeoDataFrame, area_row: pd.Series, countries: pd.DataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    country = countries[countries["country_iso3"] == area_row["country_iso3"]]
    out["city"] = area_row["name"]
    out["country_iso3"] = area_row["country_iso3"]
    out["income_group"] = country["income_group"].iloc[0] if not country.empty else "Unknown"
    out["city_key"] = area_row["area_id"]
    return out


def facility_validation(gdf: gpd.GeoDataFrame, source_urls: str) -> dict[str, float | int | str]:
    if "flood100y_wms_exposed" in gdf.columns:
        wms = gdf["flood100y_wms_exposed"].fillna(False).astype(bool).to_numpy()
    else:
        wms = gdf.get("flood100y_wms_alpha", pd.Series(0, index=gdf.index)).fillna(0).to_numpy() > 0
    depth = gdf["flood_depth_m_jrc_rp100"].to_numpy(dtype=float)
    finite = np.isfinite(depth)
    gt0 = finite & (depth > 0)
    ge015 = finite & (depth >= 0.15)
    ge05 = finite & (depth >= 0.5)
    agree_gt0 = wms == gt0
    agree_ge015 = wms == ge015
    row: dict[str, float | int | str] = {
        "n_facilities": int(len(gdf)),
        "n_depth_finite": int(finite.sum()),
        "wms_share": float(wms.mean()) if len(gdf) else math.nan,
        "raster_gt0_share": float(gt0.mean()) if len(gdf) else math.nan,
        "raster_ge015_share": float(ge015.mean()) if len(gdf) else math.nan,
        "raster_ge05_share": float(ge05.mean()) if len(gdf) else math.nan,
        "agreement_wms_vs_gt0": float(agree_gt0.mean()) if len(gdf) else math.nan,
        "agreement_wms_vs_ge015": float(agree_ge015.mean()) if len(gdf) else math.nan,
        "wms_only_gt0": int((wms & ~gt0).sum()),
        "raster_only_gt0": int((~wms & gt0).sum()),
        "both_gt0": int((wms & gt0).sum()),
        "both_false_gt0": int((~wms & ~gt0).sum()),
        "jrc_source_urls": source_urls,
    }
    return row


def aggregate_exposure(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    x = gdf.copy()
    x["heat_exposed"] = x["tasmax_hot_days"].fillna(0) >= 20
    x["flood_exposed"] = x["flood100y_exposed"].fillna(False).astype(bool)
    x["water_stress_exposed"] = x["aqueduct_bws_cat"].fillna(0) >= 3
    x["compound_exposed"] = x[["heat_exposed", "flood_exposed", "water_stress_exposed"]].sum(axis=1) >= 2
    metrics = {
        "n_facilities": ("id", "count"),
        "heat_share": ("heat_exposed", "mean"),
        "flood_share_jrc_ge015": ("flood_exposed", "mean"),
        "flood_share_jrc_gt0": ("flood100y_raster_exposed_gt0", "mean"),
        "flood_share_jrc_ge05": ("flood100y_raster_exposed_ge05", "mean"),
        "water_stress_share": ("water_stress_exposed", "mean"),
        "compound_share_jrc_ge015": ("compound_exposed", "mean"),
        "median_service_pop_5km": ("service_pop_5p0km", "median"),
        "median_flood_depth_m": ("flood_depth_m_jrc_rp100", "median"),
        "max_flood_depth_m": ("flood_depth_m_jrc_rp100", "max"),
    }
    return (
        x.groupby(["city", "country_iso3", "income_group", "facility_type"], dropna=False)
        .agg(**metrics)
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--area-id", action="append", default=[])
    parser.add_argument("--first-n", type=int, default=30)
    parser.add_argument("--rp-folder", default="RP100")
    parser.add_argument("--depth-threshold-m", type=float, default=0.15)
    parser.add_argument("--areas-csv", default=str(ROOT / "configs" / "pilot_areas.csv"))
    parser.add_argument("--country-csv", default=str(ROOT / "configs" / "country_metadata.csv"))
    parser.add_argument("--tile-cache", default=str(ROOT / "data_raw" / "jrc_flood" / "tile_extents.geojson"))
    parser.add_argument("--input-template", default=str(ROOT / "data_processed" / "{area_id}-facility_full_indicators.parquet"))
    parser.add_argument("--out-template", default=str(ROOT / "data_processed" / "{area_id}-facility_full_indicators_jrc_rp100.parquet"))
    args = parser.parse_args()

    areas = load_areas(Path(args.areas_csv), args.first_n)
    if args.area_id:
        areas = areas[areas["area_id"].isin(args.area_id)].copy()
    countries = pd.read_csv(args.country_csv)
    tiles = load_tiles(Path(args.tile_cache))

    frames: list[gpd.GeoDataFrame] = []
    validation_rows: list[dict[str, float | int | str]] = []

    for _, area in areas.iterrows():
        aid = str(area["area_id"])
        in_path = Path(args.input_template.format(area_id=aid))
        if not in_path.exists():
            print(f"skip missing {in_path}")
            continue
        print(f"sample JRC {args.rp_folder} flood depth for {aid}")
        gdf = gpd.read_parquet(in_path)
        depth, urls = sample_area_depth(gdf, area, tiles, args.rp_folder)
        gdf["flood100y_wms_exposed"] = gdf.get(
            "flood100y_exposed", pd.Series(False, index=gdf.index)
        ).fillna(False).astype(bool)
        gdf["flood_depth_m_jrc_rp100"] = depth
        gdf["flood100y_raster_exposed_gt0"] = np.isfinite(depth) & (depth > 0)
        gdf["flood100y_raster_exposed_ge015"] = np.isfinite(depth) & (depth >= args.depth_threshold_m)
        gdf["flood100y_raster_exposed_ge05"] = np.isfinite(depth) & (depth >= 0.5)
        gdf["flood100y_exposed"] = gdf["flood100y_raster_exposed_ge015"]
        gdf["flood_raster_source"] = "JRC/CEMS-GloFAS global river flood hazard maps v2.1.2 RP100 depth"
        gdf["flood_depth_threshold_m"] = args.depth_threshold_m
        gdf = add_metadata(gdf, area, countries)

        out_path = Path(args.out_template.format(area_id=aid))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(out_path, index=False)
        frames.append(gdf)

        base = facility_validation(gdf, urls)
        base.update({"area_id": aid, "city": area["name"], "facility_type": "all"})
        validation_rows.append(base)
        for ftype, sub in gdf.groupby("facility_type"):
            row = facility_validation(sub, urls)
            row.update({"area_id": aid, "city": area["name"], "facility_type": ftype})
            validation_rows.append(row)

    table_dir = ROOT / "manuscript" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validation_rows).to_csv(table_dir / "table11_jrc_rp100_flood_validation.csv", index=False)
    if frames:
        all_gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)
        all_gdf.to_parquet(ROOT / "data_processed" / "pilot_facility_indices_jrc_rp100.parquet", index=False)
        exposure = aggregate_exposure(all_gdf)
        exposure.to_csv(table_dir / "table12_jrc_rp100_facility_exposure.csv", index=False)


if __name__ == "__main__":
    main()
