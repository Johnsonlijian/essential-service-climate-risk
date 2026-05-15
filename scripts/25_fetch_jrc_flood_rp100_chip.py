from __future__ import annotations

"""Download JRC CEMS-GloFAS global river flood hazard tiles (RP100 depth) and mosaic a city chip.

Anonymous HTTP source (no CDS account):
  https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/

Tile catalogue:
  .../tile_extents.geojson

File naming (README):
  ID{tile_id}_{N/S}{lat}_{E/W}{lon}_RP100_depth.tif

Output:
  data_raw/jrc_flood/chips/{area_id}_RP100_depth.tif

`scripts/15_run_pilot_city.py` prefers this chip path when present (before global flood_depth_100y.tif).
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
import requests
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[1]
TILE_EXTENTS_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/tile_extents.geojson"
)
FTP_BASE = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard"


def _tile_url(rp_folder: str, tile_id: int, name: str) -> str:
    return f"{FTP_BASE}/{rp_folder}/ID{tile_id}_{name}_{rp_folder}_depth.tif"


def _download(url: str, dest: Path, chunk: int = 1024 * 1024) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for block in r.iter_content(chunk_size=chunk):
                if block:
                    f.write(block)


def load_tiles(cache: Path) -> gpd.GeoDataFrame:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.is_file():
        with requests.get(TILE_EXTENTS_URL, timeout=120) as r:
            r.raise_for_status()
            cache.write_bytes(r.content)
    return gpd.read_file(cache)


def bbox_from_area(row: pd.Series) -> tuple[float, float, float, float]:
    return (
        float(row["bbox_west"]),
        float(row["bbox_south"]),
        float(row["bbox_east"]),
        float(row["bbox_north"]),
    )


def intersecting_tiles(tiles: gpd.GeoDataFrame, west: float, south: float, east: float, north: float) -> gpd.GeoDataFrame:
    b = box(west, south, east, north)
    g = tiles.copy()
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    return g[g.intersects(b)].copy()


def mosaic_chip(
    tif_paths: list[Path],
    bounds: tuple[float, float, float, float],
    out_path: Path,
) -> None:
    srcs = [rasterio.open(p) for p in tif_paths]
    try:
        arr, transform = merge(srcs, bounds=bounds, nodata=srcs[0].nodata)
        profile = srcs[0].profile.copy()
        profile.update(
            height=arr.shape[1],
            width=arr.shape[2],
            transform=transform,
            compress="deflate",
            tiled=True,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr)
    finally:
        for s in srcs:
            s.close()


def run_for_area(
    area_id: str,
    tiles: gpd.GeoDataFrame,
    pilots: pd.DataFrame,
    rp_folder: str,
    tile_dir: Path,
    chip_dir: Path,
    force: bool,
) -> Path | None:
    row = pilots[pilots["area_id"] == area_id]
    if row.empty:
        print(f"unknown area_id={area_id}", file=sys.stderr)
        return None
    row = row.iloc[0]
    west, south, east, north = bbox_from_area(row)
    chip_path = chip_dir / f"{area_id}_{rp_folder}_depth.tif"
    if chip_path.is_file() and not force:
        print(f"skip existing chip {chip_path}")
        return chip_path

    hit = intersecting_tiles(tiles, west, south, east, north)
    if hit.empty:
        print(f"no tiles intersect bbox for {area_id}", file=sys.stderr)
        return None

    paths: list[Path] = []
    for _, feat in hit.iterrows():
        tid = int(feat["id"])
        name = str(feat["name"])
        url = _tile_url(rp_folder, tid, name)
        local = tile_dir / rp_folder / f"ID{tid}_{name}_{rp_folder}_depth.tif"
        if not local.is_file() or force:
            print(f"download {url}")
            try:
                _download(url, local)
            except requests.HTTPError as e:
                print(f"WARN: failed {url}: {e}", file=sys.stderr)
                continue
        paths.append(local)

    if not paths:
        print(f"no tiles downloaded for {area_id}", file=sys.stderr)
        return None

    pad = 0.002
    bounds = (west - pad, south - pad, east + pad, north + pad)
    print(f"mosaic {len(paths)} tiles -> {chip_path}")
    mosaic_chip(paths, bounds, chip_path)
    return chip_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--area-id", action="append", default=[], help="Pilot area_id (repeatable). Default: all pilot_areas.")
    parser.add_argument("--rp-folder", default="RP100", help="Subfolder under flood_hazard/, e.g. RP100, RP50.")
    parser.add_argument(
        "--tile-cache",
        default=str(ROOT / "data_raw" / "jrc_flood" / "tile_extents.geojson"),
        help="Local path for tile_extents.geojson cache.",
    )
    parser.add_argument(
        "--tile-dir",
        default=str(ROOT / "data_raw" / "jrc_flood" / "tiles"),
        help="Directory for raw per-tile GeoTIFFs.",
    )
    parser.add_argument(
        "--chip-dir",
        default=str(ROOT / "data_raw" / "jrc_flood" / "chips"),
        help="Output directory for mosaicked bbox chips.",
    )
    parser.add_argument("--pilot-csv", default=str(ROOT / "configs" / "pilot_areas.csv"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    pilots = pd.read_csv(args.pilot_csv)
    ids = list(args.area_id) if args.area_id else pilots["area_id"].tolist()
    tiles = load_tiles(Path(args.tile_cache))
    tile_dir = Path(args.tile_dir)
    chip_dir = Path(args.chip_dir)

    for aid in ids:
        run_for_area(aid, tiles, pilots, args.rp_folder, tile_dir, chip_dir, args.force)


if __name__ == "__main__":
    main()
