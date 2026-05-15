from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlencode

import geopandas as gpd
import numpy as np
import requests
from PIL import Image


WMS_URL = "https://globalfloods-ows.ecmwf.int/glofas-ows/ows.py"


def download_wms_png(bbox: list[float], out_png: str, width: int = 2048, height: int = 2048) -> None:
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": "FloodHazard100y",
        "STYLES": "",
        "SRS": "EPSG:4326",
        "BBOX": ",".join(str(x) for x in bbox),
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "FORMAT": "image/png",
        "TRANSPARENT": "TRUE",
    }
    response = requests.get(f"{WMS_URL}?{urlencode(params)}", timeout=120)
    response.raise_for_status()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    Path(out_png).write_bytes(response.content)


def assign_flood_from_png(facilities_path: str, bbox: list[float], png_path: str, out_path: str) -> None:
    gdf = gpd.read_parquet(facilities_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    west, south, east, north = bbox
    image = Image.open(png_path).convert("RGBA")
    arr = np.asarray(image)
    height, width = arr.shape[:2]
    x = ((gdf.geometry.x - west) / (east - west) * (width - 1)).round().astype(int)
    y = ((north - gdf.geometry.y) / (north - south) * (height - 1)).round().astype(int)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    alpha = np.zeros(len(gdf), dtype=np.uint8)
    alpha[valid.to_numpy()] = arr[y[valid], x[valid], 3]
    gdf["flood100y_wms_alpha"] = alpha
    gdf["flood100y_exposed"] = alpha > 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--facilities", required=True)
    parser.add_argument("--png", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=2048)
    args = parser.parse_args()
    download_wms_png(args.bbox, args.png, args.width, args.height)
    assign_flood_from_png(args.facilities, args.bbox, args.png, args.out)


if __name__ == "__main__":
    main()

