from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio


DEFAULT_GRDI_URL = (
    "https://data.naturalcapitalalliance.stanford.edu/download/global/"
    "ciesin-nasa-gpw-grdi/ciesin_nasa_povmap-grdi-v1.tif"
)


def sample_grdi(facilities_path: str, out_path: str, grdi_url: str = DEFAULT_GRDI_URL) -> None:
    gdf = gpd.read_parquet(facilities_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    with rasterio.open(grdi_url) as src:
        pts = gdf.to_crs(src.crs)
        coords = [(geom.x, geom.y) for geom in pts.geometry]
        values = [float(v[0]) for v in src.sample(coords)]
        nodata = src.nodata
    series = pd.Series(values, index=gdf.index)
    if nodata is not None:
        series = series.mask(series == nodata)
    gdf["grdi"] = series
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facilities", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--grdi-url", default=DEFAULT_GRDI_URL)
    args = parser.parse_args()
    sample_grdi(args.facilities, args.out, args.grdi_url)


if __name__ == "__main__":
    main()

