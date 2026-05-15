from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlencode

import geopandas as gpd
import requests


AQUEDUCT_BASELINE_LAYER = (
    "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/"
    "aqueduct_water_risk/FeatureServer/1/query"
)


def download_aqueduct_bbox(bbox: list[float], out_geojson: str) -> None:
    fields = [
        "gid_0",
        "name_0",
        "name_1",
        "bws_raw",
        "bws_score",
        "bws_cat",
        "bws_label",
        "w_awr_def_tot_score",
        "w_awr_def_tot_cat",
    ]
    params = {
        "f": "geojson",
        "where": "1=1",
        "outFields": ",".join(fields),
        "returnGeometry": "true",
        "spatialRel": "esriSpatialRelIntersects",
        "geometry": ",".join(str(x) for x in bbox),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
    }
    response = requests.get(f"{AQUEDUCT_BASELINE_LAYER}?{urlencode(params)}", timeout=120)
    response.raise_for_status()
    Path(out_geojson).parent.mkdir(parents=True, exist_ok=True)
    Path(out_geojson).write_bytes(response.content)


def assign_water(facilities_path: str, aqueduct_geojson: str, out_path: str) -> None:
    gdf = gpd.read_parquet(facilities_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    water = gpd.read_file(aqueduct_geojson).to_crs(gdf.crs)
    keep_cols = [
        "gid_0",
        "name_0",
        "name_1",
        "bws_raw",
        "bws_score",
        "bws_cat",
        "bws_label",
        "w_awr_def_tot_score",
        "w_awr_def_tot_cat",
        "geometry",
    ]
    joined = gpd.sjoin(gdf, water[[c for c in keep_cols if c in water.columns]], how="left", predicate="within")
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])
    joined = joined.rename(
        columns={
            "bws_raw": "aqueduct_bws_raw",
            "bws_score": "aqueduct_bws_score",
            "bws_cat": "aqueduct_bws_cat",
            "bws_label": "aqueduct_bws_label",
            "w_awr_def_tot_score": "aqueduct_overall_water_risk_score",
            "w_awr_def_tot_cat": "aqueduct_overall_water_risk_cat",
        }
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--facilities", required=True)
    parser.add_argument("--aqueduct-geojson", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    download_aqueduct_bbox(args.bbox, args.aqueduct_geojson)
    assign_water(args.facilities, args.aqueduct_geojson, args.out)


if __name__ == "__main__":
    main()

