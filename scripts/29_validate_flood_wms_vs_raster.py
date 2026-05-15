from __future__ import annotations

"""Compare WMS-derived flood mask (alpha) with JRP GeoTIFF depth exposure for one pilot parquet.

Expects an output produced with `scripts/15_run_pilot_city.py --flood-backend both` where
`flood100y_wms_alpha` is preserved and raster columns are present.
"""

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facilities", required=True, help="e.g. data_processed/shanghai-facility_heat_pop_flood.parquet")
    parser.add_argument("--out-csv", default=None, help="Optional path to write a one-row summary CSV.")
    args = parser.parse_args()

    gdf = gpd.read_parquet(args.facilities)
    need = {"flood100y_wms_alpha", "flood_depth_m_raster"}
    if not need.issubset(gdf.columns):
        missing = need - set(gdf.columns)
        raise SystemExit(f"Missing columns {missing}; run pilot with --flood-backend both and a valid RP100 chip.")

    wms = gdf["flood100y_wms_alpha"].fillna(0).to_numpy() > 0
    ras = np.isfinite(gdf["flood_depth_m_raster"].to_numpy(dtype=float)) & (
        gdf["flood_depth_m_raster"].to_numpy(dtype=float) > 0
    )
    agree = wms == ras
    print(
        "n_facilities",
        len(gdf),
        "wms_positive_rate",
        float(wms.mean()),
        "raster_positive_rate",
        float(ras.mean()),
        "agreement_rate",
        float(agree.mean()),
        "both_true",
        int((wms & ras).sum()),
        "wms_only",
        int((wms & ~ras).sum()),
        "raster_only",
        int((~wms & ras).sum()),
        "both_false",
        int((~wms & ~ras).sum()),
    )
    if args.out_csv:
        row = {
            "n_facilities": len(gdf),
            "wms_positive_rate": float(wms.mean()),
            "raster_positive_rate": float(ras.mean()),
            "agreement_rate": float(agree.mean()),
            "both_true": int((wms & ras).sum()),
            "wms_only": int((wms & ~ras).sum()),
            "raster_only": int((~wms & ras).sum()),
            "both_false": int((~wms & ~ras).sum()),
        }
        pd.DataFrame([row]).to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
