from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from cnrisk.paths import MANUSCRIPT, ensure_dirs


def make_figures(indices_path: str, out_dir: str) -> None:
    ensure_dirs()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_parquet(indices_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    gdf.plot(ax=ax, column="escri", cmap="magma_r", markersize=2, alpha=0.55, legend=True)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_axis_off()
    ax.set_title("Pilot essential-service climate risk index")
    fig.savefig(out / "fig1_pilot_global_map.png", dpi=300)
    plt.close(fig)

    summary = (
        gdf.groupby(["facility_type"], dropna=False)
        .agg(
            compound_share=("compound_exposed", "mean"),
            heat_share=("heat_exposed", "mean"),
            flood_share=("flood_exposed", "mean"),
            water_share=("water_stress_exposed", "mean"),
        )
        .reset_index()
    )
    plot_df = summary.melt("facility_type", var_name="hazard", value_name="share")
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for facility_type, data in plot_df.groupby("facility_type"):
        ax.plot(data["hazard"], data["share"], marker="o", label=facility_type)
    ax.set_ylabel("Share of facilities")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.savefig(out / "fig2_pilot_exposure_by_type.png", dpi=300)
    plt.close(fig)

    pd.DataFrame(summary).to_csv(out / "fig2_pilot_exposure_by_type.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", default="data_processed/facility_indices.parquet")
    parser.add_argument("--out-dir", default=str(MANUSCRIPT / "figures"))
    args = parser.parse_args()
    make_figures(args.indices, args.out_dir)


if __name__ == "__main__":
    main()
