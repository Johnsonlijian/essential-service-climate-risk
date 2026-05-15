from __future__ import annotations

from pathlib import Path
import math

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIG_DIR = Path("manuscript/figures")
TABLE_DIR = Path("manuscript/tables")
ROOT = Path(__file__).resolve().parents[1]


def load_indices() -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet("data_processed/pilot_facility_indices.parquet")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def city_bboxes() -> dict[str, tuple[float, float, float, float]]:
    areas = pd.read_csv(ROOT / "configs" / "pilot_areas.csv")
    return {
        row["name"]: (row["bbox_west"], row["bbox_south"], row["bbox_east"], row["bbox_north"])
        for _, row in areas.iterrows()
    }


def savefig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure1_maps(gdf: gpd.GeoDataFrame) -> None:
    boxes = city_bboxes()
    cities = [(city, boxes[city]) for city in gdf["city"].drop_duplicates() if city in boxes]
    ncols = 2
    nrows = math.ceil(len(cities) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 4.1 * nrows), constrained_layout=True)
    axes = axes.ravel()
    for ax, (city, bbox) in zip(axes, cities, strict=True):
        sub = gdf[gdf["city"] == city]
        west, south, east, north = bbox
        ax.scatter(sub.geometry.x, sub.geometry.y, c=sub["escri"], s=2, cmap="magma_r", alpha=0.55)
        flood = sub[sub["flood_exposed"]]
        ax.scatter(flood.geometry.x, flood.geometry.y, s=1.2, color="#1478a6", alpha=0.35)
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        ax.set_title(city)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    for ax in axes[len(cities):]:
        ax.axis("off")
    fig.suptitle("Figure 1. Pilot facility risk maps")
    savefig(fig, "figure1_pilot_facility_risk_maps.png")


def figure2_exposure_bars() -> None:
    table = pd.read_csv(TABLE_DIR / "table1_pilot_facility_exposure.csv")
    metrics = ["heat_share", "flood_share", "water_stress_share", "compound_share"]
    labels = ["Heat", "Flood", "Water stress", "Compound"]
    table["group"] = table["city"] + " " + table["facility_type"]
    matrix = table[metrics].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, max(4.8, 0.28 * len(table))), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(np.arange(len(table)))
    ax.set_yticklabels(table["group"])
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value > 0.55 else "black")
    fig.colorbar(im, ax=ax, label="Share of facilities")
    ax.set_title("Figure 2. Facility exposure by hazard")
    savefig(fig, "figure2_pilot_exposure_bars.png")


def figure3_grdi_inequality() -> None:
    table = pd.read_csv(TABLE_DIR / "table2_pilot_grdi_inequality.csv")
    cities = table["city"].drop_duplicates().tolist()
    ncols = 2
    nrows = math.ceil(len(cities) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.4 * nrows), sharey=True, constrained_layout=True)
    axes = axes.ravel()
    for ax, city in zip(axes, cities, strict=True):
        sub = table[table["city"] == city]
        for facility_type, group in sub.groupby("facility_type"):
            order = ["low", "middle", "high"]
            group = group.set_index("grdi_tercile").reindex(order)
            ax.plot(order, group["compound_share"], marker="o", label=facility_type)
        ax.set_title(city)
        ax.set_xlabel("GRDI tercile")
        ax.set_ylim(0, 1.05)
    for idx in range(0, len(cities), ncols):
        axes[idx].set_ylabel("Compound exposure share")
    for ax in axes[len(cities):]:
        ax.axis("off")
    axes[min(1, len(cities) - 1)].legend(frameon=False)
    fig.suptitle("Figure 3. Compound exposure across deprivation terciles")
    savefig(fig, "figure3_pilot_grdi_inequality.png")


def figure4_counterfactual() -> None:
    table = pd.read_csv(TABLE_DIR / "table5_population_weighted_counterfactual.csv")
    table["group"] = table["city"] + " " + table["facility_type"]
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(8.5, 4.3), constrained_layout=True)
    ax.scatter(x, table["observed_compound_share"], label="Observed facilities", color="#222222", zorder=3)
    ax.errorbar(
        x,
        table["counterfactual_compound_mean"],
        yerr=[
            table["counterfactual_compound_mean"] - table["counterfactual_compound_p05"],
            table["counterfactual_compound_p95"] - table["counterfactual_compound_mean"],
        ],
        fmt="o",
        color="#2f80b7",
        ecolor="#9cc9e2",
        capsize=4,
        label="Population-weighted random baseline",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(table["group"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Compound exposure share")
    ax.legend(frameon=False)
    ax.set_title("Figure 4. Facility siting against population-weighted baseline")
    savefig(fig, "figure4_pilot_counterfactual.png")


def figure5_priority(gdf: gpd.GeoDataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5), constrained_layout=True)
    markers = ["o", "^", "s", "D", "P", "X", "v", "<", ">", "*"]
    for marker, city in zip(markers, gdf["city"].drop_duplicates(), strict=False):
        sub = gdf[gdf["city"] == city].sample(min(4000, (gdf["city"] == city).sum()), random_state=8)
        ax.scatter(
            np.log1p(sub["service_pop_5p0km"]),
            sub["hazard_score"],
            s=7,
            alpha=0.35,
            marker=marker,
            label=city,
        )
    ax.set_xlabel("log(1 + population within 5 km)")
    ax.set_ylabel("Hazard score")
    ax.legend(frameon=False)
    ax.set_title("Figure 5. Adaptation priority space")
    savefig(fig, "figure5_pilot_priority_space.png")

    top = (
        gdf.sort_values("escri", ascending=False)
        .head(100)[
            [
                "city",
                "facility_type",
                "name",
                "escri",
                "hazard_score",
                "service_pop_5p0km",
                "grdi",
                "heat_exposed",
                "flood_exposed",
                "water_stress_exposed",
            ]
        ]
    )
    top.to_csv(TABLE_DIR / "supplement_top100_pilot_priority_facilities.csv", index=False)


def main() -> None:
    gdf = load_indices()
    figure1_maps(gdf)
    figure2_exposure_bars()
    figure3_grdi_inequality()
    figure4_counterfactual()
    figure5_priority(gdf)


if __name__ == "__main__":
    main()
