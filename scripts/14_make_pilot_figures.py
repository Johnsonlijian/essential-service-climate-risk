from __future__ import annotations

from pathlib import Path
import math

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


FIG_DIR = Path("manuscript/figures")
TABLE_DIR = Path("manuscript/tables")
ROOT = Path(__file__).resolve().parents[1]
ROUND5_DIR = Path("ai_autoboost/outputs/round5")


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 450,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


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
    significance_path = ROUND5_DIR / "counterfactual_significance_table.csv"
    if significance_path.exists():
        table = pd.read_csv(significance_path)
    else:
        table = pd.read_csv(TABLE_DIR / "table5_population_weighted_counterfactual.csv")
        table["empirical_p_two_sided_holm"] = np.nan
        table["significance_direction"] = "not_tested"
        table["counterfactual_compound_p025"] = table["counterfactual_compound_p05"]
        table["counterfactual_compound_p975"] = table["counterfactual_compound_p95"]

    table["group"] = table["city"] + " - " + table["facility_type"]
    table = table.sort_values("observed_minus_counterfactual", ascending=True).reset_index(drop=True)
    table["delta_pp"] = table["observed_minus_counterfactual"] * 100
    table["ci_low_pp"] = (table["observed_compound_share"] - table["counterfactual_compound_p975"]) * 100
    table["ci_high_pp"] = (table["observed_compound_share"] - table["counterfactual_compound_p025"]) * 100

    def _color(direction: str) -> str:
        if direction == "observed_above_random_after_holm":
            return "#B2182B"
        if direction == "observed_below_random_after_holm":
            return "#2166AC"
        return "#6A6A6A"

    colors = table["significance_direction"].map(_color)
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(8.8, max(8.8, 0.22 * len(table))))
    xerr = np.vstack(
        [
            table["delta_pp"] - table["ci_low_pp"],
            table["ci_high_pp"] - table["delta_pp"],
        ]
    )
    ax.errorbar(
        table["delta_pp"],
        y,
        xerr=xerr,
        fmt="none",
        ecolor="#BDBDBD",
        elinewidth=0.75,
        capsize=2.2,
        zorder=1,
    )
    ax.scatter(table["delta_pp"], y, c=colors, s=20, edgecolor="white", linewidth=0.35, zorder=3)
    ax.axvline(0, color="#222222", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(table["group"])
    ax.set_xlabel("Observed minus population-weighted random baseline (percentage points)")
    ax.set_ylabel("")
    ax.set_title("Figure 4. Facility compound exposure relative to population-weighted random baselines")
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", color="w", label="Above random after Holm", markerfacecolor="#B2182B", markersize=6),
        Line2D([0], [0], marker="o", color="w", label="Below random after Holm", markerfacecolor="#2166AC", markersize=6),
        Line2D([0], [0], marker="o", color="w", label="Not significant / not tested", markerfacecolor="#6A6A6A", markersize=6),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.subplots_adjust(left=0.34, right=0.98, top=0.96, bottom=0.06)
    savefig(fig, "figure4_pilot_counterfactual.png")


def figure5_priority(gdf: gpd.GeoDataFrame) -> None:
    table = pd.read_csv(TABLE_DIR / "table1_pilot_facility_exposure.csv").copy()
    table["group"] = table["city"] + " - " + table["facility_type"]
    table["log_service_pop_5km"] = np.log10(table["median_service_pop_5km"].clip(lower=0) + 1)
    sizes = np.clip(np.sqrt(table["n_facilities"]) * 2.0, 24, 320)

    fig, ax = plt.subplots(figsize=(7.4, 5.3), constrained_layout=True)
    scatter = ax.scatter(
        table["log_service_pop_5km"],
        table["compound_share"],
        c=table["mean_escri"],
        s=sizes,
        cmap="cividis",
        alpha=0.88,
        edgecolor="#222222",
        linewidth=0.35,
    )
    ax.set_xlabel("log10(1 + median population within 5 km)")
    ax.set_ylabel("Compound exposure share")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("Figure 5. Adaptation priority space across city-facility groups")
    ax.grid(color="#E0E0E0", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Mean ESCRI")

    for n in [500, 5_000, 25_000]:
        ax.scatter([], [], s=np.clip(np.sqrt(n) * 2.0, 24, 320), c="#D9D9D9", edgecolor="#222222", linewidth=0.35, label=f"{n:,} facilities")
    size_legend = ax.legend(
        title="Group size",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.02, 0.47),
        ncol=1,
        borderaxespad=0,
        handletextpad=0.8,
        labelspacing=1.1,
    )
    size_legend.get_title().set_fontsize(8)
    ax.add_artist(size_legend)
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
