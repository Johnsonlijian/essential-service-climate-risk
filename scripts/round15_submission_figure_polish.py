#!/usr/bin/env python3
"""Polish main-text figures for final submission readability."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "derived_tables"
FIG_DIR = ROOT / "figures"
DOCS = ROOT
OUT = ROOT / "derived_tables" / "round15_figure_polish"


BLUE = "#2166AC"
ORANGE = "#D95F02"
GREEN = "#1B9E77"
PURPLE = "#7B3294"
GREY = "#5F6368"
RED = "#B2182B"


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def wilson_interval(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def display_variant(name: str) -> str:
    mapping = {
        "equal_weight_full": "Equal-weight full",
        "without_source_confidence": "Without source confidence",
        "hazard_plus_deprivation": "Hazard + deprivation",
        "pca_hazard_full": "PCA hazard",
        "hazard_plus_service": "Hazard + service population",
        "hazard_only": "Hazard only",
        "without_flood": "Without flood",
        "without_heat": "Without heat",
        "entropy_weight_full": "Entropy-weighted",
        "without_water_stress": "Without water stress",
    }
    return mapping.get(name, name.replace("_", " "))


def display_scope(name: str) -> str:
    mapping = {
        "osm_nominatim_polygon": "OSM/Nominatim polygon",
        "facility_quantile_95": "95% facility window",
        "bbox_inner_90": "90% inner bbox",
        "bbox_inner_80": "80% inner bbox",
    }
    return mapping.get(name, name.replace("_", " "))


def figure3_grdi_dashboard() -> dict[str, object]:
    df = pd.read_csv(TABLE_DIR / "table2_pilot_grdi_inequality.csv")
    df = df.dropna(subset=["grdi_tercile"]).copy()
    order = ["low", "middle", "high"]
    types = ["health", "school"]

    weighted_rows = []
    for tercile in order:
        for facility_type in types:
            sub = df[(df["grdi_tercile"] == tercile) & (df["facility_type"] == facility_type)]
            n = float(sub["n_facilities"].sum())
            k = float((sub["compound_share"] * sub["n_facilities"]).sum())
            p = k / n if n else np.nan
            lo, hi = wilson_interval(k, n)
            weighted_rows.append(
                {
                    "grdi_tercile": tercile,
                    "facility_type": facility_type,
                    "n_facilities": int(n),
                    "compound_share_weighted": p,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    weighted = pd.DataFrame(weighted_rows)

    gradients = []
    for (city, facility_type), group in df.groupby(["city", "facility_type"]):
        p = group.set_index("grdi_tercile")["compound_share"].to_dict()
        n = group.set_index("grdi_tercile")["n_facilities"].to_dict()
        if "high" in p and "low" in p and min(n["high"], n["low"]) >= 50:
            contrast = "high - low"
            delta = p["high"] - p["low"]
            n_compare = min(n["high"], n["low"])
        elif "high" in p and "middle" in p and min(n["high"], n["middle"]) >= 50:
            contrast = "high - middle"
            delta = p["high"] - p["middle"]
            n_compare = min(n["high"], n["middle"])
        else:
            continue
        gradients.append(
            {
                "city": city,
                "facility_type": facility_type,
                "contrast": contrast,
                "delta": delta,
                "delta_pp": 100 * delta,
                "n_compare_min": int(n_compare),
                "label": f"{city} ({facility_type})",
            }
        )
    grad = pd.DataFrame(gradients)
    bottom = grad.sort_values("delta_pp").head(6)
    top = grad.sort_values("delta_pp", ascending=False).head(6)
    selected = pd.concat([bottom, top], ignore_index=True).sort_values("delta_pp")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8), dpi=300, gridspec_kw={"width_ratios": [1.0, 1.45]})

    ax = axes[0]
    x = np.arange(len(order))
    width = 0.34
    colors = {"health": BLUE, "school": ORANGE}
    for idx, facility_type in enumerate(types):
        sub = weighted[weighted["facility_type"] == facility_type].set_index("grdi_tercile").loc[order]
        y = sub["compound_share_weighted"].to_numpy()
        yerr = np.vstack([y - sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy() - y])
        offset = (idx - 0.5) * width
        bars = ax.bar(
            x + offset,
            y,
            width=width,
            color=colors[facility_type],
            label=facility_type.capitalize(),
            yerr=yerr,
            capsize=3,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, value in zip(bars, y, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#222222",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(["Low", "Middle", "High"])
    ax.set_ylim(0, 0.38)
    ax.set_ylabel("Compound exposure share")
    ax.set_xlabel("GRDI tercile")
    ax.set_title("A. Sample-weighted exposure by deprivation tercile", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)

    ax = axes[1]
    y = np.arange(len(selected))
    colors = np.where(selected["delta_pp"] >= 0, RED, BLUE)
    bars = ax.barh(y, selected["delta_pp"], color=colors, alpha=0.88)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(selected["label"])
    ax.set_xlabel("Deprivation gradient in compound exposure (percentage points)")
    ax.set_title("B. Largest city-facility deprivation gradients", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    max_abs = max(5, float(np.nanmax(np.abs(selected["delta_pp"]))))
    ax.set_xlim(-max_abs * 1.25, max_abs * 1.25)
    for bar, (_, row) in zip(bars, selected.iterrows(), strict=True):
        value = float(row["delta_pp"])
        x_text = value + (1.0 if value >= 0 else -1.0)
        ha = "left" if value >= 0 else "right"
        ax.text(
            x_text,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1f}",
            ha=ha,
            va="center",
            fontsize=7,
            color="#222222",
        )
    fig.suptitle("Compound exposure across deprivation terciles", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG_DIR / "figure3_pilot_grdi_inequality.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    weighted.to_csv(OUT / "figure3_weighted_grdi_summary.csv", index=False)
    selected.to_csv(OUT / "figure3_deprivation_gradient_selected.csv", index=False)
    return {
        "figure": str(out.relative_to(ROOT)),
        "weighted_rows": len(weighted),
        "gradient_candidates": len(grad),
        "selected_gradients": len(selected),
    }


def figure6_escri_ablation() -> dict[str, object]:
    stability = pd.read_csv(TABLE_DIR / "table20_escri_weight_ablation.csv")
    stability = stability.sort_values("spearman_rank_correlation_vs_equal_full", ascending=True).copy()
    stability["display"] = stability["variant"].map(display_variant)
    stability["stable"] = stability["spearman_rank_correlation_vs_equal_full"] >= 0.8

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=300)
    y = np.arange(len(stability))
    colors = np.where(stability["stable"], BLUE, "#A6611A")
    bars = ax.barh(y, stability["spearman_rank_correlation_vs_equal_full"], color=colors, alpha=0.9)
    ax.axvline(0.8, color=ORANGE, linestyle="--", linewidth=1.2)
    ax.text(
        0.805,
        len(stability) - 0.35,
        "0.8 stability gate",
        color=ORANGE,
        fontsize=8,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
    )
    for bar, value in zip(bars, stability["spearman_rank_correlation_vs_equal_full"], strict=True):
        ax.text(
            min(float(value) + 0.012, 1.01),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=7,
            color="#222222",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(stability["display"])
    ax.set_xlim(0.6, 1.04)
    ax.set_xlabel("Spearman rank correlation vs equal-weight ESCRI")
    ax.set_title("ESCRI city-rank stability under weighting and ablation variants")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    fig.tight_layout()
    out = FIG_DIR / "figure6_escri_ablation_rank_stability.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"figure": str(out.relative_to(ROOT)), "variants": len(stability)}


def figure9_boundary_window() -> dict[str, object]:
    boundary = pd.read_csv(TABLE_DIR / "table23_boundary_window_sensitivity.csv")
    plot = boundary[boundary["boundary_scope"] != "bbox_main"].copy()
    order = ["bbox_inner_80", "bbox_inner_90", "facility_quantile_95", "osm_nominatim_polygon"]
    plot["order"] = plot["boundary_scope"].apply(lambda x: order.index(x) if x in order else 999)
    plot = plot.sort_values("order")
    plot["display"] = plot["boundary_scope"].map(display_scope)

    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=300)
    y = np.arange(len(plot))
    h = 0.32
    bars1 = ax.barh(y - h / 2, plot["spearman_escri_rank_vs_bbox"], height=h, label="ESCRI rank", color=GREEN, alpha=0.92)
    bars2 = ax.barh(y + h / 2, plot["spearman_compound_rank_vs_bbox"], height=h, label="Compound rank", color=BLUE, alpha=0.92)
    ax.axvline(0.8, color=ORANGE, linestyle="--", linewidth=1.2)
    ax.text(
        0.805,
        len(plot) - 0.35,
        "0.8 gate",
        color=ORANGE,
        fontsize=8,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )
    for bars, values in [(bars1, plot["spearman_escri_rank_vs_bbox"]), (bars2, plot["spearman_compound_rank_vs_bbox"])]:
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                min(float(value) + 0.004, 1.006),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=7,
            )
    ax.set_yticks(y)
    ax.set_yticklabels(plot["display"])
    ax.set_xlim(0.76, 1.02)
    ax.set_xlabel("Spearman rank correlation vs bbox")
    ax.set_title("Boundary/window sensitivity of city rankings")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = FIG_DIR / "figure9_boundary_window_sensitivity.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"figure": str(out.relative_to(ROOT)), "scopes": len(plot)}


def figure10_inventory_quality() -> dict[str, object]:
    inventory = pd.read_csv(TABLE_DIR / "table24_facility_inventory_quality.csv")
    inv_city = inventory.groupby("city", as_index=False).agg(
        high_confidence_share_ge_0p75=("high_confidence_share_ge_0p75", "mean"),
        share_nn_within_50m=("share_nn_within_50m", "mean"),
        n_facilities=("n_records", "sum"),
    )
    inv_city = inv_city.sort_values("high_confidence_share_ge_0p75")
    x = inv_city["high_confidence_share_ge_0p75"]
    y = inv_city["share_nn_within_50m"]
    sizes = 30 + 140 * np.sqrt(inv_city["n_facilities"] / inv_city["n_facilities"].max())

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=300)
    ax.scatter(x, y, s=sizes, color=PURPLE, alpha=0.72, edgecolor="white", linewidth=0.5)
    ax.axvline(float(x.median()), color="#999999", linestyle=":", linewidth=0.9)
    ax.axhline(float(y.median()), color="#999999", linestyle=":", linewidth=0.9)
    ax.text(
        float(x.median()) + 0.004,
        ax.get_ylim()[1] if ax.get_ylim()[1] else 0.65,
        "median confidence",
        fontsize=7,
        color=GREY,
        va="top",
    )
    labels = {
        "New York City": (10, 8),
        "Los Angeles": (16, 4),
        "Miami": (-28, -12),
        "Istanbul": (8, 4),
        "Sao Paulo": (8, 8),
        "Cairo": (8, 6),
        "Jakarta": (8, -5),
        "Lima": (8, 3),
        "Kolkata": (8, 2),
        "Johannesburg": (8, -6),
    }
    for _, row in inv_city.iterrows():
        city = str(row["city"])
        if city not in labels:
            continue
        offset = labels[city]
        ax.annotate(
            city,
            (row["high_confidence_share_ge_0p75"], row["share_nn_within_50m"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.5,
            ha="left" if offset[0] >= 0 else "right",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.8},
            arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.45, "shrinkA": 1.0, "shrinkB": 2.5},
        )
    ax.set_xlabel("Mean high-confidence record share")
    ax.set_ylabel("Mean nearest-neighbor <=50 m share")
    ax.set_title("Facility inventory quality diagnostics")
    ax.grid(color="#E6E6E6", linewidth=0.6)
    fig.tight_layout()
    out = FIG_DIR / "figure10_inventory_quality_diagnostics.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"figure": str(out.relative_to(ROOT)), "cities": len(inv_city)}


def main() -> None:
    setup()
    summary = {
        "figure3": figure3_grdi_dashboard(),
        "figure6": figure6_escri_ablation(),
        "figure9": figure9_boundary_window(),
        "figure10": figure10_inventory_quality(),
    }
    (OUT / "round15_figure_polish_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = f"""# Round 15 Submission Figure Polish

## Completed figure edits

- Rebuilt Figure 3 from a very tall 30-city small-multiple plot into a compact two-panel main-text dashboard.
- Repositioned the Figure 6 stability-gate annotation and added numeric bar-end labels.
- Repositioned the Figure 9 stability-gate annotation, simplified boundary-scope labels and added bar-end values.
- Repositioned Figure 10 city labels with offset annotations and added median guide lines.

## Output files

- `figures/figure3_pilot_grdi_inequality.png`
- `figures/figure6_escri_ablation_rank_stability.png`
- `figures/figure9_boundary_window_sensitivity.png`
- `figures/figure10_inventory_quality_diagnostics.png`

Figure 3 remains suitable for the main text because it now summarizes the deprivation-tercile signal rather than showing a sparse 30-panel diagnostic.
"""
    (DOCS / "ROUND15_SUBMISSION_FIGURE_POLISH.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
