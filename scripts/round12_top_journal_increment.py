#!/usr/bin/env python3
"""Round 12 top-journal increments.

This script adds three computable increments that do not require redistributing
or downloading new raw third-party data:

1. ESCRI weighting/ablation robustness.
2. Service-node city typology.
3. Humid-heat reclassification summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
AI = ROOT / "ai_autoboost"
OUT = AI / "outputs" / "round12_top_journal_increment"
DOCS = AI / "docs"
TABLE_DIR = ROOT / "manuscript" / "tables"
FIG_DIR = ROOT / "manuscript" / "figures"
DATA = ROOT / "data_processed" / "pilot_facility_indices.parquet"


def clipped_minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    lo = values.quantile(0.01)
    hi = values.quantile(0.99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(values)), index=series.index)
    clipped = values.clip(lo, hi)
    return ((clipped - lo) / (hi - lo)).fillna(0).clip(0, 1)


def entropy_weights(frame: pd.DataFrame) -> np.ndarray:
    x = frame.to_numpy(dtype=float)
    x = np.clip(x, 0, None)
    col_sums = x.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    p = x / col_sums
    n = max(x.shape[0], 2)
    entropy = -(p * np.log(p + 1e-12)).sum(axis=0) / np.log(n)
    divergence = 1 - entropy
    if np.allclose(divergence.sum(), 0):
        return np.repeat(1 / x.shape[1], x.shape[1])
    return divergence / divergence.sum()


def first_pc_score(frame: pd.DataFrame) -> pd.Series:
    x = frame.to_numpy(dtype=float)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-12)
    cov = np.cov(x, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    pc = vecs[:, np.argmax(vals)]
    score = x @ pc
    if np.corrcoef(score, frame.mean(axis=1))[0, 1] < 0:
        score = -score
    return clipped_minmax(pd.Series(score, index=frame.index))


def classify_effect_size(delta: float) -> str:
    ad = abs(delta)
    if ad < 0.02:
        return "small"
    if ad < 0.05:
        return "moderate"
    return "large"


def escri_ablation(facilities: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = facilities.copy()
    df["heat_score_r12"] = clipped_minmax(df["tasmax_hot_days"])
    df["flood_score_r12"] = pd.to_numeric(df["flood_exposed"], errors="coerce").fillna(0).astype(float)
    water_source = df["aqueduct_bws_score"] if "aqueduct_bws_score" in df.columns else df["aqueduct_bws_cat"]
    df["water_score_r12"] = clipped_minmax(water_source)
    df["service_score_r12"] = clipped_minmax(np.log1p(pd.to_numeric(df["service_pop_5p0km"], errors="coerce").fillna(0)))
    df["vulnerability_score_r12"] = clipped_minmax(df["grdi"])
    df["confidence_r12"] = pd.to_numeric(df["source_confidence"], errors="coerce").fillna(0.5).clip(0, 1)

    hazard_cols = ["heat_score_r12", "flood_score_r12", "water_score_r12"]
    hazard_matrix = df[hazard_cols].fillna(0)
    ew = entropy_weights(hazard_matrix)
    pca_hazard = first_pc_score(hazard_matrix)

    variants = {
        "equal_weight_full": hazard_matrix.mean(axis=1)
        * (0.5 + 0.5 * df["service_score_r12"])
        * (0.5 + 0.5 * df["vulnerability_score_r12"])
        * df["confidence_r12"],
        "entropy_weight_full": (hazard_matrix @ ew)
        * (0.5 + 0.5 * df["service_score_r12"])
        * (0.5 + 0.5 * df["vulnerability_score_r12"])
        * df["confidence_r12"],
        "pca_hazard_full": pca_hazard
        * (0.5 + 0.5 * df["service_score_r12"])
        * (0.5 + 0.5 * df["vulnerability_score_r12"])
        * df["confidence_r12"],
        "hazard_only": hazard_matrix.mean(axis=1),
        "hazard_plus_service": hazard_matrix.mean(axis=1) * (0.5 + 0.5 * df["service_score_r12"]),
        "hazard_plus_deprivation": hazard_matrix.mean(axis=1) * (0.5 + 0.5 * df["vulnerability_score_r12"]),
        "without_source_confidence": hazard_matrix.mean(axis=1)
        * (0.5 + 0.5 * df["service_score_r12"])
        * (0.5 + 0.5 * df["vulnerability_score_r12"]),
        "without_heat": df[["flood_score_r12", "water_score_r12"]].mean(axis=1)
        * (0.5 + 0.5 * df["service_score_r12"])
        * (0.5 + 0.5 * df["vulnerability_score_r12"])
        * df["confidence_r12"],
        "without_flood": df[["heat_score_r12", "water_score_r12"]].mean(axis=1)
        * (0.5 + 0.5 * df["service_score_r12"])
        * (0.5 + 0.5 * df["vulnerability_score_r12"])
        * df["confidence_r12"],
        "without_water_stress": df[["heat_score_r12", "flood_score_r12"]].mean(axis=1)
        * (0.5 + 0.5 * df["service_score_r12"])
        * (0.5 + 0.5 * df["vulnerability_score_r12"])
        * df["confidence_r12"],
    }

    city_scores = pd.DataFrame({"city": df["city"]})
    for name, score in variants.items():
        city_scores[name] = score
    city_means = city_scores.groupby("city", as_index=False).mean(numeric_only=True)

    base_rank = city_means["equal_weight_full"].rank(ascending=False, method="min")
    base_top = set(city_means.nlargest(3, "equal_weight_full")["city"])
    rows: list[dict[str, object]] = []
    for name in variants:
        rank = city_means[name].rank(ascending=False, method="min")
        top = set(city_means.nlargest(3, name)["city"])
        rows.append(
            {
                "variant": name,
                "spearman_rank_correlation_vs_equal_full": float(base_rank.corr(rank, method="spearman")),
                "top3_overlap_vs_equal_full": len(base_top & top),
                "max_absolute_rank_shift": int((base_rank - rank).abs().max()),
                "mean_absolute_rank_shift": float((base_rank - rank).abs().mean()),
                "top3_cities": "; ".join(city_means.nlargest(3, name)["city"].tolist()),
                "weight_heat": float(ew[0]) if name == "entropy_weight_full" else "",
                "weight_flood": float(ew[1]) if name == "entropy_weight_full" else "",
                "weight_water": float(ew[2]) if name == "entropy_weight_full" else "",
            }
        )
    stability = pd.DataFrame(rows)
    return city_means, stability


def service_node_typology() -> pd.DataFrame:
    city = pd.read_csv(TABLE_DIR / "table3_pilot_city_ranking.csv")
    cf = pd.read_csv(TABLE_DIR / "cee_counterfactual_classification.csv")
    heat = pd.read_csv(TABLE_DIR / "table13_nasa_power_heat_local_humid_sensitivity.csv")
    flood = pd.read_csv(TABLE_DIR / "table11_jrc_rp100_flood_validation.csv")

    city_flags = []
    compound_cut = city["compound_share"].median()
    service_cut = city["median_service_pop_5km"].median()
    escri_cut = city["mean_escri"].median()
    for _, row in city.iterrows():
        sub = cf[cf["city"] == row["city"]]
        above = int((sub["counterfactual_class"] == "siting_amplification").sum())
        below = int((sub["counterfactual_class"].isin(["relative_buffering", "siting_protection_or_underconcentration"])).sum())
        large_abs = int((sub["observed_minus_counterfactual"].abs() >= 0.05).sum())
        heat_sub = heat[heat["city"] == row["city"]]
        heat_uplift = np.average(
            heat_sub["apparent_abs35_share_20d"] - heat_sub["tmax_abs35_share_20d"],
            weights=heat_sub["n_facilities"],
        ) if not heat_sub.empty else np.nan
        flood_sub = flood[(flood["city"] == row["city"]) & (flood["facility_type"] == "all")]
        flood_agreement = float(flood_sub["agreement_wms_vs_ge015"].iloc[0]) if not flood_sub.empty else np.nan
        source_sensitive = bool((pd.notna(heat_uplift) and heat_uplift >= 0.25) or (pd.notna(flood_agreement) and flood_agreement < 0.85))

        if above > 0 and below > 0:
            typology = "mixed_divergence"
        elif above > 0:
            typology = "facility_overconcentration"
        elif below > 0:
            typology = "facility_underconcentration"
        elif row["compound_share"] >= compound_cut and row["median_service_pop_5km"] >= service_cut:
            typology = "area_wide_service_burden"
        elif source_sensitive:
            typology = "source_sensitive_screening"
        else:
            typology = "lower_priority_or_no_detected_divergence"

        city_flags.append(
            {
                "city": row["city"],
                "n_facilities": int(row["n_facilities"]),
                "compound_share": row["compound_share"],
                "mean_escri": row["mean_escri"],
                "median_service_pop_5km": row["median_service_pop_5km"],
                "above_random_groups": above,
                "below_random_groups": below,
                "large_counterfactual_departures": large_abs,
                "humid_heat_abs35_share_uplift": heat_uplift,
                "jrc_wms_flood_agreement_ge015": flood_agreement,
                "source_sensitive": source_sensitive,
                "service_node_typology": typology,
                "high_compound": bool(row["compound_share"] >= compound_cut),
                "high_service_population": bool(row["median_service_pop_5km"] >= service_cut),
                "high_escri": bool(row["mean_escri"] >= escri_cut),
            }
        )
    return pd.DataFrame(city_flags)


def humid_heat_reclassification() -> pd.DataFrame:
    heat = pd.read_csv(TABLE_DIR / "table13_nasa_power_heat_local_humid_sensitivity.csv")
    rows = []
    for city, sub in heat.groupby("city"):
        weights = sub["n_facilities"]
        dry = np.average(sub["tmax_abs35_share_20d"], weights=weights)
        apparent = np.average(sub["apparent_abs35_share_20d"], weights=weights)
        uplift = apparent - dry
        rows.append(
            {
                "city": city,
                "n_facilities": int(sub["n_facilities"].sum()),
                "weighted_dry_bulb_abs35_share": dry,
                "weighted_apparent_heat_abs35_share": apparent,
                "absolute_share_uplift": uplift,
                "risk_reclassification": "large_humid_heat_uplift"
                if uplift >= 0.25
                else "moderate_humid_heat_uplift"
                if uplift >= 0.10
                else "limited_humid_heat_uplift",
                "mean_apparent_minus_tmax_abs35_days": np.average(sub["mean_apparent_minus_tmax_abs35_days"], weights=weights),
            }
        )
    return pd.DataFrame(rows).sort_values("absolute_share_uplift", ascending=False)


def write_figures(stability: pd.DataFrame, typology: pd.DataFrame, heat_reclass: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    stab = stability.sort_values("spearman_rank_correlation_vs_equal_full")
    plt.figure(figsize=(8, 5), dpi=300)
    plt.barh(stab["variant"], stab["spearman_rank_correlation_vs_equal_full"], color="#386cb0")
    plt.axvline(0.8, color="#d95f02", linestyle="--", linewidth=1, label="0.8 stability gate")
    plt.xlabel("Spearman rank correlation vs equal-weight ESCRI")
    plt.ylabel("ESCRI variant")
    plt.title("ESCRI city-rank stability under weighting and ablation variants")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure6_escri_ablation_rank_stability.png", dpi=300)
    plt.close()

    type_counts = typology["service_node_typology"].value_counts().sort_values()
    plt.figure(figsize=(8, 4.8), dpi=300)
    plt.barh(type_counts.index, type_counts.values, color="#1b9e77")
    plt.xlabel("Number of cities")
    plt.ylabel("Service-node typology")
    plt.title("Thirty-city service-node climate-risk typology")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure7_service_node_typology.png", dpi=300)
    plt.close()

    heat_top = heat_reclass.head(12).sort_values("absolute_share_uplift")
    plt.figure(figsize=(8, 5), dpi=300)
    plt.barh(heat_top["city"], heat_top["absolute_share_uplift"], color="#e6550d")
    plt.xlabel("Apparent heat minus dry-bulb exposed share")
    plt.ylabel("City")
    plt.title("Largest humid-heat reclassification shifts")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure8_humid_heat_reclassification.png", dpi=300)
    plt.close()


def write_report(stability: pd.DataFrame, typology: pd.DataFrame, heat_reclass: pd.DataFrame) -> None:
    stable_count = int((stability["spearman_rank_correlation_vs_equal_full"] >= 0.8).sum())
    unstable = stability[stability["spearman_rank_correlation_vs_equal_full"] < 0.8]["variant"].tolist()
    typology_counts = typology["service_node_typology"].value_counts().to_dict()
    large_heat = int((heat_reclass["risk_reclassification"] == "large_humid_heat_uplift").sum())
    report = f"""# ROUND12_TOP_JOURNAL_INCREMENT_REPORT

Generated from `ai_autoboost/code/round12_top_journal_increment/round12_top_journal_increment.py`.

## Completed real calculations

1. ESCRI weighting and ablation robustness across 10 index variants.
2. Service-node city typology combining area-wide burden, counterfactual divergence, humid-heat uplift and flood-source sensitivity.
3. Humid-heat reclassification summary using the existing NASA POWER sensitivity table.

## Key results

- ESCRI variants passing the Spearman >= 0.8 city-rank stability gate: {stable_count} of {len(stability)}.
- Variants below the stability gate: {', '.join(unstable) if unstable else 'none'}.
- Service-node typology counts: {json.dumps(typology_counts, ensure_ascii=False)}.
- Cities with large humid-heat uplift (apparent heat minus dry-bulb absolute-35C exposed share >= 0.25): {large_heat}.

## Submission-facing interpretation

The new results support a stronger top-journal narrative: the manuscript is not only a 30-city open-data pilot, but a reproducible service-node framework with explicit index robustness checks and a city typology that separates area-wide burden, facility overconcentration, facility underconcentration and source-sensitive screening cases. The ESCRI ablation results should be reported as a robustness check rather than as proof that the index is universally optimal.

## Files generated

- `manuscript/tables/table20_escri_weight_ablation.csv`
- `manuscript/tables/table21_service_node_typology.csv`
- `manuscript/tables/table22_humid_heat_reclassification.csv`
- `manuscript/figures/figure6_escri_ablation_rank_stability.png`
- `manuscript/figures/figure7_service_node_typology.png`
- `manuscript/figures/figure8_humid_heat_reclassification.png`
"""
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "ROUND12_TOP_JOURNAL_INCREMENT_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    facilities = pd.read_parquet(DATA)

    city_means, stability = escri_ablation(facilities)
    typology = service_node_typology()
    heat_reclass = humid_heat_reclassification()

    city_means.to_csv(OUT / "escri_variant_city_scores.csv", index=False)
    stability.to_csv(OUT / "escri_weight_ablation.csv", index=False)
    typology.to_csv(OUT / "service_node_typology.csv", index=False)
    heat_reclass.to_csv(OUT / "humid_heat_reclassification.csv", index=False)

    stability.to_csv(TABLE_DIR / "table20_escri_weight_ablation.csv", index=False)
    typology.to_csv(TABLE_DIR / "table21_service_node_typology.csv", index=False)
    heat_reclass.to_csv(TABLE_DIR / "table22_humid_heat_reclassification.csv", index=False)

    write_figures(stability, typology, heat_reclass)
    write_report(stability, typology, heat_reclass)

    summary = {
        "facility_records": int(len(facilities)),
        "escri_variants": int(len(stability)),
        "rank_stable_variants_spearman_ge_0p8": int((stability["spearman_rank_correlation_vs_equal_full"] >= 0.8).sum()),
        "typology_counts": typology["service_node_typology"].value_counts().to_dict(),
        "large_humid_heat_uplift_cities": int((heat_reclass["risk_reclassification"] == "large_humid_heat_uplift").sum()),
        "outputs": [
            "table20_escri_weight_ablation.csv",
            "table21_service_node_typology.csv",
            "table22_humid_heat_reclassification.csv",
            "figure6_escri_ablation_rank_stability.png",
            "figure7_service_node_typology.png",
            "figure8_humid_heat_reclassification.png",
        ],
    }
    (OUT / "round12_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
