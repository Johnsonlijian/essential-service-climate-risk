from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "manuscript" / "tables"
REPORT_DIR = ROOT / "manuscript"


def dominant_pathway(row: pd.Series) -> str:
    """Screening pathway labels for figure planning.

    The thresholds are intentionally simple and transparent:
    heat and water are dominant when at least half of audited facilities are
    exposed; flood is dominant when at least 15% are flagged, because the
    current flood layer is a provisional binary screening layer and flood
    shares are generally lower.
    """

    parts: list[str] = []
    if row["heat_share"] >= 0.5:
        parts.append("heat")
    if row["flood_share"] >= 0.15:
        parts.append("flood")
    if row["water_stress_share"] >= 0.5:
        parts.append("water")
    return "+".join(parts) if parts else "threshold_limited"


def counterfactual_class(row: pd.Series) -> str:
    obs = row["observed_compound_share"]
    cf = row["counterfactual_compound_mean"]
    p05 = row["counterfactual_compound_p05"]
    p95 = row["counterfactual_compound_p95"]
    if obs == 0 and cf == 0:
        return "threshold_limited_no_signal"
    if obs >= 0.95 and cf >= 0.95 and p05 <= obs <= p95:
        return "diffuse_high_risk"
    if obs > p95:
        return "siting_amplification"
    if obs < p05:
        return "relative_buffering"
    return "population_like"


def md_table(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a Markdown table without optional deps."""

    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.3f}")
    rows = []
    headers = [str(c) for c in view.columns]
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in view.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in view.columns) + " |")
    return "\n".join(rows)


def main() -> None:
    t1 = pd.read_csv(TABLE_DIR / "table1_pilot_facility_exposure.csv")
    t5 = pd.read_csv(TABLE_DIR / "table5_population_weighted_counterfactual.csv")
    t6 = pd.read_csv(TABLE_DIR / "table6_threshold_and_scope_sensitivity.csv")
    t9 = pd.read_csv(TABLE_DIR / "table9_worldpop2020_service_sensitivity.csv")
    t10 = pd.read_csv(TABLE_DIR / "table10_nas_era5_heat_validation.csv")

    pathways = t1.copy()
    pathways["dominant_pathway_screen"] = pathways.apply(dominant_pathway, axis=1)
    pathways["pathway_screening_rule"] = "heat>=0.5; flood>=0.15; water>=0.5"
    pathways.to_csv(TABLE_DIR / "cee_hazard_pathways_by_facility_group.csv", index=False)

    city_priority = (
        t1.groupby("city", as_index=False)
        .agg(
            n_facilities=("n_facilities", "sum"),
            heat_share=("heat_share", "mean"),
            flood_share=("flood_share", "mean"),
            water_stress_share=("water_stress_share", "mean"),
            compound_share=("compound_share", "mean"),
            median_service_pop_5km=("median_service_pop_5km", "median"),
            mean_escri=("mean_escri", "mean"),
            median_grdi=("median_grdi", "median"),
        )
    )
    exposure_cut = city_priority["compound_share"].median()
    service_cut = city_priority["median_service_pop_5km"].median()

    def quadrant(row: pd.Series) -> str:
        high_exp = row["compound_share"] >= exposure_cut
        high_service = row["median_service_pop_5km"] >= service_cut
        if high_exp and high_service:
            return "high_exposure_high_service_pressure"
        if high_exp:
            return "high_exposure_lower_service_pressure"
        if high_service:
            return "lower_exposure_high_service_pressure"
        return "lower_exposure_lower_service_pressure"

    city_priority["priority_quadrant"] = city_priority.apply(quadrant, axis=1)
    city_priority["compound_median_cut"] = exposure_cut
    city_priority["service_pop_5km_median_cut"] = service_cut
    city_priority.sort_values(["mean_escri", "compound_share"], ascending=False).to_csv(
        TABLE_DIR / "cee_city_priority_quadrants.csv", index=False
    )

    cf = t5.copy()
    cf["counterfactual_class"] = cf.apply(counterfactual_class, axis=1)
    cf["above_p95"] = cf["observed_compound_share"] > cf["counterfactual_compound_p95"]
    cf["below_p05"] = cf["observed_compound_share"] < cf["counterfactual_compound_p05"]
    cf.sort_values("observed_minus_counterfactual", ascending=False).to_csv(
        TABLE_DIR / "cee_counterfactual_classification.csv", index=False
    )

    sens = (
        t6.groupby(["subset", "heat_days_threshold", "water_cat_threshold"], as_index=False)
        .agg(
            n_groups=("compound_share", "size"),
            mean_compound_share=("compound_share", "mean"),
            median_compound_share=("compound_share", "median"),
            max_compound_share=("compound_share", "max"),
        )
        .sort_values(["subset", "heat_days_threshold", "water_cat_threshold"])
    )
    sens.to_csv(TABLE_DIR / "cee_threshold_scope_sensitivity_summary.csv", index=False)

    compared = t9[t9["status"] == "compared"]
    overlapping_era5 = t10[t10["in_30_city_pilot"] == True]  # noqa: E712
    top_cities = city_priority.sort_values("mean_escri", ascending=False).head(8)
    top_cf = cf.sort_values("observed_minus_counterfactual", ascending=False).head(8)

    report = f"""# CEE Revision Data Summary

Generated by `scripts/28_make_cee_revision_tables.py`.

## Core Numbers

- Total audited facility records: {int(t1['n_facilities'].sum()):,}
- Cities: {t1['city'].nunique()}
- Facility groups: {len(t1)}
- WorldPop compared cities: {len(compared)}
- WorldPop median 2020/2022 ratio: {compared['median_ratio_2020_to_2022'].median():.3f}
- WorldPop median facility-level correlation: {compared['pearson_corr_facility_service_pop'].median():.3f}
- Overlapping NAS ERA5 files readable: {int(overlapping_era5['readable_files'].sum()):,}/{int(overlapping_era5['nas_files_total'].sum()):,}

## Top Cities By Mean ESCRI

{md_table(top_cities[['city', 'n_facilities', 'compound_share', 'median_service_pop_5km', 'mean_escri', 'priority_quadrant']])}

## Top Positive Counterfactual Differences

{md_table(top_cf[['city', 'facility_type', 'observed_compound_share', 'counterfactual_compound_mean', 'counterfactual_compound_p05', 'counterfactual_compound_p95', 'observed_minus_counterfactual', 'counterfactual_class']])}

## Generated Tables

- `manuscript/tables/cee_hazard_pathways_by_facility_group.csv`
- `manuscript/tables/cee_city_priority_quadrants.csv`
- `manuscript/tables/cee_counterfactual_classification.csv`
- `manuscript/tables/cee_threshold_scope_sensitivity_summary.csv`
"""
    (REPORT_DIR / "CEE_revision_data_summary.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
