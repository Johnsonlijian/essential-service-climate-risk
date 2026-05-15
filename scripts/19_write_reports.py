from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def md_table(df: pd.DataFrame, cols: list[str], float_cols: list[str] | None = None, n: int | None = None) -> str:
    float_cols = float_cols or []
    view = df[cols].copy()
    if n is not None:
        view = view.head(n)
    for col in float_cols:
        if col in view.columns:
            view[col] = view[col].astype(float).map(lambda x: f"{x:.3f}")
    return view.to_markdown(index=False)


def write(path: str, text: str) -> None:
    out = ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def main() -> None:
    table1 = pd.read_csv(ROOT / "manuscript/tables/table1_pilot_facility_exposure.csv")
    ranking = pd.read_csv(ROOT / "manuscript/tables/table3_pilot_city_ranking.csv")
    counter = pd.read_csv(ROOT / "manuscript/tables/table5_population_weighted_counterfactual.csv")
    scope = pd.read_csv(ROOT / "manuscript/tables/table7_facility_scope_audit.csv")
    osm = pd.read_csv(ROOT / "manuscript/tables/table8_osm_overture_facility_audit.csv")
    sens = pd.read_csv(ROOT / "manuscript/tables/table6_threshold_and_scope_sensitivity.csv")

    facility_total = int(ranking["n_facilities"].sum())
    city_count = ranking.shape[0]
    city_names = ranking["city"].tolist()
    city_list = ", ".join(city_names[:-1]) + f" and {city_names[-1]}" if len(city_names) > 1 else city_names[0]
    city_span_text = (
        f"{city_count} cities across Asia, Africa, Europe, North America and South America"
        if city_count >= 12
        else f"{city_count} cities spanning {city_list}"
    )
    top3 = ranking.head(3)["city"].tolist()
    top3_text = ", ".join(top3[:-1]) + f" and {top3[-1]}" if len(top3) > 1 else top3[0]
    top = ranking.iloc[0]
    highest_counter = counter.sort_values("observed_minus_counterfactual", ascending=False).head(4)
    positive_counter = highest_counter[highest_counter["observed_minus_counterfactual"].astype(float) > 0]
    positive_counter_text = (
        ", ".join(
            f"{row.city} {row.facility_type}"
            for row in positive_counter.itertuples(index=False)
        )
        if not positive_counter.empty
        else "no pilot group"
    )

    empirical = f"""# Empirical Pilot Results

Run date: {date.today().isoformat()}

## Scope

This pilot processed **{city_count} cities** and **{facility_total:,} Overture-derived facility records**. The analysis covers schools and health-related facilities in {city_list}.

Processed layers:

- Facilities: Overture Maps Places `2026-04-15.0`
- OSM validation smoke test: Overpass API for selected cities
- Population: WorldPop 2022 constrained 1 km population
- Heat: Open-Meteo Archive 2025 daily maximum temperature, using facilities with at least 20 days above 35C as the main pilot heat threshold
- Flood: GloFAS/JRC `FloodHazard100y` WMS, sampled as binary flood-zone exposure
- Water stress: WRI Aqueduct 4.0 Baseline Annual FeatureServer
- Deprivation: Global Gridded Relative Deprivation Index

## City-Level Ranking

{md_table(ranking, ["city", "n_facilities", "heat_share", "flood_share", "water_stress_share", "compound_share", "mean_escri"], ["heat_share", "flood_share", "water_stress_share", "compound_share", "mean_escri"])}

The highest mean ESCRI city in this pilot is **{top["city"]}**. The top three cities by mean ESCRI are **{top3_text}**. These rankings reflect the selected pilot thresholds and should be interpreted as screening priorities rather than definitive local engineering risk estimates.

## Facility-Type Exposure

{md_table(table1, ["city", "facility_type", "n_facilities", "heat_share", "flood_share", "water_stress_share", "compound_share"], ["heat_share", "flood_share", "water_stress_share", "compound_share"])}

## Population-Weighted Counterfactual

{md_table(counter, ["city", "facility_type", "observed_compound_share", "counterfactual_compound_mean", "observed_minus_counterfactual"], ["observed_compound_share", "counterfactual_compound_mean", "observed_minus_counterfactual"])}

Largest positive observed-minus-counterfactual differences:

{md_table(highest_counter, ["city", "facility_type", "observed_compound_share", "counterfactual_compound_mean", "observed_minus_counterfactual"], ["observed_compound_share", "counterfactual_compound_mean", "observed_minus_counterfactual"])}

Interpretation: {positive_counter_text} show the largest positive observed-minus-counterfactual differences in this run. Cities with area-wide compound exposure, such as Dhaka under the current thresholds, may show little difference from a population-weighted random baseline because the hazard combination is already widespread.

## Figures and Tables

Main pilot figures:

- `manuscript/figures/figure1_pilot_facility_risk_maps.png`
- `manuscript/figures/figure2_pilot_exposure_bars.png`
- `manuscript/figures/figure3_pilot_grdi_inequality.png`
- `manuscript/figures/figure4_pilot_counterfactual.png`
- `manuscript/figures/figure5_pilot_priority_space.png`

Main pilot tables:

- `manuscript/tables/table1_pilot_facility_exposure.csv`
- `manuscript/tables/table2_pilot_grdi_inequality.csv`
- `manuscript/tables/table3_pilot_city_ranking.csv`
- `manuscript/tables/table5_population_weighted_counterfactual.csv`
- `manuscript/tables/table6_threshold_and_scope_sensitivity.csv`
- `manuscript/tables/table7_facility_scope_audit.csv`
- `manuscript/tables/table8_osm_overture_facility_audit.csv`

## Status

These results are now a credible multi-city pilot, not yet a global paper. They justify scaling because the pipeline produces interpretable differences across cities and because at least some cities show facility-location differences beyond a population-weighted baseline.
"""
    write("10_EMPIRICAL_PILOT_RESULTS.md", empirical)

    baseline_sens = sens[
        (sens["heat_days_threshold"] == 20)
        & (sens["water_cat_threshold"] == 3)
        & (sens["subset"].isin(["all_classes", "school_all_health_core", "source_confidence_ge_0p6"]))
    ]
    robustness = f"""# Robustness and Data Quality

## Facility Scope Audit

The facility registry is intentionally split into core, auxiliary and broad classes. Schools are retained broadly, while health facilities should be reported in both broad and core-only forms because pharmacies, dentists and laboratories may inflate the health denominator.

{md_table(scope, ["city", "facility_type", "facility_scope", "n", "median_confidence"], ["median_confidence"], n=40)}

## OSM vs Overture Audit

OSM is used as validation/fallback rather than the main denominator. Overture generally returns more facilities than OSM core tags in the smoke-test cities, which supports using Overture as the primary scalable source while auditing coverage.

{md_table(osm, ["city", "overture_school", "overture_health", "osm_school_core", "osm_health_core", "osm_to_overture_school_ratio", "osm_to_overture_health_ratio"], ["osm_to_overture_school_ratio", "osm_to_overture_health_ratio"])}

## Threshold and Scope Sensitivity

The full sensitivity table is in `manuscript/tables/table6_threshold_and_scope_sensitivity.csv`. The baseline sensitivity slice below compares all classes, health-core-only and high-confidence subsets under the main pilot thresholds.

{md_table(baseline_sens, ["city", "facility_type", "subset", "n_facilities", "heat_share", "flood_share", "water_share", "compound_share"], ["heat_share", "flood_share", "water_share", "compound_share"], n=60)}

## Key Reviewer Risks

1. **Facility category noise**: answer with core-only health sensitivity and source-confidence subsets.
2. **Threshold dependence**: answer with heat-day and water-category threshold sensitivity.
3. **Population-density confounding**: answer with population-weighted counterfactual siting.
4. **Flood WMS limitation**: final paper should replace WMS-derived binary exposure with original hazard raster depth products where possible.
5. **Pilot-to-global overclaiming**: label all current findings as a {city_count}-city pilot until global or high-confidence multinational expansion is complete.
"""
    write("11_ROBUSTNESS_AND_DATA_QUALITY.md", robustness)

    strategy = f"""# Global Expansion and Submission Strategy

## Current Readiness

The project has crossed the feasibility threshold. It has a working environment, repeatable scripts, {city_count} processed pilot cities, facility/population/hazard/vulnerability overlays, counterfactual tests, figures and manuscript text.

## What Makes the Full Paper Strong

The full paper should not simply add more cities. It should scale three ideas:

1. essential-service facilities are a distinct exposure layer;
2. compound exposure changes adaptation priorities;
3. facility exposure can differ from population-weighted expectations.

## Required Before Nature-Family Submission

1. Expand to at least 40-100 cities or a high-confidence global facility subset.
2. Replace or validate the WMS flood proxy with original JRC/GloFAS hazard rasters or another documented flood product.
3. Add ERA5-HEAT UTCI or a comparable human heat-stress metric after CDS credentials are configured.
4. Run core-only, high-confidence and threshold sensitivity for all included cities.
5. Decide whether the manuscript is global, Global South-focused, or megacity-focused.

## Best Journal Fit

- Nature Communications: if the analysis becomes global or broadly multinational.
- Science Advances: if the paper emphasizes the general framework and cross-regional inequality.
- Nature Cities: if the city-level adaptation and planning story becomes strongest.
- Nature Water: if flood and water stress are foregrounded.
- Communications Earth & Environment: strong fallback target.

## Recommended Next Iteration

Process an additional tranche of cities from underrepresented regions and hazard regimes, then run a go/no-go review. If the counterfactual and robustness results remain informative, expand globally or to a high-confidence multinational sample.
"""
    write("12_GLOBAL_EXPANSION_AND_SUBMISSION_STRATEGY.md", strategy)

    paper = f"""# Compound climate risks to essential public service facilities worldwide

## Abstract

Schools and health facilities are central to social resilience, yet climate-risk assessments usually count exposed people or assets rather than the public-service infrastructure through which societies educate, care and respond. Here we develop a reproducible open-data framework to quantify compound exposure of essential public-service facilities to heat stress, river flooding and water stress. In a pilot covering {city_span_text}, we combine Overture Maps facility data with WorldPop population, daily heat indicators, GloFAS/JRC 100-year flood hazard, WRI Aqueduct water stress and a gridded deprivation index. The pilot identifies strong contrasts across cities: {top3_text} have the highest mean essential-service climate-risk scores, while counterfactual tests identify {positive_counter_text} as having the largest positive facility exposure differences relative to population-weighted random locations. These results demonstrate the feasibility of facility-level essential-service climate-risk assessment and motivate expansion to a global high-confidence facility subset.

## Introduction

Climate hazards increasingly disrupt the everyday systems that allow societies to function. Extreme heat changes whether children can safely learn, whether patients and staff can remain healthy inside medical buildings, and whether emergency services can operate. Flooding damages buildings, cuts access roads, interrupts electricity and water supply, and can turn schools and health facilities from places of care into places that require rescue. Water stress further weakens hygiene, cooling and service continuity. These risks are usually assessed through population exposure, economic assets or built-up area, but essential public-service facilities form a distinct layer of climate resilience.

Schools and health facilities deserve direct assessment because they provide services beyond their building footprints, they are often expected to function during disasters, and their locations reflect planning decisions and public investment histories. A facility-level perspective therefore asks whether societies can maintain education, health care and emergency support as climate hazards intensify.

Recent open geospatial data now make this assessment possible. Overture Maps provides cloud-native data on places and buildings. WorldPop and GHSL describe population and settlement patterns. Heat, flood and water-risk products from Open-Meteo/ERA-style data, GloFAS/JRC and WRI Aqueduct can be joined to facilities, while deprivation data add equity context. This pilot tests whether these layers can produce defensible facility-level findings before scaling to a global analysis.

## Results

### Multi-city facility registry

The {city_count}-city pilot extracted **{facility_total:,}** Overture-derived facility records. Facility-scope audits show that health facilities include both core facilities and auxiliary facilities, so all health results should be reported with core-only sensitivity.

### Facility exposure

{md_table(table1, ["city", "facility_type", "n_facilities", "heat_share", "flood_share", "water_stress_share", "compound_share"], ["heat_share", "flood_share", "water_stress_share", "compound_share"])}

{top3_text} show the highest mean ESCRI under the main pilot thresholds. City-specific patterns differ: some cities are dominated by heat-water compound exposure, some by flood-water combinations, and others by heat-flood exposure. Low compound exposure in this pilot should be interpreted as threshold-specific rather than as proof of low local vulnerability.

### Service-catchment exposure

Service-population estimates were computed from WorldPop 2022 within 1 km, 5 km and 10 km circular catchments. These are facility-level exposure-intensity measures rather than unique served-population counts because catchments overlap. Dense cities such as Dhaka and Kolkata have very large median 5 km service-population values, highlighting the scale of potential service disruption.

### Counterfactual facility siting

{md_table(counter, ["city", "facility_type", "observed_compound_share", "counterfactual_compound_mean", "observed_minus_counterfactual"], ["observed_compound_share", "counterfactual_compound_mean", "observed_minus_counterfactual"])}

The largest positive observed-minus-counterfactual differences occur for {positive_counter_text}. Area-wide compound-risk cities can show little observed-minus-random difference because the selected hazard combination is widespread across the population surface.

## Discussion

The pilot supports the paper's core premise: essential public-service facilities can be treated as a distinct climate-risk layer. The analysis also shows why a global paper must include data-quality auditing and counterfactual tests. Some cities are dominated by area-wide hazards, while others show facility-type or siting differences. These differences matter for adaptation planning because schools and health facilities anchor education, care and emergency response.

The pilot should not be overclaimed. The heat metric is based on a fixed 35C threshold, which may understate locally meaningful heat stress in cooler climates and miss humidity. The flood layer is currently sampled from a WMS map and should be replaced or validated with original hazard rasters for final publication. Overture facility categories include auxiliary health-related services, requiring core-only sensitivity. Despite these limitations, the workflow is now reproducible and ready for multinational scaling.

## Methods

Facilities were extracted from Overture Maps Places `2026-04-15.0` using category filters for schools and health-related facilities. Service-population intensity was computed from WorldPop 2022 constrained 1 km population within 1 km, 5 km and 10 km circular buffers. Heat exposure was measured as at least 20 days in 2025 with daily maximum temperature above 35C from Open-Meteo Archive grids. Flood exposure was sampled from the GloFAS/JRC `FloodHazard100y` WMS layer as a binary facility-level indicator. Water stress was assigned from WRI Aqueduct 4.0 Baseline Annual polygons. Deprivation was sampled from the Global Gridded Relative Deprivation Index. Compound exposure was defined as exposure to at least two of heat, flood and water stress. Counterfactual tests sampled population-weighted random locations from WorldPop cells within each city bbox and compared simulated compound exposure with observed facility exposure.
"""
    write("manuscript/paper.md", paper)


if __name__ == "__main__":
    main()
