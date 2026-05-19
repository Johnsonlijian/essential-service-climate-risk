# Essential-Service Climate-Risk Reproducibility Package

This is a public reproducibility package for the 30-city essential public-service climate-risk pilot.

Important boundary: this repository intentionally excludes raw third-party data, active submission manuscripts, reviewer drafts, private logs, credentials, and files with unclear redistribution rights.

## Contents

- `REPRODUCIBLE_RUNBOOK.md`: commands and expected outputs.
- `DATASETS_AND_LINKS.csv`: source registry for external datasets.
- `CITATION.cff`: citation metadata draft.
- `scripts/`: public-safe analysis and figure-generation scripts copied from the private project.
- `derived_tables/`: non-sensitive derived manuscript tables.
- `figures/`: generated manuscript figures.
- `PUBLIC_RELEASE_BLOCKERS.md`: release-scope notes and exclusions.
- `METHOD_TRANSPARENCY.md`: public-safe explanation of facility taxonomy, ESCRI construction, city-window counterfactual interpretation and multi-scale layer limits.

## Round 12 top-journal increments

This public package now includes public-safe Round 12 outputs used to strengthen the manuscript for a Q1/top-journal route:

- `scripts/round12_top_journal_increment.py`
- `derived_tables/table20_escri_weight_ablation.csv`
- `derived_tables/table21_service_node_typology.csv`
- `derived_tables/table22_humid_heat_reclassification.csv`
- `figures/figure6_escri_ablation_rank_stability.png`
- `figures/figure7_service_node_typology.png`
- `figures/figure8_humid_heat_reclassification.png`

## Round 13 boundary and inventory-validation outputs

The package also includes public-safe Round 13 outputs addressing boundary and facility-inventory reviewer risks:

- `scripts/round13_boundary_inventory_validation.py`
- `derived_tables/table23_boundary_window_sensitivity.csv`
- `derived_tables/table24_facility_inventory_quality.csv`
- `derived_tables/table25_osm_boundary_fetch_status.csv`
- `figures/figure9_boundary_window_sensitivity.png`
- `figures/figure10_inventory_quality_diagnostics.png`

## Round 15 submission-figure polish

The package includes the public-safe figure polish script used for the final submission-facing figures:

- `scripts/round15_submission_figure_polish.py`
- `figures/figure3_pilot_grdi_inequality.png`
- `figures/figure6_escri_ablation_rank_stability.png`
- `figures/figure9_boundary_window_sensitivity.png`
- `figures/figure10_inventory_quality_diagnostics.png`

Round 15 rebuilds Figure 3 as a compact two-panel GRDI dashboard and repositions labels/legends in Figures 6, 9 and 10 to improve main-text readability.

## Intended remote

`https://github.com/Johnsonlijian/essential-service-climate-risk`

Repository creation and push were explicitly approved by the author on 2026-05-15.
