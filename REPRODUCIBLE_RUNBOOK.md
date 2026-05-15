# REPRODUCIBLE_RUNBOOK

## Environment

Use Python 3.11+ with common geospatial/scientific packages such as `pandas`, `numpy`, `geopandas`, `rasterio`, `shapely`, `matplotlib`, `requests` and `scipy`.

## Non-destructive audit and derived-output regeneration

From a full private/local project checkout with raw source data available:

```powershell
python scripts\00_check_environment.py
python scripts\16_run_all_pilots.py --only-missing
```

Counterfactual outputs can be regenerated directly from the full local project using:

```powershell
python scripts\13_counterfactual_population_weighted.py
```

This public package does not redistribute raw third-party datasets. The `derived_tables/` and `figures/` folders provide non-sensitive derived outputs for verification and reuse.

## Expected key outputs

- 621,781 processed facility records in the private processed table.
- 60 city-facility counterfactual comparisons.
- 27 Holm-corrected empirical departures from the population-weighted random-location baseline.

## Public release boundary

Do not commit raw third-party datasets. Instead, provide source URLs, scripts, derived non-sensitive tables and generated figures.
