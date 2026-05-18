# Method Transparency Notes

This public-safe note summarizes the documented screening workflow for the 30-city essential public-service facility climate-risk pilot. It is provided to make the public reproducibility package easier to audit without redistributing raw third-party datasets or the active submission manuscript.

## Facility inventory scope

Facility records are derived from Overture Maps Places categories configured in the private workflow.

| Group | Included category terms | Interpretation |
|---|---|---|
| Education-service records | school, primary school, secondary school, elementary school, middle school, high school, kindergarten, preschool, university, college | Education-service facility records, not a verified census of all schools. |
| Health-service records | hospital, clinic, doctor, dentist, pharmacy, medical center, healthcare, nursing home, urgent care | Health-service facility records; broader than hospital-only facilities. |

The workflow checks duplicate facility IDs and coordinate validity, but ID-level checks do not guarantee full semantic or spatial de-duplication across source providers.

## Spatial support

The pilot uses reproducible longitude-latitude city windows listed in `configs/pilot_areas.csv`. These windows are not official administrative boundaries or built-up-area masks. Counterfactual results should therefore be interpreted as departures from a population-weighted within-city-window baseline.

## ESCRI screening index

The essential-service climate-risk index (ESCRI) is a transparent screening index:

```text
ESCRI_i = H_i * (0.5 + 0.5 S_i) * (0.5 + 0.5 V_i) * C_i
H_i = (H_heat_i + H_flood_i + H_water_i) / 3
```

Where:

| Term | Definition |
|---|---|
| `H_heat_i` | Percentile-clipped min-max score of hot-day counts. |
| `H_flood_i` | Main-threshold flood-exposure flag. |
| `H_water_i` | Percentile-clipped min-max score of Aqueduct water-stress score, falling back to category where needed. |
| `S_i` | Percentile-clipped min-max score of `log(1 + P_i)`, where `P_i` is 5 km WorldPop service-population intensity. |
| `V_i` | Percentile-clipped min-max score of gridded deprivation. |
| `C_i` | Overture/source-confidence term clipped to 0-1. |

Equal hazard weighting is deliberately used for auditability. The index is not an optimized physical-damage model and should not be interpreted as a facility failure probability.

## Counterfactual interpretation

The counterfactual analysis samples population-weighted WorldPop cells within each city window. Observed compound-exposure share is compared with replicate-level random-location shares using empirical plus-one corrected P values and Holm correction across 60 city-facility comparisons.

Practical magnitude is interpreted separately from statistical departure:

| Absolute difference | Interpretation |
|---:|---|
| < 0.02 | Small |
| 0.02 to < 0.05 | Moderate |
| >= 0.05 | Large |

The test is not a causal siting analysis.

## Multi-scale layer caution

| Layer | Spatial support | Facility interpretation | Limitation |
|---|---|---|---|
| Heat | Gridded weather layer | Local heat screen | Not a building thermal-condition model. |
| Flood | JRC/CEMS-GloFAS RP100 screening layer | Fluvial flood screen near facility locations | Does not fully represent pluvial or coastal flooding. |
| Water stress | Aqueduct basin-scale layer | Background basin water-stress context | Not a facility-level water-service interruption probability. |
| Service population | WorldPop buffers | Population intensity around facility records | Not a unique served-population catchment. |
| Deprivation | Gridded deprivation layer | Social vulnerability context | Not household-level vulnerability. |
| Source confidence | Overture/source metadata | Data reliability modifier | Does not validate all POI categories manually. |
