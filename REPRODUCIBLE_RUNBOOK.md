# REPRODUCIBLE_RUNBOOK

## Environment

Use Python 3.11+ with geospatial packages listed in `ai_autoboost/code/requirements.txt`.

## Non-destructive audit and derived-output regeneration

From the private project root:

```powershell
python scripts\00_check_environment.py
python scripts\16_run_all_pilots.py --only-missing
python ai_autoboost\code\run_all_experiments.py
```

Round 5 counterfactual outputs can be regenerated directly:

```powershell
python ai_autoboost\code\round5_literature_counterfactual\round5_counterfactual_replicates.py --reps 5000
python ai_autoboost\code\round5_literature_counterfactual\round5_summarize_counterfactual.py
```

## Expected key outputs

- 621,781 processed facility records in the private processed table.
- 300,000 Round 5 counterfactual replicate rows.
- 60 city-facility counterfactual comparisons.
- 27 Holm-corrected empirical departures from the population-weighted random-location baseline.

## Public release boundary

Do not commit raw third-party datasets. Instead, provide source URLs, scripts, derived non-sensitive tables and generated figures.
