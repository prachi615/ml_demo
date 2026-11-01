# `src/storm_prediction/pipelines/`

The two pipeline definitions. Each is a `@dsl.pipeline` function that wires components together into a graph of steps.
`__init__.py` exposes them as `PIPELINES` for the scripts and tests.

## `training_pipeline.py` (`storm-forecast-training`): monthly

Runs on the 1st of every month (04:00 UTC) and once on every deploy. It has two stages, and evaluation only starts after training has finished.

**Training stage**
```
ingest → build features → time split ─┬─ train linear_regression → score on test ─┐
                                      ├─ train ridge             → score on test ─┤
                                      ├─ train random_forest     → score on test ─┼→ best of 4 → baseline gate ─→ write leaderboard (BigQuery)
                                      └─ train xgboost           → score on test ─┘
```
The four branches run in parallel. The best of the four is the model with the lowest validation RMSE.
**Baseline gate:** its test RMSE must beat the Linear Regression baseline by more than 2% (`min_improvement_over_baseline_pct`). If Linear Regression itself wins, the gate fails.

**Evaluation stage**
```
best of 4 ─→ evaluate current champion on the same test weeks ─→ decide promotion ─→ [gate passed AND beats champion?] register as champion
                                                                       └─→ model_evaluations (BigQuery)
```
The new model replaces the champion only if it passed the baseline gate **and** its test RMSE is lower than the champion's, by at least `min_improvement_over_champion_pct` (default 0).
On the very first run there is no champion, so passing the baseline gate is enough. A model that fails the gate is never promoted.

## `batch_inference_pipeline.py` (`storm-forecast-batch-inference`)

```
ingest → build features → forecast rows ─┬→ batch predict → BigQuery (storm_predictions)
load champion ───────────────────────────┘
```
Runs weekly (Mondays 06:00 UTC) with whichever model is champion at that moment.

## Notes

- Data steps have caching turned off, because BigQuery content changes between runs.
- Run IDs come from `dsl.PIPELINE_JOB_NAME_PLACEHOLDER` and are stored with every BigQuery row for lineage.
- Compile with `scripts/compile_pipelines.py`; run with `scripts/run_pipeline.py`.
