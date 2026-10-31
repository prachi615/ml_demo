# `src/storm_prediction/pipelines/`

The three pipeline definitions. Each is a `@dsl.pipeline` function that wires components together into a graph of steps.
`__init__.py` exposes them as `PIPELINES` for the scripts and tests.

## `training_pipeline.py` (`storm-forecast-training`)

```
ingest → build features → time split ─┬─ train linear_regression → evaluate ─┐
                                      ├─ train ridge             → evaluate ─┤
                                      ├─ train random_forest     → evaluate ─┼→ select best ─┬→ write leaderboard (BigQuery)
                                      └─ train xgboost           → evaluate ─┘               └→ [beats baselines?] register model
```
The four training branches run in parallel. A model is registered only if a challenger beats the best linear baseline by `min_improvement_pct`.

## `evaluation_pipeline.py` (`storm-forecast-evaluation`)

```
ingest → build features → recent weeks ─┬→ evaluate champion ─┬→ health check (BigQuery) → [flagged and auto_retrain?] trigger training
load champion ──────────────────────────┘  naive baselines ───┘
```
Runs weekly. The health check flags the champion if its RMSE degrades by more than `max_rmse_degradation_pct` against training, or if it is no better than a naive forecast.

## `batch_inference_pipeline.py` (`storm-forecast-batch-inference`)

```
ingest → build features → forecast rows ─┬→ batch predict → BigQuery (storm_predictions)
load champion ───────────────────────────┘
```
Runs weekly, after the evaluation pipeline.

## Notes

- Data steps have caching turned off, because BigQuery content changes between runs.
- Run IDs come from `dsl.PIPELINE_JOB_NAME_PLACEHOLDER` and are stored with every BigQuery row for lineage.
- Compile with `scripts/compile_pipelines.py`; run with `scripts/run_pipeline.py`.
