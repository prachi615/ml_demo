# `configs/`

One YAML file per environment. `scripts/run_pipeline.py` reads them and turns them into pipeline parameters.

| File | Used for | Differences |
|---|---|---|
| `dev.yaml` | the `dev` branch | loads 2010+ data only (faster, cheaper), smaller random forest / XGBoost |
| `rc.yaml` | release candidates | identical to prod, so the RC is tested with production settings |
| `prod.yaml` | `main` / production | full history (2000+) |

## Placeholders

`${GCP_PROJECT_ID}` is replaced with the environment variable of the same name when the file is loaded. CI sets it from the GitHub environment.
Loading fails with a clear error if a placeholder can't be resolved.

## Sections

| Key | Meaning |
|---|---|
| `project_id`, `region`, `pipeline_root`, `service_account` | where and as whom pipelines run |
| `bigquery.*` | dataset (must be in `US`) and table names |
| `data.*` | years to load (`end_year: 0` = current year), validation/test split sizes |
| `model.*` | Model Registry name, optional serving image |
| `training.*` | the two promotion margins and per-model hyperparameters: `min_improvement_over_baseline_pct` (beat Linear Regression by more than this, default 2) and `min_improvement_over_champion_pct` (beat the current champion by at least this, default 0) |
| `schedules.*` | cron expressions (UTC): `training` = monthly retraining, `batch_inference` = weekly forecast |

To change a model's hyperparameters, edit `training.hyperparameters.<model>`. No code change is needed.
