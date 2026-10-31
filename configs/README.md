# `configs/`

One YAML file per environment. `scripts/run_pipeline.py` reads them and turns them into pipeline parameters.

| File | Used for | Differences |
|---|---|---|
| `dev.yaml` | the `dev` branch | loads 2010+ data only (faster, cheaper), smaller models, no auto-retraining |
| `rc.yaml` | release candidates | same as prod, but auto-retraining off |
| `prod.yaml` | `main` / production | full history (2000+), auto-retraining on |

## Placeholders

`${GCP_PROJECT_ID}` is replaced with the environment variable of the same name when the file is loaded. CI sets it from the GitHub environment.
Loading fails with a clear error if a placeholder can't be resolved.

## Sections

| Key | Meaning |
|---|---|
| `project_id`, `region`, `pipeline_root`, `service_account` | where and as whom pipelines run |
| `bigquery.*` | dataset (must be in `US`) and table names |
| `data.*` | years to load, validation/test split sizes |
| `model.*` | Model Registry name, optional serving image |
| `training.*` | registration gate (`min_improvement_pct`) and per-model hyperparameters |
| `evaluation.*` | monitoring window, allowed RMSE degradation, auto-retraining switch |
| `schedules.*` | cron expressions (UTC) for the evaluation and batch inference schedules |

To change a model's hyperparameters, edit `training.hyperparameters.<model>`. No code change is needed.
