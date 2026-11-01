# `src/storm_prediction/components/`

Kubeflow Pipelines (KFP) components. Each function decorated with `@dsl.component` becomes one step that runs in its own container on Vertex AI.
Steps pass data to each other as **artifacts** (files in GCS, such as parquet datasets and model directories) and **parameters** (strings and numbers).

## `data.py`: getting data out of BigQuery

| Component | Does |
|---|---|
| `ingest_raw_storm_events` | runs `sql/ingest_raw.sql` (public NOAA data → our raw table) |
| `build_feature_table` | runs `sql/build_features.sql` (raw → weekly features) |
| `extract_training_splits` | labelled rows → train / val / test parquet, split by time |
| `extract_scoring_rows` | forecast rows (next week, one per state), for batch inference |

## `training.py`: model building

| Component | Does |
|---|---|
| `train_model` | trains one model type and returns its validation metrics |
| `evaluate_model` | scores one trained model on the test split |
| `select_best_model` | picks the best of the four on validation RMSE and checks it beats Linear Regression on test RMSE by more than 2% |
| `write_leaderboard` | appends the run's leaderboard to BigQuery |
| `register_model` | uploads the promoted model to the Vertex AI Model Registry and moves the `champion` alias to it |

## `evaluation.py`: new model vs champion (runs after training)

| Component | Does |
|---|---|
| `evaluate_champion` | loads the current `champion` and scores it on the same test split as the new model; reports `none` on the first run |
| `decide_promotion` | promotes the new model only if it passed the baseline gate and beats the champion; logs the decision to BigQuery and to a Markdown report |

## `inference.py`: predictions

| Component | Does |
|---|---|
| `load_registered_model` | downloads `<model>@champion` from the registry |
| `batch_predict` | forecasts next week per state and appends the results to BigQuery with a risk level |

## Conventions

- Every component uses the project image (`config.PIPELINE_IMAGE`) with `install_kfp_package=False`, because KFP is already in the image.
- Imports go **inside** the function body, because KFP sends only the function's source code to the container.
- Docstrings follow Google style. KFP copies the `Args` descriptions into the compiled pipeline, so they also appear in the Vertex AI console.
