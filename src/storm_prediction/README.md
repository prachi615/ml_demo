# `src/storm_prediction/`

The Python package. It is installed into the pipeline container image, and every pipeline step imports from it.

| Path | Contents |
|---|---|
| `sql/` | BigQuery SQL: raw ingestion and feature engineering |
| `components/` | Kubeflow Pipelines components, one per pipeline step |
| `pipelines/` | the training, evaluation and batch inference pipeline definitions |
| `models.py` | the four models (2 linear baselines, random forest, XGBoost) and their shared preprocessing |
| `features.py` | the list of feature columns and target, shared by training and inference |
| `metrics.py` | RMSE, MAE, R², WAPE, Poisson deviance, bias |
| `data.py` | time-based train/val/test split and risk-level labels |
| `gcp.py` | small BigQuery and Cloud Storage helpers |
| `config.py` | component image name and config file loading |

## Design

- **Components are thin.** The ML logic lives in the plain modules (`models.py`, `features.py`, `metrics.py`, `data.py`). Those modules need no GCP access and are unit tested in `tests/`.
- **Imports inside components.** KFP sends only each component function's source code to the container, so a component's imports are written inside the function.
- **One feature list.** `features.py` defines the columns that `sql/build_features.sql` creates. Training and batch inference both read that list, so they can't drift apart.
