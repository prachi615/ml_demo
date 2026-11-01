# `tests/`

Unit tests. They run locally and in CI **without any GCP access**: all data is synthetic, and BigQuery and Vertex AI are never called.

```bash
make test
```

| File | Checks |
|---|---|
| `conftest.py` | fixture: a synthetic feature table with the same schema as BigQuery |
| `test_models.py` | the four models train and predict non-negative counts; unseen states work; metrics, the > 2% Linear Regression gate and the champion rule are correct |
| `test_data.py` | time splits are chronological and don't overlap; risk labels; SQL templates render completely and contain no same-week leakage |
| `test_pipelines.py` | both pipelines compile; training has four training branches; evaluation runs after training and gates registration; the baseline gate feeds the decision |

## Not covered

Anything that needs real GCP: the SQL running against live BigQuery, Model Registry uploads, and pipeline runs.
Those are exercised by the training run that each dev deploy starts.
