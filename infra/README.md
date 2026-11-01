# `infra/`

One-time setup of the Google Cloud resources the project needs.

## `setup_gcp.sh`

Run it once per environment, each with its own GCP project:

```bash
PROJECT_ID=my-storm-dev GITHUB_REPO=prachi615/ml_demo ./infra/setup_gcp.sh
```

| Resource | Name | Why |
|---|---|---|
| APIs | Vertex AI, BigQuery, Storage, Artifact Registry, IAM | services the pipelines use |
| GCS bucket | `gs://<project>-storm-ml` | pipeline artifacts and staging files |
| BigQuery dataset | `storm_ml` (location `US`) | raw, feature, leaderboard, evaluation and prediction tables |
| Artifact Registry (Docker) | `storm-ml` | the component container image |
| Artifact Registry (KFP) | `storm-pipelines` | compiled pipeline templates |
| Service account | `vertex-pipelines` | identity the pipelines run as |
| Service account | `github-deployer` | identity GitHub Actions deploys as |
| Workload Identity Federation | pool `github`, provider `github-oidc` | keyless login from GitHub Actions, limited to `GITHUB_REPO` |

At the end, the script prints the values to put in the matching GitHub environment.
The script is safe to re-run.

Requires the `gcloud` and `bq` CLIs, logged in as a project owner (or with equivalent rights).
