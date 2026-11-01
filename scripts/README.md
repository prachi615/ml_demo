# `scripts/`

Command-line entry points, used by CI, by the Makefile and by people running the pipelines by hand.

## `compile_pipelines.py`

Compiles the two pipelines from Python into YAML templates that Vertex AI can run. It can also upload them to Artifact Registry.

```bash
PIPELINE_IMAGE=us-central1-docker.pkg.dev/my-proj/storm-ml/storm-pipeline:sha-abc python scripts/compile_pipelines.py --output-dir build
```

| Option | Meaning |
|---|---|
| `--output-dir` | where to write `*_pipeline.yaml` (default `build`) |
| `--registry-host` | Artifact Registry KFP repo URL; if given, templates are uploaded |
| `--tag` | extra tag(s) for the uploaded templates (repeatable) |

`PIPELINE_IMAGE` decides which container image every step runs in, so set it before compiling.

## `run_pipeline.py`

Submits one pipeline run, or creates a recurring schedule. The parameters come from `configs/<env>.yaml`.

```bash
python scripts/run_pipeline.py --pipeline training --config configs/dev.yaml
```

```bash
python scripts/run_pipeline.py --pipeline training --config configs/prod.yaml --schedule "0 4 1 * *"
```

```bash
python scripts/run_pipeline.py --pipeline batch_inference --config configs/prod.yaml --schedule "0 6 * * 1"
```

| Option | Meaning |
|---|---|
| `--pipeline` | `training` or `batch_inference` |
| `--config` | environment config file |
| `--template` | compiled YAML or Artifact Registry URL (default: `build/<pipeline>.yaml`) |
| `--schedule` | cron expression; replaces any existing schedule with the same name |
| `--no-cache` | disable Vertex AI step caching |
| `--sync` | wait for the run to finish |

Both scripts need `gcloud auth application-default login` (or a service account) and `GCP_PROJECT_ID` exported.
