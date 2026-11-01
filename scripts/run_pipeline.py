"""Submit a compiled pipeline to Vertex AI Pipelines, or (re)create its schedule.

Pipeline parameters are built from an environment config (``configs/<env>.yaml``),
so the same compiled template runs in dev, rc and prod.

Examples:

    python scripts/run_pipeline.py --pipeline training --config configs/dev.yaml
    python scripts/run_pipeline.py --pipeline training --config configs/prod.yaml --schedule "0 4 1 * *"
    python scripts/run_pipeline.py --pipeline batch_inference --config configs/prod.yaml \
        --template https://us-central1-kfp.pkg.dev/<project>/storm-pipelines/<template>/latest \
        --schedule "0 6 * * 1"
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from google.cloud import aiplatform

from storm_prediction.config import load_config

# Display name prefix of each pipeline's runs and schedules.
DISPLAY_NAMES = {
    "training": "storm-forecast-training",
    "batch_inference": "storm-forecast-batch-inference",
}
# Local compiled templates used when --template is not given (see `make compile`).
DEFAULT_TEMPLATES = {
    "training": "build/training_pipeline.yaml",
    "batch_inference": "build/batch_inference_pipeline.yaml",
}


def common_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the parameters shared by both pipelines from a config.

    Args:
        cfg: Parsed environment config (``configs/<env>.yaml``).

    Returns:
        Pipeline parameter name -> value.
    """
    bq, data = cfg["bigquery"], cfg["data"]
    return {
        "project": cfg["project_id"],
        "location": cfg["region"],
        "bq_location": bq["location"],
        "bq_dataset": bq["dataset"],
        "raw_table": bq["raw_table"],
        "features_table": bq["features_table"],
        "start_year": data["start_year"],
        "end_year": data["end_year"],
        "refresh_raw_data": data["refresh_raw_data"],
        "model_display_name": cfg["model"]["display_name"],
    }


def training_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the training pipeline parameters from a config.

    Args:
        cfg: Parsed environment config.

    Returns:
        Pipeline parameter name -> value. Hyperparameters are passed as one JSON string.
    """
    tr = cfg["training"]
    params = {
        **common_params(cfg),
        "leaderboard_table": cfg["bigquery"]["leaderboard_table"],
        "evaluation_table": cfg["bigquery"]["evaluation_table"],
        "val_weeks": cfg["data"]["val_weeks"],
        "test_weeks": cfg["data"]["test_weeks"],
        "min_improvement_over_baseline_pct": tr["min_improvement_over_baseline_pct"],
        "min_improvement_over_champion_pct": tr["min_improvement_over_champion_pct"],
        "hyperparameters_json": json.dumps(tr["hyperparameters"]),
    }
    if cfg["model"].get("serving_container_image_uri"):
        params["serving_container_image_uri"] = cfg["model"]["serving_container_image_uri"]
    return params


def build_params(pipeline: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the parameter values for the chosen pipeline.

    Args:
        pipeline: ``training`` or ``batch_inference``.
        cfg: Parsed environment config.

    Returns:
        Pipeline parameter name -> value.

    Raises:
        ValueError: For an unknown pipeline name.
    """
    if pipeline == "training":
        return training_params(cfg)
    if pipeline == "batch_inference":
        return {**common_params(cfg), "predictions_table": cfg["bigquery"]["predictions_table"]}
    raise ValueError(pipeline)


def main() -> None:
    """Parse the command line, then submit one run or (re)create a schedule.

    With ``--schedule``, any existing schedule with the same display name is
    deleted first, so redeploying never creates duplicate schedules.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True, choices=sorted(DISPLAY_NAMES))
    parser.add_argument("--config", required=True)
    parser.add_argument("--template", help="Local YAML or Artifact Registry template URL")
    parser.add_argument(
        "--schedule", help="Cron expression; creates/replaces a schedule instead of a one-off run"
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--sync", action="store_true", help="Wait for the run to finish")
    args = parser.parse_args()

    # Expands ${GCP_PROJECT_ID} etc. from the environment.
    cfg = load_config(args.config)
    aiplatform.init(
        project=cfg["project_id"], location=cfg["region"], staging_bucket=cfg.get("staging_bucket")
    )

    display_name = DISPLAY_NAMES[args.pipeline]
    job = aiplatform.PipelineJob(
        display_name=f"{display_name}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}",
        template_path=args.template or DEFAULT_TEMPLATES[args.pipeline],
        pipeline_root=cfg["pipeline_root"],
        parameter_values=build_params(args.pipeline, cfg),
        enable_caching=not args.no_cache,
        labels={"env": cfg["env"], "use_case": "storm-forecast"},
    )
    service_account = cfg.get("service_account") or None

    if args.schedule:
        # Replace, don't duplicate: delete the old schedule before creating the new one.
        schedule_name = f"{display_name}-schedule"
        for existing in aiplatform.PipelineJobSchedule.list(filter=f'display_name="{schedule_name}"'):
            print(f"deleting old schedule {existing.resource_name}")
            existing.delete()
        schedule = job.create_schedule(
            display_name=schedule_name,
            cron=f"TZ={cfg.get('timezone', 'UTC')} {args.schedule}",
            max_concurrent_run_count=1,
            service_account=service_account,
        )
        print(f"created schedule {schedule.resource_name} ({args.schedule})")
        return

    job.submit(service_account=service_account)
    print(f"submitted {job.resource_name}")
    if args.sync:
        job.wait()


if __name__ == "__main__":
    main()
