"""Runtime configuration shared by components, pipelines and scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Container image every pipeline component runs in. It is resolved at
# *compile* time, so CI sets PIPELINE_IMAGE to the freshly pushed image tag
# before compiling the pipelines.
PIPELINE_IMAGE: str = os.environ.get(
    "PIPELINE_IMAGE",
    "us-central1-docker.pkg.dev/your-gcp-project/storm-ml/storm-pipeline:latest",
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load an environment YAML config (configs/dev.yaml, configs/prod.yaml)."""
    with open(path, encoding="utf-8") as fh:
        text = os.path.expandvars(fh.read())  # supports ${GCP_PROJECT_ID} etc.
    cfg = yaml.safe_load(text)
    unresolved = [k for k, v in cfg.items() if isinstance(v, str) and "${" in v]
    if unresolved:
        raise ValueError(f"Unresolved environment variables in {path}: {unresolved}")
    return cfg
