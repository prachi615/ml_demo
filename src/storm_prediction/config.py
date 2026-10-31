"""Runtime configuration shared by components, pipelines and scripts.

Two kinds of configuration live here:

1. ``PIPELINE_IMAGE``: the container image every pipeline component runs in.
   KFP bakes the image name into the compiled pipeline YAML, so it must be known
   at *compile* time. CI sets the ``PIPELINE_IMAGE`` environment variable to the
   freshly pushed image tag before compiling.
2. ``load_config``: reads an environment file such as ``configs/dev.yaml`` and
   expands ``${VAR}`` placeholders from the environment (e.g. ``${GCP_PROJECT_ID}``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Fallback value is only a placeholder for local compilation / unit tests.
PIPELINE_IMAGE: str = os.environ.get(
    "PIPELINE_IMAGE",
    "us-central1-docker.pkg.dev/your-gcp-project/storm-ml/storm-pipeline:latest",
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load an environment YAML config and expand environment variables in it.

    Args:
        path: Path to a config file, e.g. ``configs/dev.yaml``.

    Returns:
        The parsed config as a nested dictionary.

    Raises:
        ValueError: If a top-level value still contains an unresolved ``${...}``
            placeholder, which usually means an environment variable such as
            ``GCP_PROJECT_ID`` was not exported.
    """
    with open(path, encoding="utf-8") as fh:
        # expandvars replaces ${VAR} with its value and leaves unknown variables as-is.
        text = os.path.expandvars(fh.read())
    cfg = yaml.safe_load(text)

    # Fail early instead of creating resources literally named "${GCP_PROJECT_ID}".
    unresolved = [k for k, v in cfg.items() if isinstance(v, str) and "${" in v]
    if unresolved:
        raise ValueError(f"Unresolved environment variables in {path}: {unresolved}")
    return cfg
