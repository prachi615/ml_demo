"""Compile all pipelines to KFP YAML and optionally push them to Artifact Registry.

Compiling turns each Python ``@dsl.pipeline`` into a YAML template that Vertex AI
Pipelines can run. CI runs this with ``PIPELINE_IMAGE`` set to the image it just
built, then publishes the templates to an Artifact Registry "KFP" repository.

Examples:

    PIPELINE_IMAGE=<image:tag> python scripts/compile_pipelines.py --output-dir build
    python scripts/compile_pipelines.py --output-dir build \
        --registry-host https://us-central1-kfp.pkg.dev/<project>/storm-pipelines --tag <git-sha>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from kfp import compiler


def main() -> None:
    """Compile every pipeline in ``PIPELINES`` and optionally upload it.

    The component image is read from the ``PIPELINE_IMAGE`` environment variable
    when the pipeline modules are imported, so it must be set before running.
    Uploaded templates are tagged ``latest`` plus any ``--tag`` values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="build")
    parser.add_argument("--registry-host", default="", help="Artifact Registry KFP repo URL")
    parser.add_argument("--tag", action="append", default=[], help="Extra template tag(s)")
    args = parser.parse_args()

    # Import after argparse so PIPELINE_IMAGE from the environment is honoured.
    from storm_prediction.pipelines import PIPELINES

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Component image: {os.environ.get('PIPELINE_IMAGE', '<default>')}")

    for name, (pipeline_func, file_name) in PIPELINES.items():
        path = out / file_name
        compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=str(path))
        print(f"compiled {name:16s} -> {path}")

        if args.registry_host:
            from kfp.registry import RegistryClient

            client = RegistryClient(host=args.registry_host)
            template_name, version = client.upload_pipeline(
                file_name=str(path), tags=["latest", *args.tag]
            )
            print(f"uploaded {template_name}@{version} tags={['latest', *args.tag]}")


if __name__ == "__main__":
    main()
