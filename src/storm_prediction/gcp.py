"""Thin wrappers around BigQuery / GCS used by pipeline components."""

from __future__ import annotations

import logging
import os
from importlib import resources
from string import Template

import pandas as pd

logger = logging.getLogger(__name__)


def render_sql(name: str, **params) -> str:
    """Render a SQL template from storm_prediction/sql using $placeholders."""
    text = resources.files("storm_prediction").joinpath("sql", name).read_text(encoding="utf-8")
    return Template(text).substitute({k: str(v) for k, v in params.items()})


def run_query(sql: str, project: str, location: str) -> None:
    from google.cloud import bigquery

    client = bigquery.Client(project=project, location=location)
    job = client.query(sql)
    job.result()
    logger.info("BigQuery job %s done (%s bytes processed)", job.job_id, job.total_bytes_processed)


def read_query(sql: str, project: str, location: str) -> pd.DataFrame:
    from google.cloud import bigquery

    client = bigquery.Client(project=project, location=location)
    return client.query(sql).to_dataframe(create_bqstorage_client=True)


def write_dataframe(
    df: pd.DataFrame, table_id: str, project: str, location: str, append: bool = True
) -> None:
    from google.cloud import bigquery

    client = bigquery.Client(project=project, location=location)
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND" if append else "WRITE_TRUNCATE",
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION] if append else None,
    )
    client.load_table_from_dataframe(df, table_id, job_config=job_config).result()
    logger.info("Wrote %d rows to %s", len(df), table_id)


def download_gcs_dir(gcs_uri: str, local_dir: str) -> None:
    """Copy every blob under gs://bucket/prefix into local_dir."""
    from google.cloud import storage

    bucket_name, _, prefix = gcs_uri.removeprefix("gs://").partition("/")
    prefix = prefix.rstrip("/") + "/"
    client = storage.Client()
    os.makedirs(local_dir, exist_ok=True)
    found = False
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        rel = blob.name[len(prefix):]
        if not rel or rel.endswith("/"):
            continue
        dest = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        blob.download_to_filename(dest)
        found = True
    if not found:
        raise FileNotFoundError(f"No files found under {gcs_uri}")
