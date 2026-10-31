"""Thin wrappers around BigQuery and Cloud Storage used by pipeline components.

Google Cloud client libraries are imported inside each function so that the rest
of the package (and its unit tests) can be imported without GCP credentials.
"""

from __future__ import annotations

import logging
import os
from importlib import resources
from string import Template

import pandas as pd

logger = logging.getLogger(__name__)


def render_sql(name: str, **params) -> str:
    """Load a SQL template shipped with the package and fill in its placeholders.

    Templates use ``$name`` placeholders (``string.Template``) instead of ``{}`` so
    that BigQuery syntax containing braces never clashes with formatting.

    Args:
        name: File name inside ``storm_prediction/sql``, e.g. ``"build_features.sql"``.
        **params: Values for the placeholders, e.g. ``project="my-proj"``.

    Returns:
        The rendered SQL string.

    Raises:
        KeyError: If the template uses a placeholder that was not supplied.
    """
    text = resources.files("storm_prediction").joinpath("sql", name).read_text(encoding="utf-8")
    return Template(text).substitute({k: str(v) for k, v in params.items()})


def run_query(sql: str, project: str, location: str) -> None:
    """Run a SQL statement (DDL/DML) in BigQuery and wait until it finishes.

    Args:
        sql: The statement to run, e.g. a ``CREATE OR REPLACE TABLE``.
        project: GCP project that runs (and is billed for) the job.
        location: BigQuery location, e.g. ``"US"``.

    Raises:
        google.api_core.exceptions.GoogleAPIError: If the job fails.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project, location=location)
    job = client.query(sql)
    job.result()  # blocks until done and raises on failure
    logger.info("BigQuery job %s done (%s bytes processed)", job.job_id, job.total_bytes_processed)


def read_query(sql: str, project: str, location: str) -> pd.DataFrame:
    """Run a SELECT in BigQuery and return the result as a DataFrame.

    Uses the BigQuery Storage API, which is much faster than the REST API for
    the tens of thousands of rows in the feature table.

    Args:
        sql: A SELECT statement.
        project: GCP project that runs the query.
        location: BigQuery location, e.g. ``"US"``.

    Returns:
        The query result.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project, location=location)
    return client.query(sql).to_dataframe(create_bqstorage_client=True)


def write_dataframe(
    df: pd.DataFrame, table_id: str, project: str, location: str, append: bool = True
) -> None:
    """Load a DataFrame into a BigQuery table, creating the table if needed.

    Args:
        df: Rows to write.
        table_id: Fully qualified table, ``project.dataset.table``.
        project: GCP project that runs the load job.
        location: BigQuery location, e.g. ``"US"``.
        append: Append rows (default). In append mode, new columns are added to
            the table schema automatically. If False, the table is overwritten.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project, location=location)
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND" if append else "WRITE_TRUNCATE",
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION] if append else None,
    )
    client.load_table_from_dataframe(df, table_id, job_config=job_config).result()
    logger.info("Wrote %d rows to %s", len(df), table_id)


def download_gcs_dir(gcs_uri: str, local_dir: str) -> None:
    """Copy every file under a Cloud Storage prefix into a local directory.

    Used to fetch a registered model's artifacts (``model.joblib``) before scoring.

    Args:
        gcs_uri: Source prefix, e.g. ``gs://bucket/path/to/model``.
        local_dir: Destination directory; created if missing. Sub-folders are kept.

    Raises:
        FileNotFoundError: If nothing exists under ``gcs_uri``.
    """
    from google.cloud import storage

    bucket_name, _, prefix = gcs_uri.removeprefix("gs://").partition("/")
    prefix = prefix.rstrip("/") + "/"
    client = storage.Client()
    os.makedirs(local_dir, exist_ok=True)
    found = False
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        rel = blob.name[len(prefix):]
        if not rel or rel.endswith("/"):  # skip "directory" placeholder objects
            continue
        dest = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        blob.download_to_filename(dest)
        found = True
    if not found:
        raise FileNotFoundError(f"No files found under {gcs_uri}")
