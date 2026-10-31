"""Storm event forecasting on Vertex AI Pipelines.

The package holds everything that runs inside the pipeline container:

* ``sql/``         BigQuery SQL templates (raw ingestion and feature engineering)
* ``components/``  Kubeflow Pipelines (KFP) components, i.e. the pipeline steps
* ``pipelines/``   the three pipeline definitions (training, evaluation, batch inference)
* ``models.py``, ``features.py``, ``metrics.py``, ``data.py``
                   plain-Python ML logic, unit tested without any GCP access
* ``gcp.py``       thin BigQuery / Cloud Storage helpers
* ``config.py``    image name and YAML config loading
"""

__version__ = "0.1.0"
