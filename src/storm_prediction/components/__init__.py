"""Kubeflow Pipelines components (the individual pipeline steps).

Each component runs in its own container on Vertex AI, using the project image.
Components stay thin: they read their inputs, call tested library code
(``storm_prediction.models``, ``features``, ``metrics`` ...) and write outputs.

* ``data``        BigQuery ingestion, feature table, train/val/test and scoring extracts
* ``training``    train / evaluate a model, pick the best one, register it
* ``monitoring``  load a registered model, naive baselines, health check, retraining trigger
* ``inference``   batch prediction to BigQuery
"""
