"""Kubeflow Pipelines components (the individual pipeline steps).

Each component runs in its own container on Vertex AI, using the project image.
Components stay thin: they read their inputs, call tested library code
(``storm_prediction.models``, ``features``, ``metrics`` ...) and write outputs.

* ``data``        BigQuery ingestion, feature table, train/val/test and scoring extracts
* ``training``    train / test-score a model, pick the best of four, register the champion
* ``evaluation``  after training: score the champion on the same test data, decide promotion
* ``inference``   load the champion, batch prediction to BigQuery
"""
