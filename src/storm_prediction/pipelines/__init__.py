"""The two Vertex AI pipelines and a registry used by the scripts and tests.

``PIPELINES`` maps a short name to ``(pipeline function, compiled YAML file name)``.
``scripts/compile_pipelines.py`` and ``scripts/run_pipeline.py`` use these names.
"""

from storm_prediction.pipelines.batch_inference_pipeline import batch_inference_pipeline
from storm_prediction.pipelines.training_pipeline import training_pipeline

PIPELINES = {
    "training": (training_pipeline, "training_pipeline.yaml"),
    "batch_inference": (batch_inference_pipeline, "batch_inference_pipeline.yaml"),
}
