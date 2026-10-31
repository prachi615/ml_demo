"""Pipeline registry: name -> (pipeline function, compiled file name)."""

from storm_prediction.pipelines.batch_inference_pipeline import batch_inference_pipeline
from storm_prediction.pipelines.evaluation_pipeline import evaluation_pipeline
from storm_prediction.pipelines.training_pipeline import training_pipeline

PIPELINES = {
    "training": (training_pipeline, "training_pipeline.yaml"),
    "evaluation": (evaluation_pipeline, "evaluation_pipeline.yaml"),
    "batch_inference": (batch_inference_pipeline, "batch_inference_pipeline.yaml"),
}
