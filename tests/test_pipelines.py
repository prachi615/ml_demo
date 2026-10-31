"""Tests that the KFP pipelines compile and have the expected shape."""

import yaml
from kfp import compiler

from storm_prediction.pipelines import PIPELINES


def test_all_pipelines_compile(tmp_path):
    """All three pipelines compile to valid KFP YAML (catches wiring errors early)."""
    for name, (pipeline_func, file_name) in PIPELINES.items():
        path = tmp_path / file_name
        compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=str(path))
        spec = yaml.safe_load(path.read_text())
        assert spec["pipelineInfo"]["name"].startswith("storm-forecast"), name


def test_training_pipeline_trains_four_models(tmp_path):
    """The compiled training DAG contains exactly four training tasks."""
    func, file_name = PIPELINES["training"]
    path = tmp_path / file_name
    compiler.Compiler().compile(pipeline_func=func, package_path=str(path))
    tasks = yaml.safe_load(path.read_text())["root"]["dag"]["tasks"]
    train_tasks = [t for t in tasks if t.startswith("train-model")]
    assert len(train_tasks) == 4
