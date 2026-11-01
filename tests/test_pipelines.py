"""Tests that the KFP pipelines compile and have the expected shape."""

import yaml
from kfp import compiler

from storm_prediction.pipelines import PIPELINES


def test_all_pipelines_compile(tmp_path):
    """Both pipelines compile to valid KFP YAML (catches wiring errors early)."""
    for name, (pipeline_func, file_name) in PIPELINES.items():
        path = tmp_path / file_name
        compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=str(path))
        spec = yaml.safe_load(path.read_text())
        assert spec["pipelineInfo"]["name"].startswith("storm-forecast"), name


def test_only_training_and_batch_inference_pipelines_exist():
    """There is no separate evaluation pipeline: evaluation is part of training."""
    assert set(PIPELINES) == {"training", "batch_inference"}


def test_training_pipeline_trains_four_models(tmp_path):
    """The compiled training DAG contains exactly four training tasks."""
    func, file_name = PIPELINES["training"]
    path = tmp_path / file_name
    compiler.Compiler().compile(pipeline_func=func, package_path=str(path))
    tasks = yaml.safe_load(path.read_text())["root"]["dag"]["tasks"]
    train_tasks = [t for t in tasks if t.startswith("train-model")]
    assert len(train_tasks) == 4


def test_evaluation_runs_after_training_and_gates_registration(tmp_path):
    """Champion evaluation waits for model selection; registration sits behind the decision."""
    func, file_name = PIPELINES["training"]
    path = tmp_path / file_name
    compiler.Compiler().compile(pipeline_func=func, package_path=str(path))
    tasks = yaml.safe_load(path.read_text())["root"]["dag"]["tasks"]

    assert "select-best-model" in tasks["evaluate-champion"]["dependentTasks"]
    assert {"select-best-model", "evaluate-champion"} <= set(tasks["decide-promotion"]["dependentTasks"])
    # register_model lives inside the "new model is better" condition, not at the top level
    assert "register-model" not in tasks
    condition = next(t for name, t in tasks.items() if name.startswith("condition"))
    assert "decide-promotion" in condition["dependentTasks"]


def test_baseline_gate_feeds_the_promotion_decision(tmp_path):
    """The linear regression gate (default 2%) is wired from model selection into the decision."""
    func, file_name = PIPELINES["training"]
    path = tmp_path / file_name
    compiler.Compiler().compile(pipeline_func=func, package_path=str(path))
    spec = yaml.safe_load(path.read_text())

    params = spec["root"]["inputDefinitions"]["parameters"]
    assert params["min_improvement_over_baseline_pct"]["defaultValue"] == 2.0

    decision_inputs = spec["root"]["dag"]["tasks"]["decide-promotion"]["inputs"]["parameters"]
    assert decision_inputs["beats_baseline"]["taskOutputParameter"] == {
        "outputParameterKey": "beats_baseline",
        "producerTask": "select-best-model",
    }
