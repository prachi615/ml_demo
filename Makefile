ENV ?= dev
CONFIG := configs/$(ENV).yaml
REGION ?= us-central1
IMAGE ?= $(REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/storm-ml/storm-pipeline:latest

.PHONY: install lint test compile image run-training run-evaluation run-batch

install:
	pip install -r requirements-dev.txt && pip install --no-deps -e .

lint:
	ruff check src scripts tests

test:
	pytest --cov=storm_prediction

compile:
	PIPELINE_IMAGE=$(IMAGE) python scripts/compile_pipelines.py --output-dir build

image:
	gcloud builds submit --tag $(IMAGE) .

run-training: compile
	python scripts/run_pipeline.py --pipeline training --config $(CONFIG)

run-evaluation: compile
	python scripts/run_pipeline.py --pipeline evaluation --config $(CONFIG) --training-template build/training_pipeline.yaml

run-batch: compile
	python scripts/run_pipeline.py --pipeline batch_inference --config $(CONFIG)
