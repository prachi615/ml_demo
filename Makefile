# Developer shortcuts. Usage: make <target> [ENV=dev|rc|prod]
# Targets that talk to GCP need GCP_PROJECT_ID exported and `gcloud auth` done.

ENV ?= dev
CONFIG := configs/$(ENV).yaml
REGION ?= us-central1
IMAGE ?= $(REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/storm-ml/storm-pipeline:latest

.PHONY: install lint test compile image run-training run-evaluation run-batch

# Install runtime + dev dependencies and the package in editable mode.
install:
	pip install -r requirements-dev.txt && pip install --no-deps -e .

# Static checks (style, imports, common bugs).
lint:
	ruff check src scripts tests

# Unit tests + pipeline compilation tests, with coverage.
test:
	pytest --cov=storm_prediction

# Compile the three pipelines to build/*.yaml using $(IMAGE) for every component.
compile:
	PIPELINE_IMAGE=$(IMAGE) python scripts/compile_pipelines.py --output-dir build

# Build and push the component image with Cloud Build (no local Docker needed).
image:
	gcloud builds submit --tag $(IMAGE) .

# Submit one run of a pipeline to Vertex AI using configs/$(ENV).yaml.
run-training: compile
	python scripts/run_pipeline.py --pipeline training --config $(CONFIG)

run-evaluation: compile
	python scripts/run_pipeline.py --pipeline evaluation --config $(CONFIG) --training-template build/training_pipeline.yaml

run-batch: compile
	python scripts/run_pipeline.py --pipeline batch_inference --config $(CONFIG)
