# Single image used by every Vertex AI Pipelines component.
#
# It contains the Python dependencies (KFP, scikit-learn, XGBoost, Google Cloud
# clients) and the storm_prediction package, so each component can simply
# `from storm_prediction... import ...`. CI builds it, tags it with the commit
# SHA and compiles the pipelines against that exact tag.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgomp is required by XGBoost / scikit-learn OpenMP builds
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Dependencies first: this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Then the project code (changes often, so it comes last).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# Components are invoked by KFP's executor (python -m kfp.dsl.executor_main)
ENTRYPOINT []
