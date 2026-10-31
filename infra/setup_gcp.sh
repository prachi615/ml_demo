#!/usr/bin/env bash
# One-time GCP bootstrap for the storm forecasting project.
#   usage: PROJECT_ID=my-proj GITHUB_REPO=owner/repo ./infra/setup_gcp.sh
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
: "${GITHUB_REPO:?set GITHUB_REPO (owner/repo) for Workload Identity Federation}"
REGION="${REGION:-us-central1}"
BQ_LOCATION="${BQ_LOCATION:-US}"          # must be US: NOAA public dataset lives in US
BQ_DATASET="${BQ_DATASET:-storm_ml}"
BUCKET="gs://${PROJECT_ID}-storm-ml"
PIPELINES_SA="vertex-pipelines@${PROJECT_ID}.iam.gserviceaccount.com"
DEPLOYER_SA="github-deployer@${PROJECT_ID}.iam.gserviceaccount.com"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"

gcloud config set project "$PROJECT_ID"

echo ">> Enabling APIs"
gcloud services enable aiplatform.googleapis.com bigquery.googleapis.com bigquerystorage.googleapis.com \
  storage.googleapis.com artifactregistry.googleapis.com iamcredentials.googleapis.com \
  cloudresourcemanager.googleapis.com sts.googleapis.com

echo ">> Storage + BigQuery"
gcloud storage buckets create "$BUCKET" --location="$REGION" --uniform-bucket-level-access || true
bq --location="$BQ_LOCATION" mk --dataset --description "Storm forecasting (NOAA)" "${PROJECT_ID}:${BQ_DATASET}" || true

echo ">> Artifact Registry (Docker images + KFP pipeline templates)"
gcloud artifacts repositories create storm-ml --repository-format=docker --location="$REGION" || true
gcloud artifacts repositories create storm-pipelines --repository-format=kfp --location="$REGION" || true

echo ">> Service accounts"
gcloud iam service-accounts create vertex-pipelines --display-name="Vertex AI Pipelines runtime" || true
gcloud iam service-accounts create github-deployer --display-name="GitHub Actions deployer" || true

for role in roles/aiplatform.user roles/bigquery.dataEditor roles/bigquery.jobUser \
            roles/bigquery.readSessionUser roles/storage.objectAdmin roles/artifactregistry.reader; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:${PIPELINES_SA}" \
    --role="$role" --condition=None --quiet >/dev/null
done
# pipelines SA submits retraining runs as itself
gcloud iam service-accounts add-iam-policy-binding "$PIPELINES_SA" \
  --member="serviceAccount:${PIPELINES_SA}" --role=roles/iam.serviceAccountUser --quiet >/dev/null

for role in roles/aiplatform.user roles/artifactregistry.writer roles/storage.objectAdmin; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:${DEPLOYER_SA}" \
    --role="$role" --condition=None --quiet >/dev/null
done
gcloud iam service-accounts add-iam-policy-binding "$PIPELINES_SA" \
  --member="serviceAccount:${DEPLOYER_SA}" --role=roles/iam.serviceAccountUser --quiet >/dev/null

echo ">> Workload Identity Federation for GitHub Actions"
gcloud iam workload-identity-pools create github --location=global --display-name="GitHub" || true
gcloud iam workload-identity-pools providers create-oidc github-oidc --location=global \
  --workload-identity-pool=github --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='${GITHUB_REPO}'" || true
gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER_SA" \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/attribute.repository/${GITHUB_REPO}" \
  --quiet >/dev/null

cat <<MSG

Done. Configure the matching GitHub environment (dev / rc / prod) with:
  vars.GCP_PROJECT_ID   = ${PROJECT_ID}
  secrets.WIF_PROVIDER  = projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/providers/github-oidc
  secrets.DEPLOYER_SA   = ${DEPLOYER_SA}
MSG
