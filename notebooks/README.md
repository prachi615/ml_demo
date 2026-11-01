# `notebooks/`

## `local_training.ipynb`: get real metrics without Vertex AI

Runs the **training stage** of the project in Google Colab (or Jupyter on your laptop) against the **free BigQuery sandbox**.
It uses the project's own SQL, features, models and metrics, so the numbers match what the production pipeline would produce.

**Output:**
- a leaderboard of the 4 models (validation and test RMSE, MAE, R², WAPE, bias);
- the best of the 4, and the result of the "beat Linear Regression by more than 2%" check;
- reference forecasts: last week, same week last year, 3-year same-week average;
- errors by state and by month;
- XGBoost feature importance;
- CSV downloads of the results.

**Not included** (needs Vertex AI): Model Registry, champion check, schedules, batch inference.

---

## Step 1: Create a BigQuery sandbox project (free, no credit card)

1. Open https://console.cloud.google.com/bigquery and sign in with a Google account.
2. If asked, accept the terms of service and choose your country.
3. Create a project: click the project picker at the top → **New project**, give it a name (e.g. `storm-forecast-sandbox`) → **Create**.
4. Make sure the new project is selected. A **"Sandbox"** badge near the top confirms you're in the free tier with no billing.
5. Note the **project ID**. It is shown in the project picker and can differ from the name, e.g. `storm-forecast-sandbox-123456`.

Sandbox limits: 1 TB of queries and 10 GB of storage per month, and tables are deleted after 60 days. This notebook uses a few GB of queries and well under 1 GB of storage.

## Step 2: Open the notebook in Colab

Use either:
- **From GitHub** (once the code is pushed): go to https://colab.research.google.com → **File → Open notebook → GitHub**, enter `prachi615/ml_demo` and pick `notebooks/local_training.ipynb`.
- **Upload the file:** https://colab.research.google.com → **File → Upload notebook** → choose `local_training.ipynb`.

## Step 3: Run it

1. In the first code cell, set `PROJECT_ID` to your sandbox project ID.
2. Choose how the notebook gets the project code (`CODE_SOURCE`):
   - `"github"` clones `prachi615/ml_demo` (needs the code pushed first);
   - `"upload"` asks you to upload a `.zip` of your `ml_demo` folder.
3. Click **Runtime → Run all**. When the login pop-up appears, choose your Google account and allow access.

The whole run takes about 5–15 minutes on a free Colab machine, mostly for Random Forest and XGBoost.

## Running in Jupyter on your laptop instead

```bash
pip install -r requirements.txt jupyterlab
```

```bash
gcloud auth application-default login
```

```bash
jupyter lab notebooks/local_training.ipynb
```

Run the cells from the `notebooks/` folder; the notebook then finds the project code one level up.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Access Denied` / `bigquery.jobs.create` | `PROJECT_ID` is wrong, or a different project is selected; use the exact project ID |
| `Not found: Dataset ... was not found in location US` | Leave `LOCATION = "US"`; the public dataset is in US |
| Error mentioning the BigQuery Storage API | Change `.to_dataframe()` to `.to_dataframe(create_bqstorage_client=False)` in step 8 |
| Schema check (step 6) fails | NOAA changed column names; the error lists them. The SQL in `src/storm_prediction/sql/` needs updating |
