# `src/storm_prediction/sql/`

BigQuery SQL templates. Components render them with `gcp.render_sql()`, which fills `$project`, `$dataset` and the other placeholders, and then run them in BigQuery.

## `ingest_raw.sql`

Copies NOAA Storm Events from the public dataset `bigquery-public-data.noaa_historic_severe_storms.storms_*` into `storm_ml.storm_events_raw`. Along the way it:

- keeps the 50 states and DC (drops marine zones and territories);
- adds direct and indirect injuries and deaths together;
- fills missing damage with 0;
- partitions the table by month and clusters it by `state, event_type`.

Placeholders: `$project`, `$dataset`, `$raw_table`, `$start_year`, `$end_year`.

## `build_features.sql`

Builds `storm_ml.storm_features_weekly` with one row per **state × week** (Monday to Sunday):

1. Counts events per state and week, overall and by type (tornado, hail, wind, flood).
2. Builds a full state × week calendar so weeks without storms appear as 0.
3. Adds lags, rolling means and standard deviations, same-week-last-years averages, and recent injuries, deaths and damage.
4. Adds one **forecast row** per state for the week after the last complete week. Its `event_count` is `NULL` and `is_forecast_row` is `TRUE`; batch inference scores these rows.
5. Drops the first 3 years of each state, because the 3-year lags aren't available there yet.

**No leakage:** every window ends at `1 PRECEDING`, so a row's features use only earlier weeks. A unit test checks this.

Placeholders: `$project`, `$dataset`, `$raw_table`, `$features_table`.

> The public dataset lives in the `US` multi-region, so the `storm_ml` dataset must also be in `US`.
