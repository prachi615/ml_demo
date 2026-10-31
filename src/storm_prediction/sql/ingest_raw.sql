-- -----------------------------------------------------------------------------
-- Ingest NOAA Storm Events (public BigQuery dataset) into the project's own
-- BigQuery dataset as a clean, partitioned "raw" table.
--
-- Source : bigquery-public-data.noaa_historic_severe_storms.storms_YYYY
--          (one table per year, NOAA NCEI Storm Events Database)
-- Target : `$project.$dataset.$raw_table`
--
-- NOTE: the public dataset lives in the US multi-region, so `$dataset`
--       must also be created in location US (cross-region CTAS is not allowed).
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE `$project.$dataset.$raw_table`
PARTITION BY DATE_TRUNC(event_date, MONTH)
CLUSTER BY state, event_type
OPTIONS (
  description = "NOAA Storm Events (US states + DC), cleaned. Loaded by the Vertex AI storm pipelines."
)
AS
SELECT
  CAST(event_id AS STRING)                                   AS event_id,
  CAST(episode_id AS STRING)                                 AS episode_id,
  INITCAP(TRIM(state))                                       AS state,
  state_fips_code,
  event_type,
  DATE(event_begin_time)                                     AS event_date,
  event_begin_time,
  event_end_time,
  COALESCE(injuries_direct, 0) + COALESCE(injuries_indirect, 0) AS injuries,
  COALESCE(deaths_direct, 0) + COALESCE(deaths_indirect, 0)     AS deaths,
  COALESCE(damage_property, 0)                               AS damage_property,
  COALESCE(damage_crops, 0)                                  AS damage_crops,
  magnitude,
  magnitude_type,
  event_latitude,
  event_longitude,
  CURRENT_TIMESTAMP()                                        AS ingested_at
FROM `bigquery-public-data.noaa_historic_severe_storms.storms_*`
WHERE _TABLE_SUFFIX BETWEEN '$start_year' AND '$end_year'
  -- keep the 50 states + DC (drops marine zones, territories, lakes)
  AND SAFE_CAST(state_fips_code AS INT64) BETWEEN 1 AND 56
  AND event_begin_time IS NOT NULL;
