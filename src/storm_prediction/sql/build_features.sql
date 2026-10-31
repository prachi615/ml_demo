-- -----------------------------------------------------------------------------
-- Weekly, state-level feature table for storm-event forecasting.
--
-- One row = (state, week_start). Target `event_count` = number of severe storm
-- events reported in that state during that ISO week (Mon-Sun).
--
-- Every feature only uses weeks STRICTLY BEFORE `week_start`
-- (LAG / "N PRECEDING AND 1 PRECEDING" windows), so there is no target leakage.
--
-- The table also contains one "forecast" row per state for the week after the
-- last complete week of data (event_count IS NULL, is_forecast_row = TRUE).
-- Batch inference scores those rows.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE `$project.$dataset.$features_table`
PARTITION BY DATE_TRUNC(week_start, MONTH)
CLUSTER BY state
OPTIONS (description = "Weekly storm features per state (leak-free lags). Built by Vertex AI pipelines.")
AS
WITH
bounds AS (
  SELECT
    MIN(DATE_TRUNC(event_date, WEEK(MONDAY))) AS min_week,
    -- Monday of the last fully observed week
    DATE_SUB(DATE_TRUNC(DATE_ADD(MAX(event_date), INTERVAL 1 DAY), WEEK(MONDAY)), INTERVAL 1 WEEK)
      AS last_complete_week
  FROM `$project.$dataset.$raw_table`
),

weekly AS (
  SELECT
    state,
    DATE_TRUNC(event_date, WEEK(MONDAY))                                     AS week_start,
    COUNT(*)                                                                 AS event_count,
    COUNTIF(event_type = 'Tornado')                                          AS tornado_count,
    COUNTIF(event_type = 'Hail')                                             AS hail_count,
    COUNTIF(event_type IN ('Thunderstorm Wind', 'High Wind', 'Strong Wind')) AS wind_count,
    COUNTIF(event_type IN ('Flash Flood', 'Flood', 'Heavy Rain'))            AS flood_count,
    SUM(injuries)                                                            AS injuries,
    SUM(deaths)                                                              AS deaths,
    SUM(damage_property)                                                     AS damage_property
  FROM `$project.$dataset.$raw_table`
  GROUP BY state, week_start
),

states AS (SELECT DISTINCT state FROM weekly),

-- Dense calendar so weeks with zero events become explicit zeros.
spine AS (
  SELECT s.state, w AS week_start, b.last_complete_week
  FROM states AS s
  CROSS JOIN bounds AS b
  CROSS JOIN UNNEST(
    GENERATE_DATE_ARRAY(b.min_week, DATE_ADD(b.last_complete_week, INTERVAL 1 WEEK), INTERVAL 1 WEEK)
  ) AS w
),

filled AS (
  SELECT
    sp.state,
    sp.week_start,
    sp.week_start > sp.last_complete_week                                    AS is_forecast_row,
    IF(sp.week_start > sp.last_complete_week, NULL, COALESCE(w.event_count, 0)) AS event_count,
    COALESCE(w.event_count, 0)     AS event_count_filled,
    COALESCE(w.tornado_count, 0)   AS tornado_count,
    COALESCE(w.hail_count, 0)      AS hail_count,
    COALESCE(w.wind_count, 0)      AS wind_count,
    COALESCE(w.flood_count, 0)     AS flood_count,
    COALESCE(w.injuries, 0)        AS injuries,
    COALESCE(w.deaths, 0)          AS deaths,
    COALESCE(w.damage_property, 0) AS damage_property
  FROM spine AS sp
  LEFT JOIN weekly AS w
    ON w.state = sp.state AND w.week_start = sp.week_start
),

features AS (
  SELECT
    state,
    week_start,
    is_forecast_row,
    event_count,

    -- calendar / seasonality
    EXTRACT(MONTH FROM week_start)                                   AS month,
    SIN(2 * ACOS(-1) * EXTRACT(ISOWEEK FROM week_start) / 52.0)      AS week_of_year_sin,
    COS(2 * ACOS(-1) * EXTRACT(ISOWEEK FROM week_start) / 52.0)      AS week_of_year_cos,

    -- autoregressive lags of the target
    LAG(event_count_filled, 1)   OVER w AS lag_1,
    LAG(event_count_filled, 2)   OVER w AS lag_2,
    LAG(event_count_filled, 3)   OVER w AS lag_3,
    LAG(event_count_filled, 4)   OVER w AS lag_4,
    LAG(event_count_filled, 52)  OVER w AS lag_52,
    (LAG(event_count_filled, 52) OVER w
      + LAG(event_count_filled, 104) OVER w
      + LAG(event_count_filled, 156) OVER w) / 3.0                   AS same_week_mean_3y,

    -- rolling statistics (exclude current week)
    AVG(event_count_filled)    OVER (w ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING)  AS rolling_mean_4,
    AVG(event_count_filled)    OVER (w ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING) AS rolling_mean_12,
    STDDEV(event_count_filled) OVER (w ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING) AS rolling_std_12,
    AVG(event_count_filled)    OVER (w ROWS BETWEEN 52 PRECEDING AND 1 PRECEDING) AS rolling_mean_52,
    AVG(event_count_filled)    OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS state_expanding_mean,

    -- event-type mix of the recent past
    SUM(tornado_count) OVER (w ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS tornado_sum_4,
    SUM(hail_count)    OVER (w ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS hail_sum_4,
    SUM(wind_count)    OVER (w ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS wind_sum_4,
    SUM(flood_count)   OVER (w ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS flood_sum_4,

    -- impact of the recent past
    SUM(injuries) OVER (w ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING)              AS injuries_sum_12,
    SUM(deaths)   OVER (w ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING)              AS deaths_sum_12,
    LN(1 + SUM(damage_property) OVER (w ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING)) AS log_damage_sum_12
  FROM filled
  WINDOW w AS (PARTITION BY state ORDER BY week_start)
)

SELECT
  *,
  CURRENT_TIMESTAMP() AS feature_built_at
FROM features
-- require 3 full years of history so every lag is populated
WHERE same_week_mean_3y IS NOT NULL;
