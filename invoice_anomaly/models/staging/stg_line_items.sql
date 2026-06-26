-- staging.stg_line_items
{{ config(materialized='view', schema='staging') }}

SELECT
    id                          AS line_item_id,
    invoice_id,
    TRIM(description)           AS description,
    CAST(quantity   AS DOUBLE)  AS quantity,
    CAST(unit_price AS DOUBLE)  AS unit_price,
    CAST(total      AS DOUBLE)  AS total,
    UPPER(TRIM(category_code))  AS category_code,

    -- unit price buckets for anomaly detection
    CASE
        WHEN unit_price < 100      THEN 'micro'
        WHEN unit_price < 1000     THEN 'small'
        WHEN unit_price < 10000    THEN 'medium'
        WHEN unit_price < 100000   THEN 'large'
        ELSE                            'xlarge'
    END                         AS unit_price_bucket

FROM {{ ref('raw_line_items') }}
