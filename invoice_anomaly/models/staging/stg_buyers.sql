-- staging.stg_buyers
{{ config(materialized='view', schema='staging') }}

SELECT
    id          AS buyer_id,
    TRIM(name)  AS buyer_name,
    department,
    UPPER(TRIM(country)) AS country,
    CAST(created_at AS TIMESTAMP) AS buyer_created_at
FROM {{ ref('raw_buyers') }}
