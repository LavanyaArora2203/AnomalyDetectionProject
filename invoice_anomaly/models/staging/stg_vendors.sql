-- staging.stg_vendors
{{ config(materialized='view', schema='staging') }}

WITH source AS (SELECT * FROM {{ ref('raw_vendors') }})

SELECT
    id                                          AS vendor_id,
    TRIM(name)                                  AS vendor_name,
    tax_id,
    UPPER(TRIM(country))                        AS country,
    category                                    AS vendor_category,
    CAST(is_active AS BOOLEAN)                  AS is_active,
    CAST(created_at AS TIMESTAMP)               AS vendor_created_at,

    -- shell vendor signals
    bank_account,
    shell_cluster_id,
    (shell_cluster_id IS NOT NULL)              AS is_shell_vendor,

    -- high-risk jurisdiction flag
    (UPPER(TRIM(country)) IN ('BVI','KY','PA','SC','VG'))   AS is_high_risk_country

FROM source
