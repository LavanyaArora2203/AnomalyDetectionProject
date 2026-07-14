{{ config(materialized='view') }}

SELECT
    vendor_id,
    vendor_name,
    vendor_category,
    bank_account_id,
    vendor_address,
    CAST(onboarding_date AS DATE) AS onboarding_date,
    payment_terms
FROM {{ ref('vendors') }}