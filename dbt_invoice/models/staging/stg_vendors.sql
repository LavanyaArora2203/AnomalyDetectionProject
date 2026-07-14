{{ config(materialized='view') }}

SELECT

    vendor_id,

    TRIM(vendor_name) AS vendor_name,

    vendor_category,

    bank_account_id,

    TRIM(vendor_address) AS vendor_address,

    onboarding_date,

    payment_terms

FROM {{ ref('raw_vendors') }}