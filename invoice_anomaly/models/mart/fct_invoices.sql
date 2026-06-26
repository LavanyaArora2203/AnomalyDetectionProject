-- mart.fct_invoices
-- Central fact table: one row per invoice, staging cols + enriched vendor/buyer dims.
-- Materialized as TABLE — downstream feature models ref this.
{{ config(materialized='table', schema='mart') }}

SELECT
    i.*,
    v.vendor_name,
    v.vendor_category,
    v.country          AS vendor_country,
    v.is_active        AS vendor_is_active,
    v.is_shell_vendor,
    v.is_high_risk_country,
    v.bank_account,
    v.shell_cluster_id,

    b.buyer_name,
    b.department       AS buyer_department,
    b.country          AS buyer_country

FROM {{ ref('stg_invoices') }}  i
LEFT JOIN {{ ref('stg_vendors') }} v ON i.vendor_id = v.vendor_id
LEFT JOIN {{ ref('stg_buyers') }}  b ON i.buyer_id  = b.buyer_id
