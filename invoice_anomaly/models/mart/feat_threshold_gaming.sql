-- mart.feat_threshold_gaming
-- ──────────────────────────────────────────────────────────────────────────────
-- Detects approval-threshold evasion.
-- Key idea: cluster invoices that land suspiciously close to known thresholds
-- and look for repeated patterns from the same vendor.
-- ──────────────────────────────────────────────────────────────────────────────
{{ config(materialized='table', schema='mart') }}

WITH base AS (
    SELECT * FROM {{ ref('fct_invoices') }}
),

-- Tag which threshold each invoice is nearest to (from below)
threshold_proximity AS (
    SELECT
        invoice_id,
        vendor_id,
        buyer_id,
        amount,
        issue_date,
        status,
        fraud_category,
        is_fraud,

        -- Distance below each common threshold
        5000   - amount AS gap_to_5k,
        10000  - amount AS gap_to_10k,
        25000  - amount AS gap_to_25k,
        50000  - amount AS gap_to_50k,
        100000 - amount AS gap_to_100k,

        -- Nearest threshold (from below, within $500 window)
        CASE
            WHEN amount BETWEEN 4500  AND 4999.99  THEN 5000
            WHEN amount BETWEEN 9500  AND 9999.99  THEN 10000
            WHEN amount BETWEEN 24500 AND 24999.99 THEN 25000
            WHEN amount BETWEEN 49500 AND 49999.99 THEN 50000
            WHEN amount BETWEEN 99500 AND 99999.99 THEN 100000
            ELSE NULL
        END                                         AS nearest_threshold,

        -- How tight is the gap? (<$50 is very suspicious, <$200 is suspicious)
        CASE
            WHEN amount BETWEEN 4950  AND 4999.99  THEN 'tight'
            WHEN amount BETWEEN 9950  AND 9999.99  THEN 'tight'
            WHEN amount BETWEEN 24950 AND 24999.99 THEN 'tight'
            WHEN amount BETWEEN 49950 AND 49999.99 THEN 'tight'
            WHEN amount BETWEEN 99950 AND 99999.99 THEN 'tight'
            WHEN amount BETWEEN 4500  AND 4949.99  THEN 'near'
            WHEN amount BETWEEN 9500  AND 9949.99  THEN 'near'
            WHEN amount BETWEEN 24500 AND 24949.99 THEN 'near'
            WHEN amount BETWEEN 49500 AND 49949.99 THEN 'near'
            WHEN amount BETWEEN 99500 AND 99949.99 THEN 'near'
            ELSE 'none'
        END                                         AS threshold_proximity_band

    FROM base
),

-- Count how many times each vendor has hit the same threshold zone
vendor_threshold_counts AS (
    SELECT
        *,
        COUNT(*) OVER (
            PARTITION BY vendor_id, nearest_threshold
        )                                           AS vendor_hits_same_threshold,

        -- Recency: how many threshold-zone invoices from this vendor in last 90 days?
        SUM(CASE WHEN nearest_threshold IS NOT NULL THEN 1 ELSE 0 END) OVER (
            PARTITION BY vendor_id
            ORDER BY issue_date
            RANGE BETWEEN INTERVAL '90' DAY PRECEDING AND CURRENT ROW
        )                                           AS vendor_threshold_hits_90d,

        -- Rank within vendor × threshold group by recency
        ROW_NUMBER() OVER (
            PARTITION BY vendor_id, nearest_threshold
            ORDER BY issue_date
        )                                           AS vendor_threshold_hit_seq,

        -- Buyer-level: how many threshold invoices approved by same buyer?
        COUNT(*) OVER (
            PARTITION BY buyer_id, nearest_threshold
        )                                           AS buyer_threshold_approvals

    FROM threshold_proximity
)

SELECT
    *,
    -- Final composite flag
    CASE
        WHEN nearest_threshold IS NOT NULL
         AND (vendor_hits_same_threshold >= 3
              OR threshold_proximity_band = 'tight'
              OR vendor_threshold_hits_90d >= 5)
        THEN TRUE
        ELSE FALSE
    END                                             AS flag_threshold_gaming

FROM vendor_threshold_counts
