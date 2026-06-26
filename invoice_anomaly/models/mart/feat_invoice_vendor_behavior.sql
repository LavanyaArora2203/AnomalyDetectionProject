-- mart.feat_invoice_vendor_behavior
-- ──────────────────────────────────────────────────────────────────────────────
-- Per-invoice vendor behavioral features derived with window functions.
-- Core signals for duplicate detection, velocity anomalies, and amount deviation.
-- ──────────────────────────────────────────────────────────────────────────────
{{ config(materialized='table', schema='mart') }}

WITH base AS (
    SELECT * FROM {{ ref('fct_invoices') }}
),

-- ── 1. Vendor-level rolling aggregates ───────────────────────────────────────
vendor_windows AS (
    SELECT
        invoice_id,
        vendor_id,
        amount,
        issue_date,
        status,
        fraud_category,
        is_fraud,

        -- ── Rolling 30-day invoice count per vendor ──
        COUNT(*) OVER (
            PARTITION BY vendor_id
            ORDER BY issue_date
            RANGE BETWEEN INTERVAL '30' DAY PRECEDING AND CURRENT ROW
        )                                               AS vendor_inv_count_30d,

        -- ── Rolling 90-day invoice count per vendor ──
        COUNT(*) OVER (
            PARTITION BY vendor_id
            ORDER BY issue_date
            RANGE BETWEEN INTERVAL '90' DAY PRECEDING AND CURRENT ROW
        )                                               AS vendor_inv_count_90d,

        -- ── Rolling 30-day total spend per vendor ──
        SUM(amount) OVER (
            PARTITION BY vendor_id
            ORDER BY issue_date
            RANGE BETWEEN INTERVAL '30' DAY PRECEDING AND CURRENT ROW
        )                                               AS vendor_total_spend_30d,

        -- ── All-time vendor average and stddev ──
        AVG(amount) OVER (
            PARTITION BY vendor_id
        )                                               AS vendor_avg_amount_alltime,

        STDDEV(amount) OVER (
            PARTITION BY vendor_id
        )                                               AS vendor_stddev_amount_alltime,

        -- ── Vendor invoice rank by amount (for top-N flagging) ──
        RANK() OVER (
            PARTITION BY vendor_id
            ORDER BY amount DESC
        )                                               AS vendor_amount_rank,

        -- ── Lag: previous invoice amount from same vendor ──
        LAG(amount, 1) OVER (
            PARTITION BY vendor_id
            ORDER BY issue_date
        )                                               AS vendor_prev_invoice_amount,

        LAG(issue_date, 1) OVER (
            PARTITION BY vendor_id
            ORDER BY issue_date
        )                                               AS vendor_prev_invoice_date,

        -- ── Lead: next invoice amount (detects duplicate pairs from both sides) ──
        LEAD(amount, 1) OVER (
            PARTITION BY vendor_id
            ORDER BY issue_date
        )                                               AS vendor_next_invoice_amount,

        -- ── Running cumulative spend per vendor ──
        SUM(amount) OVER (
            PARTITION BY vendor_id
            ORDER BY issue_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )                                               AS vendor_cumulative_spend

    FROM base
),

-- ── 2. Derived anomaly signals ────────────────────────────────────────────────
enriched AS (
    SELECT
        *,

        -- Z-score: how far is this invoice from the vendor's typical amount?
        CASE
            WHEN vendor_stddev_amount_alltime > 0
            THEN (amount - vendor_avg_amount_alltime) / vendor_stddev_amount_alltime
            ELSE 0
        END                                             AS vendor_amount_zscore,

        -- Amount delta vs previous invoice (same vendor)
        ABS(amount - COALESCE(vendor_prev_invoice_amount, amount))
                                                        AS amount_delta_from_prev,

        -- Near-duplicate signal: current vs prev differ by < $1
        CASE
            WHEN vendor_prev_invoice_amount IS NOT NULL
             AND ABS(amount - vendor_prev_invoice_amount) < 1.0
             AND ABS(issue_date - vendor_prev_invoice_date) <= 14
            THEN TRUE ELSE FALSE
        END                                             AS flag_near_duplicate_prev,

        CASE
            WHEN vendor_next_invoice_amount IS NOT NULL
             AND ABS(amount - vendor_next_invoice_amount) < 1.0
            THEN TRUE ELSE FALSE
        END                                             AS flag_near_duplicate_next,

        -- Velocity spike: >3x the 90-day average daily rate
        CASE
            WHEN vendor_inv_count_90d > (vendor_inv_count_30d * 3.5)
            THEN TRUE ELSE FALSE
        END                                             AS flag_velocity_spike,

        -- Days since last invoice from same vendor
        DATEDIFF('day',
            COALESCE(vendor_prev_invoice_date, issue_date),
            issue_date
        )                                               AS days_since_prev_vendor_invoice

    FROM vendor_windows
)

SELECT * FROM enriched
