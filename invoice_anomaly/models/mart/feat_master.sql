-- mart.feat_master
-- ──────────────────────────────────────────────────────────────────────────────
-- Master ML feature table: one row per invoice, all signals joined.
-- Direct input to training pipeline: 40+ features + is_fraud ground-truth label.
-- ──────────────────────────────────────────────────────────────────────────────
{{ config(materialized='table', schema='mart') }}

WITH fct AS (SELECT * FROM {{ ref('fct_invoices') }}),
     vb  AS (SELECT * FROM {{ ref('feat_invoice_vendor_behavior') }}),
     tg  AS (SELECT * FROM {{ ref('feat_threshold_gaming') }}),
     sh  AS (SELECT * FROM {{ ref('feat_shell_vendor_signals') }})

SELECT
    -- ── Identity ──────────────────────────────────────────────────────────
    fct.invoice_id,
    fct.vendor_id,
    fct.buyer_id,
    fct.issue_date,
    fct.issue_year,
    fct.issue_month,
    fct.issue_dow,
    fct.issue_dom,
    fct.vendor_category,
    fct.buyer_department,

    -- ── Raw invoice features ───────────────────────────────────────────────
    fct.amount,
    fct.currency,
    fct.status,
    fct.payment_method,
    fct.days_to_payment,
    fct.days_past_due,
    fct.is_weekend_issue,
    fct.is_weekend_payment,
    fct.is_month_start,
    fct.is_month_end,
    fct.is_round_1k,
    fct.is_round_500,

    -- ── Vendor behavior features ───────────────────────────────────────────
    vb.vendor_inv_count_30d,
    vb.vendor_inv_count_90d,
    vb.vendor_total_spend_30d,
    vb.vendor_avg_amount_alltime,
    vb.vendor_stddev_amount_alltime,
    vb.vendor_amount_zscore,
    vb.vendor_amount_rank,
    vb.vendor_prev_invoice_amount,
    vb.days_since_prev_vendor_invoice,
    vb.amount_delta_from_prev,
    vb.vendor_cumulative_spend,
    vb.flag_near_duplicate_prev,
    vb.flag_near_duplicate_next,
    vb.flag_velocity_spike,

    -- ── Threshold gaming features ──────────────────────────────────────────
    tg.nearest_threshold,
    tg.threshold_proximity_band,
    tg.vendor_hits_same_threshold,
    tg.vendor_threshold_hits_90d,
    tg.vendor_threshold_hit_seq,
    tg.buyer_threshold_approvals,
    tg.flag_threshold_gaming,

    -- ── Shell vendor features ──────────────────────────────────────────────
    sh.vendors_sharing_bank,
    sh.total_amount_same_bank,
    sh.invoice_count_same_bank,
    sh.ein_root,
    sh.vendors_sharing_ein_root,
    sh.v_total_invoices,
    sh.v_avg_amount,
    sh.v_wire_ratio,
    sh.v_rolling_30d_amount,
    sh.days_vendor_to_first_invoice,
    sh.shell_suspicion_score,
    sh.flag_shell_vendor,
    fct.is_high_risk_country,
    fct.is_shell_vendor,

    -- ── Composite rule-based fraud score (0–N, higher = more suspicious) ──
    (
        CASE WHEN vb.flag_near_duplicate_prev OR vb.flag_near_duplicate_next THEN 3 ELSE 0 END +
        CASE WHEN tg.flag_threshold_gaming                                    THEN 3 ELSE 0 END +
        CASE WHEN sh.flag_shell_vendor                                        THEN 4 ELSE 0 END +
        CASE WHEN vb.flag_velocity_spike                                      THEN 2 ELSE 0 END +
        CASE WHEN vb.vendor_amount_zscore > 3                                 THEN 2 ELSE 0 END +
        CASE WHEN fct.is_round_1k                                             THEN 1 ELSE 0 END +
        CASE WHEN fct.is_weekend_payment                                      THEN 1 ELSE 0 END +
        CASE WHEN fct.is_high_risk_country                                    THEN 2 ELSE 0 END
    )                                                   AS rule_based_fraud_score,

    -- ── Ground-truth labels ────────────────────────────────────────────────
    fct.fraud_category,
    fct.is_fraud,
    fct.anomaly_type

FROM fct
LEFT JOIN vb ON fct.invoice_id = vb.invoice_id
LEFT JOIN tg ON fct.invoice_id = tg.invoice_id
LEFT JOIN sh ON fct.invoice_id = sh.invoice_id
