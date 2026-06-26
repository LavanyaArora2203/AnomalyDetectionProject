-- mart.feat_shell_vendor_signals
-- ──────────────────────────────────────────────────────────────────────────────
-- Network-level signals for shell vendor / payment diversion detection.
-- Joins vendor metadata to detect shared bank accounts, EIN patterns,
-- and behavioural clustering without knowing ground-truth labels.
-- ──────────────────────────────────────────────────────────────────────────────
{{ config(materialized='table', schema='mart') }}

WITH base AS (
    SELECT * FROM {{ ref('fct_invoices') }}
),

-- ── How many vendors share each bank account? ─────────────────────────────
bank_account_sharing AS (
    SELECT
        bank_account,
        COUNT(DISTINCT vendor_id)       AS vendors_sharing_bank,
        SUM(amount)                     AS total_amount_same_bank,
        COUNT(*)                        AS invoice_count_same_bank,
        MIN(issue_date)                 AS first_invoice_same_bank,
        MAX(issue_date)                 AS last_invoice_same_bank
    FROM base
    WHERE bank_account IS NOT NULL
    GROUP BY 1
),

-- ── EIN root clustering: vendors whose tax_id shares first 6 chars ───────
-- (proxy for "same real entity, rotated EIN")
vendor_ein_clusters AS (
    SELECT
        v.vendor_id,
        LEFT(v.tax_id, 6)               AS ein_root,
        COUNT(*) OVER (
            PARTITION BY LEFT(v.tax_id, 6)
        )                               AS vendors_sharing_ein_root
    FROM {{ ref('stg_vendors') }} v
),

-- ── Per-vendor behavioural baseline ──────────────────────────────────────
vendor_behavior AS (
    SELECT
        b.vendor_id,
        COUNT(*)                         AS total_invoices,
        SUM(b.amount)                    AS total_billed,
        AVG(b.amount)                    AS avg_invoice_amount,
        STDDEV(b.amount)                 AS stddev_invoice_amount,
        MIN(b.issue_date)                AS first_invoice_date,
        MAX(b.issue_date)                AS last_invoice_date,
        COUNT(DISTINCT b.buyer_id)       AS distinct_buyers,
        COUNT(DISTINCT b.payment_method) AS distinct_pay_methods,
        SUM(CASE WHEN b.payment_method = 'wire' THEN 1 ELSE 0 END) AS wire_count,
        SUM(CASE WHEN b.status = 'paid'         THEN 1 ELSE 0 END) AS paid_count,
        DATEDIFF('day',
            MIN(CAST(v.vendor_created_at AS DATE)),
            MIN(b.issue_date)
        )                                AS days_vendor_to_first_invoice
    FROM base b
    LEFT JOIN main_staging.stg_vendors v ON b.vendor_id = v.vendor_id
    GROUP BY b.vendor_id
),

-- ── Main join ─────────────────────────────────────────────────────────────
joined AS (
    SELECT
        b.*,
        ba.vendors_sharing_bank,
        ba.total_amount_same_bank,
        ba.invoice_count_same_bank,
        ec.ein_root,
        ec.vendors_sharing_ein_root,
        vb.total_invoices                               AS v_total_invoices,
        vb.total_billed                                 AS v_total_billed,
        vb.avg_invoice_amount                           AS v_avg_amount,
        vb.stddev_invoice_amount                        AS v_stddev_amount,
        vb.distinct_buyers                              AS v_distinct_buyers,
        vb.wire_count                                   AS v_wire_count,
        vb.paid_count                                   AS v_paid_count,
        vb.days_vendor_to_first_invoice,

        -- Wire-only ratio
        CASE
            WHEN vb.total_invoices > 0
            THEN vb.wire_count * 1.0 / vb.total_invoices
            ELSE 0
        END                                             AS v_wire_ratio,

        -- Rolling 30-day amount per vendor (velocity)
        SUM(b.amount) OVER (
            PARTITION BY b.vendor_id
            ORDER BY b.issue_date
            RANGE BETWEEN INTERVAL '30' DAY PRECEDING AND CURRENT ROW
        )                                               AS v_rolling_30d_amount,

        -- Rank vendor invoices by amount within the shared-bank group
        RANK() OVER (
            PARTITION BY b.bank_account
            ORDER BY b.amount DESC
        )                                               AS rank_within_bank_group,

        -- Cumulative vendor billing (for sudden ramp detection)
        SUM(b.amount) OVER (
            PARTITION BY b.vendor_id
            ORDER BY b.issue_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )                                               AS v_cumulative_billed

    FROM base b
    LEFT JOIN bank_account_sharing ba ON b.bank_account = ba.bank_account
    LEFT JOIN vendor_ein_clusters   ec ON b.vendor_id   = ec.vendor_id
    LEFT JOIN vendor_behavior       vb ON b.vendor_id   = vb.vendor_id
)

SELECT
    *,

    -- ── Composite shell vendor score (0–5, higher = more suspicious) ────────
    (
        CASE WHEN COALESCE(vendors_sharing_bank, 0)    >= 2    THEN 2 ELSE 0 END +
        CASE WHEN COALESCE(vendors_sharing_ein_root, 0) >= 2   THEN 1 ELSE 0 END +
        CASE WHEN COALESCE(v_wire_ratio, 0)             >= 0.9  THEN 1 ELSE 0 END +
        CASE WHEN COALESCE(is_high_risk_country, FALSE)         THEN 1 ELSE 0 END +
        CASE WHEN COALESCE(days_vendor_to_first_invoice, 999) <= 30 THEN 1 ELSE 0 END
    )                                                   AS shell_suspicion_score,

    CASE
        WHEN vendors_sharing_bank >= 2
          OR vendors_sharing_ein_root >= 3
          OR (v_wire_ratio >= 0.9 AND is_high_risk_country)
        THEN TRUE
        ELSE FALSE
    END                                                 AS flag_shell_vendor

FROM joined
