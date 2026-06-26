-- mart.invoice_ml_features
-- ML-ready feature table built on top of fact_invoice_features.
-- Adds statistical, temporal and behavioral features for anomaly detection.

{{ config(materialized='table', schema='mart') }}

{% set approval_threshold = 5000 %}

WITH vendor_stats AS (

    SELECT
        vendor_id,
        AVG(amount) AS vendor_avg_amount,
        STDDEV_SAMP(amount) AS vendor_std_amount

    FROM {{ ref('fact_invoice_features') }}
    GROUP BY vendor_id

),

base AS (

    SELECT
        f.*,

        --------------------------------------------------------------------
        -- Vendor Amount Z-Score
        --------------------------------------------------------------------
        CASE
            WHEN vs.vendor_std_amount IS NULL
                 OR vs.vendor_std_amount = 0
            THEN 0

            ELSE
                (f.amount - vs.vendor_avg_amount)
                / vs.vendor_std_amount
        END AS vendor_amount_zscore,

        --------------------------------------------------------------------
        -- Rolling Average Days to Payment
        --------------------------------------------------------------------
        AVG(f.days_to_payment)
        OVER (
            PARTITION BY f.vendor_id
            ORDER BY f.issue_date
            ROWS BETWEEN 10 PRECEDING AND CURRENT ROW
        ) AS rolling_avg_days_to_payment,

        --------------------------------------------------------------------
        -- Distance from Approval Threshold
        --------------------------------------------------------------------
        CASE
            WHEN f.amount <= {{ approval_threshold }}
            THEN {{ approval_threshold }} - f.amount
            ELSE 0
        END AS amount_to_threshold,

        --------------------------------------------------------------------
        -- Submission Hour
        --------------------------------------------------------------------
        EXTRACT(HOUR FROM f.created_at)
            AS submission_hour,

        --------------------------------------------------------------------
        -- Submission Day of Week
        --------------------------------------------------------------------
        EXTRACT(DOW FROM f.created_at)
            AS submission_dow,

        --------------------------------------------------------------------
        -- Vendors Sharing Same Bank Account
        --------------------------------------------------------------------
        COUNT(DISTINCT f.vendor_id)
        OVER (
            PARTITION BY f.bank_account
        ) AS bank_account_vendor_count

    FROM {{ ref('fact_invoice_features') }} f

    LEFT JOIN vendor_stats vs
        ON f.vendor_id = vs.vendor_id

)

SELECT *
FROM base;