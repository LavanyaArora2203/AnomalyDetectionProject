-- staging.stg_invoices
-- Cast types, normalise nulls, derive simple flags, clean anomaly_type.
{{ config(materialized='view', schema='staging') }}

WITH source AS (
    SELECT * FROM {{ ref('raw_invoices') }}
)

SELECT
    id                                              AS invoice_id,
    vendor_id,
    buyer_id,
    invoice_number,

    -- amounts
    CAST(amount AS DOUBLE)                          AS amount,
    UPPER(TRIM(currency))                           AS currency,

    -- dates
    CAST(issue_date    AS DATE)                     AS issue_date,
    CAST(due_date      AS DATE)                     AS due_date,
    CAST(payment_date  AS DATE)                     AS payment_date,
    CAST(created_at    AS TIMESTAMP)                AS created_at,

    -- derived date parts (used heavily in window functions downstream)
    EXTRACT(YEAR  FROM CAST(issue_date AS DATE))    AS issue_year,
    EXTRACT(MONTH FROM CAST(issue_date AS DATE))    AS issue_month,
    EXTRACT(DOW   FROM CAST(issue_date AS DATE))    AS issue_dow,   -- 0=Sun, 6=Sat
    EXTRACT(DAY   FROM CAST(issue_date AS DATE))    AS issue_dom,

    -- payment timing
    CASE
        WHEN payment_date IS NOT NULL AND due_date IS NOT NULL
        THEN DATEDIFF('day', CAST(due_date AS DATE), CAST(payment_date AS DATE))
    END                                             AS days_past_due,   -- negative = early

    CASE
        WHEN payment_date IS NOT NULL AND issue_date IS NOT NULL
        THEN DATEDIFF('day', CAST(issue_date AS DATE), CAST(payment_date AS DATE))
    END                                             AS days_to_payment,

    -- status / method
    LOWER(TRIM(status))                             AS status,
    LOWER(TRIM(payment_method))                     AS payment_method,

    -- label hierarchy
    CASE
        WHEN anomaly_type IS NULL OR anomaly_type = ''  THEN 'clean'
        ELSE anomaly_type
    END                                             AS anomaly_type,

    CASE
        WHEN anomaly_type LIKE 'fraud_%'            THEN TRUE
        ELSE FALSE
    END                                             AS is_fraud,

    CASE
        WHEN anomaly_type = 'fraud_duplicate'               THEN 'duplicate'
        WHEN anomaly_type LIKE 'fraud_threshold_%'          THEN 'threshold_gaming'
        WHEN anomaly_type LIKE 'fraud_shell_%'              THEN 'shell_vendor'
        WHEN anomaly_type IN ('round_amount','weekend_payment','potential_duplicate')
                                                            THEN 'soft_anomaly'
        ELSE 'clean'
    END                                             AS fraud_category,

    -- simple flags
    (EXTRACT(DOW FROM CAST(issue_date AS DATE)) IN (0, 6))      AS is_weekend_issue,
    (EXTRACT(DOW FROM CAST(payment_date AS DATE)) IN (0, 6))    AS is_weekend_payment,
    (EXTRACT(DAY FROM CAST(issue_date AS DATE)) <= 5)           AS is_month_start,
    (EXTRACT(DAY FROM CAST(issue_date AS DATE)) >= 25)          AS is_month_end,
    (amount = ROUND(amount / 1000) * 1000)                      AS is_round_1k,
    (amount = ROUND(amount / 500)  * 500)                       AS is_round_500

FROM source
