{{ config(materialized='view') }}

SELECT

    i.invoice_id,

    i.vendor_id,

    v.vendor_name,

    v.vendor_category,

    v.bank_account_id,

    i.invoice_date,

    i.submission_timestamp,

    i.payment_date,

    i.due_date,

    i.amount,

    i.quantity,

    i.unit_price,

    i.tax_amount,

    i.discount,

    i.currency,

    i.payment_terms,

    i.payment_method,

    i.invoice_status,

    i.department,

    i.requester,

    i.approver,

    i.purchase_order_id,

    i.contract_id,

    i.description,

    --------------------------------------------------------------------
    -- Simple Date Features
    --------------------------------------------------------------------

    DATE_DIFF('day', i.invoice_date, i.payment_date)
        AS days_to_payment,

    DATE_DIFF('day', i.due_date, i.payment_date)
        AS days_past_due,

    EXTRACT(YEAR FROM i.invoice_date)
        AS invoice_year,

    EXTRACT(MONTH FROM i.invoice_date)
        AS invoice_month,

    EXTRACT(DAY FROM i.invoice_date)
        AS invoice_day,

    EXTRACT(DOW FROM i.invoice_date)
        AS invoice_weekday,

    --------------------------------------------------------------------
    -- Simple Flags
    --------------------------------------------------------------------

    CASE

        WHEN i.discount > 0

        THEN 1

        ELSE 0

    END AS has_discount,

    CASE

        WHEN i.tax_amount > 0

        THEN 1

        ELSE 0

    END AS has_tax

FROM {{ ref('raw_invoices') }} i

LEFT JOIN {{ ref('stg_vendors') }} v

ON i.vendor_id = v.vendor_id