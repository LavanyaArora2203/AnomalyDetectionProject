{{ config(materialized='view') }}

SELECT

    invoice_id,

    vendor_id,

    CAST(invoice_date AS DATE) AS invoice_date,

    CAST(submission_timestamp AS TIMESTAMP) AS submission_timestamp,

    CAST(payment_date AS DATE) AS payment_date,

    CAST(due_date AS DATE) AS due_date,

    CAST(amount AS DOUBLE) AS amount,

    CAST(quantity AS INTEGER) AS quantity,

    CAST(unit_price AS DOUBLE) AS unit_price,

    currency,

    requester,

    approver,

    purchase_order_id,

    contract_id,

    CAST(payment_terms AS INTEGER) AS payment_terms,

    payment_method,

    invoice_status,

    department,

    description,

    CAST(tax_amount AS DOUBLE) AS tax_amount,

    CAST(discount AS DOUBLE) AS discount

FROM {{ ref('invoices') }}