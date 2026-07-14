{{ config(materialized='table') }}

WITH base AS (

SELECT *

FROM {{ ref('stg_invoices') }}

),

features AS (

SELECT

---------------------------------------------------------------------
-- Original Invoice Information
---------------------------------------------------------------------

invoice_id,

vendor_id,

vendor_name,

vendor_category,

bank_account_id,

invoice_date,

submission_timestamp,

payment_date,

due_date,

amount,

quantity,

unit_price,

discount,

tax_amount,

currency,

payment_terms,

payment_method,

invoice_status,

department,

requester,

approver,

purchase_order_id,

contract_id,

description,

days_to_payment,

days_past_due,

invoice_year,

invoice_month,

invoice_day,

invoice_weekday,

has_discount,

has_tax,

---------------------------------------------------------------------
-- Vendor Behaviour Features
---------------------------------------------------------------------

COUNT(*)

OVER(

PARTITION BY vendor_id

)

AS vendor_invoice_count,

AVG(amount)

OVER(

PARTITION BY vendor_id

)

AS vendor_avg_amount,

STDDEV(amount)

OVER(

PARTITION BY vendor_id

)

AS vendor_std_amount,

MIN(amount)

OVER(

PARTITION BY vendor_id

)

AS vendor_min_amount,

MAX(amount)

OVER(

PARTITION BY vendor_id

)

AS vendor_max_amount,

---------------------------------------------------------------------
-- Historical Features
---------------------------------------------------------------------

ROW_NUMBER()

OVER(

PARTITION BY vendor_id

ORDER BY invoice_date

)

AS vendor_invoice_number,

LAG(amount)

OVER(

PARTITION BY vendor_id

ORDER BY invoice_date

)

AS previous_invoice_amount,

LAG(invoice_date)

OVER(

PARTITION BY vendor_id

ORDER BY invoice_date

)

AS previous_invoice_date,

LAG(days_to_payment)

OVER(

PARTITION BY vendor_id

ORDER BY invoice_date

)

AS previous_payment_delay,

---------------------------------------------------------------------
-- Rolling Window Features
---------------------------------------------------------------------

AVG(amount)

OVER(

PARTITION BY vendor_id

ORDER BY invoice_date

ROWS BETWEEN 4 PRECEDING
AND CURRENT ROW

)

AS rolling_avg_amount,

STDDEV(amount)

OVER(

PARTITION BY vendor_id

ORDER BY invoice_date

ROWS BETWEEN 4 PRECEDING
AND CURRENT ROW

)

AS rolling_std_amount,

SUM(amount)

OVER(

PARTITION BY vendor_id

ORDER BY invoice_date

ROWS BETWEEN 4 PRECEDING
AND CURRENT ROW

)

AS rolling_total_amount,

AVG(days_to_payment)

OVER(

PARTITION BY vendor_id

ORDER BY invoice_date

ROWS BETWEEN 4 PRECEDING
AND CURRENT ROW

)

AS rolling_avg_days_to_payment,

---------------------------------------------------------------------
-- Running Totals
---------------------------------------------------------------------

SUM(amount)

OVER(

PARTITION BY vendor_id

ORDER BY invoice_date

)

AS cumulative_vendor_spend,

---------------------------------------------------------------------
-- Bank Account Features
---------------------------------------------------------------------

COUNT(*)

OVER(

PARTITION BY bank_account_id

)

AS bank_account_vendor_count,

---------------------------------------------------------------------
-- Round Number Indicators
---------------------------------------------------------------------

CASE

WHEN MOD(amount,100)=0

THEN 1

ELSE 0

END

AS is_round_100,

CASE

WHEN MOD(amount,1000)=0

THEN 1

ELSE 0

END

AS is_round_1000

FROM base

)

SELECT

*,

---------------------------------------------------------------------
-- Time Since Previous Invoice
---------------------------------------------------------------------

DATE_DIFF(

'day',

previous_invoice_date,

invoice_date

)

AS days_since_previous_invoice,

---------------------------------------------------------------------
-- Vendor Amount Z Score
---------------------------------------------------------------------

CASE

WHEN vendor_std_amount IS NULL

OR vendor_std_amount=0

THEN 0

ELSE

(amount-vendor_avg_amount)

/vendor_std_amount

END

AS vendor_amount_zscore,

---------------------------------------------------------------------
-- Amount Threshold Distance
---------------------------------------------------------------------

CASE

WHEN amount<=50000

THEN 50000-amount

ELSE NULL

END

AS amount_to_threshold_distance,

---------------------------------------------------------------------
-- Submission Time Features
---------------------------------------------------------------------

EXTRACT(

HOUR

FROM submission_timestamp

)

AS submission_hour,

EXTRACT(

DOW

FROM submission_timestamp

)

AS submission_day_of_week

FROM features