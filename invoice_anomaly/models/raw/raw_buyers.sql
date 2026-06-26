{{ config(materialized='view', schema='raw') }}
SELECT * FROM main.raw_buyers
