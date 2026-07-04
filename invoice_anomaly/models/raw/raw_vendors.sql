{{ config(materialized='view', schema='raw') }}
SELECT * FROM main.vendors
