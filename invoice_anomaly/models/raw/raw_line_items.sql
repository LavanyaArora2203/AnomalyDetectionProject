{{ config(materialized='view', schema='raw') }}
SELECT * FROM main.line_items
