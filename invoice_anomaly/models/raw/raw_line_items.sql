{{ config(materialized='view', schema='raw') }}
SELECT * FROM main.raw_line_items
