-- raw.raw_invoices
-- Pass-through view over the seeded DuckDB table.
-- No transformations here — this is the source-of-truth snapshot.
{{ config(materialized='view', schema='raw') }}

SELECT * FROM main.raw_invoices
