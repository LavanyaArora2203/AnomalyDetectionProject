import duckdb
con = duckdb.connect("invoice_anomaly/dev.duckdb")

tables = con.execute("SHOW TABLES").fetchall()
print("Tables in DuckDB:")
for t in tables:
    print(f"  {t[0]} → {con.execute(f'SELECT COUNT(*) FROM {t[0]}').fetchone()[0]} rows")

con.close()