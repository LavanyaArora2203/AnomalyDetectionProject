import duckdb
import glob
import os

# Connect to your existing DuckDB file
con = duckdb.connect("invoice_anomaly/dev.duckdb")  # update filename

# Find all CSV files in the project folder
csv_files = glob.glob("dataset/*.csv")  # or use "data/*.csv" if in a subfolder

for csv_path in csv_files:
    table_name = os.path.splitext(os.path.basename(csv_path))[0]  # filename without .csv
    print(f"Loading {csv_path} → table: {table_name}")
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" AS
        SELECT * FROM read_csv_auto('{csv_path}')
    """)

con.close()
print("Done! All CSVs loaded.")