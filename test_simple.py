import sys
import os
os.environ['DUCKDB_NO_PROGRESS_BAR'] = '1'
sys.path.insert(0, 'src')

import duckdb

# Direct DuckDB test
print('Testing direct DuckDB query...')
conn = duckdb.connect()

# Query with small limit
result = conn.execute("""
    SELECT block_number, maker_asset_id, maker_amount 
    FROM read_parquet('/root/prediction-market-analysis/data/polymarket/trades/[!._]*.parquet')
    WHERE block_number >= 63950000 AND block_number < 63960000
    LIMIT 10
""").fetchall()

print(f'Got {len(result)} rows')
for row in result[:3]:
    print(f'  Block {row[0]}: amount={row[2]}')

print('Direct DuckDB test passed!')
