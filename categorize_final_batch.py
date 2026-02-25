import duckdb
from pathlib import Path

markets_dir = Path('/root/prediction-market-analysis/data/polymarket/markets')
valid_files = [str(f) for f in markets_dir.glob('*.parquet') if not f.name.startswith('._')]
read_con = duckdb.connect()
write_con = duckdb.connect('/root/combarbbot/polymarket.db')

def categorize(q):
    q = q.lower()
    if any(w in q for w in ['trump', 'biden', 'election', 'president', 'congress', 'senate', 'vote']):
        return ('politics', 'us-congress') if 'congress' in q or 'senate' in q else ('politics', 'us-election')
    if any(w in q for w in ['nba', 'basketball']): return 'sports', 'nba'
    if any(w in q for w in ['nfl', 'football', 'super bowl']): return 'sports', 'nfl'
    if any(w in q for w in ['mlb', 'baseball']): return 'sports', 'mlb'
    if any(w in q for w in ['soccer', 'fifa']): return 'sports', 'soccer'
    if any(w in q for w in ['ufc', 'mma', 'boxing']): return 'sports', 'mma'
    if any(w in q for w in ['bitcoin', 'btc']): return 'crypto', 'bitcoin'
    if any(w in q for w in ['ethereum', 'eth']): return 'crypto', 'ethereum'
    if any(w in q for w in ['solana', 'xrp', 'doge']): return 'crypto', 'altcoins'
    if any(w in q for w in ['temperature', 'weather']): return 'weather', 'temperature'
    if any(w in q for w in ['fed', 'interest rate']): return 'finance', 'fed'
    return 'other', 'misc'

total = 0
for s in range(1000000, 1350000, 5000):
    df = read_con.execute(f"SELECT id, question, slug FROM read_parquet({valid_files}) WHERE CAST(id AS INTEGER) >= {s} AND CAST(id AS INTEGER) < {s+5000}").fetchall()
    for r in df:
        c, sc = categorize((r[1] or '') + ' ' + (r[2] or ''))
        try: write_con.execute("INSERT OR REPLACE INTO market_categories VALUES (?,?,?,1.0,CURRENT_TIMESTAMP)", [r[0], c, sc])
        except: pass
    total += len(df)
    if s % 50000 == 0: print(f'{s}: total={total}')
write_con.close()
print(f'Done: {total}')
