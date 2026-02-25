"""Simple API server for querying market categories."""

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import duckdb
from pathlib import Path
import uvicorn

app = FastAPI(title="Market Categories API")

DB_PATH = "/root/combarbbot/polymarket.db"
MARKETS_DIR = Path("/root/prediction-market-analysis/data/polymarket/markets")

def get_db():
    return duckdb.connect(DB_PATH, read_only=True)

def get_market_files():
    return [str(f) for f in MARKETS_DIR.glob("*.parquet") if not f.name.startswith("._")]

@app.get("/")
def root():
    return {"status": "ok", "service": "market-categories-api"}

@app.get("/stats")
def get_stats():
    """Get overall categorization statistics."""
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM market_categories").fetchone()[0]
        by_category = conn.execute(
            "SELECT category, COUNT(*) as cnt FROM market_categories GROUP BY category ORDER BY cnt DESC"
        ).fetchall()
        by_subcategory = conn.execute(
            "SELECT category, subcategory, COUNT(*) as cnt FROM market_categories GROUP BY category, subcategory ORDER BY cnt DESC LIMIT 20"
        ).fetchall()
    
    return {
        "total_categorized": total,
        "by_category": {r[0]: r[1] for r in by_category},
        "top_subcategories": [{"category": r[0], "subcategory": r[1], "count": r[2]} for r in by_subcategory]
    }

@app.get("/categories/{category}")
def get_by_category(
    category: str,
    subcategory: str = None,
    limit: int = Query(default=100, le=10000),
    offset: int = 0
):
    """Get market IDs by category."""
    with get_db() as conn:
        if subcategory:
            results = conn.execute(
                "SELECT market_id FROM market_categories WHERE category = ? AND subcategory = ? ORDER BY market_id LIMIT ? OFFSET ?",
                [category, subcategory, limit, offset]
            ).fetchall()
        else:
            results = conn.execute(
                "SELECT market_id FROM market_categories WHERE category = ? ORDER BY market_id LIMIT ? OFFSET ?",
                [category, limit, offset]
            ).fetchall()
    
    return {"category": category, "subcategory": subcategory, "count": len(results), "market_ids": [r[0] for r in results]}

# IMPORTANT: /markets/details must come BEFORE /markets/{market_id}
@app.get("/markets/details")
def get_market_details(
    category: str,
    subcategory: str = None,
    min_volume: float = 0,
    limit: int = Query(default=100, le=1000)
):
    """Get full market details (joined with parquet) by category."""
    with get_db() as conn:
        # Get market IDs from category
        if subcategory:
            cat_results = conn.execute(
                "SELECT market_id FROM market_categories WHERE category = ? AND subcategory = ? LIMIT ?",
                [category, subcategory, limit * 2]
            ).fetchall()
        else:
            cat_results = conn.execute(
                "SELECT market_id FROM market_categories WHERE category = ? LIMIT ?",
                [category, limit * 2]
            ).fetchall()
        
        if not cat_results:
            return {"markets": []}
        
        market_ids = [r[0] for r in cat_results]
    
    # Use separate connection for parquet query
    parquet_conn = duckdb.connect()
    placeholders = ",".join(f"'{mid}'" for mid in market_ids)
    
    # Join with parquet for full details
    market_files = get_market_files()
    query = f"""
        SELECT m.id, m.question, m.slug, COALESCE(m.volume, 0) as volume
        FROM read_parquet({market_files}) m
        WHERE m.id IN ({placeholders})
        AND COALESCE(m.volume, 0) >= {min_volume}
        ORDER BY COALESCE(m.volume, 0) DESC
        LIMIT {limit}
    """
    
    results = parquet_conn.execute(query).fetchall()
    parquet_conn.close()
    
    # Get categories for results
    result_ids = [r[0] for r in results]
    if result_ids:
        with get_db() as conn:
            placeholders = ",".join("?" * len(result_ids))
            cat_map = {
                r[0]: {"category": r[1], "subcategory": r[2]}
                for r in conn.execute(
                    f"SELECT market_id, category, subcategory FROM market_categories WHERE market_id IN ({placeholders})",
                    result_ids
                ).fetchall()
            }
    else:
        cat_map = {}
    
    markets = []
    for r in results:
        cat = cat_map.get(r[0], {})
        markets.append({
            "id": r[0],
            "question": r[1],
            "slug": r[2],
            "volume": float(r[3]),
            "category": cat.get("category"),
            "subcategory": cat.get("subcategory")
        })
    
    return {"markets": markets}

@app.get("/markets/{market_id}")
def get_market(market_id: str):
    """Get category for a single market."""
    with get_db() as conn:
        result = conn.execute(
            "SELECT market_id, category, subcategory, confidence FROM market_categories WHERE market_id = ?",
            [market_id]
        ).fetchone()
    
    if result is None:
        return JSONResponse(status_code=404, content={"error": "Market not found"})
    
    return {"market_id": result[0], "category": result[1], "subcategory": result[2], "confidence": result[3]}

@app.post("/markets/batch")
def get_markets_batch(market_ids: list[str]):
    """Get categories for multiple markets."""
    if not market_ids:
        return {"results": []}
    
    with get_db() as conn:
        placeholders = ",".join("?" * len(market_ids))
        results = conn.execute(
            f"SELECT market_id, category, subcategory, confidence FROM market_categories WHERE market_id IN ({placeholders})",
            market_ids
        ).fetchall()
    
    return {
        "results": [
            {"market_id": r[0], "category": r[1], "subcategory": r[2], "confidence": r[3]}
            for r in results
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8420)
