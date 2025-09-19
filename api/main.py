import os
import asyncio
from typing import Optional, List, Dict, Any

import asyncpg
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# Prefer env var, else fall back to your constants default if present
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rushhour2")


async def make_pool() -> asyncpg.Pool:
    return await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)


app = FastAPI(title="RushHour2 API", version="0.1.0")

# Allow local dev frontends by default
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    app.state.pool = await make_pool()


@app.on_event("shutdown")
async def _shutdown() -> None:
    pool = getattr(app.state, "pool", None)
    if pool:
        await pool.close()


def _safe_int(v) -> Optional[int]:
    return int(v) if v is not None else None


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/api/scraping-status")
async def get_scraping_status() -> Dict[str, Any]:
    """Get information about the last scraping run."""
    pool: asyncpg.Pool = app.state.pool
    async with pool.acquire() as conn:
        # Get the most recent scraped_at timestamp
        last_scraped = await conn.fetchval(
            "SELECT MAX(scraped_at) FROM listings WHERE scraped_at IS NOT NULL"
        )
        
        # Get total listings count
        total_listings = await conn.fetchval("SELECT COUNT(*) FROM listings")
        
        # Get count of listings scraped in the last 24 hours
        recent_listings = await conn.fetchval(
            "SELECT COUNT(*) FROM listings WHERE scraped_at >= NOW() - INTERVAL '24 hours'"
        )
    
    return {
        "last_scraped": last_scraped.isoformat() if last_scraped else None,
        "total_listings": int(total_listings or 0),
        "recent_listings": int(recent_listings or 0),
    }


@app.get("/api/listings")
async def get_listings(
    offset: int = Query(0, ge=0),
    limit: int = Query(24, ge=1, le=100),
    sort: str = Query("score", pattern=r"^(score|price|effective_price|commute|sqft)$"),
    order: str = Query("desc", pattern=r"^(asc|desc)$"),
) -> Dict[str, Any]:
    sort_map = {
        "score": "l.score",
        "price": "COALESCE(l.effective_price, l.price)",
        "effective_price": "l.effective_price",
        "commute": "COALESCE(l.commute_duration_google_min, l.commute_duration_heur_min)",
        "sqft": "COALESCE(l.sqft, l.ocr_sqft_extracted)",
    }
    order_sql = "ASC" if order.lower() == "asc" else "DESC"
    sort_sql = sort_map.get(sort, "l.score")

    sql = f"""
        SELECT
          l.id,
          l.url,
          l.combined_address,
          l.listing_name,
          l.building_address,
          l.price,
          l.effective_price,
          l.beds,
          l.baths,
          COALESCE(l.sqft, l.ocr_sqft_extracted) AS sqft,
          l.ocr_sqft_confidence,
          COALESCE(l.commute_duration_google_min, l.commute_duration_heur_min) AS commute_min,
          l.score,
          p.url AS photo_url,
          (
            SELECT COALESCE(array_agg(lp.url ORDER BY lp.position), ARRAY[]::text[])
            FROM listing_photos lp
            WHERE lp.listing_id = l.id
          ) AS photo_urls
        FROM listings l
        LEFT JOIN LATERAL (
          SELECT url FROM listing_photos WHERE listing_id = l.id ORDER BY position LIMIT 1
        ) p ON TRUE
        WHERE TRUE
        ORDER BY {sort_sql} {order_sql} NULLS LAST, l.id {order_sql}
        OFFSET $1
        LIMIT $2
    """

    count_sql = "SELECT COUNT(*) FROM listings"

    pool: asyncpg.Pool = app.state.pool
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, offset, limit)
        total = await conn.fetchval(count_sql)

    def to_dict(r: asyncpg.Record) -> Dict[str, Any]:
        return {
            "id": int(r["id"]),
            "url": r["url"],
            "combined_address": r["combined_address"],
            "listing_name": r["listing_name"],
            "building_address": r["building_address"],
            "price": _safe_int(r["price"]),
            "effective_price": _safe_int(r["effective_price"]),
            "beds": float(r["beds"]) if r["beds"] is not None else None,
            "baths": float(r["baths"]) if r["baths"] is not None else None,
            "sqft": _safe_int(r["sqft"]),
            "ocr_sqft_confidence": float(r["ocr_sqft_confidence"]) if r["ocr_sqft_confidence"] is not None else None,
            "commute_min": _safe_int(r["commute_min"]),
            "score": _safe_int(r["score"]),
            "photo_url": r["photo_url"],
            "photo_urls": list(r["photo_urls"]) if r["photo_urls"] is not None else [],
        }

    return {
        "offset": offset,
        "limit": limit,
        "total": int(total or 0),
        "items": [to_dict(r) for r in rows],
    }


# Dev entry: uvicorn api.main:app --reload --port 8000
