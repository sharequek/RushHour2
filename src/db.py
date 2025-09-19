import os
import sys
import json
from typing import Any, Dict, Tuple
from decimal import Decimal

import asyncpg
import asyncio

try:
    from .constants import DEFAULT_DATABASE_URL
except ImportError:
    from constants import DEFAULT_DATABASE_URL

DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


SCHEMA_SQL = """
-- Main listings table with logically organized columns
CREATE TABLE IF NOT EXISTS listings (
  -- Primary key and core identifiers
  id SERIAL PRIMARY KEY,
  url TEXT NOT NULL,
  combined_address TEXT UNIQUE NOT NULL,
  listing_name TEXT,
  building_address TEXT,
  
  -- Pricing information
  price INTEGER,
  effective_price INTEGER,
  price_per_sqft NUMERIC,
  free_months INTEGER,
  lease_term_months INTEGER,
  
  -- Property details
  sqft INTEGER,
  rooms NUMERIC,
  beds NUMERIC,
  baths NUMERIC,
  
  -- Availability and market info
  availability_date DATE,
  availability_now BOOLEAN,
  days_on_market INTEGER,
  last_change_amount INTEGER,
  last_change_pct NUMERIC,
  last_change_date DATE,
  
  -- Description and location
  description TEXT,
  google_maps TEXT,
  latitude DOUBLE PRECISION,
  longitude DOUBLE PRECISION,
  
  -- Commute information
  commute_distance_heur_km NUMERIC(6,2),
  commute_duration_heur_min INTEGER,
  commute_distance_google_km NUMERIC(6,2),
  commute_duration_google_min INTEGER,
  commute_calculated_at TIMESTAMPTZ,
  
  -- LLM-extracted fee information
  broker_fee_amount NUMERIC(8,2),
  broker_fee_pct NUMERIC(5,2),
  application_fee NUMERIC(8,2),
  
  -- Building amenities (boolean flags)
  has_elevator BOOLEAN DEFAULT FALSE,
  has_doorman BOOLEAN DEFAULT FALSE,
  has_concierge BOOLEAN DEFAULT FALSE,
  has_gym BOOLEAN DEFAULT FALSE,
  has_roof_deck BOOLEAN DEFAULT FALSE,
  has_pool BOOLEAN DEFAULT FALSE,
  has_garden BOOLEAN DEFAULT FALSE,
  has_courtyard BOOLEAN DEFAULT FALSE,
  has_bike_room BOOLEAN DEFAULT FALSE,
  has_live_in_super BOOLEAN DEFAULT FALSE,
  
  -- Unit features (boolean flags)
  has_dishwasher BOOLEAN DEFAULT FALSE,
  has_washer_dryer BOOLEAN DEFAULT FALSE,
  has_hardwood_floors BOOLEAN DEFAULT FALSE,
  has_central_air BOOLEAN DEFAULT FALSE,
  has_private_outdoor BOOLEAN DEFAULT FALSE,
  has_balcony BOOLEAN DEFAULT FALSE,
  has_terrace BOOLEAN DEFAULT FALSE,
  has_storage BOOLEAN DEFAULT FALSE,
  has_wheelchair_access BOOLEAN DEFAULT FALSE,
  has_laundry_in_building BOOLEAN DEFAULT FALSE,
  has_laundry_in_unit BOOLEAN DEFAULT FALSE,
  has_fireplace BOOLEAN DEFAULT FALSE,
  has_stainless_steel_appliances BOOLEAN DEFAULT FALSE,
  has_oversized_windows BOOLEAN DEFAULT FALSE,
  has_high_ceilings BOOLEAN DEFAULT FALSE,
  has_natural_light BOOLEAN DEFAULT FALSE,
  is_new_construction BOOLEAN DEFAULT FALSE,
  
  -- Parking & transportation (boolean flags)
  has_parking BOOLEAN DEFAULT FALSE,
  has_garage_parking BOOLEAN DEFAULT FALSE,
  has_valet_parking BOOLEAN DEFAULT FALSE,
  
  -- Policies (boolean flags)
  pets_allowed BOOLEAN DEFAULT FALSE,
  guarantors_accepted BOOLEAN DEFAULT FALSE,
  is_smoke_free BOOLEAN DEFAULT FALSE,
  allows_subletting BOOLEAN DEFAULT FALSE,
  
  -- Financial policies (boolean flags)
  has_broker_fee BOOLEAN DEFAULT FALSE,
  has_application_fee BOOLEAN DEFAULT FALSE,
  no_fee BOOLEAN DEFAULT FALSE,
  is_rent_stabilized BOOLEAN DEFAULT FALSE,
  
  -- Processing metadata
  feature_flags_populated_at TIMESTAMPTZ,
  llm_enrichment_completed_at TIMESTAMPTZ,
  llm_enrichment_occurred BOOLEAN DEFAULT FALSE,
  llm_enriched_fields JSONB,
  ocr_sqft_completed_at TIMESTAMPTZ,
  ocr_sqft_extracted INTEGER,
  ocr_sqft_source_text TEXT,
  ocr_sqft_source_photo_id INTEGER,
  ocr_sqft_confidence NUMERIC(3,2),
  ocr_sqft_engine TEXT,
  
  -- Processing timing tracking
  scrape_duration_ms INTEGER,
  commute_duration_ms INTEGER,
  llm_enrichment_duration_ms INTEGER,
  ocr_sqft_duration_ms INTEGER,
  
  -- Raw data and timestamps
  raw JSONB NOT NULL DEFAULT '{}'::jsonb,
  scraped_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  
  -- Scoring (for future use)
  score INTEGER
);

-- Photos table (one-to-many)
CREATE TABLE IF NOT EXISTS listing_photos (
  id SERIAL PRIMARY KEY,
  listing_id INTEGER NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
  url TEXT NOT NULL,
  position INTEGER NOT NULL,
  type TEXT,
  added_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (listing_id, url),
  UNIQUE (listing_id, position)
);

-- Features table (policies, home features, amenities, etc.)
CREATE TABLE IF NOT EXISTS listing_features (
  id SERIAL PRIMARY KEY,
  listing_id INTEGER NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  sublabel TEXT,
  added_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for optimal performance
-- Core lookup indexes
CREATE INDEX IF NOT EXISTS idx_listings_combined_address ON listings (combined_address);
CREATE INDEX IF NOT EXISTS idx_listings_scraped_at ON listings (scraped_at);
CREATE INDEX IF NOT EXISTS idx_listings_price ON listings (price) WHERE price IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_listings_beds_baths ON listings (beds, baths) WHERE beds IS NOT NULL OR baths IS NOT NULL;

-- Location-based indexes
CREATE INDEX IF NOT EXISTS idx_listings_location ON listings (latitude, longitude) WHERE latitude IS NOT NULL AND longitude IS NOT NULL;

-- Commute indexes
CREATE INDEX IF NOT EXISTS idx_listings_commute_heuristic ON listings (commute_calculated_at) WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_listings_commute_google ON listings (commute_distance_google_km, commute_duration_google_min) WHERE latitude IS NOT NULL AND longitude IS NOT NULL;

-- LLM enrichment indexes
CREATE INDEX IF NOT EXISTS idx_listings_llm_enrichment ON listings (llm_enrichment_completed_at, llm_enrichment_occurred);
CREATE INDEX IF NOT EXISTS idx_listings_llm_pending ON listings (id) WHERE llm_enrichment_completed_at IS NULL;

-- Feature flags indexes (for common queries)
CREATE INDEX IF NOT EXISTS idx_listings_elevator_doorman ON listings (has_elevator, has_doorman) WHERE has_elevator IS NOT NULL OR has_doorman IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_listings_gym_pool ON listings (has_gym, has_pool) WHERE has_gym IS NOT NULL OR has_pool IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_listings_pets_smoke_free ON listings (pets_allowed, is_smoke_free) WHERE pets_allowed IS NOT NULL OR is_smoke_free IS NOT NULL;

-- Score index (for future ranking queries)
CREATE INDEX IF NOT EXISTS idx_listings_score ON listings (score) WHERE score IS NOT NULL;

-- Foreign key indexes
CREATE INDEX IF NOT EXISTS idx_listing_photos_listing_id ON listing_photos (listing_id);
CREATE INDEX IF NOT EXISTS idx_listing_features_listing_id ON listing_features (listing_id);

-- Ensure uniqueness per listing on (name, COALESCE(sublabel, ''))
CREATE UNIQUE INDEX IF NOT EXISTS ux_listing_features_unique
  ON listing_features (listing_id, name, COALESCE(sublabel, ''));
"""


UPSERT_SQL = """
INSERT INTO listings (
  url, combined_address, listing_name, building_address, price,
  effective_price, price_per_sqft, free_months, lease_term_months, sqft,
  rooms, beds, baths, availability_date, availability_now, days_on_market,
  last_change_amount, last_change_pct, last_change_date, description,
  google_maps, latitude, longitude, raw
) VALUES (
  $1,$2,$3,$4,$5,
  $6,$7,$8,$9,$10,
  $11,$12,$13,$14,$15,$16,
  $17,$18,$19,$20,$21,
  $22,$23,$24
)
ON CONFLICT (combined_address) DO UPDATE SET
  -- Update all fields when monitored fields change
  url = EXCLUDED.url,
  listing_name = EXCLUDED.listing_name,
  building_address = EXCLUDED.building_address,
  price = EXCLUDED.price,
  effective_price = EXCLUDED.price,
  price_per_sqft = EXCLUDED.price_per_sqft,
  free_months = EXCLUDED.free_months,
  lease_term_months = EXCLUDED.lease_term_months,
  sqft = EXCLUDED.sqft,
  rooms = EXCLUDED.rooms,
  beds = EXCLUDED.beds,
  baths = EXCLUDED.baths,
  availability_date = EXCLUDED.availability_date,
  availability_now = EXCLUDED.availability_now,
  days_on_market = EXCLUDED.days_on_market,
  last_change_amount = EXCLUDED.last_change_amount,
  last_change_pct = EXCLUDED.last_change_pct,
  last_change_date = EXCLUDED.last_change_date,
  description = EXCLUDED.description,
  google_maps = EXCLUDED.google_maps,
  latitude = EXCLUDED.latitude,
  longitude = EXCLUDED.longitude,
  raw = EXCLUDED.raw,
  scraped_at = now(),
  -- Only reset enrichment timestamps when fields that could affect LLM enrichment change
  -- (description changes or when LLM-extracted fields are updated)
  feature_flags_populated_at = CASE 
    WHEN listings.description IS DISTINCT FROM EXCLUDED.description THEN NULL
    ELSE listings.feature_flags_populated_at
  END,
  llm_enrichment_completed_at = CASE 
    WHEN listings.description IS DISTINCT FROM EXCLUDED.description OR
         listings.sqft IS DISTINCT FROM EXCLUDED.sqft OR
         listings.beds IS DISTINCT FROM EXCLUDED.beds OR
         listings.baths IS DISTINCT FROM EXCLUDED.baths OR
         listings.broker_fee_amount IS DISTINCT FROM EXCLUDED.broker_fee_amount OR
         listings.broker_fee_pct IS DISTINCT FROM EXCLUDED.broker_fee_pct OR
         listings.application_fee IS DISTINCT FROM EXCLUDED.application_fee
    THEN NULL
    ELSE listings.llm_enrichment_completed_at
  END,
  llm_enrichment_occurred = CASE 
    WHEN listings.description IS DISTINCT FROM EXCLUDED.description OR
         listings.sqft IS DISTINCT FROM EXCLUDED.sqft OR
         listings.beds IS DISTINCT FROM EXCLUDED.beds OR
         listings.baths IS DISTINCT FROM EXCLUDED.baths OR
         listings.broker_fee_amount IS DISTINCT FROM EXCLUDED.broker_fee_amount OR
         listings.broker_fee_pct IS DISTINCT FROM EXCLUDED.broker_fee_pct OR
         listings.application_fee IS DISTINCT FROM EXCLUDED.application_fee
    THEN false
    ELSE listings.llm_enrichment_occurred
  END,
  llm_enriched_fields = CASE 
    WHEN listings.description IS DISTINCT FROM EXCLUDED.description OR
         listings.sqft IS DISTINCT FROM EXCLUDED.sqft OR
         listings.beds IS DISTINCT FROM EXCLUDED.beds OR
         listings.baths IS DISTINCT FROM EXCLUDED.baths OR
         listings.broker_fee_amount IS DISTINCT FROM EXCLUDED.broker_fee_amount OR
         listings.broker_fee_pct IS DISTINCT FROM EXCLUDED.broker_fee_pct OR
         listings.application_fee IS DISTINCT FROM EXCLUDED.application_fee
    THEN NULL
    ELSE listings.llm_enriched_fields
  END,
  -- Reset OCR timestamps when listing is updated (so OCR can re-run)
  ocr_sqft_completed_at = NULL,
  ocr_sqft_extracted = NULL,
  ocr_sqft_source_photo_id = NULL,
  ocr_sqft_source_text = NULL,
  ocr_sqft_engine = NULL,
  -- Reset timing fields when listing is updated (so timing can be re-measured)
  scrape_duration_ms = NULL,
  commute_duration_ms = NULL,
  llm_enrichment_duration_ms = NULL,
  ocr_sqft_duration_ms = NULL
WHERE 
  -- Price changes
  listings.price != EXCLUDED.price OR
  listings.effective_price != EXCLUDED.effective_price OR
  -- Availability changes
  listings.availability_now IS DISTINCT FROM EXCLUDED.availability_now OR
  listings.availability_date IS DISTINCT FROM EXCLUDED.availability_date OR
  -- Last change data
  listings.last_change_amount IS DISTINCT FROM EXCLUDED.last_change_amount OR
  listings.last_change_pct IS DISTINCT FROM EXCLUDED.last_change_pct OR
  listings.last_change_date IS DISTINCT FROM EXCLUDED.last_change_date OR
  -- Lease terms
  listings.lease_term_months IS DISTINCT FROM EXCLUDED.lease_term_months OR
  listings.free_months IS DISTINCT FROM EXCLUDED.free_months
RETURNING id;
"""


async def make_pool() -> asyncpg.Pool:
    return await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)


async def upsert_listing(pool: asyncpg.Pool, row: Dict[str, Any]) -> Tuple[int, bool, bool]:
    """
    Upsert a listing and return (listing_id, was_updated, was_new).
    
    Returns:
        Tuple[int, bool, bool]: (listing_id, True if updated, False if skipped due to no changes, True if new listing)
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Check if listing exists before upsert (within transaction to prevent race conditions)
            existing_id = await conn.fetchval(
                "SELECT id FROM listings WHERE combined_address = $1",
                row["combined_address"]
            )
            
            # Try the upsert with WHERE clause
            listing_id = await conn.fetchval(
                UPSERT_SQL,
                row["url"],
                row["combined_address"],
                row.get("listing_name"),
                row.get("building_address"),
                row.get("price"),
                row.get("effective_price"),
                row.get("price_per_sqft"),
                row.get("free_months"),
                row.get("lease_term_months"),
                row.get("sqft"),
                row.get("rooms"),
                row.get("beds"),
                row.get("baths"),
                row.get("availability_date"),
                row.get("availability_now"),
                row.get("days_on_market"),
                row.get("last_change_amount"),
                Decimal(str(row.get("last_change_pct"))) if row.get("last_change_pct") is not None else None,
                row.get("last_change_date"),
                row.get("description"),
                    row.get("google_maps"),
                    row.get("latitude"),
                    row.get("longitude"),
                json.dumps(row.get("raw", {}), ensure_ascii=False),
            )
            
            # Determine what happened
            if existing_id is None:
                # This was a new listing
                was_new = True
                was_updated = False  # New listings are not "updated"
            elif listing_id is None:
                # Listing exists but no changes detected (WHERE clause prevented update)
                listing_id = existing_id
                was_new = False
                was_updated = False
            else:
                # Listing exists and was updated
                was_new = False
                was_updated = True
            
            return int(listing_id), was_updated, was_new


async def upsert_listing_transaction(conn, row: Dict[str, Any]) -> Tuple[int, bool, bool]:
    """
    Upsert a listing within an existing transaction and return (listing_id, was_updated, was_new).
    
    Returns:
        Tuple[int, bool, bool]: (listing_id, True if updated, False if skipped due to no changes, True if new listing)
    """
    # Check if listing exists before upsert
    existing_id = await conn.fetchval(
        "SELECT id FROM listings WHERE combined_address = $1",
        row["combined_address"]
    )
    
    # Try the upsert with WHERE clause
    listing_id = await conn.fetchval(
        UPSERT_SQL,
        row["url"],
        row["combined_address"],
        row.get("listing_name"),
        row.get("building_address"),
        row.get("price"),
        row.get("effective_price"),
        row.get("price_per_sqft"),
        row.get("free_months"),
        row.get("lease_term_months"),
        row.get("sqft"),
        row.get("rooms"),
        row.get("beds"),
        row.get("baths"),
        row.get("availability_date"),
        row.get("availability_now"),
        row.get("days_on_market"),
        row.get("last_change_amount"),
        Decimal(str(row.get("last_change_pct"))) if row.get("last_change_pct") is not None else None,
        row.get("last_change_date"),
        row.get("description"),
            row.get("google_maps"),
            row.get("latitude"),
            row.get("longitude"),
        json.dumps(row.get("raw", {}), ensure_ascii=False),
    )
    
    # Determine what happened
    if existing_id is None:
        # This was a new listing
        was_new = True
        was_updated = False  # New listings are not "updated"
    elif listing_id is None:
        # Listing exists but no changes detected (WHERE clause prevented update)
        listing_id = existing_id
        was_new = False
        was_updated = False
    else:
        # Listing exists and was updated
        was_new = False
        was_updated = True
    
    return int(listing_id), was_updated, was_new


async def update_scrape_duration(pool: asyncpg.Pool, listing_id: int, duration_ms: int) -> None:
    """Update the scrape duration for a listing."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE listings SET scrape_duration_ms = $1 WHERE id = $2",
            duration_ms, listing_id
        )


async def update_commute_duration(pool: asyncpg.Pool, listing_id: int, duration_ms: int) -> None:
    """Update the commute duration for a listing."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE listings SET commute_duration_ms = $1 WHERE id = $2",
            duration_ms, listing_id
        )


async def update_llm_enrichment_duration(pool: asyncpg.Pool, listing_id: int, duration_ms: int) -> None:
    """Update the LLM enrichment duration for a listing."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE listings SET llm_enrichment_duration_ms = $1 WHERE id = $2",
            duration_ms, listing_id
        )


async def update_ocr_sqft_duration(pool: asyncpg.Pool, listing_id: int, duration_ms: int) -> None:
    """Update the OCR square footage duration for a listing."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE listings SET ocr_sqft_duration_ms = $1 WHERE id = $2",
            duration_ms, listing_id
        )


async def replace_listing_photos_transaction(conn, listing_id: int, photos: list[tuple[str, str]]) -> None:
    """Replace listing photos within an existing transaction."""
    await conn.execute("DELETE FROM listing_photos WHERE listing_id = $1", listing_id)
    if photos:
        await conn.executemany(
            "INSERT INTO listing_photos (listing_id, url, position, type) VALUES ($1, $2, $3, $4)",
            [(listing_id, url, idx + 1, type) for idx, (url, type) in enumerate(photos)]
        )


async def replace_listing_features_transaction(conn, listing_id: int, features: list[tuple[str, str | None]]) -> None:
    """Replace listing features within an existing transaction."""
    await conn.execute("DELETE FROM listing_features WHERE listing_id = $1", listing_id)
    if features:
        # Deduplicate incoming features to avoid unique conflicts
        seen: set[tuple[str, str | None]] = set()
        rows: list[tuple[int, str, str | None]] = []
        for name, sublabel in features:
            key = (name or "", sublabel or None)
            if key in seen:
                continue
            seen.add(key)
            rows.append((listing_id, name, sublabel))
        if rows:
            await conn.executemany(
                "INSERT INTO listing_features (listing_id, name, sublabel) VALUES ($1, $2, $3)",
                rows
            )


async def clear_listings(pool: asyncpg.Pool) -> None:
  """Clear all listings and related data from the database."""
  async with pool.acquire() as conn:
    # Truncate both with CASCADE to satisfy FK
    await conn.execute("TRUNCATE TABLE listing_photos, listing_features, listings RESTART IDENTITY CASCADE;")

async def drop_and_recreate_schema(pool: asyncpg.Pool) -> None:
  """Drop and recreate all tables (use with caution - clears all data)."""
  async with pool.acquire() as conn:
    # Drop existing tables if they exist (for clean recreation)
    await conn.execute("DROP TABLE IF EXISTS listing_photos CASCADE;")
    await conn.execute("DROP TABLE IF EXISTS listing_features CASCADE;")
    await conn.execute("DROP TABLE IF EXISTS listings CASCADE;")
    # Recreate schema
    await conn.execute(SCHEMA_SQL)


async def replace_listing_photos(pool: asyncpg.Pool, listing_id: int, photos: list[tuple[str, str]]) -> None:
  async with pool.acquire() as conn:
    async with conn.transaction():
      await conn.execute("DELETE FROM listing_photos WHERE listing_id = $1", listing_id)
      if photos:
            await conn.executemany(
        "INSERT INTO listing_photos (listing_id, url, position, type) VALUES ($1, $2, $3, $4)",
        [(listing_id, url, idx + 1, type) for idx, (url, type) in enumerate(photos)]
    )


async def replace_listing_features(pool: asyncpg.Pool, listing_id: int, features: list[tuple[str, str | None]]) -> None:
  async with pool.acquire() as conn:
    async with conn.transaction():
      await conn.execute("DELETE FROM listing_features WHERE listing_id = $1", listing_id)
      if features:
        # Deduplicate incoming features to avoid unique conflicts
        seen: set[tuple[str, str | None]] = set()
        rows: list[tuple[int, str, str | None]] = []
        for name, sublabel in features:
          key = (name or "", sublabel or None)
          if key in seen:
            continue
          seen.add(key)
          rows.append((listing_id, name, sublabel))
        if rows:
          await conn.executemany(
            "INSERT INTO listing_features (listing_id, name, sublabel) VALUES ($1, $2, $3)",
            rows
          )


def _parse_args(argv: list[str]) -> str:
  args = set(argv[1:])
  if "--clear-listings" in args:
    return "clear-listings"
  return ""


if __name__ == "__main__":
  action = _parse_args(sys.argv)
  if action == "clear-listings":
    async def _run():
      pool = await make_pool()
      await ensure_schema(pool)
      await clear_listings(pool)
      await pool.close()
      print("{\"event\": \"db_cleared\", \"tables\": [\"listing_photos\", \"listing_features\", \"listings\"]}")
    asyncio.run(_run())
