#!/usr/bin/env python3
"""
Script to run OCR extraction and scoring on existing listings in the database.
This is useful when OCR processing was interrupted or needs to be completed.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import with proper module path
from src.db import make_pool, ensure_schema
from src.ocr_extractor import process_all_listings_sqft_ocr
from src.scoring import score_all_listings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("ocr_scoring")

async def main():
    """Run OCR extraction and scoring on all listings."""
    start_time = time.time()
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    log.info({"event": "starting_ocr_and_scoring"})
    
    # Create database connection
    pool = await make_pool()
    await ensure_schema(pool)
    
    try:
        # Run OCR extraction on all listings
        log.info({"event": "starting_ocr_extraction"})
        await process_all_listings_sqft_ocr(pool, config.get("ocr", {}))
        log.info({"event": "ocr_extraction_complete"})
        
        # Run scoring on all listings
        log.info({"event": "starting_scoring"})
        await score_all_listings(pool, config)
        log.info({"event": "scoring_complete"})
        
        # Log final statistics
        async with pool.acquire() as conn:
            total_listings = await conn.fetchval('SELECT COUNT(*) FROM listings')
            ocr_count = await conn.fetchval('SELECT COUNT(*) FROM listings WHERE ocr_sqft_extracted IS NOT NULL')
            scored_count = await conn.fetchval('SELECT COUNT(*) FROM listings WHERE score IS NOT NULL')
            
            log.info({
                "event": "processing_complete",
                "total_listings": total_listings,
                "ocr_extracted": ocr_count,
                "scored_listings": scored_count,
                "total_time_seconds": round(time.time() - start_time, 2),
                "total_time_minutes": round((time.time() - start_time) / 60, 2)
            })
            
            print(f"\n✅ Processing Complete!")
            print(f"   📊 Total listings: {total_listings}")
            print(f"   📐 OCR extracted: {ocr_count}")
            print(f"   🎯 Scored listings: {scored_count}")
            print(f"   ⏱️  Total time: {round((time.time() - start_time) / 60, 2)} minutes")
        
    except Exception as e:
        log.error({"event": "processing_error", "error": str(e)})
        raise
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(main())
