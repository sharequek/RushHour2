#!/usr/bin/env python3
"""
Script to fix existing bedroom/bathroom data inconsistencies in the database.
This script identifies and corrects data that may have been corrupted by the previous LLM enrichment bug.
"""

import asyncio
import asyncpg
import os
import sys
from typing import List, Tuple, Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import DATABASE_URL


async def analyze_bedroom_data(pool: asyncpg.Pool) -> Dict[str, Any]:
    """Analyze current bedroom/bathroom data to identify inconsistencies."""
    async with pool.acquire() as conn:
        # Get overall statistics
        stats = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_listings,
                COUNT(beds) as beds_not_null,
                COUNT(CASE WHEN beds = 0 THEN 1 END) as studios,
                COUNT(CASE WHEN beds IS NULL THEN 1 END) as beds_null,
                COUNT(CASE WHEN beds > 0 THEN 1 END) as multi_bed,
                COUNT(baths) as baths_not_null,
                COUNT(CASE WHEN baths IS NULL THEN 1 END) as baths_null
            FROM listings
        """)
        
        # Find listings with inconsistent data
        inconsistent = await conn.fetch("""
            SELECT 
                id, url, combined_address, beds, baths, 
                llm_enrichment_occurred, llm_enriched_fields
            FROM listings 
            WHERE (beds IS NULL AND baths IS NOT NULL)  -- Missing beds but has baths
               OR (beds = 0 AND baths IS NULL)         -- Studio with missing baths
               OR (beds IS NULL AND baths IS NULL)     -- Both missing
            ORDER BY id
            LIMIT 20
        """)
        
        return {
            'stats': dict(stats),
            'inconsistent_listings': [dict(row) for row in inconsistent]
        }


async def fix_studio_inconsistencies(pool: asyncpg.Pool) -> Tuple[int, int]:
    """Fix studio apartments that have inconsistent data."""
    async with pool.acquire() as conn:
        # Find studios that might have been corrupted
        corrupted_studios = await conn.fetch("""
            SELECT id, url, combined_address, beds, baths, description
            FROM listings 
            WHERE (beds IS NULL OR beds = 0) 
              AND description IS NOT NULL
              AND (
                  LOWER(description) LIKE '%studio%' 
                  OR LOWER(description) LIKE '%0 bed%'
                  OR LOWER(description) LIKE '%0 bedroom%'
                  OR LOWER(description) LIKE '%efficiency%'
                  OR LOWER(description) LIKE '%alcove%'
              )
            ORDER BY id
        """)
        
        fixed_count = 0
        updated_count = 0
        
        for listing in corrupted_studios:
            listing_id = listing['id']
            current_beds = listing['beds']
            current_baths = listing['baths']
            
            # Determine what the correct values should be
            should_be_studio = (
                current_beds is None or 
                (current_beds == 0 and current_baths is None)
            )
            
            if should_be_studio:
                # Update to be a proper studio
                await conn.execute("""
                    UPDATE listings 
                    SET beds = 0.0, 
                        baths = COALESCE(baths, 1.0),
                        llm_enrichment_completed_at = NULL,
                        llm_enrichment_occurred = false
                    WHERE id = $1
                """, listing_id)
                
                fixed_count += 1
                print(f"✅ Fixed studio: {listing['combined_address']} (ID: {listing_id})")
            
            updated_count += 1
        
        return fixed_count, updated_count


async def fix_missing_bedroom_data(pool: asyncpg.Pool) -> int:
    """Fix listings with missing bedroom data by analyzing descriptions."""
    async with pool.acquire() as conn:
        # Find listings with missing bedroom data
        missing_bedrooms = await conn.fetch("""
            SELECT id, url, combined_address, beds, baths, description
            FROM listings 
            WHERE beds IS NULL 
              AND description IS NOT NULL
              AND description != ''
            ORDER BY id
            LIMIT 50  -- Process in batches
        """)
        
        fixed_count = 0
        
        for listing in missing_bedrooms:
            listing_id = listing['id']
            description = listing['description'].lower()
            
            # Try to extract bedroom info from description
            beds = None
            if 'studio' in description or '0 bed' in description:
                beds = 0.0
            elif '1 bed' in description or '1 bedroom' in description:
                beds = 1.0
            elif '2 bed' in description or '2 bedroom' in description:
                beds = 2.0
            elif '3 bed' in description or '3 bedroom' in description:
                beds = 3.0
            
            if beds is not None:
                await conn.execute("""
                    UPDATE listings 
                    SET beds = $1,
                        llm_enrichment_completed_at = NULL,
                        llm_enrichment_occurred = false
                    WHERE id = $2
                """, beds, listing_id)
                
                fixed_count += 1
                print(f"✅ Fixed bedrooms: {listing['combined_address']} -> {beds} beds (ID: {listing_id})")
        
        return fixed_count


async def reset_llm_enrichment_for_corrupted_data(pool: asyncpg.Pool) -> int:
    """Reset LLM enrichment for listings that may have corrupted data."""
    async with pool.acquire() as conn:
        # Find listings that might have been corrupted by LLM
        corrupted = await conn.fetch("""
            SELECT id, url, combined_address, beds, baths
            FROM listings 
            WHERE llm_enrichment_occurred = true
              AND (
                  (beds IS NULL AND baths IS NOT NULL)  -- Missing beds but has baths
                  OR (beds = 0 AND baths IS NULL)       -- Studio with missing baths
                  OR (beds IS NULL AND baths IS NULL)   -- Both missing
              )
            ORDER BY id
        """)
        
        reset_count = 0
        
        for listing in corrupted:
            listing_id = listing['id']
            
            # Reset LLM enrichment so it can re-run with the fixed logic
            await conn.execute("""
                UPDATE listings 
                SET llm_enrichment_completed_at = NULL,
                    llm_enrichment_occurred = false,
                    llm_enriched_fields = NULL
                WHERE id = $1
            """, listing_id)
            
            reset_count += 1
            print(f"🔄 Reset LLM enrichment: {listing['combined_address']} (ID: {listing_id})")
        
        return reset_count


async def main():
    """Main function to run the data cleanup."""
    print("🔍 Analyzing bedroom/bathroom data inconsistencies...")
    
    # Connect to database
    pool = await asyncpg.create_pool(DATABASE_URL)
    
    try:
        # Analyze current state
        analysis = await analyze_bedroom_data(pool)
        
        print("\n📊 Current Data Analysis:")
        print(f"Total listings: {analysis['stats']['total_listings']}")
        print(f"Listings with beds: {analysis['stats']['beds_not_null']}")
        print(f"Studios (beds = 0): {analysis['stats']['studios']}")
        print(f"Missing beds: {analysis['stats']['beds_null']}")
        print(f"Multi-bedroom: {analysis['stats']['multi_bed']}")
        print(f"Listings with baths: {analysis['stats']['baths_not_null']}")
        print(f"Missing baths: {analysis['stats']['baths_null']}")
        
        if analysis['inconsistent_listings']:
            print(f"\n⚠️  Found {len(analysis['inconsistent_listings'])} listings with inconsistent data:")
            for listing in analysis['inconsistent_listings'][:5]:  # Show first 5
                print(f"  - ID {listing['id']}: {listing['combined_address']} (beds: {listing['beds']}, baths: {listing['baths']})")
            if len(analysis['inconsistent_listings']) > 5:
                print(f"  ... and {len(analysis['inconsistent_listings']) - 5} more")
        
        # Ask user if they want to proceed with fixes
        print("\n" + "="*60)
        response = input("Do you want to proceed with fixing the data inconsistencies? (y/N): ").strip().lower()
        
        if response != 'y':
            print("❌ Data cleanup cancelled.")
            return
        
        print("\n🔧 Starting data cleanup...")
        
        # Fix studio inconsistencies
        print("\n1. Fixing studio apartment inconsistencies...")
        fixed_studios, total_studios = await fix_studio_inconsistencies(pool)
        print(f"   Fixed {fixed_studios}/{total_studios} studios")
        
        # Fix missing bedroom data
        print("\n2. Fixing missing bedroom data...")
        fixed_bedrooms = await fix_missing_bedroom_data(pool)
        print(f"   Fixed {fixed_bedrooms} listings with missing bedroom data")
        
        # Reset LLM enrichment for corrupted data
        print("\n3. Resetting LLM enrichment for corrupted data...")
        reset_count = await reset_llm_enrichment_for_corrupted_data(pool)
        print(f"   Reset LLM enrichment for {reset_count} listings")
        
        # Final analysis
        print("\n4. Final analysis...")
        final_analysis = await analyze_bedroom_data(pool)
        
        print(f"\n🎉 Data cleanup completed!")
        print(f"Final stats:")
        print(f"  Studios: {final_analysis['stats']['studios']}")
        print(f"  Missing beds: {final_analysis['stats']['beds_null']}")
        print(f"  Missing baths: {final_analysis['stats']['baths_null']}")
        
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
