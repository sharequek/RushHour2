#!/usr/bin/env python3
"""
Script to fix remaining listings with null bedroom data.
This addresses the 124 listings that still have beds = null after the initial cleanup.
"""

import asyncio
import asyncpg
import os
import sys
import re
from typing import List, Tuple, Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import DATABASE_URL


async def analyze_null_bedroom_listings(pool: asyncpg.Pool) -> Dict[str, Any]:
    """Analyze listings with null bedroom data to understand the scope."""
    async with pool.acquire() as conn:
        # Get overall statistics
        stats = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_listings,
                COUNT(beds) as beds_not_null,
                COUNT(CASE WHEN beds = 0 THEN 1 END) as studios,
                COUNT(CASE WHEN beds IS NULL THEN 1 END) as beds_null,
                COUNT(CASE WHEN beds > 0 THEN 1 END) as multi_bed
            FROM listings
        """)
        
        # Find listings with null beds
        null_beds = await conn.fetch("""
            SELECT id, url, combined_address, beds, baths, description, 
                   llm_enrichment_occurred, llm_enriched_fields
            FROM listings 
            WHERE beds IS NULL
            ORDER BY id
            LIMIT 20
        """)
        
        return {
            'stats': dict(stats),
            'null_beds_listings': [dict(row) for row in null_beds]
        }


async def extract_bedroom_from_description(description: str) -> int:
    """Extract bedroom count from listing description using pattern matching."""
    if not description:
        return None
    
    desc_lower = description.lower()
    
    # Studio indicators
    if any(term in desc_lower for term in ['studio', 'efficiency', 'alcove', '0 bed', '0 bedroom']):
        return 0
    
    # Bedroom patterns
    patterns = [
        (r'(\d+)\s*bed', 1),           # "1 bed", "2 bed", etc.
        (r'(\d+)\s*bedroom', 1),       # "1 bedroom", "2 bedroom", etc.
        (r'(\d+)\s*br', 1),            # "1br", "2br", etc.
        (r'(\d+)\s*bedroom', 1),       # "1 bedroom", "2 bedroom", etc.
    ]
    
    for pattern, group_idx in patterns:
        match = re.search(pattern, desc_lower)
        if match:
            try:
                beds = int(match.group(group_idx))
                # Sanity check: reasonable bedroom count
                if 0 <= beds <= 10:
                    return beds
            except (ValueError, IndexError):
                continue
    
    return None


async def fix_null_bedrooms_with_descriptions(pool: asyncpg.Pool) -> int:
    """Fix listings with null bedrooms by analyzing their descriptions."""
    async with pool.acquire() as conn:
        # Find listings with null beds but descriptions
        null_beds_with_desc = await conn.fetch("""
            SELECT id, url, combined_address, beds, baths, description
            FROM listings 
            WHERE beds IS NULL 
              AND description IS NOT NULL 
              AND description != ''
            ORDER BY id
        """)
        
        fixed_count = 0
        
        for listing in null_beds_with_desc:
            listing_id = listing['id']
            description = listing['description']
            
            # Try to extract bedroom info
            beds = extract_bedroom_from_description(description)
            
            if beds is not None:
                await conn.execute("""
                    UPDATE listings 
                    SET beds = $1,
                        llm_enrichment_completed_at = NULL,
                        llm_enrichment_occurred = false
                    WHERE id = $2
                """, beds, listing_id)
                
                fixed_count += 1
                print(f"✅ Fixed bedrooms from description: {listing['combined_address']} -> {beds} beds (ID: {listing_id})")
        
        return fixed_count


async def fix_null_bedrooms_with_url_patterns(pool: asyncpg.Pool) -> int:
    """Fix listings with null bedrooms by analyzing URL patterns and other clues."""
    async with pool.acquire() as conn:
        # Find listings with null beds
        null_beds = await conn.fetch("""
            SELECT id, url, combined_address, beds, baths, description
            FROM listings 
            WHERE beds IS NULL
            ORDER BY id
        """)
        
        fixed_count = 0
        
        for listing in null_beds:
            listing_id = listing['id']
            url = listing['url']
            combined_address = listing['combined_address']
            
            # Try to infer from URL patterns (StreetEasy often has bedroom info in URLs)
            beds = None
            
            # Common StreetEasy URL patterns
            if 'studio' in url.lower() or '0-bed' in url.lower():
                beds = 0
            elif '1-bed' in url.lower() or '1-bedroom' in url.lower():
                beds = 1
            elif '2-bed' in url.lower() or '2-bedroom' in url.lower():
                beds = 2
            elif '3-bed' in url.lower() or '3-bedroom' in url.lower():
                beds = 3
            
            # If still no match, try address patterns
            if beds is None:
                address_lower = combined_address.lower()
                if any(term in address_lower for term in ['studio', 'efficiency']):
                    beds = 0
                elif any(term in address_lower for term in ['1 bed', '1 bedroom']):
                    beds = 1
                elif any(term in address_lower for term in ['2 bed', '2 bedroom']):
                    beds = 2
            
            if beds is not None:
                await conn.execute("""
                    UPDATE listings 
                    SET beds = $1,
                        llm_enrichment_completed_at = NULL,
                        llm_enrichment_occurred = false
                    WHERE id = $2
                """, beds, listing_id)
                
                fixed_count += 1
                print(f"✅ Fixed bedrooms from URL/address: {combined_address} -> {beds} beds (ID: {listing_id})")
        
        return fixed_count


async def set_default_bedrooms_for_remaining(pool: asyncpg.Pool) -> int:
    """Set default bedroom values for remaining listings with null beds."""
    async with pool.acquire() as conn:
        # Find remaining listings with null beds
        remaining = await conn.fetch("""
            SELECT id, url, combined_address, beds, baths
            FROM listings 
            WHERE beds IS NULL
            ORDER BY id
        """)
        
        if not remaining:
            return 0
        
        print(f"\n⚠️  Found {len(remaining)} listings that still have null bedrooms.")
        print("These listings don't have enough information to determine bedroom count.")
        print("Options:")
        print("1. Set all to 1 bedroom (most common)")
        print("2. Set all to studio (0 bedrooms)")
        print("3. Leave as null (manual review needed)")
        
        choice = input("\nEnter choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            default_beds = 1
            reason = "1 bedroom (most common default)"
        elif choice == "2":
            default_beds = 0
            reason = "studio (0 bedrooms)"
        else:
            print("❌ No changes made. Manual review needed.")
            return 0
        
        # Apply the default
        await conn.execute("""
            UPDATE listings 
            SET beds = $1,
                llm_enrichment_completed_at = NULL,
                llm_enrichment_occurred = false
            WHERE beds IS NULL
        """, default_beds)
        
        print(f"✅ Set {len(remaining)} listings to {reason}")
        return len(remaining)


async def main():
    """Main function to fix remaining null bedroom data."""
    print("🔍 Analyzing remaining null bedroom data...")
    
    # Connect to database
    pool = await asyncpg.create_pool(DATABASE_URL)
    
    try:
        # Analyze current state
        analysis = await analyze_null_bedroom_listings(pool)
        
        print("\n📊 Current Data Analysis:")
        print(f"Total listings: {analysis['stats']['total_listings']}")
        print(f"Listings with beds: {analysis['stats']['beds_not_null']}")
        print(f"Studios (beds = 0): {analysis['stats']['studios']}")
        print(f"Missing beds: {analysis['stats']['beds_null']}")
        print(f"Multi-bedroom: {analysis['stats']['multi_bed']}")
        
        if analysis['null_beds_listings']:
            print(f"\n⚠️  Sample of listings with null bedrooms:")
            for listing in analysis['null_beds_listings'][:5]:
                print(f"  - ID {listing['id']}: {listing['combined_address']}")
                if listing['description']:
                    desc_preview = listing['description'][:100] + "..." if len(listing['description']) > 100 else listing['description']
                    print(f"    Description: {desc_preview}")
            if len(analysis['null_beds_listings']) > 5:
                print(f"  ... and {len(analysis['null_beds_listings']) - 5} more")
        
        # Ask user if they want to proceed
        print("\n" + "="*60)
        response = input("Do you want to proceed with fixing the remaining null bedrooms? (y/N): ").strip().lower()
        
        if response != 'y':
            print("❌ Fix cancelled.")
            return
        
        print("\n🔧 Starting comprehensive bedroom fix...")
        
        # Step 1: Fix from descriptions
        print("\n1. Fixing bedrooms from descriptions...")
        fixed_from_desc = await fix_null_bedrooms_with_descriptions(pool)
        print(f"   Fixed {fixed_from_desc} listings from descriptions")
        
        # Step 2: Fix from URL/address patterns
        print("\n2. Fixing bedrooms from URL/address patterns...")
        fixed_from_url = await fix_null_bedrooms_with_url_patterns(pool)
        print(f"   Fixed {fixed_from_url} listings from URL/address patterns")
        
        # Step 3: Handle remaining nulls
        print("\n3. Handling remaining null bedrooms...")
        fixed_default = await set_default_bedrooms_for_remaining(pool)
        print(f"   Fixed {fixed_default} listings with default values")
        
        # Final analysis
        print("\n4. Final analysis...")
        final_analysis = await analyze_null_bedroom_listings(pool)
        
        print(f"\n🎉 Bedroom fix completed!")
        print(f"Final stats:")
        print(f"  Studios: {final_analysis['stats']['studios']}")
        print(f"  Missing beds: {final_analysis['stats']['beds_null']}")
        
        if final_analysis['stats']['beds_null'] == 0:
            print("✅ All listings now have bedroom data!")
        else:
            print(f"⚠️  {final_analysis['stats']['beds_null']} listings still have null bedrooms")
            print("   These may need manual review or additional data sources")
        
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())

