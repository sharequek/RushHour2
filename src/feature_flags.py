"""
Robust feature flag system for rental listings.

This module handles the conversion of listing_features data into boolean flags
and coordinates with LLM enrichment for maximum accuracy.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Set, Tuple

from asyncpg import Pool

from .utils import load_config

log = logging.getLogger(__name__)


class FeatureFlagMapper:
    """
    Robust mapper that converts listing_features into boolean flags.
    
    Uses comprehensive feature name matching with fuzzy logic for maximum accuracy.
    """
    
    # Comprehensive feature mapping with multiple variations
    FEATURE_MAPPINGS = {
        # Building amenities
        'has_elevator': [
            'elevator', 'elevators'
        ],
        'has_doorman': [
            'doorman', 'doorman (full-time)', 'doorman (part-time)', 
            'doorman (virtual)', 'doorman (full-time, virtual)', 
            'doorman (part-time, virtual)', 'full-time doorman', 
            'part-time doorman', 'virtual doorman'
        ],
        'has_concierge': [
            'concierge', 'concierge service', 'concierge services'
        ],
        'has_gym': [
            'gym', 'fitness center', 'fitness room', 'fitness facility',
            'exercise room', 'workout room', 'health club'
        ],
        'has_roof_deck': [
            'roof deck', 'rooftop deck', 'roof terrace', 'rooftop terrace',
            'rooftop access', 'roof access', 'rooftop amenities'
        ],
        'has_pool': [
            'swimming pool', 'pool', 'pools', 'indoor pool', 'outdoor pool'
        ],
        'has_garden': [
            'garden', 'gardens', 'community garden', 'landscaped garden',
            'garden area', 'outdoor garden'
        ],
        'has_courtyard': [
            'courtyard', 'courtyards', 'interior courtyard', 'private courtyard'
        ],
        'has_bike_room': [
            'bike room', 'bicycle room', 'bike storage', 'bicycle storage',
            'bike area', 'cycling room'
        ],
        'has_live_in_super': [
            'live-in super', 'live in super', 'resident super', 
            'superintendent', 'live-in superintendent', 'on-site super'
        ],
        
        # Unit features
        'has_dishwasher': [
            'dishwasher', 'dish washer', 'built-in dishwasher'
        ],
        'has_washer_dryer': [
            'washer/dryer', 'washer dryer', 'washer and dryer',
            'laundry in unit', 'in-unit laundry', 'w/d', 'wd',
            'washing machine', 'dryer'
        ],
        'has_hardwood_floors': [
            'hardwood floors', 'hardwood flooring', 'wood floors',
            'wooden floors', 'hardwood', 'parquet floors'
        ],
        'has_central_air': [
            'central air', 'central air conditioning', 'central a/c',
            'central hvac', 'forced air', 'ducted air'
        ],
        'has_private_outdoor': [
            'private outdoor space', 'private outdoor', 'outdoor space',
            'private terrace', 'private balcony', 'private garden',
            'private patio', 'private deck'
        ],
        'has_balcony': [
            'balcony', 'balconies', 'private outdoor space (balcony)',
            'juliet balcony', 'french balcony'
        ],
        'has_terrace': [
            'terrace', 'terraces', 'private outdoor space (terrace)',
            'private terrace', 'outdoor terrace'
        ],
        'has_storage': [
            'storage space', 'storage', 'storage room', 'storage area',
            'storage space (locker/cage)', 'storage space (cold storage)',
            'locker', 'cage storage', 'basement storage', 'closet space'
        ],
        'has_wheelchair_access': [
            'wheelchair access', 'wheelchair accessible', 'ada compliant',
            'handicap accessible', 'disabled access', 'accessibility'
        ],
        'has_laundry_in_building': [
            'laundry in building', 'building laundry', 'laundry room',
            'laundry facility', 'communal laundry', 'shared laundry'
        ],
        'has_laundry_in_unit': [
            'laundry in unit', 'in-unit laundry', 'unit laundry',
            'private laundry', 'washer/dryer in unit'
        ],
        'has_fireplace': [
            'fireplace', 'fireplaces', 'wood burning fireplace',
            'gas fireplace', 'electric fireplace', 'decorative fireplace'
        ],
        'has_stainless_steel_appliances': [
            'stainless steel appliances', 'stainless steel appliance',
            'stainless appliances', 'stainless steel', 'ss appliances'
        ],
        'has_oversized_windows': [
            'oversized windows', 'oversized window', 'large windows',
            'huge windows', 'expansive windows', 'floor to ceiling windows',
            'floor-to-ceiling windows', 'floor to ceiling window',
            'floor-to-ceiling window'
        ],
        'has_high_ceilings': [
            'high ceilings', 'high ceiling', 'soaring ceilings',
            'soaring ceiling', '10-foot ceilings', '10 foot ceilings',
            'tall ceilings', 'vaulted ceilings'
        ],
        'has_natural_light': [
            'natural light', 'abundant natural light', 'abundance of natural light',
            'natural light in living area', 'light-filled interiors',
            'light-filled layouts', 'airy interiors'
        ],
        'is_new_construction': [
            'new construction', 'newly constructed', 'new development',
            'newly renovated', 'renovated', 'new building'
        ],
        
        # Parking & transportation
        'has_parking': [
            'parking', 'parking space', 'parking spot', 'car parking'
        ],
        'has_garage_parking': [
            'parking (garage)', 'garage parking', 'parking garage',
            'indoor parking', 'covered parking', 'parking (assigned, garage)'
        ],
        'has_valet_parking': [
            'parking (valet)', 'valet parking', 'parking (garage, valet)',
            'valet service', 'attended parking'
        ],
        
        # Policies
        'pets_allowed': [
            'pets allowed', 'pet friendly', 'dogs allowed', 'cats allowed',
            'pets ok', 'pets welcome', 'pets allowed (cats and dogs allowed)'
        ],
        'guarantors_accepted': [
            'guarantors accepted', 'guarantor ok', 'guarantor accepted',
            'co-signer accepted', 'guarantor welcome'
        ],
        'is_smoke_free': [
            'smoke-free', 'smoke free', 'no smoking', 'non-smoking',
            'smoking prohibited'
        ],
        'allows_subletting': [
            'subletting allowed', 'subletting ok', 'sublet allowed',
            'subletting permitted', 'sublet ok'
        ],
        
        # Financial (these will be primarily LLM-extracted)
        'has_broker_fee': [
            'broker fee', 'brokers fee', 'broker\'s fee', 'brokerage fee'
        ],
        'has_application_fee': [
            'application fee', 'app fee', 'application cost'
        ],
        'no_fee': [
            'no fee', 'no broker fee', 'fee free', 'zero fee',
            'no brokerage fee'
        ],
        'is_rent_stabilized': [
            'rent stabilized', 'rent stabilization', 'stabilized rent',
            'rent controlled', 'rent control'
        ]
    }
    
    @classmethod
    def normalize_feature_name(cls, name: str, sublabel: Optional[str] = None) -> str:
        """Normalize feature name for consistent matching."""
        normalized = name.lower().strip()
        if sublabel:
            sublabel_clean = sublabel.lower().strip()
            normalized = f"{normalized} ({sublabel_clean})"
        return normalized
    
    @classmethod
    def map_features_to_flags(cls, features: List[Tuple[str, Optional[str]]]) -> Dict[str, bool]:
        """
        Map a list of (name, sublabel) features to boolean flags.
        
        Returns a dict of flag_name -> True for detected features.
        Only returns True values for maximum accuracy.
        """
        flags = {}
        
        # Normalize all input features
        normalized_features = set()
        for name, sublabel in features:
            normalized = cls.normalize_feature_name(name, sublabel)
            normalized_features.add(normalized)
        
        # Check each flag mapping
        for flag_name, feature_patterns in cls.FEATURE_MAPPINGS.items():
            flag_detected = False
            
            for pattern in feature_patterns:
                pattern_lower = pattern.lower().strip()
                
                # Exact match
                if pattern_lower in normalized_features:
                    flag_detected = True
                    break
                
                # Partial match for complex patterns
                for normalized_feature in normalized_features:
                    if pattern_lower in normalized_feature or normalized_feature in pattern_lower:
                        # Additional validation for partial matches
                        if cls._validate_partial_match(pattern_lower, normalized_feature):
                            flag_detected = True
                            break
                
                if flag_detected:
                    break
            
            if flag_detected:
                flags[flag_name] = True
        
        return flags
    
    @classmethod
    def _validate_partial_match(cls, pattern: str, feature: str) -> bool:
        """
        Validate partial matches to avoid false positives.
        
        Returns True only if the match is semantically valid.
        """
        # Avoid false positives for short patterns
        if len(pattern) < 4:
            return pattern == feature
        
        # Special cases for common false positives
        false_positive_pairs = [
            ('pool', 'carpool'),
            ('garden', 'kindergarten'),
            ('storage', 'cold storage')  # This should match
        ]
        
        for false_pattern, false_feature in false_positive_pairs:
            if pattern == false_pattern and false_feature in feature:
                if false_pattern != 'storage':  # storage is actually valid
                    return False
        
        # For most cases, partial match is valid if one contains the other
        return True


async def populate_feature_flags_from_scraped_data(pool: Pool) -> None:
    """
    Populate boolean flags from existing listing_features data.
    
    This is the first phase - high accuracy from structured data.
    """
    start_time = time.time()
    
    log.info({"event": "feature_flags_population_start", "source": "listing_features"})
    
    async with pool.acquire() as conn:
        # Get listings with their features that haven't been processed yet (new and updated)
        query = """
        SELECT l.id, 
               ARRAY_AGG(lf.name) as feature_names,
               ARRAY_AGG(lf.sublabel) as feature_sublabels
        FROM listings l
        LEFT JOIN listing_features lf ON l.id = lf.listing_id
        WHERE l.feature_flags_populated_at IS NULL
        GROUP BY l.id
        ORDER BY l.id
        """
        
        rows = await conn.fetch(query)
        
        if not rows:
            log.info({"event": "feature_flags_population_complete", "reason": "no_listings_to_process", "note": "all_listings_already_processed_during_scraping"})
            return
        
        log.info({"event": "feature_flags_population_processing", "listings_count": len(rows)})
        
        updated_count = 0
        flags_set_count = 0
        
        for row in rows:
            listing_id = row['id']
            feature_names = row['feature_names'] or []
            feature_sublabels = row['feature_sublabels'] or []
            
            # Combine names and sublabels
            features = []
            for i, name in enumerate(feature_names):
                if name:  # Skip None values
                    sublabel = feature_sublabels[i] if i < len(feature_sublabels) else None
                    features.append((name, sublabel))
            
            # Map features to boolean flags
            flags = FeatureFlagMapper.map_features_to_flags(features)
            
            if flags:
                # Build dynamic UPDATE query
                flag_updates = []
                params = []
                param_idx = 1
                
                for flag_name, flag_value in flags.items():
                    flag_updates.append(f"{flag_name} = ${param_idx}")
                    params.append(flag_value)
                    param_idx += 1
                
                # Add timestamp and listing_id
                flag_updates.append(f"feature_flags_populated_at = now()")
                # Don't add "now()" as a parameter, it's a SQL function
                params.append(listing_id)
                
                update_query = f"""
                UPDATE listings 
                SET {', '.join(flag_updates)}
                WHERE id = ${param_idx}
                """
                
                await conn.execute(update_query, *params)
                flags_set_count += len(flags)
            else:
                # Mark as processed even if no flags were set
                await conn.execute(
                    "UPDATE listings SET feature_flags_populated_at = now() WHERE id = $1",
                    listing_id
                )
            
            updated_count += 1
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        log.info({
            "event": "feature_flags_population_complete",
            "source": "listing_features",
            "listings_processed": updated_count,
            "total_flags_set": flags_set_count,
            "avg_flags_per_listing": round(flags_set_count / max(1, updated_count), 1),
            "ms": elapsed_ms
        })


async def populate_feature_flags_for_listing(pool: Pool, listing_id: int, features: List[Tuple[str, Optional[str]]]) -> None:
    """
    Populate feature flags for a single listing (used during scraping).
    
    Args:
        pool: Database connection pool
        listing_id: ID of the listing to update
        features: List of (name, sublabel) tuples from listing_features
    """
    flags = FeatureFlagMapper.map_features_to_flags(features)
    
    if not flags:
        # Mark as processed even if no flags were set
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE listings SET feature_flags_populated_at = now() WHERE id = $1",
                listing_id
            )
        return
    
    # Build dynamic UPDATE query
    flag_updates = []
    params = []
    param_idx = 1
    
    for flag_name, flag_value in flags.items():
        flag_updates.append(f"{flag_name} = ${param_idx}")
        params.append(flag_value)
        param_idx += 1
    
    # Add timestamp and listing_id
    flag_updates.append(f"feature_flags_populated_at = now()")
    params.append(listing_id)
    
    update_query = f"""
    UPDATE listings 
    SET {', '.join(flag_updates)}
    WHERE id = ${param_idx}
    """
    
    async with pool.acquire() as conn:
        await conn.execute(update_query, *params)
    
    log.debug({
        "event": "feature_flags_populated",
        "listing_id": listing_id,
        "flags_set": len(flags),
        "flags": list(flags.keys())
    })


async def populate_feature_flags_for_listing_transaction(conn, listing_id: int, features: List[Tuple[str, Optional[str]]]) -> None:
    """
    Populate feature flags for a single listing within an existing transaction.
    
    Args:
        conn: Database connection (within transaction)
        listing_id: ID of the listing to update
        features: List of (name, sublabel) tuples from listing_features
    """
    flags = FeatureFlagMapper.map_features_to_flags(features)
    
    if not flags:
        # Mark as processed even if no flags were set
        await conn.execute(
            "UPDATE listings SET feature_flags_populated_at = now() WHERE id = $1",
            listing_id
        )
        return
    
    # Build dynamic UPDATE query
    flag_updates = []
    params = []
    param_idx = 1
    
    for flag_name, flag_value in flags.items():
        flag_updates.append(f"{flag_name} = ${param_idx}")
        params.append(flag_value)
        param_idx += 1
    
    # Add timestamp and listing_id
    flag_updates.append(f"feature_flags_populated_at = now()")
    params.append(listing_id)
    
    update_query = f"""
    UPDATE listings 
    SET {', '.join(flag_updates)}
    WHERE id = ${param_idx}
    """
    
    await conn.execute(update_query, *params)
    
    log.debug({
        "event": "feature_flags_populated",
        "listing_id": listing_id,
        "flags_set": len(flags),
        "flags": list(flags.keys())
    })


async def get_feature_flag_coverage_stats(pool: Pool) -> Dict[str, any]:
    """Get statistics on feature flag coverage for monitoring."""
    async with pool.acquire() as conn:
        # Get overall stats
        overall_stats = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_listings,
                COUNT(CASE WHEN feature_flags_populated_at IS NOT NULL THEN 1 END) as flags_populated,
                COUNT(CASE WHEN llm_enrichment_completed_at IS NOT NULL THEN 1 END) as llm_enriched
            FROM listings
        """)
        
        # Get flag-specific stats
        flag_columns = [
            'has_elevator', 'has_doorman', 'has_concierge', 'has_gym', 'has_roof_deck',
            'has_pool', 'has_garden', 'has_courtyard', 'has_bike_room', 'has_live_in_super',
            'has_dishwasher', 'has_washer_dryer', 'has_hardwood_floors', 'has_central_air',
            'has_private_outdoor', 'has_balcony', 'has_terrace', 'has_storage',
            'has_wheelchair_access', 'has_laundry_in_building', 'has_laundry_in_unit',
            'has_fireplace', 'has_parking', 'has_garage_parking', 'has_valet_parking',
            'pets_allowed', 'guarantors_accepted', 'is_smoke_free', 'allows_subletting',
            'has_broker_fee', 'has_application_fee', 'no_fee'
        ]
        
        flag_stats = {}
        for flag in flag_columns:
            stats = await conn.fetchrow(f"""
                SELECT 
                    COUNT(CASE WHEN {flag} = true THEN 1 END) as true_count,
                    COUNT(CASE WHEN {flag} = false THEN 1 END) as false_count,
                    COUNT(CASE WHEN {flag} IS NULL THEN 1 END) as null_count
                FROM listings
            """)
            flag_stats[flag] = {
                'true': stats['true_count'],
                'false': stats['false_count'],
                'null': stats['null_count']
            }
        
        return {
            'overall': dict(overall_stats),
            'flags': flag_stats
        }
