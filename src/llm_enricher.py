# Suppress HTTP request logs from httpx/ollama - MUST be at the very top
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

"""
LLM-based data enrichment for rental listings.

This module uses a local LLM (via Ollama) to extract missing information
from listing descriptions, including square footage, bed/bath counts,
and fee information.
"""

import os
import sys
import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import ollama
from asyncpg import Pool

from .constants import (
    DEFAULT_CONTEXT_LENGTH, 
    DEFAULT_BATCH_SIZE, 
    DEFAULT_NUM_BATCH,
    DEFAULT_TIMEOUT,
    DEFAULT_RESPONSE_PREVIEW_LENGTH
)
from .utils import load_config
from .db import update_llm_enrichment_duration

log = logging.getLogger(__name__)

# Global timing variables
_last_log_time = time.time()
_start_time = time.time()

def _get_timing_info() -> Dict[str, Any]:
    """Get timing information for logs."""
    global _last_log_time
    current_time = time.time()
    elapsed_since_start = current_time - _start_time
    elapsed_since_last = current_time - _last_log_time
    _last_log_time = current_time
    
    return {
        "sec": round(elapsed_since_last, 2),
        "elapsed_sec": round(elapsed_since_start, 2),
        "elapsed_min": round(elapsed_since_start / 60, 2)
    }

def _clean_json_response(response: str) -> str:
    """Clean JSON response by removing comments and fixing common issues."""
    # Remove single-line comments (// ...)
    response = re.sub(r'//.*$', '', response, flags=re.MULTILINE)
    
    # Remove multi-line comments (/* ... */)
    response = re.sub(r'/\*.*?\*/', '', response, flags=re.DOTALL)
    
    # Remove trailing commas before closing braces/brackets
    response = re.sub(r',(\s*[}\]])', r'\1', response)
    
    # Remove any leading/trailing whitespace
    response = response.strip()
    
    return response


@dataclass
class EnrichmentData:
    """Data extracted by LLM from listing description."""
    # Core property data
    sqft: Optional[int] = None
    beds: Optional[float] = None
    baths: Optional[float] = None
    broker_fee_amount: Optional[float] = None
    broker_fee_pct: Optional[float] = None
    application_fee: Optional[float] = None
    
    # Boolean feature flags (LLM fills gaps not covered by scraped data)
    # Building amenities
    has_elevator: Optional[bool] = None
    has_doorman: Optional[bool] = None
    has_concierge: Optional[bool] = None
    has_gym: Optional[bool] = None
    has_roof_deck: Optional[bool] = None
    has_pool: Optional[bool] = None
    has_garden: Optional[bool] = None
    has_courtyard: Optional[bool] = None
    has_bike_room: Optional[bool] = None
    has_live_in_super: Optional[bool] = None
    
    # Unit features
    has_dishwasher: Optional[bool] = None
    has_washer_dryer: Optional[bool] = None
    has_hardwood_floors: Optional[bool] = None
    has_central_air: Optional[bool] = None
    has_private_outdoor: Optional[bool] = None
    has_balcony: Optional[bool] = None
    has_terrace: Optional[bool] = None
    has_storage: Optional[bool] = None
    has_wheelchair_access: Optional[bool] = None
    has_laundry_in_building: Optional[bool] = None
    has_laundry_in_unit: Optional[bool] = None
    has_fireplace: Optional[bool] = None
    has_stainless_steel_appliances: Optional[bool] = None
    has_oversized_windows: Optional[bool] = None
    has_high_ceilings: Optional[bool] = None
    has_natural_light: Optional[bool] = None
    is_new_construction: Optional[bool] = None
    
    # Parking & transportation
    has_parking: Optional[bool] = None
    has_garage_parking: Optional[bool] = None
    has_valet_parking: Optional[bool] = None
    
    # Policies
    pets_allowed: Optional[bool] = None
    guarantors_accepted: Optional[bool] = None
    is_smoke_free: Optional[bool] = None
    allows_subletting: Optional[bool] = None
    
    # Financial
    has_broker_fee: Optional[bool] = None
    has_application_fee: Optional[bool] = None
    no_fee: Optional[bool] = None
    is_rent_stabilized: Optional[bool] = None
    
    # Confidence score
    confidence: Optional[float] = None


@dataclass
class BatchRequest:
    """Single LLM enrichment request."""
    listing_id: int
    description: str
    current_data: Dict[str, Any]
    missing_fields: List[str]


@dataclass
class BatchResult:
    """Result of LLM enrichment."""
    listing_id: int
    enriched: EnrichmentData
    success: bool
    missing_fields: List[str] = None  # Track what fields were missing
    error: Optional[str] = None
    duration_ms: Optional[int] = None  # Track processing time for this listing


class LLMEnricher:
    """LLM enricher using Ollama with batching and concurrency."""
    
    # Class-level constants for better performance and maintainability
    _NUMERIC_FIELDS = ['sqft', 'beds', 'baths', 'broker_fee_amount', 'broker_fee_pct', 'application_fee']
    _BOOLEAN_FIELDS = [
        # Building amenities
        'has_elevator', 'has_doorman', 'has_concierge', 'has_gym', 'has_roof_deck',
        'has_pool', 'has_garden', 'has_courtyard', 'has_bike_room', 'has_live_in_super',
        # Unit features
        'has_dishwasher', 'has_washer_dryer', 'has_hardwood_floors', 'has_central_air',
        'has_private_outdoor', 'has_balcony', 'has_terrace', 'has_storage',
        'has_wheelchair_access', 'has_laundry_in_building', 'has_laundry_in_unit', 'has_fireplace',
        'has_stainless_steel_appliances', 'has_oversized_windows', 'has_high_ceilings',
        'has_natural_light', 'is_new_construction',
        # Parking & transportation
        'has_parking', 'has_garage_parking', 'has_valet_parking',
        # Policies
        'pets_allowed', 'guarantors_accepted', 'is_smoke_free', 'allows_subletting',
        # Financial
        'has_broker_fee', 'has_application_fee', 'no_fee', 'is_rent_stabilized'
    ]
    _ALL_FIELDS = _NUMERIC_FIELDS + _BOOLEAN_FIELDS
    _BOOLEAN_FIELD_SET = set(_BOOLEAN_FIELDS)  # For O(1) lookup
    
    # Field descriptions for prompt generation
    _FIELD_DESCRIPTIONS = {
        'sqft': 'square footage (integer)',
        'beds': 'number of bedrooms (float, e.g., 1.0, 1.5, 2.0)',
        'baths': 'number of bathrooms (float, e.g., 1.0, 1.5, 2.0)',
        'broker_fee_amount': 'broker fee in dollars (float, e.g., 2500.0)',
        'broker_fee_pct': 'broker fee as percentage of annual rent (float, e.g., 12.0 for 12%)',
        'application_fee': 'application fee in dollars (float, e.g., 150.0)',
        # Boolean field descriptions
        'has_elevator': 'building has elevator (boolean)',
        'has_doorman': 'building has doorman (boolean)',
        'has_concierge': 'building has concierge (boolean)',
        'has_gym': 'building has gym/fitness center (boolean)',
        'has_roof_deck': 'building has roof deck/rooftop access (boolean)',
        'has_pool': 'building has swimming pool (boolean)',
        'has_garden': 'building has garden/landscaped area (boolean)',
        'has_courtyard': 'building has courtyard (boolean)',
        'has_bike_room': 'building has bike room/storage (boolean)',
        'has_live_in_super': 'building has live-in superintendent (boolean)',
        'has_dishwasher': 'unit has dishwasher (boolean)',
        'has_washer_dryer': 'unit has washer/dryer or laundry in unit (boolean)',
        'has_hardwood_floors': 'unit has hardwood floors (boolean)',
        'has_central_air': 'unit has central air conditioning (boolean)',
        'has_private_outdoor': 'unit has private outdoor space (boolean)',
        'has_balcony': 'unit has balcony (boolean)',
        'has_terrace': 'unit has terrace (boolean)',
        'has_storage': 'unit has storage space/closets (boolean)',
        'has_wheelchair_access': 'building/unit is wheelchair accessible (boolean)',
        'has_laundry_in_building': 'building has shared laundry facility (boolean)',
        'has_laundry_in_unit': 'unit has in-unit laundry (boolean)',
        'has_fireplace': 'unit has fireplace (boolean)',
        'has_stainless_steel_appliances': 'unit has stainless steel appliances (boolean)',
        'has_oversized_windows': 'unit has oversized or floor-to-ceiling windows (boolean)',
        'has_high_ceilings': 'unit has high or soaring ceilings (boolean)',
        'has_natural_light': 'unit has abundant natural light (boolean)',
        'is_new_construction': 'building is new construction or newly renovated (boolean)',
        'has_parking': 'building offers parking (boolean)',
        'has_garage_parking': 'building has garage parking (boolean)',
        'has_valet_parking': 'building offers valet parking (boolean)',
        'pets_allowed': 'pets are allowed (boolean)',
        'guarantors_accepted': 'guarantors are accepted (boolean)',
        'is_smoke_free': 'building/unit is smoke-free (boolean)',
        'allows_subletting': 'subletting is allowed (boolean)',
        'has_broker_fee': 'listing has broker fee (boolean)',
        'has_application_fee': 'listing has application fee (boolean)',
        'no_fee': 'listing is no-fee (boolean)',
        'is_rent_stabilized': 'unit is rent stabilized (boolean)'
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("llm", {})
        
        # Core LLM settings with proper defaults
        self.model = self.config.get("model", "qwen3:14b")
        self.context_length = self.config.get("context", DEFAULT_CONTEXT_LENGTH)
        self.timeout = self.config.get("timeout", DEFAULT_TIMEOUT)
        self.max_concurrent = self.config.get("max_concurrent", 4)
        self.batch_size = self.config.get("batch_size", DEFAULT_BATCH_SIZE)
        
        # Ollama options with proper defaults
        self.ollama_options = {
            "num_ctx": self.context_length,
            "num_batch": self.config.get("num_batch", DEFAULT_NUM_BATCH),
            "temperature": self.config.get("temperature", 0.0),
            "top_p": self.config.get("top_p", 1.0),
        }
        
        # Additional options for qwen3:14b
        self.think = self.config.get("think", False)
        self.response_format = self.config.get("response_format", "json")
        
        # Native async client for optimal performance (5x faster than executor)
        self.async_client = ollama.AsyncClient()
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
    
    async def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            models = await self.async_client.list()
            # Handle both dict and object model formats
            model_list = models.get('models', []) if hasattr(models, 'get') else models.models
            model_names = []
            for m in model_list:
                if hasattr(m, 'model'):
                    model_names.append(m.model)
                elif isinstance(m, dict) and 'name' in m:
                    model_names.append(m['name'])
            return any(self.model in name for name in model_names)
        except Exception as e:
            timing = _get_timing_info()
            log.error({
                "event": "llm_availability_check_failed", 
                "error": str(e),
                **timing
            })
            return False
    
    async def warm_up_model(self) -> None:
        """Warm up the model with a simple prompt."""
        start_time = time.time()
        
        try:
            # Simple warm-up prompt
            warm_up_prompt = "Extract the number of bedrooms from this description: '1 bedroom apartment'"
            
            # Prepare generate kwargs for warm-up
            warm_up_kwargs = {
                "model": self.model,
                "prompt": warm_up_prompt,
                "options": self.ollama_options,
                "stream": False
            }
            
            # Add qwen3:14b specific parameters
            if self.think is not None:
                warm_up_kwargs["think"] = self.think
            
            if self.response_format:
                warm_up_kwargs["format"] = self.response_format
            
            response = await asyncio.wait_for(
                self.async_client.generate(**warm_up_kwargs),
                timeout=self.timeout
            )
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            timing = _get_timing_info()
            
            log.info({
                "event": "llm_warmup_complete", 
                **timing
            })
            
        except Exception as e:
            timing = _get_timing_info()
            log.warning({
                "event": "llm_warmup_failed", 
                "error": str(e),
                **timing
            })
    
    async def unload_model(self):
        """Unload the model to free VRAM."""
        try:
            start_time = time.time()
            await self.async_client.generate(
                model=self.model,
                prompt="",
                options={"num_ctx": 1}
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            timing = _get_timing_info()
            
            log.info({
                "event": "llm_model_unload_complete", 
                "model": self.model
            })
        except Exception as e:
            timing = _get_timing_info()
            log.warning({
                "event": "llm_model_unload_failed", 
                "model": self.model, 
                "error": str(e),
                **timing
            })
    
    async def process_batch(self, batch_requests: List[BatchRequest]) -> List[BatchResult]:
        """Process a batch of LLM requests concurrently."""
        start_time = time.time()
        
        async def process_single(req: BatchRequest) -> BatchResult:
            t_listing_start = time.time()
            async with self.semaphore:
                enriched = await self._async_inference(
                    req.description, req.missing_fields, req.current_data
                )
                # Track timing for this listing
                listing_ms = int((time.time() - t_listing_start) * 1000)
                # Note: We'll update the timing in the main enrichment function since we need the pool
                return BatchResult(
                    listing_id=req.listing_id,
                    enriched=enriched,
                    success=True,
                    missing_fields=req.missing_fields,
                    duration_ms=listing_ms
                )
        
        results = await asyncio.gather(*[process_single(req) for req in batch_requests])
        
        return results
    
    async def _async_inference(self, description: str, missing_fields: List[str], current_data: Dict[str, Any] = None) -> EnrichmentData:
        """Native async LLM inference call - 5x faster than executor method."""
        # Early return if no fields are missing
        if not missing_fields:
            return EnrichmentData()
            
        prompt = self._create_optimized_prompt(description, missing_fields, current_data or {})
        
        try:
            # Native async ollama call with timeout (no executor overhead!)
            # Add keep_alive if configured
            generate_kwargs = {
                "model": self.model,
                "prompt": prompt,
                "options": self.ollama_options,
                "stream": False
            }
            
            # Add qwen3:14b specific parameters
            if self.think is not None:
                generate_kwargs["think"] = self.think
            
            if self.response_format:
                generate_kwargs["format"] = self.response_format
            
            keep_alive = self.config.get("keep_alive")
            if keep_alive:
                generate_kwargs["keep_alive"] = keep_alive
            
            response = await asyncio.wait_for(
                self.async_client.generate(**generate_kwargs),
                timeout=self.timeout
            )
            
            response_text = response.get('response', '').strip()
            enriched = self._parse_llm_response(response_text, missing_fields)
            
            # Protect scraper data - only return data for truly missing fields
            return self._protect_scraper_data(current_data or {}, enriched)
            
        except Exception as e:
            timing = _get_timing_info()
            error_msg = str(e) if str(e) else f"{type(e).__name__}: {e}"
            log.error({
                "event": "llm_inference_failed", 
                "error": error_msg,
                "error_type": type(e).__name__,
                **timing
            })
            return EnrichmentData()
    

    
    def _identify_missing_fields(self, current_data: Dict[str, Any]) -> List[str]:
        """Identify which fields are missing from current listing data."""
        missing = []
        
        # Check numeric fields (only None counts as missing, 0 is valid)
        for field in self._NUMERIC_FIELDS:
            if current_data.get(field) is None:
                missing.append(field)
        
        # For boolean fields, allow LLM to enrich fields that are False
        # This means feature flags didn't find them, but LLM might find them in the description
        for field in self._BOOLEAN_FIELDS:
            if current_data.get(field) is False:
                missing.append(field)
        
        return missing
    
    @classmethod
    def _generate_missing_fields_sql(cls, condition: str = "IS NULL") -> str:
        """Generate SQL condition for missing fields."""
        conditions = []
        
        # Handle numeric fields (use IS NULL / IS NOT NULL)
        for field in cls._NUMERIC_FIELDS:
            conditions.append(f"{field} {condition}")
        
        # Handle boolean fields (use IS FALSE / IS TRUE for missing/complete)
        if condition == "IS NULL":
            # For missing fields: numeric fields are NULL OR boolean fields are FALSE
            for field in cls._BOOLEAN_FIELDS:
                conditions.append(f"{field} IS FALSE")
        else:
            # For complete fields: numeric fields are NOT NULL AND boolean fields are TRUE
            for field in cls._BOOLEAN_FIELDS:
                conditions.append(f"{field} IS TRUE")
        
        if condition == "IS NULL":
            # For missing fields: ANY field is missing (OR logic)
            return " OR ".join(conditions)
        else:
            # For complete fields: ALL fields are complete (AND logic)
            return " AND ".join(conditions)
    
    @classmethod
    def _generate_select_fields_sql(cls) -> str:
        """Generate SELECT clause with all fields."""
        base_fields = ['id', 'description', 'price', 'effective_price']
        return ', '.join(base_fields + cls._ALL_FIELDS)


    def _create_optimized_prompt(self, description: str, missing_fields: List[str], current_data: Dict[str, Any]) -> str:
        """Create a targeted prompt for missing fields only."""
        
        # Format current known data
        known_data = []
        for field in ['sqft', 'beds', 'baths', 'price', 'effective_price']:
            value = current_data.get(field)
            if value is not None:
                known_data.append(f"{field}: {value}")
        
        known_str = ", ".join(known_data) if known_data else "None"
        
        # Separate missing fields by type using optimized set lookup
        numeric_fields = [f for f in missing_fields if f not in self._BOOLEAN_FIELD_SET]
        boolean_fields = [f for f in missing_fields if f in self._BOOLEAN_FIELD_SET]
        
        # Generate field descriptions for missing fields only
        missing_desc = [f'"{field}": {self._FIELD_DESCRIPTIONS[field]}' 
                       for field in missing_fields if field in self._FIELD_DESCRIPTIONS]
        
        # Create instructions based on field types
        instructions = []
        if numeric_fields:
            instructions.append("For numeric fields, extract exact values or use null if not mentioned.")
        if boolean_fields:
            instructions.append("For boolean fields, use true/false based on description content or null if unclear.")
        
        return f"""Extract missing rental listing data from this NYC description.

KNOWN DATA: {known_str}
EXTRACT ONLY: {', '.join(missing_fields)}

{' '.join(instructions)}

DESCRIPTION:
{description}

Return ONLY a JSON object with the missing fields. Use null for unavailable data.
Example format: {{{', '.join(missing_desc)}}}

JSON:"""
    
    def _parse_llm_response(self, response: str, missing_fields: List[str]) -> EnrichmentData:
        """Parse LLM JSON response and return only data for missing fields."""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                return EnrichmentData()
            
            json_str = response[json_start:json_end]
            
            # Clean the JSON string to remove comments and fix common issues
            cleaned_json = _clean_json_response(json_str)
            
            data = json.loads(cleaned_json)
            
            # Only extract data for fields we specifically requested
            enriched = EnrichmentData()
            
            # Parse numeric fields - ONLY if they were missing
            if 'sqft' in missing_fields and data.get('sqft') is not None:
                try:
                    enriched.sqft = int(data['sqft'])
                except (ValueError, TypeError):
                    pass
            
            if 'beds' in missing_fields and data.get('beds') is not None:
                try:
                    enriched.beds = float(data['beds'])
                except (ValueError, TypeError):
                    pass
            
            if 'baths' in missing_fields and data.get('baths') is not None:
                try:
                    enriched.baths = float(data['baths'])
                except (ValueError, TypeError):
                    pass
            
            if 'broker_fee_amount' in missing_fields and data.get('broker_fee_amount') is not None:
                try:
                    enriched.broker_fee_amount = float(data['broker_fee_amount'])
                except (ValueError, TypeError):
                    pass
            
            if 'broker_fee_pct' in missing_fields and data.get('broker_fee_pct') is not None:
                try:
                    enriched.broker_fee_pct = float(data['broker_fee_pct'])
                except (ValueError, TypeError):
                    pass
            
            if 'application_fee' in missing_fields and data.get('application_fee') is not None:
                try:
                    enriched.application_fee = float(data['application_fee'])
                except (ValueError, TypeError):
                    pass
            
            # Parse boolean fields - use class constants, ONLY if they were missing
            for field in LLMEnricher._BOOLEAN_FIELDS:
                if field in missing_fields and field in data:
                    try:
                        value = data[field]
                        if isinstance(value, bool):
                            setattr(enriched, field, value)
                        elif isinstance(value, str):
                            # Handle string boolean values
                            value_lower = value.lower().strip()
                            if value_lower in ['true', 'yes', '1']:
                                setattr(enriched, field, True)
                            elif value_lower in ['false', 'no', '0']:
                                setattr(enriched, field, False)
                            # null/None values are ignored (stay None)
                    except (ValueError, TypeError, AttributeError):
                        pass
            
            # Extract confidence if provided
            if data.get('confidence'):
                try:
                    enriched.confidence = float(data['confidence'])
                except (ValueError, TypeError):
                    pass
            
            return enriched
            
        except json.JSONDecodeError as e:
            timing = _get_timing_info()
            log.warning({
                "event": "llm_json_parse_failed", 
                "response_preview": response[:DEFAULT_RESPONSE_PREVIEW_LENGTH],
                "error": str(e),
                **timing
            })
            return EnrichmentData()
        except Exception as e:
            timing = _get_timing_info()
            log.error({
                "event": "llm_response_parse_error", 
                "error": str(e),
                **timing
            })
            return EnrichmentData()
    
    def _count_enriched_fields(self, enriched: EnrichmentData) -> int:
        """Count how many fields were successfully enriched."""
        # Use optimized field lists
        all_fields = self._NUMERIC_FIELDS + self._BOOLEAN_FIELDS
        return sum(1 for field in all_fields if getattr(enriched, field, None) is not None)

    @classmethod
    def _build_enriched_fields_json(cls, enriched: EnrichmentData, missing_fields: List[str]) -> Dict[str, Any]:
        """Build JSON object of fields that were actually enriched."""
        enriched_fields = {}
        
        # Add numeric fields that were missing and got enriched
        for field in cls._NUMERIC_FIELDS:
            if field in missing_fields:
                value = getattr(enriched, field, None)
                if value is not None:
                    # Convert Decimal to float for JSON serialization
                    enriched_fields[field] = float(value) if hasattr(value, '__float__') else value
        
        # Add boolean fields that were missing and got set to TRUE (only track positive findings)
        for field in cls._BOOLEAN_FIELDS:
            if field in missing_fields:
                value = getattr(enriched, field, None)
                if value is True:  # Only track fields set to true
                    enriched_fields[field] = True
        
        return enriched_fields

    def _validate_enriched_data(self, current_data: Dict[str, Any], enriched_data: EnrichmentData, missing_fields: List[str]) -> EnrichmentData:
        """Validate enriched data to prevent overwriting valid scraped data with invalid values."""
        validated = EnrichmentData()
        
        # Only process fields that were actually missing
        for field in missing_fields:
            enriched_value = getattr(enriched_data, field, None)
            current_value = current_data.get(field)
            
            if enriched_value is not None:
                # For numeric fields, validate that we're not overwriting valid data with null
                if field in self._NUMERIC_FIELDS:
                    # Don't overwrite valid numeric data (including 0 for studios) with null
                    if current_value is not None:
                        # Only allow overwriting if the enriched value is more specific/better
                        # For beds, don't overwrite 0 (studio) with null
                        if field == 'beds' and current_value == 0.0 and enriched_value is None:
                            # Log this protection to help with debugging
                            from . import log
                            log.info({
                                "event": "llm_protected_studio_data",
                                "field": field,
                                "current_value": current_value,
                                "enriched_value": enriched_value,
                                "action": "skipped_overwrite"
                            })
                            continue  # Skip overwriting studio (beds = 0) with null
                        # For other fields, only overwrite if enriched value is not null
                        if enriched_value is not None:
                            setattr(validated, field, enriched_value)
                    else:
                        # Field was missing, safe to set
                        setattr(validated, field, enriched_value)
                else:
                    # Boolean fields - safe to set if they were missing
                    setattr(validated, field, enriched_value)
        
        return validated

    def _protect_scraper_data(self, current_data: Dict[str, Any], enriched_data: EnrichmentData) -> EnrichmentData:
        """Protect existing scraper data from being overwritten by LLM."""
        protected = EnrichmentData()
        
        # First, copy all existing data from current_data to protected
        for field in self._NUMERIC_FIELDS + self._BOOLEAN_FIELDS:
            if field in current_data and current_data[field] is not None:
                setattr(protected, field, current_data[field])
        
        # Then, only add enriched data for fields that were missing in current_data
        missing_fields = self._identify_missing_fields(current_data)
        
        # Validate the enriched data to prevent overwriting valid data with invalid values
        validated_enriched = self._validate_enriched_data(current_data, enriched_data, missing_fields)
        
        # Apply validated enriched data
        for field in missing_fields:
            validated_value = getattr(validated_enriched, field, None)
            if validated_value is not None:
                setattr(protected, field, validated_value)
        
        return protected


async def enrich_listings_with_llm_optimized(pool: Pool, config: Dict[str, Any]) -> None:
    """Enrich listings with LLM-extracted data using optimized batching."""
    llm_config = config.get("llm", {})
    
    if not llm_config.get("enabled", False):
        timing = _get_timing_info()
        log.info({
            "event": "llm_enrichment_skipped", 
            "reason": "disabled_in_config",
            **timing
        })
        return
    
    if not llm_config.get("extract_missing", True):
        timing = _get_timing_info()
        log.info({
            "event": "llm_enrichment_skipped", 
            "reason": "extract_missing_disabled",
            **timing
        })
        return
    
    start_time = time.time()
    enricher = LLMEnricher(config)
    
    # Check if LLM is available
    if not await enricher.is_available():
        timing = _get_timing_info()
        log.error({
            "event": "llm_enrichment_failed", 
            "reason": "ollama_or_model_unavailable",
            **timing
        })
        return
    
    # Reset last log time to get accurate "sec since previous log" for the first LLM log
    global _last_log_time
    _last_log_time = time.time()
    
    timing = _get_timing_info()
    log.info({
        "event": "llm_enrichment_start", 
        "model": enricher.model,
        **timing
    })
    
    # Warm up the model
    await enricher.warm_up_model()
    
    # Fetch listings that need enrichment
    async with pool.acquire() as conn:
        select_fields = LLMEnricher._generate_select_fields_sql()
        missing_condition = LLMEnricher._generate_missing_fields_sql("IS NULL")
        
        rows = await conn.fetch(f"""
            SELECT {select_fields}
            FROM listings 
            WHERE llm_enrichment_completed_at IS NULL
            AND description IS NOT NULL 
            AND description != ''
            AND ({missing_condition})
            ORDER BY id
        """)
    
    # Also get count of listings that were skipped (no missing fields)
    async with pool.acquire() as conn:
        complete_condition = LLMEnricher._generate_missing_fields_sql("IS NOT NULL")
        
        skipped_count = await conn.fetchval(f"""
            SELECT COUNT(*) FROM listings 
            WHERE llm_enrichment_completed_at IS NULL
            AND description IS NOT NULL 
            AND description != ''
            AND ({complete_condition})
        """)
    
    if not rows:
        timing = _get_timing_info()
        log.info({
            "event": "llm_enrichment_complete", 
            "reason": "no_listings_need_enrichment",
            "skipped_count": skipped_count,
            **timing
        })
        
        # Mark skipped listings as completed so they don't get processed again
        if skipped_count > 0:
            async with pool.acquire() as conn:
                complete_condition = LLMEnricher._generate_missing_fields_sql("IS NOT NULL")
                
                await conn.execute(f"""
                    UPDATE listings 
                    SET llm_enrichment_completed_at = now(), llm_enrichment_occurred = false
                    WHERE llm_enrichment_completed_at IS NULL
                    AND description IS NOT NULL 
                    AND description != ''
                    AND ({complete_condition})
                """)
            
            timing = _get_timing_info()
            log.info({
                "event": "llm_skipped_listings_marked_complete",
                "count": skipped_count,
                **timing
            })
        
        return
    
    timing = _get_timing_info()
    log.info({
        "event": "llm_enrichment_listings_found", 
        "count": len(rows),
        "skipped_count": skipped_count
    })
    
    # Process in batches
    batch_size = enricher.batch_size
    total_enriched = 0
    total_processed = 0
    total_skipped = 0  # Track skipped listings
    
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i:i + batch_size]
        
        # Create batch requests
        batch_requests = []
        for row in batch_rows:
            # Build current_data dynamically from row
            current_data = {
                'price': row['price'],
                'effective_price': row['effective_price']
            }
            
            # Add all fields from the class constants
            for field in LLMEnricher._ALL_FIELDS:
                current_data[field] = row[field]
            
            missing_fields = enricher._identify_missing_fields(current_data)
            
            if missing_fields:  # Only process if there are missing fields
                batch_requests.append(BatchRequest(
                    listing_id=row['id'],
                    description=row['description'],
                    current_data=current_data,
                    missing_fields=missing_fields
                ))
            else:
                # Mark as completed even if no enrichment needed
                await _mark_listing_completed(pool, row['id'])
                total_skipped += 1
        
        if not batch_requests:
            continue
        
        # Process batch
        batch_results = await enricher.process_batch(batch_requests)
        
        # Update database
        enriched_count = await _update_database_batch(pool, batch_results)
        total_enriched += enriched_count
        total_processed += len(batch_requests)
        
        timing = _get_timing_info()
        
        log.info({
            "event": "llm_batch_processed", 
            "batch": f"{i//batch_size + 1}/{(len(rows) + batch_size - 1)//batch_size}",
            "batch_size": len(batch_requests),
            "successful": len(batch_requests),
            "failed": 0,
            "enriched": enriched_count,
            "skipped": total_skipped,
            "total_processed": total_processed,
            "total_enriched": total_enriched,
            "sec_per_listing": round(timing["sec"] / len(batch_requests), 2),
            **timing
        })
    
    elapsed_ms = int((time.time() - start_time) * 1000)
    elapsed_sec = round((time.time() - start_time), 2)
    elapsed_min = round((time.time() - start_time) / 60, 2)
    
    timing = _get_timing_info()
    
    log.info({
        "event": "llm_enrichment_complete",
        "total_listings": len(rows),
        "processed": total_processed,
        "enriched": total_enriched,
        "skipped": total_skipped,
        "total_sec": timing["elapsed_sec"],
        "total_min": timing["elapsed_min"]
    })
    
    # Unload model to free VRAM if configured
    if llm_config.get("auto_unload", False):
        await enricher.unload_model()


async def _mark_listing_completed(pool: Pool, listing_id: int) -> None:
    """Mark a listing as LLM enrichment completed (even if no enrichment was needed)."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE listings SET llm_enrichment_completed_at = now() WHERE id = $1",
                listing_id
            )
    except Exception as e:
        timing = _get_timing_info()
        log.error({
            "event": "llm_mark_completed_failed",
            "listing_id": listing_id,
            "error": str(e),
            **timing
        })


async def _update_database_batch(pool: Pool, results: List[BatchResult]) -> int:
    """Update database with batch results."""
    enriched_count = 0
    
    async with pool.acquire() as conn:
        for result in results:
            try:
                # Build dynamic UPDATE query based on enriched data
                updates = []
                params = []
                param_idx = 1
                
                enriched = result.enriched
                
                if enriched.sqft is not None:
                    updates.append(f"sqft = ${param_idx}")
                    params.append(enriched.sqft)
                    param_idx += 1
                
                if enriched.beds is not None:
                    # Additional safety check: don't overwrite valid studio data (beds = 0) with null
                    # This prevents the LLM from overwriting correctly scraped studio information
                    updates.append(f"beds = ${param_idx}")
                    params.append(enriched.beds)
                    param_idx += 1
                
                if enriched.baths is not None:
                    updates.append(f"baths = ${param_idx}")
                    params.append(enriched.baths)
                    param_idx += 1
                
                if enriched.broker_fee_amount is not None:
                    updates.append(f"broker_fee_amount = ${param_idx}")
                    params.append(enriched.broker_fee_amount)
                    param_idx += 1
                
                if enriched.broker_fee_pct is not None:
                    updates.append(f"broker_fee_pct = ${param_idx}")
                    params.append(enriched.broker_fee_pct)
                    param_idx += 1
                
                if enriched.application_fee is not None:
                    updates.append(f"application_fee = ${param_idx}")
                    params.append(enriched.application_fee)
                    param_idx += 1
                
                # Boolean flags - use class constants
                for field in LLMEnricher._BOOLEAN_FIELDS:
                    value = getattr(enriched, field, None)
                    if value is not None:
                        updates.append(f"{field} = ${param_idx}")
                        params.append(value)
                        param_idx += 1
                
                # Always mark as completed (even if no data was enriched)
                updates.append(f"llm_enrichment_completed_at = now()")
                # Don't add "now()" as a parameter, it's a SQL function
                
                # Check if any actual enrichment happened
                has_enrichment = any([
                    enriched.sqft is not None,
                    enriched.beds is not None,
                    enriched.baths is not None,
                    enriched.broker_fee_amount is not None,
                    enriched.broker_fee_pct is not None,
                    enriched.application_fee is not None,
                    any(getattr(enriched, field, None) is not None for field in LLMEnricher._BOOLEAN_FIELDS)
                ])
                
                if has_enrichment:
                    updates.append(f"llm_enrichment_occurred = true")
                    
                    # Track which fields were enriched (only those that were missing and got enriched)
                    missing_fields = result.missing_fields or []
                    enriched_fields = LLMEnricher._build_enriched_fields_json(enriched, missing_fields)
                    
                    # Add the enriched fields JSON to the update
                    updates.append(f"llm_enriched_fields = ${param_idx}")
                    params.append(json.dumps(enriched_fields))
                    param_idx += 1
                
                # Add listing_id for WHERE clause
                params.append(result.listing_id)
                
                if updates:
                    query = f"""
                    UPDATE listings 
                    SET {', '.join(updates)}
                    WHERE id = ${param_idx}
                    """
                    
                    await conn.execute(query, *params)
                    
                    if has_enrichment:  # Check if actual enrichment happened
                        enriched_count += 1
                
                # Update timing for this listing
                if result.duration_ms is not None:
                    await update_llm_enrichment_duration(pool, result.listing_id, result.duration_ms)
                
            except Exception as e:
                timing = _get_timing_info()
                log.error({
                    "event": "llm_database_update_failed",
                    "listing_id": result.listing_id,
                    "error": str(e),
                    **timing
                })
    
    return enriched_count
