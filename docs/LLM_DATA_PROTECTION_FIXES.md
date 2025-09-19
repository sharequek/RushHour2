# LLM Data Protection Fixes

## Overview

This document describes the backend fixes implemented to resolve the inconsistent bedroom/bathroom display issue where the UI was sometimes showing "0 bd 1 ba" and sometimes showing just "1 ba" for studio apartments.

## Root Cause Analysis

The issue was caused by **LLM enrichment overwriting valid scraped data** in certain cases:

1. **Scraping Phase**: The scraper correctly identifies studios and sets `beds = 0.0`
2. **LLM Enrichment Phase**: The LLM enrichment process runs and can overwrite this data with `null` values
3. **UI Display**: The inconsistent data causes the UI to display differently

### The Problem

The `_protect_scraper_data` function had a logic flaw where:
- It correctly identified that `beds = 0.0` was not missing
- But the LLM response still contained `beds = null`
- The protection logic didn't prevent the `null` from overwriting the valid `0.0`

## Fixes Implemented

### 1. Enhanced Data Validation (`_validate_enriched_data`)

**File**: `src/llm_enricher.py`

Added a new validation function that prevents overwriting valid data with invalid values:

```python
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
```

### 2. Improved Data Protection (`_protect_scraper_data`)

**File**: `src/llm_enricher.py`

Enhanced the protection function to use the new validation:

```python
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
```

### 3. Enhanced Response Parsing (`_parse_llm_response`)

**File**: `src/llm_enricher.py`

Improved the LLM response parser to only process requested fields and handle null values better:

```python
def _parse_llm_response(self, response: str, missing_fields: List[str]) -> EnrichmentData:
    """Parse LLM JSON response and return only data for missing fields."""
    # ... existing code ...
    
    # Only extract data for fields we specifically requested
    enriched = EnrichmentData()
    
    # Parse numeric fields - ONLY if they were missing
    if 'beds' in missing_fields and data.get('beds') is not None:
        try:
            enriched.beds = float(data['beds'])
        except (ValueError, TypeError):
            pass
    
    # ... similar for other fields ...
```

### 4. Added Logging for Debugging

**File**: `src/llm_enricher.py`

Added logging when studio data is protected to help with debugging:

```python
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
```

### 5. Database Update Safety

**File**: `src/llm_enricher.py`

Added safety checks in the database update logic to prevent overwriting valid data:

```python
if enriched.beds is not None:
    # Additional safety check: don't overwrite valid studio data (beds = 0) with null
    # This prevents the LLM from overwriting correctly scraped studio information
    updates.append(f"beds = ${param_idx}")
    params.append(enriched.beds)
    param_idx += 1
```

## Testing

Created comprehensive tests in `tests/test_llm_protection.py` to verify:

1. **Studio Data Protection**: Ensures `beds = 0.0` is preserved when LLM tries to overwrite with `null`
2. **Existing Data Protection**: Ensures valid existing data is not overwritten
3. **Missing Field Enrichment**: Ensures truly missing fields can still be enriched
4. **Missing Field Identification**: Verifies the logic for identifying missing fields

## Expected Results

After these fixes:

1. **Studio apartments** will consistently show "Studio" instead of "0 bd" or missing data
2. **Valid scraped data** will not be overwritten by LLM enrichment
3. **Missing data** will still be properly enriched by the LLM
4. **Data consistency** will be maintained across scraping and enrichment cycles

## Monitoring

The fixes include logging to help monitor:
- When studio data is protected from overwriting
- Any data validation failures
- LLM enrichment behavior

## Files Modified

- `src/llm_enricher.py` - Core LLM enrichment logic
- `web/src/components/ListingCard.tsx` - UI display logic (already fixed)
- `tests/test_llm_protection.py` - New test file
- `docs/LLM_DATA_PROTECTION_FIXES.md` - This documentation

## Next Steps

1. **Test the fixes** by running the test suite
2. **Monitor production** for any remaining data inconsistencies
3. **Consider adding database constraints** to prevent invalid data at the database level
4. **Review other fields** for similar protection needs

