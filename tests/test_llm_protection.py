#!/usr/bin/env python3
"""
Test LLM data protection logic to ensure scraped data is not overwritten incorrectly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_enricher import LLMEnricher, EnrichmentData


def test_protect_studio_data():
    """Test that studio data (beds = 0.0) is protected from being overwritten with null."""
    enricher = LLMEnricher({})
    
    # Simulate current data with studio apartment
    current_data = {
        'beds': 0.0,  # Studio apartment
        'baths': 1.0,
        'sqft': 500
    }
    
    # Simulate LLM trying to overwrite with null
    enriched_data = EnrichmentData()
    enriched_data.beds = None  # LLM says "no beds mentioned"
    enriched_data.baths = 1.5  # LLM found 1.5 baths
    
    # Test protection
    protected = enricher._protect_scraper_data(current_data, enriched_data)
    
    # Verify that beds = 0.0 is preserved (not overwritten with null)
    assert protected.beds == 0.0, f"Expected beds = 0.0, got {protected.beds}"
    assert protected.baths == 1.5, f"Expected baths = 1.5, got {protected.baths}"
    assert protected.sqft == 500, f"Expected sqft = 500, got {protected.sqft}"
    
    print("✅ Studio data protection test passed")


def test_protect_existing_data():
    """Test that existing valid data is protected from being overwritten."""
    enricher = LLMEnricher({})
    
    # Simulate current data with valid information
    current_data = {
        'beds': 2.0,
        'baths': 1.0,
        'sqft': 800
    }
    
    # Simulate LLM trying to overwrite with different values
    enriched_data = EnrichmentData()
    enriched_data.beds = 1.0  # LLM says "1 bedroom"
    enriched_data.baths = None  # LLM says "no bathrooms mentioned"
    
    # Test protection
    protected = enricher._protect_scraper_data(current_data, enriched_data)
    
    # Verify that existing data is preserved
    assert protected.beds == 2.0, f"Expected beds = 2.0, got {protected.beds}"
    assert protected.baths == 1.0, f"Expected baths = 1.0, got {protected.baths}"
    assert protected.sqft == 800, f"Expected sqft = 800, got {protected.sqft}"
    
    print("✅ Existing data protection test passed")


def test_allow_missing_field_enrichment():
    """Test that truly missing fields can be enriched."""
    enricher = LLMEnricher({})
    
    # Simulate current data with missing fields
    current_data = {
        'beds': 1.0,
        'baths': None,  # Missing
        'sqft': None    # Missing
    }
    
    # Simulate LLM providing missing data
    enriched_data = EnrichmentData()
    enriched_data.beds = None  # Not missing, should be ignored
    enriched_data.baths = 1.5  # Missing, should be set
    enriched_data.sqft = 600   # Missing, should be set
    
    # Test protection
    protected = enricher._protect_scraper_data(current_data, enriched_data)
    
    # Verify that missing fields are enriched
    assert protected.beds == 1.0, f"Expected beds = 1.0, got {protected.beds}"
    assert protected.baths == 1.5, f"Expected baths = 1.5, got {protected.baths}"
    assert protected.sqft == 600, f"Expected sqft = 600, got {protected.sqft}"
    
    print("✅ Missing field enrichment test passed")


def test_identify_missing_fields():
    """Test that missing fields are correctly identified."""
    enricher = LLMEnricher({})
    
    # Test with various data combinations
    test_cases = [
        # (current_data, expected_missing_fields)
        (
            {'beds': 0.0, 'baths': 1.0, 'sqft': 500},
            []  # No missing fields
        ),
        (
            {'beds': None, 'baths': 1.0, 'sqft': 500},
            ['beds']  # beds is missing
        ),
        (
            {'beds': 0.0, 'baths': None, 'sqft': None},
            ['baths', 'sqft']  # baths and sqft are missing
        ),
        (
            {'beds': None, 'baths': None, 'sqft': None},
            ['beds', 'baths', 'sqft']  # All are missing
        )
    ]
    
    for current_data, expected_missing in test_cases:
        missing = enricher._identify_missing_fields(current_data)
        assert set(missing) == set(expected_missing), \
            f"Expected {expected_missing}, got {missing} for data {current_data}"
    
    print("✅ Missing field identification test passed")


if __name__ == "__main__":
    print("Running LLM data protection tests...")
    
    try:
        test_identify_missing_fields()
        test_protect_studio_data()
        test_protect_existing_data()
        test_allow_missing_field_enrichment()
        
        print("\n🎉 All tests passed! LLM data protection is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

