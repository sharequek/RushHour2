#!/usr/bin/env python3
"""OCR Regression Test Suite - Prevents regressions as we fix OCR issues."""

import pytest
import re
from pathlib import Path
from src.ocr_extractor import (
    extract_sqft_from_text, 
    _get_dimension_evidence, 
    get_paddle_cfg,
    run_hybrid_ocr,
    extract_explicit_sqft_only
)
import tempfile
import asyncio
import aiohttp

class TestOCRRegression:
    """Test suite to prevent OCR regressions."""
    
    def test_letter_misreading_artifact(self):
        """Test OCR pattern: 8.PL×6.11 -> 62 sqft (pattern-based, not listing-specific)."""
        test_cases = [
            ("8.PL×6.11", 62, "8.PL×6.11"),
        ]
        
        paddle_cfg = get_paddle_cfg()
        
        for text, expected_sqft, expected_source in test_cases:
            dim_count, total_sqft, source_text, confidence = _get_dimension_evidence(text, paddle_cfg)
            
            assert dim_count == 1, f"Should detect 1 dimension in '{text}'"
            assert total_sqft == expected_sqft, f"SQFT should be {expected_sqft} for '{text}'"
            # Handle escaped quotes in source_text for comparison
            normalized_source = source_text.replace('\\"', '"')
            assert expected_source in normalized_source, f"Source should contain '{expected_source}' for '{text}', got: {source_text}"

    def test_dimension_pattern_extraction(self):
        """Test dimension pattern extraction for various formats."""
        test_cases = [
            # Feet and inches patterns
            ("11'3\" x 14'", 158, "11'3\""),  # 11.25 × 14 = 157.5 ≈ 158
            ("11'3\" X 14'", 158, "11'3\""),  # Uppercase X
            ("11'3\"×14'", 158, "11'3\""),   # × symbol
            
            # Feet with inches in second dimension
            ("13' X 11'6\"", 150, "13'×11'6\""),  # 13 × 11.5 = 149.5 ≈ 150
            ("13' x 11'6\"", 150, "13'×11'6\""),  # Lowercase x
            ("13'×11'6\"", 150, "13'×11'6\""),    # × symbol
        ]

        paddle_cfg = get_paddle_cfg()

        for text, expected_sqft, expected_source in test_cases:
            dim_count, total_sqft, source_text, confidence = _get_dimension_evidence(text, paddle_cfg)

            assert dim_count == 1, f"Should detect 1 dimension in '{text}'"
            assert total_sqft == expected_sqft, f"SQFT should be {expected_sqft} for '{text}'"
            # Check that the source text contains the expected pattern (ignoring escaped quotes)
            normalized_source = source_text.replace('\\"', '"').replace("\\'", "'")
            assert expected_source in normalized_source, f"Source should contain '{expected_source}' for '{text}', got: '{normalized_source}'"

    def test_rounding_edge_cases(self):
        """Test rounding edge cases to ensure consistent behavior."""
        test_cases = [
            # Test cases with .5 values to verify rounding behavior
            ("10' x 10'", 100, "10'×10'"),  # 10 × 10 = 100.0 → 100
            ("10'6\" x 10'6\"", 110, "10'6\"×10'6\""),  # 10.5 × 10.5 = 110.25 → 110
            ("11' x 11'", 121, "11'×11'"),  # 11 × 11 = 121.0 → 121
            ("11'6\" x 11'6\"", 132, "11'6\"×11'6\""),  # 11.5 × 11.5 = 132.25 → 132
        ]

        paddle_cfg = get_paddle_cfg()

        for text, expected_sqft, expected_source in test_cases:
            dim_count, total_sqft, source_text, confidence = _get_dimension_evidence(text, paddle_cfg)

            assert dim_count == 1, f"Should detect 1 dimension in '{text}'"
            assert total_sqft == expected_sqft, f"SQFT should be {expected_sqft} for '{text}'"
            assert expected_source in source_text, f"Source should contain '{expected_source}' for '{text}'"

    def test_letter_o_misreading_artifact(self):
        """Test OCR pattern: IO"×11'9" -> 129 sqft (pattern-based, not listing-specific)."""
        test_cases = [
            ("IO\"×11'9\"", 129, "11'0\""),
        ]
        
        paddle_cfg = get_paddle_cfg()
        
        for text, expected_sqft, expected_source in test_cases:
            dim_count, total_sqft, source_text, confidence = _get_dimension_evidence(text, paddle_cfg)
            
            assert dim_count == 1, f"Should detect 1 dimension in '{text}'"
            assert total_sqft == expected_sqft, f"SQFT should be {expected_sqft} for '{text}'"
            # Handle escaped quotes in source_text for comparison
            normalized_source = source_text.replace('\\"', '"')
            assert expected_source in normalized_source, f"Source should contain '{expected_source}' for '{text}', got: {source_text}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
