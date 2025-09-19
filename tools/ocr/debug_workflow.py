#!/usr/bin/env python3
"""
OCR Debug Workflow Script

This script automates the process of:
1. Testing individual listings with OCR
2. Identifying extraction issues
3. Adding fixes to the OCR extractor
4. Creating/updating regression tests
5. Running regression tests to verify fixes

Usage:
    python scripts/ocr_debug_workflow.py --listings 15,26,21
    python scripts/ocr_debug_workflow.py --test-pattern "15'-6\"14-7\""
    python scripts/ocr_debug_workflow.py --add-fix --pattern "15'-6\"14-7\"" --expected 226
"""

import sys
import asyncio
import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Add project root to path
sys.path.append('.')

import aiohttp
from asyncpg import create_pool

from src.ocr_extractor import (
    run_hybrid_ocr, 
    tesseract_texts_from_path, 
    get_ocr_instance, 
    extract_text_from_ocr_result,
    get_paddle_cfg,
    get_tesseract_cfg,
    get_hybrid_cfg,
    _get_dimension_evidence,
    extract_explicit_sqft_only,
    _calculate_explicit_sqft_confidence,
    download_image
)

class OCRDebugWorkflow:
    def __init__(self):
        self.db_pool = None
        self.session = None
        self.workflow_log = []
        
    async def __aenter__(self):
        # Initialize database connection
        self.db_pool = await create_pool(
            host='localhost',
            port=5432,
            user='postgres',
            password='postgres',
            database='rushhour2'
        )
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.db_pool:
            await self.db_pool.close()
        if self.session:
            await self.session.close()
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        self.workflow_log.append(log_entry)
    
    async def download_image(self, url: str, temp_dir: Path) -> Optional[Path]:
        """Download image from URL to temporary directory using the actual implementation."""
        return await download_image(self.session, url, temp_dir)
    
    async def test_listing_ocr(self, listing_id: int) -> Dict:
        """Test OCR on a specific listing's floor plans."""
        self.log(f"Testing Listing {listing_id}")
        
        # Get floor plan photos for this listing
        async with self.db_pool.acquire() as conn:
            photos = await conn.fetch("""
                SELECT id, url FROM listing_photos 
                WHERE listing_id = $1 AND type = 'floor_plan'
                ORDER BY position
            """, listing_id)
        
        if not photos:
            self.log(f"No floor plan photos found for listing {listing_id}", "WARNING")
            return {"listing_id": listing_id, "photos": [], "results": []}
        
        self.log(f"Found {len(photos)} floor plan photo(s)")
        
        results = []
        for i, photo in enumerate(photos):
            photo_id = photo['id']
            photo_url = photo['url']
            
            self.log(f"Processing photo {i+1} (ID: {photo_id})")
            
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    image_path = await self.download_image(photo_url, temp_path)
                    
                    if image_path is None:
                        self.log("Failed to download image", "ERROR")
                        continue
                    
                    # Get detailed OCR output from all engines
                    detailed_results = await self.get_detailed_ocr_output(image_path)
                    
                    # Test hybrid OCR on the image
                    sqft, src, engine, confidence = run_hybrid_ocr(image_path)
                    
                    result = {
                        "photo_id": photo_id,
                        "photo_url": photo_url,
                        "sqft": sqft,
                        "source_text": src,
                        "engine": engine,
                        "confidence": confidence,
                        "detailed_ocr": detailed_results
                    }
                    
                    results.append(result)
                    
                    self.log(f"Hybrid OCR Result: sqft={sqft}, src='{src}', engine={engine}, confidence={confidence}")
                    
            except Exception as e:
                self.log(f"Error processing photo {photo_id}: {e}", "ERROR")
                results.append({
                    "photo_id": photo_id,
                    "photo_url": photo_url,
                    "error": str(e)
                })
        
        return {
            "listing_id": listing_id,
            "photos": [{"id": p['id'], "url": p['url']} for p in photos],
            "results": results
        }
    
    async def get_detailed_ocr_output(self, image_path: Path) -> Dict:
        """Get detailed OCR output from all engines and modes by reusing the actual OCR logic."""
        self.log("Getting detailed OCR output from all engines...")
        
        detailed_results = {}
        
        # Add configuration information
        paddle_cfg = get_paddle_cfg()
        tesseract_cfg = get_tesseract_cfg()
        hybrid_cfg = get_hybrid_cfg()
        
        detailed_results['configuration'] = {
            'paddleocr': {
                'enabled': paddle_cfg.get('enabled', True),
                'preprocess_enabled': paddle_cfg.get('preprocess_enabled', True),
                'min_side': paddle_cfg.get('preprocess_min_side', 1800),
                'max_scale': paddle_cfg.get('preprocess_max_scale', 2.4),
                'contrast': paddle_cfg.get('preprocess_contrast', 1.35),
                'sharpen_percent': paddle_cfg.get('preprocess_sharpen_percent', 120)
            },
            'tesseract': {
                'enabled': tesseract_cfg.get('enabled', True),
                'psm': tesseract_cfg.get('psm', 11),
                'oem': tesseract_cfg.get('oem', 1),
                'min_word_conf': tesseract_cfg.get('min_word_conf', 50),
                'whitelist': tesseract_cfg.get('whitelist', '')
            },
            'hybrid': {
                'explicit_sqft_min': hybrid_cfg.get('explicit_sqft_min', 64),
                'explicit_sqft_max': hybrid_cfg.get('explicit_sqft_max', 1200),
                'dimension_total_min': hybrid_cfg.get('dimension_total_min', 15),
                'dimension_total_max': hybrid_cfg.get('dimension_total_max', 1200),
                'tesseract_skip_paddle_conf': hybrid_cfg.get('tesseract_skip_paddle_conf', 0.80),
                'sqft_confidence_threshold': hybrid_cfg.get('sqft_confidence_threshold', 0.40),
                'dimension_confidence_threshold': hybrid_cfg.get('dimension_confidence_threshold', 0.60)
            }
        }
        
        try:
            # Use the actual run_hybrid_ocr function from src/ocr_extractor.py
            self.log("Running hybrid OCR using actual implementation...")
            hybrid_result = run_hybrid_ocr(image_path)
            
            # Extract the raw OCR results from the hybrid function
            # run_hybrid_ocr returns (sqft, source_text, engine_used, confidence)
            sqft, source_text, engine_used, confidence = hybrid_result
            detailed_results['hybrid_result'] = {
                'sqft': sqft,
                'source_text': source_text,
                'confidence': confidence,
                'engine_used': engine_used or 'unknown'
            }
            
            # Get individual engine results for debugging
            self.log("Getting individual engine results for debugging...")
            
            # PaddleOCR
            self.log("Running PaddleOCR...")
            paddle_ocr = get_ocr_instance()
            paddle_result = paddle_ocr.predict(str(image_path))
            paddle_texts = extract_text_from_ocr_result(paddle_result)
            detailed_results['paddleocr'] = {
                'raw_result': paddle_result,
                'extracted_texts': paddle_texts,
                'full_text': ' '.join(paddle_texts) if paddle_texts else ''
            }
            self.log(f"PaddleOCR extracted {len(paddle_texts)} text blocks")
            
            # Tesseract - use the actual tesseract_texts_from_path function
            self.log("Running Tesseract...")
            tesseract_texts = tesseract_texts_from_path(image_path)
            detailed_results['tesseract'] = {
                'extracted_texts': tesseract_texts,
                'full_text': ' '.join(tesseract_texts) if tesseract_texts else ''
            }
            self.log(f"Tesseract extracted {len(tesseract_texts)} text blocks")
            
            # Test explicit SQFT and dimension extraction on each engine's output
            self.log("Testing explicit SQFT and dimension extraction on each engine's output...")
            config = get_paddle_cfg()
            
            # Test PaddleOCR output
            paddle_full_text = detailed_results['paddleocr']['full_text']
            if paddle_full_text:
                # Debug pattern matching
                pattern_debug = self.debug_pattern_matching(paddle_full_text)
                detailed_results['paddleocr']['pattern_debug'] = pattern_debug
                
                # Test explicit SQFT first
                explicit_sqft, explicit_source = extract_explicit_sqft_only(paddle_full_text, config)
                if explicit_sqft:
                    explicit_conf = _calculate_explicit_sqft_confidence(explicit_sqft, explicit_source)
                    detailed_results['paddleocr']['explicit_sqft'] = {
                        'sqft': explicit_sqft,
                        'source_text': explicit_source,
                        'confidence': explicit_conf
                    }
                    self.log(f"PaddleOCR explicit SQFT: {explicit_sqft} sqft (confidence: {explicit_conf})")
                
                # Test dimension extraction
                paddle_dim_count, paddle_sqft, paddle_source, paddle_conf = _get_dimension_evidence(paddle_full_text, config)
                detailed_results['paddleocr']['dimension_extraction'] = {
                    'dimension_count': paddle_dim_count,
                    'total_sqft': paddle_sqft,
                    'source_text': paddle_source,
                    'confidence': paddle_conf
                }
                self.log(f"PaddleOCR dimension extraction: count={paddle_dim_count}, sqft={paddle_sqft}")
            
            # Test Tesseract output
            tesseract_full_text = detailed_results['tesseract']['full_text']
            if tesseract_full_text:
                # Debug pattern matching
                pattern_debug = self.debug_pattern_matching(tesseract_full_text)
                detailed_results['tesseract']['pattern_debug'] = pattern_debug
                
                # Test explicit SQFT first
                explicit_sqft, explicit_source = extract_explicit_sqft_only(tesseract_full_text, config)
                if explicit_sqft:
                    explicit_conf = _calculate_explicit_sqft_confidence(explicit_sqft, explicit_source)
                    detailed_results['tesseract']['explicit_sqft'] = {
                        'sqft': explicit_sqft,
                        'source_text': explicit_source,
                        'confidence': explicit_conf
                    }
                    self.log(f"Tesseract explicit SQFT: {explicit_sqft} sqft (confidence: {explicit_conf})")
                
                # Test dimension extraction
                tesseract_dim_count, tesseract_sqft, tesseract_source, tesseract_conf = _get_dimension_evidence(tesseract_full_text, config)
                detailed_results['tesseract']['dimension_extraction'] = {
                    'dimension_count': tesseract_dim_count,
                    'total_sqft': tesseract_sqft,
                    'source_text': tesseract_source,
                    'confidence': tesseract_conf
                }
                self.log(f"Tesseract dimension extraction: count={tesseract_dim_count}, sqft={tesseract_sqft}")
            
        except Exception as e:
            self.log(f"Error getting detailed OCR output: {e}", "ERROR")
            detailed_results['error'] = str(e)
        
        return detailed_results
    
    def debug_pattern_matching(self, text: str) -> Dict:
        """Debug which patterns are matching in the text by reusing actual patterns from ocr_extractor.py."""
        if not text:
            return {}
        
        debug_results = {}
        
        # Test explicit SQFT patterns - reuse the actual patterns from extract_explicit_sqft_only
        explicit_patterns = [
            # Standard formats (from extract_explicit_sqft_only)
            r'(\d{2,5})\s*(?:SQ\.?\s*FT|SQFT|SF|FT²|FT2)',
            r'(\d{2,5})\s*(?:SQUARE\s*FEET|SQUARE\s*FOOT)',
            r'(\d{2,5})\s*(?:SF|S\.F\.)',
            # Metric conversions (from extract_explicit_sqft_only)
            r'(\d{2,5})\s*(?:M²|SQM|SQUARE\s*METERS)',
        ]
        
        debug_results['explicit_patterns'] = []
        for i, pattern in enumerate(explicit_patterns):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                debug_results['explicit_patterns'].append({
                    'pattern_index': i,
                    'pattern': pattern,
                    'matches': [{
                        'text': match.group(0),
                        'start': match.start(),
                        'end': match.end(),
                        'groups': list(match.groups())
                    }]
                })
        
        # Test dimension patterns - reuse the actual patterns from _get_dimension_evidence
        # These are the exact patterns from src/ocr_extractor.py line 1093+
        dimension_patterns = [
            # Complex: 15' 6" x 10' 3"
            r'(\d{1,2})\s*[\'′]\s*(\d{1,2})?\s*["″]?\s*[xX×]\s*(\d{1,2})\s*[\'′]\s*(\d{1,2})?\s*["″]?',
            # Dash notation: 11'-6" x 10'-3"
            r'(\d{1,2})\s*[\'′]\s*-\s*(\d{1,2})\s*["″\\"]\s*[xX×]\s*(\d{1,2})\s*[\'′]\s*-\s*(\d{1,2})\s*["″\\"]',
            # Mixed dash notation: 15'4" x 11'-6" or 11'-6" x 15'4"
            r'(\d{1,2})\s*[\'′]\s*(\d{1,2})?\s*["″]?\s*[xX×]\s*(\d{1,2})\s*[\'′]\s*-\s*(\d{1,2})\s*["″]',
            r'(\d{1,2})\s*[\'′]\s*-\s*(\d{1,2})\s*["″]\s*[xX×]\s*(\d{1,2})\s*[\'′]\s*(\d{1,2})?\s*["″]?',
            # Simple: 15' x 10'
            r'(\d{1,2})\s*[\'′]?\s*[xX×]\s*(\d{1,2})\s*[\'′]?',
            # Feet implied: 15.5 x 10.5
            r'(\d{1,2}(?:\.\d+)?)\s*[xX×]\s*(\d{1,2}(?:\.\d+)?)',
            # OCR artifacts: feet-inches without apostrophes (e.g., 221"X148" -> 22'1" X 14'8")
            r'(\d{2,2})(\d{1,1})["″]\s*[xX×]\s*(\d{1,2})(\d{1,1})["″]',
            # OCR artifacts: feet-inches without any markers (e.g., 9SX97 -> 9'3" X 9'7")
            r'(\d{1,1})(\d{1,1})[sS][xX×](\d{1,1})(\d{1,1})',
            # NEW: OCR artifacts from Listing 15 - missing space and quotes: 15'-6"14-7" -> 15'-6" x 14'-7"
            r'(\d{1,2})\s*[\'′]\s*-\s*(\d{1,2})\s*["″](\d{1,2})\s*[\'′]\s*-\s*(\d{1,2})\s*["″]',
            # NEW: OCR artifacts from Listing 15 - missing space between dimensions: 15'-6"14-7" -> 15'-6" x 14'-7"
            r'(\d{1,2})\s*[\'′]\s*-\s*(\d{1,2})\s*["″]\s*(\d{1,2})\s*[\'′]\s*-\s*(\d{1,2})\s*["″]',
            # NEW: OCR artifacts from Listing 15 - missing space between dimensions (no separator): 15'-6"14-7" -> 15'-6" x 14'-7"
            r'(\d{1,2})\s*[\'′]\s*-\s*(\d{1,2})\s*["″](\d{1,2})\s*[\'′]\s*-\s*(\d{1,2})\s*["″]',
            # NEW: OCR artifacts from Listing 15 - missing apostrophe in second dimension: 15'-6"14-7" -> 15'-6" x 14'-7"
            r'(\d{1,2})\s*[\'′]\s*-\s*(\d{1,2})\s*["″](\d{1,2})\s*-\s*(\d{1,2})\s*["″]',
            # NEW: OCR artifacts from Listing 26 - missing leading digit and apostrophes: 2-0×10-8 -> 21'-0" X 10'-8"
            r'(\d{1,2})\s*-\s*(\d{1,2})\s*[xX×]\s*(\d{1,2})\s*-\s*(\d{1,2})',
            # NEW: OCR artifacts - missing apostrophes: 15-6×14-7 -> 15'-6" x 14'-7"
            r'(\d{1,2})\s*-\s*(\d{1,2})\s*[xX×]\s*(\d{1,2})\s*-\s*(\d{1,2})',
        ]
        
        debug_results['dimension_patterns'] = []
        for i, pattern in enumerate(dimension_patterns):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                debug_results['dimension_patterns'].append({
                    'pattern_index': i,
                    'pattern': pattern,
                    'matches': [{
                        'text': match.group(0),
                        'start': match.start(),
                        'end': match.end(),
                        'groups': list(match.groups())
                    }]
                })
        
        return debug_results

    async def test_multiple_listings(self, listing_ids: List[int]) -> List[Dict]:
        """Test OCR on multiple listings."""
        self.log(f"Testing {len(listing_ids)} listings: {listing_ids}")
        
        results = []
        for listing_id in listing_ids:
            result = await self.test_listing_ocr(listing_id)
            results.append(result)
        
        return results
    
    def test_text_pattern(self, text: str) -> Dict:
        """Test a specific text pattern with OCR extraction."""
        self.log(f"Testing text pattern: '{text}'")
        
        config = get_paddle_cfg()
        
        # Test explicit SQFT first
        explicit_sqft, explicit_source = extract_explicit_sqft_only(text, config)
        explicit_conf = None
        if explicit_sqft:
            explicit_conf = _calculate_explicit_sqft_confidence(explicit_sqft, explicit_source)
        
        # Test dimension extraction
        count, total, source, confidence = _get_dimension_evidence(text, config)
        
        # Debug pattern matching
        pattern_debug = self.debug_pattern_matching(text)
        
        result = {
            "input_text": text,
            "explicit_sqft": {
                "sqft": explicit_sqft,
                "source_text": explicit_source,
                "confidence": explicit_conf
            },
            "dimension_extraction": {
                "dimension_count": count,
                "total_sqft": total,
                "source_text": source,
                "confidence": confidence
            },
            "pattern_debug": pattern_debug
        }
        
        if explicit_sqft:
            self.log(f"Explicit SQFT: {explicit_sqft} sqft (confidence: {explicit_conf})")
        self.log(f"Dimension extraction: count={count}, total={total}, src='{source}', confidence={confidence}")
        
        return result
    
    def generate_regression_test(self, pattern: str, expected_sqft: int, test_name: str = None) -> str:
        """Generate a regression test for a pattern (pattern-based, not listing-specific)."""
        if test_name is None:
            # Generate test name from pattern (sanitized for Python function name)
            sanitized_pattern = re.sub(r'[^a-zA-Z0-9]', '_', pattern)
            # Limit length to avoid overly long function names
            if len(sanitized_pattern) > 30:
                sanitized_pattern = sanitized_pattern[:30]
            test_name = f"test_{sanitized_pattern}_pattern"
        
        # Escape quotes in pattern for the test case
        escaped_pattern = pattern.replace('"', '\\"')
        
        # Clean the pattern for the docstring (remove problematic Unicode characters)
        clean_pattern = pattern.replace('×', 'x').replace('′', "'").replace('″', '"')
        
        test_code = f'''    def {test_name}(self):
        """Test OCR pattern: {clean_pattern} -> {expected_sqft} sqft (pattern-based, not listing-specific)."""
        test_cases = [
            ("{escaped_pattern}", {expected_sqft}, "{escaped_pattern}"),
        ]
        
        paddle_cfg = get_paddle_cfg()
        
        for text, expected_sqft, expected_source in test_cases:
            dim_count, total_sqft, source_text, confidence = _get_dimension_evidence(text, paddle_cfg)
            
            assert dim_count == 1, f"Should detect 1 dimension in '{{text}}'"
            assert total_sqft == expected_sqft, f"SQFT should be {{expected_sqft}} for '{{text}}'"
            # Handle escaped quotes in source_text for comparison
            normalized_source = source_text.replace('\\\\"', '"')
            assert expected_source in normalized_source, f"Source should contain '{{expected_source}}' for '{{text}}', got: {{source_text}}"
'''
        return test_code
    
    def validate_pattern_for_regression_test(self, pattern: str, expected_sqft: int) -> bool:
        """Validate that a pattern is appropriate for a regression test."""
        # Check if pattern is too specific to a listing
        listing_specific_indicators = [
            'listing', 'floor_plan', 'photo', 'image', 'url',
            'address', 'building', 'apartment', 'unit'
        ]
        
        pattern_lower = pattern.lower()
        for indicator in listing_specific_indicators:
            if indicator in pattern_lower:
                self.log(f"Warning: Pattern contains listing-specific term '{indicator}'", "WARNING")
                return False
        
        # Check if pattern is too generic
        if len(pattern.strip()) < 3:
            self.log("Error: Pattern is too short for a regression test", "ERROR")
            return False
        
        # Check if expected SQFT is reasonable
        if expected_sqft < 10 or expected_sqft > 10000:
            self.log(f"Warning: Expected SQFT {expected_sqft} seems unusual", "WARNING")
        
        # Test the pattern to ensure it actually works
        config = get_paddle_cfg()
        count, total, source, confidence = _get_dimension_evidence(pattern, config)
        
        if count == 0:
            self.log(f"Warning: Pattern '{pattern}' doesn't extract any dimensions", "WARNING")
            return False
        
        if total != expected_sqft:
            self.log(f"Warning: Pattern '{pattern}' extracts {total} sqft, not {expected_sqft}", "WARNING")
            return False
        
        return True
    
    def add_regression_test(self, pattern: str, expected_sqft: int, test_name: str = None):
        """Add a regression test to the test file (pattern-based, not listing-specific)."""
        # Validate the pattern first
        if not self.validate_pattern_for_regression_test(pattern, expected_sqft):
            self.log("Pattern validation failed. Not adding regression test.", "ERROR")
            return False
        
        test_file = Path("tests/test_ocr_regression.py")
        
        if not test_file.exists():
            self.log("Test file not found", "ERROR")
            return False
        
        # Read the test file
        with open(test_file, 'r') as f:
            content = f.read()
        
        # Find the TestRecentOCRFixes class
        class_pattern = r'class TestRecentOCRFixes:'
        class_match = re.search(class_pattern, content, re.DOTALL)
        
        if not class_match:
            self.log("TestRecentOCRFixes class not found", "ERROR")
            return False
        
        # Generate the test code
        test_code = self.generate_regression_test(pattern, expected_sqft, test_name)
        
        # Find the end of the class (before the next class or end of file)
        class_start = class_match.end()
        remaining_content = content[class_start:]
        
        # Find the end of the class
        next_class_match = re.search(r'\nclass ', remaining_content)
        if next_class_match:
            class_end = class_start + next_class_match.start()
        else:
            class_end = len(content)
        
        # Insert the test before the end of the class
        new_content = (
            content[:class_end] + 
            "\n" + test_code + "\n" +
            content[class_end:]
        )
        
        # Write the updated content
        with open(test_file, 'w') as f:
            f.write(new_content)
        
        self.log(f"Added regression test for pattern: {pattern}")
        return True
    
    def run_regression_tests(self) -> bool:
        """Run all regression tests."""
        self.log("Running regression tests...")
        
        import subprocess
        import sys
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/test_ocr_regression.py", "-v"
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                self.log("All regression tests passed! ✅")
                return True
            else:
                self.log("Some regression tests failed! ❌", "ERROR")
                self.log(f"Test output:\n{result.stdout}\n{result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Error running regression tests: {e}", "ERROR")
            return False
    
    def save_workflow_log(self, filename: str = None):
        """Save the workflow log to a file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ocr_debug_workflow_{timestamp}.log"
        
        # Enable log collection and collect current logs
        # self._save_log_enabled = True # This line is removed as per the new_code
        # self.workflow_log = [] # This line is removed as per the new_code
        
        with open(filename, 'w') as f:
            f.write('\n'.join(self.workflow_log))
        
        self.log(f"Workflow log saved to {filename}")
    
    def save_detailed_ocr_output(self, results: List[Dict], filename: str = None):
        """Save detailed OCR output to a JSON file for analysis."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detailed_ocr_output_{timestamp}.json"
        
        # Clean up the results for JSON serialization
        clean_results = []
        for result in results:
            clean_result = {
                'listing_id': result['listing_id'],
                'photos': result['photos'],
                'results': []
            }
            
            for photo_result in result['results']:
                clean_photo_result = {
                    'photo_id': photo_result['photo_id'],
                    'photo_url': photo_result['photo_url'],
                    'sqft': photo_result['sqft'],
                    'source_text': photo_result['source_text'],
                    'engine': photo_result['engine'],
                    'confidence': photo_result['confidence']
                }
                
                if 'detailed_ocr' in photo_result:
                    # Clean up detailed OCR data for JSON
                    detailed = photo_result['detailed_ocr']
                    clean_detailed = {}
                    
                    # Hybrid result
                    if 'hybrid_result' in detailed:
                        clean_detailed['hybrid_result'] = detailed['hybrid_result']
                    
                    # PaddleOCR results
                    if 'paddleocr' in detailed:
                        paddle = detailed['paddleocr']
                        clean_detailed['paddleocr'] = {
                            'full_text': paddle['full_text'],
                            'extracted_texts': paddle['extracted_texts']
                        }
                        if 'dimension_extraction' in paddle:
                            clean_detailed['paddleocr']['dimension_extraction'] = paddle['dimension_extraction']
                        if 'explicit_sqft' in paddle:
                            clean_detailed['paddleocr']['explicit_sqft'] = paddle['explicit_sqft']
                        if 'pattern_debug' in paddle:
                            clean_detailed['paddleocr']['pattern_debug'] = paddle['pattern_debug']
                    
                    # Tesseract results (simplified structure)
                    if 'tesseract' in detailed:
                        tesseract = detailed['tesseract']
                        clean_detailed['tesseract'] = {
                            'full_text': tesseract['full_text'],
                            'extracted_texts': tesseract['extracted_texts']
                        }
                        if 'dimension_extraction' in tesseract:
                            clean_detailed['tesseract']['dimension_extraction'] = tesseract['dimension_extraction']
                        if 'explicit_sqft' in tesseract:
                            clean_detailed['tesseract']['explicit_sqft'] = tesseract['explicit_sqft']
                        if 'pattern_debug' in tesseract:
                            clean_detailed['tesseract']['pattern_debug'] = tesseract['pattern_debug']
                    
                    # Configuration
                    if 'configuration' in detailed:
                        clean_detailed['configuration'] = detailed['configuration']
                    
                    clean_photo_result['detailed_ocr'] = clean_detailed
                
                if 'error' in photo_result:
                    clean_photo_result['error'] = photo_result['error']
                
                clean_result['results'].append(clean_photo_result)
            
            clean_results.append(clean_result)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        self.log(f"Detailed OCR output saved to {filename}")
        return filename

async def main():
    parser = argparse.ArgumentParser(description="OCR Debug Workflow")
    parser.add_argument("--listings", type=str, help="Comma-separated list of listing IDs to test")
    parser.add_argument("--test-pattern", type=str, help="Test a specific text pattern")
    parser.add_argument("--test-pattern-file", type=str, help="Test a text pattern from a file (avoids PowerShell quote issues)")
    parser.add_argument("--debug-patterns-file", type=str, help="Debug pattern matching for text from a file")
    parser.add_argument("--add-fix", action="store_true", help="Add a fix for a pattern")
    parser.add_argument("--pattern", type=str, help="Pattern to add fix for (with --add-fix)")
    parser.add_argument("--pattern-file", type=str, help="Pattern file to add fix for (with --add-fix, avoids PowerShell quote issues)")
    parser.add_argument("--expected", type=int, help="Expected SQFT for pattern (with --add-fix)")
    parser.add_argument("--test-name", type=str, help="Test name for regression test (with --add-fix)")
    parser.add_argument("--run-tests", action="store_true", help="Run regression tests after adding fix")
    parser.add_argument("--save-log", type=str, help="Save workflow log to file")
    parser.add_argument("--save-detailed", type=str, help="Save detailed OCR output to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Show full detailed OCR output")
    
    args = parser.parse_args()
    
    async with OCRDebugWorkflow() as workflow:
        if args.listings:
            # Test specific listings
            listing_ids = [int(x.strip()) for x in args.listings.split(',')]
            results = await workflow.test_multiple_listings(listing_ids)
            
            # Print summary
            print("\n" + "="*60)
            print("OCR TEST RESULTS SUMMARY")
            print("="*60)
            for result in results:
                listing_id = result['listing_id']
                print(f"\nListing {listing_id}:")
                for photo_result in result['results']:
                    if 'error' in photo_result:
                        print(f"  ❌ Error: {photo_result['error']}")
                    else:
                        sqft = photo_result['sqft']
                        engine = photo_result['engine']
                        confidence = photo_result['confidence']
                        if sqft is not None:
                            print(f"  ✅ {sqft} sqft ({engine}, {confidence:.2f} conf)")
                        else:
                            print(f"  ❌ No extraction ({engine})")
                        
                        # Show detailed OCR breakdown
                        if 'detailed_ocr' in photo_result:
                            detailed = photo_result['detailed_ocr']
                            print(f"    📊 Detailed OCR Analysis:")
                            
                            # PaddleOCR results
                            if 'paddleocr' in detailed:
                                paddle = detailed['paddleocr']
                                print(f"      🚣 PaddleOCR:")
                                if args.verbose:
                                    print(f"        Full Text: '{paddle['full_text']}'")
                                    print(f"        Raw Result: {paddle['raw_result']}")
                                else:
                                    print(f"        Text: '{paddle['full_text'][:100]}{'...' if len(paddle['full_text']) > 100 else ''}'")
                                if 'dimension_extraction' in paddle:
                                    dim_ext = paddle['dimension_extraction']
                                    print(f"        Dimensions: {dim_ext['dimension_count']} found, {dim_ext['total_sqft']} sqft")
                            
                                # Show Tesseract results
                                if 'tesseract' in detailed:
                                    print(f"      🔍 Tesseract:")
                                    tesseract_data = detailed['tesseract']
                                    if 'error' in tesseract_data:
                                        print(f"        ❌ {tesseract_data['error']}")
                                    else:
                                        if args.verbose:
                                            print(f"        Text: '{tesseract_data['full_text']}'")
                                        else:
                                            text_preview = tesseract_data['full_text'][:80] + ('...' if len(tesseract_data['full_text']) > 80 else '')
                                            print(f"        Text: '{text_preview}'")
                                        
                                        # Show dimension extraction results for Tesseract
                                        if 'dimension_extraction' in tesseract_data:
                                            dim_data = tesseract_data['dimension_extraction']
                                            if dim_data['dimension_count'] > 0:
                                                print(f"        Dimensions: {dim_data['dimension_count']} found, {dim_data['total_sqft']} sqft")
                                                if args.verbose:
                                                    print(f"          Source: '{dim_data['source_text']}'")
                                            else:
                                                print(f"        Dimensions: No dimensions found")
        
        elif args.test_pattern:
            # Test a specific text pattern
            result = workflow.test_text_pattern(args.test_pattern)
            print(f"\nPattern test result: {result}")
        
        elif args.test_pattern_file:
            # Test a text pattern from a file (avoids PowerShell quote issues)
            try:
                # Try different encodings
                encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1']
                text = None
                
                for encoding in encodings:
                    try:
                        with open(args.test_pattern_file, 'r', encoding=encoding) as f:
                            text = f.read().strip()
                        break
                    except UnicodeDecodeError:
                        continue
                
                if text is None:
                    print(f"Error: Could not read file '{args.test_pattern_file}' with any encoding")
                    return
                
                result = workflow.test_text_pattern(text)
                print(f"\nPattern test result for '{text}':")
                print(json.dumps(result, indent=2))
            except FileNotFoundError:
                print(f"Error: File '{args.test_pattern_file}' not found")
            except Exception as e:
                print(f"Error reading file: {e}")
        
        elif args.debug_patterns_file:
            # Debug pattern matching for text from a file
            try:
                # Try different encodings
                encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1']
                text = None
                
                for encoding in encodings:
                    try:
                        with open(args.debug_patterns_file, 'r', encoding=encoding) as f:
                            text = f.read().strip()
                        break
                    except UnicodeDecodeError:
                        continue
                
                if text is None:
                    print(f"Error: Could not read file '{args.debug_patterns_file}' with any encoding")
                    return
                
                pattern_debug = workflow.debug_pattern_matching(text)
                print(f"\nPattern debug result for '{text}':")
                print(json.dumps(pattern_debug, indent=2))
            except FileNotFoundError:
                print(f"Error: File '{args.debug_patterns_file}' not found")
            except Exception as e:
                print(f"Error reading file: {e}")
        
        elif args.add_fix:
            # Add a fix for a pattern
            pattern = None
            
            # Get pattern from file or direct argument
            if args.pattern_file:
                try:
                    # Try different encodings
                    encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1']
                    for encoding in encodings:
                        try:
                            with open(args.pattern_file, 'r', encoding=encoding) as f:
                                pattern = f.read().strip()
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if pattern is None:
                        print(f"Error: Could not read file '{args.pattern_file}' with any encoding")
                        return
                except FileNotFoundError:
                    print(f"Error: File '{args.pattern_file}' not found")
                    return
                except Exception as e:
                    print(f"Error reading file: {e}")
                    return
            elif args.pattern:
                pattern = args.pattern
            else:
                print("Error: --pattern or --pattern-file and --expected are required with --add-fix")
                return
            
            if not args.expected:
                print("Error: --expected is required with --add-fix")
                return
            
            success = workflow.add_regression_test(
                pattern, 
                args.expected, 
                args.test_name
            )
            
            if success and args.run_tests:
                workflow.run_regression_tests()
        
        elif args.run_tests:
            # Run regression tests
            workflow.run_regression_tests()
        
        else:
            # Interactive mode
            print("OCR Debug Workflow - Interactive Mode")
            print("Commands:")
            print("  test <listing_id> - Test a specific listing")
            print("  pattern <text> - Test a text pattern")
            print("  add-fix <pattern> <expected_sqft> - Add regression test")
            print("  run-tests - Run regression tests")
            print("  quit - Exit")
            
            while True:
                try:
                    command = input("\n> ").strip()
                    if command == "quit":
                        break
                    elif command.startswith("test "):
                        listing_id = int(command.split()[1])
                        result = await workflow.test_listing_ocr(listing_id)
                        print(f"Result: {result}")
                    elif command.startswith("pattern "):
                        pattern = command.split(" ", 1)[1]
                        result = workflow.test_text_pattern(pattern)
                        print(f"Result: {result}")
                    elif command.startswith("add-fix "):
                        parts = command.split()
                        if len(parts) >= 3:
                            pattern = parts[1]
                            expected_sqft = int(parts[2])
                            workflow.add_regression_test(pattern, expected_sqft)
                        else:
                            print("Usage: add-fix <pattern> <expected_sqft>")
                    elif command == "run-tests":
                        workflow.run_regression_tests()
                    else:
                        print("Unknown command")
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")
        
        # Save log if requested
        if args.save_log:
            workflow.save_workflow_log(args.save_log)
        
        # Save detailed OCR output if requested
        if args.save_detailed and args.listings:
            workflow.save_detailed_ocr_output(results, args.save_detailed)

if __name__ == "__main__":
    asyncio.run(main())
