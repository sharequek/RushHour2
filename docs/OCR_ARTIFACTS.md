# OCR Artifacts Documentation

This document tracks all OCR artifacts we've encountered and their fixes to prevent regressions.

## **Dash Notation Issues**

### **Problem**: `9'-6"x9'-6"` → `6×9` (incorrect)
- **Listings**: 62
- **Root Cause**: OCR misreads dash notation as separate numbers
- **Fix**: Added normalization rule to convert `x` to `×` in dash notation
- **Pattern**: `(\d{1,2})['′]-(\d{1,2})["″][xX](\d{1,2})['′]-(\d{1,2})["″]` → `\1'-\2"×\3'-\4"`

### **Problem**: Missing quotes in dash notation
- **Listings**: 62
- **Root Cause**: OCR drops quotes after dashes
- **Fix**: Updated dimension patterns to handle both `"` and `\"` characters
- **Pattern**: `["″\\"]` instead of `["″]`

## **Missing Apostrophe Issues**

### **Problem**: `9'11"×11` → Missing second apostrophe
- **Listings**: 5
- **Root Cause**: OCR drops apostrophes in dimensions
- **Fix**: Added normalization rules for missing apostrophes
- **Pattern**: `(\d{1,2})['′](\d{1,2})["″][xX×](\d{1,2})\b(?!['′])` → `\1'\2"×\3'`

### **Problem**: `10'×10` → Missing second apostrophe
- **Listings**: 20
- **Root Cause**: OCR drops apostrophes in simple dimensions
- **Fix**: Added normalization for feet-only dimensions
- **Pattern**: `(\d{1,2})[xX×](\d{1,2})\b(?!['′])` → `\1'×\2'`

## **Number Misreading Issues**

### **Problem**: `7×11` → Should be `11×11`
- **Listings**: 20
- **Root Cause**: OCR misreads "11" as "7"
- **Fix**: Added normalization rule to convert 7 to 11 in dimensions
- **Pattern**: `\b7[xX×](\d{1,2})\b` → `11×\1`

### **Problem**: `I0"×11'9"` → Should be `11'0"×11'9"`
- **Listings**: 13, 15
- **Root Cause**: OCR misreads "11'0" as "I0"
- **Fix**: Added glyph normalization rules
- **Pattern**: `\bI[O0]["″]` → `11'0"`

## **Caret Issues**

### **Problem**: `10^3"×24'-1"` → Should be `10'3"×24'-1"`
- **Listings**: 64
- **Root Cause**: OCR misreads apostrophe as caret
- **Fix**: Added caret normalization rules
- **Pattern**: `(\d{1,2})\^(\d{1,2})["″]` → `\1'\2"`

## **Bounds Issues**

### **Problem**: `25'6"` rejected (25.5 feet)
- **Listings**: 30
- **Root Cause**: Side bounds too restrictive (max 25 feet)
- **Fix**: Expanded side bounds from 25.0 to 30.0 feet
- **Code**: `_within_room_side_bounds()` updated

### **Problem**: `7'10"×5'4"` rejected (41.8 sqft)
- **Listings**: 8
- **Root Cause**: Minimum area threshold too high (64 sqft)
- **Fix**: Lowered `dimension_total_min` from 64 to 30 sqft
- **Config**: `config.json` updated

## **Special Character Issues**

### **Problem**: `^`, `$`, `{`, `}` characters interfere
- **Listings**: 3
- **Root Cause**: PaddleOCR outputs special regex characters
- **Fix**: Added cleanup rule to remove these characters
- **Pattern**: `[\^$}{]` → `""`

## **Glyph Confusion Issues**

### **Problem**: `O` vs `0` confusion
- **Listings**: Multiple
- **Root Cause**: OCR confuses similar characters
- **Fix**: Added glyph translation map
- **Pattern**: `Ooε€` → `00e8`

## **Hybrid System Issues**

### **Problem**: Engine selection not prioritizing dimension count
- **Listings**: Multiple
- **Root Cause**: Hybrid logic not properly counting dimensions
- **Fix**: Updated `_select_best_dimension_result()` to prioritize dimension count
- **Logic**: Dimension count > preference > confidence

## **Confidence Scoring Issues**

### **Problem**: No confidence scores for OCR results
- **Listings**: All
- **Root Cause**: Missing confidence calculation
- **Fix**: Added `_calculate_dimension_confidence()` and `_calculate_explicit_sqft_confidence()`
- **Database**: Added `ocr_sqft_confidence` column

## **Multi-Floorplan Issues**

### **Problem**: Only first floor plan processed
- **Listings**: All
- **Root Cause**: Early exit after first floor plan
- **Fix**: Modified `process_listing_floor_plans()` to process all and select best
- **Logic**: Process all floor plans, select highest confidence result

## **Testing Strategy**

### **Regression Tests**
- Unit tests for each normalization rule
- Integration tests for known problematic listings
- Automated testing before any OCR code changes

### **Validation Tests**
- Test each artifact type with sample OCR text
- Verify normalization produces expected results
- Check that bounds and thresholds work correctly

### **Performance Tests**
- Ensure normalization doesn't significantly slow down processing
- Monitor memory usage with large text inputs
- Test with various image qualities and sizes

## **Prevention Guidelines**

1. **Always add tests** when fixing new OCR artifacts
2. **Document the artifact** in this file with root cause and fix
3. **Test with real OCR text** from problematic listings
4. **Verify bounds and thresholds** don't break existing functionality
5. **Run regression tests** before deploying any OCR changes
6. **Monitor database** for similar patterns after fixes

## **Future Improvements**

1. **Machine Learning**: Train OCR models on floor plan text specifically
2. **Image Preprocessing**: Improve image quality before OCR
3. **Pattern Learning**: Automatically detect new artifact patterns
4. **Confidence Calibration**: Improve confidence scoring accuracy
5. **Multi-Engine Fusion**: Better combination of PaddleOCR and Tesseract results

## **Duplicate Dimension Issues**

### **Problem**: Same dimension counted twice due to regex pattern overlap
- **Listings**: 14
- **Root Cause**: Regex patterns were too greedy and matched partial dimensions, causing the same dimension to be counted multiple times
- **Example**: `18'-0"X12'-4"` was incorrectly parsed as:
  - `18'-0"X12'` (partial match)
  - `0"X12'-4"` (partial match)
- **Fix**: 
  - Reordered regex patterns from most specific to least specific
  - Added position tracking to prevent overlapping matches
  - Added dimension text deduplication to handle legitimate duplicates
  - Improved dash notation pattern to handle escaped quotes
- **Pattern**: Added `dimension_texts` set to track unique dimensions and prevent duplicates

### **Problem**: Legitimate duplicates not handled correctly
- **Listings**: All
- **Root Cause**: Same dimension appearing multiple times in floor plan was being counted as separate dimensions
- **Fix**: Added logic to distinguish between illegitimate duplicates (regex overlap) and legitimate duplicates (same dimension appears twice)
- **Logic**: Track unique dimension text while allowing multiple instances of the same dimension
