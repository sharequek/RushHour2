# OCR Regression Prevention System

This document explains how to set up and use the OCR regression prevention system to avoid regressions as we fix more OCR issues.

## **System Components**

### **1. Regression Test Suite** (`tests/test_ocr_regression.py`)
- Unit tests for each normalization rule
- Integration tests for known problematic listings
- Tests for bounds, thresholds, and confidence scoring

### **2. Artifact Documentation** (`docs/OCR_ARTIFACTS.md`)
- Comprehensive documentation of all OCR artifacts encountered
- Root causes and fixes for each issue
- Prevention guidelines and testing strategies

### **3. Validation Script** (`scripts/validate_ocr_fixes.py`)
- Tests OCR on known problematic floor plan images
- Validates database results match expected values
- Generates detailed reports with pass/fail status

### **4. Pre-commit Hook** (`scripts/pre_commit_ocr_check.py`)
- Automatically runs tests before committing OCR changes
- Prevents regressions from being committed
- Only runs when OCR-related files are changed

### **5. Quality Monitor** (`scripts/monitor_ocr_quality.py`)
- Tracks OCR performance over time
- Detects new problematic patterns
- Generates quality reports with recommendations

## **Setup Instructions**

### **1. Install Dependencies**
```bash
pip install pytest pytest-asyncio
```

### **2. Set Up Pre-commit Hook**
```bash
# Make the pre-commit script executable
chmod +x scripts/pre_commit_ocr_check.py

# Add to .git/hooks/pre-commit (create if it doesn't exist)
echo '#!/bin/bash' > .git/hooks/pre-commit
echo 'python scripts/pre_commit_ocr_check.py' >> .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### **3. Create Test Directories**
```bash
mkdir -p tests
mkdir -p docs
mkdir -p scripts
```

## **Usage Guidelines**

### **When Fixing New OCR Issues**

1. **Document the Issue**
   - Add to `docs/OCR_ARTIFACTS.md`
   - Include root cause, fix, and affected listings

2. **Add Regression Tests**
   - Add test cases to `tests/test_ocr_regression.py`
   - Test both the normalization and the final result

3. **Update Validation Script**
   - Add problematic listings to `scripts/validate_ocr_fixes.py`
   - Include expected results and image URLs

4. **Test Before Committing**
   - Run `python -m pytest tests/test_ocr_regression.py -v`
   - Run `python scripts/validate_ocr_fixes.py`
   - Fix any failures before committing

### **Regular Monitoring**

1. **Daily Quality Check**
   ```bash
   python scripts/monitor_ocr_quality.py
   ```

2. **Weekly Validation**
   ```bash
   python scripts/validate_ocr_fixes.py
   ```

3. **Before Major Changes**
   ```bash
   # Run all tests
   python -m pytest tests/test_ocr_regression.py -v
   python scripts/validate_ocr_fixes.py
   python scripts/monitor_ocr_quality.py
   ```

## **Test Categories**

### **Unit Tests**
- Individual normalization rules
- Pattern matching
- Bounds checking
- Confidence calculation

### **Integration Tests**
- End-to-end OCR processing
- Hybrid engine selection
- Multi-floorplan processing
- Database updates

### **Regression Tests**
- Known problematic listings
- Specific OCR artifacts
- Edge cases and boundary conditions

## **Adding New Tests**

### **For New Normalization Rules**
```python
def test_new_normalization_rule(self):
    """Test new normalization rule for specific artifact."""
    test_cases = [
        ("original_ocr_text", "expected_normalized_text"),
        ("another_example", "expected_result"),
    ]
    
    for original, expected in test_cases:
        paddle_cfg = get_paddle_cfg()
        dim_count, total_sqft, source_text, confidence = _get_dimension_evidence(original, paddle_cfg)
        
        # Verify the normalization worked
        assert "expected_pattern" in source_text
        assert "unwanted_pattern" not in source_text
```

### **For New Problematic Listings**
```python
def test_listing_X_floorplan(self):
    """Test listing X's floor plan specifically."""
    paddle_text = "actual_ocr_text_from_listing"
    
    paddle_cfg = get_paddle_cfg()
    dim_count, total_sqft, source_text, confidence = _get_dimension_evidence(paddle_text, paddle_cfg)
    
    # Verify expected results
    assert dim_count == expected_count
    assert total_sqft == expected_sqft
    assert "expected_dimension" in source_text
```

## **Monitoring and Alerts**

### **Quality Metrics**
- High confidence rate (>90%)
- Medium confidence rate (70-90%)
- Low confidence rate (<70%)
- Issue detection rate

### **Pattern Detection**
- New problematic patterns
- Recurring issues
- Performance degradation
- Confidence score trends

### **Automated Alerts**
- Pre-commit hook failures
- Validation script failures
- Quality report warnings
- New pattern detection

## **Best Practices**

### **Before Making OCR Changes**
1. Run existing tests to establish baseline
2. Document the issue you're fixing
3. Add test cases for the specific problem
4. Implement the fix
5. Run all tests to ensure no regressions
6. Update documentation

### **When Adding New Features**
1. Add comprehensive tests
2. Update validation script
3. Test with real OCR data
4. Monitor performance impact
5. Document the changes

### **Regular Maintenance**
1. Review quality reports weekly
2. Update test cases as needed
3. Monitor for new patterns
4. Optimize performance
5. Update documentation

## **Troubleshooting**

### **Test Failures**
1. Check if the failure is expected (new issue)
2. Verify the test case is correct
3. Check if normalization rules need updating
4. Ensure bounds and thresholds are appropriate

### **Validation Failures**
1. Check if image URLs are still valid
2. Verify expected results are correct
3. Check if OCR engines have changed
4. Update test cases if needed

### **Performance Issues**
1. Monitor test execution time
2. Check for memory leaks
3. Optimize normalization rules
4. Consider caching results

## **Future Enhancements**

1. **Machine Learning Integration**
   - Train models on floor plan text
   - Automatic pattern detection
   - Confidence score calibration

2. **Continuous Integration**
   - Automated testing on every commit
   - Performance regression detection
   - Quality trend analysis

3. **Advanced Monitoring**
   - Real-time quality alerts
   - Predictive issue detection
   - Automated fix suggestions

4. **Test Data Management**
   - Centralized test image repository
   - Automated test case generation
   - Version control for test data

