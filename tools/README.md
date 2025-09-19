# Tools Directory

This directory contains specialized tools for development and debugging.

## OCR Tools

### `ocr/debug_workflow.py`
Main OCR debugging script with comprehensive analysis capabilities.

### `ocr/debug.ps1`
PowerShell wrapper for OCR debugging.

**Quick Start:**
```powershell
# Test a listing
.\tools\ocr\debug.ps1 -Listings "15" -Verbose

# Debug pattern matching
.\tools\ocr\debug.ps1 -DebugPatternsFile "tools/data/common_patterns.txt"
```

## Data

### `data/common_patterns.txt`
Common problematic OCR patterns for testing and debugging.

## Reports

### `reports/`
Directory for generated reports and analysis outputs.

## Documentation

For detailed OCR debugging documentation, see `docs/OCR_DEBUG_WORKFLOW.md`.
