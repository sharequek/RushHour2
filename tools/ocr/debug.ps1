#!/usr/bin/env pwsh
<#
.SYNOPSIS
    OCR Debug Workflow PowerShell Wrapper

.DESCRIPTION
    This script provides an easy-to-use interface for the OCR debug workflow.
    It automates testing listings, fixing extraction issues, and maintaining regression tests.

.PARAMETER Listings
    Comma-separated list of listing IDs to test

.PARAMETER Pattern
    Test a specific text pattern

.PARAMETER AddFix
    Add a fix for a pattern

.PARAMETER Expected
    Expected SQFT for pattern (required with AddFix)

.PARAMETER TestName
    Test name for regression test (optional)

.PARAMETER RunTests
    Run regression tests after adding fix

.PARAMETER SaveLog
    Save workflow log to file

.PARAMETER SaveDetailed
    Save detailed OCR output to JSON file

.PARAMETER Verbose
    Show full detailed OCR output

.PARAMETER Interactive
    Run in interactive mode

.EXAMPLE
    .\scripts\ocr_debug.ps1 -Listings "15,26,21"
    
.EXAMPLE
    .\scripts\ocr_debug.ps1 -Pattern "15'-6`"14-7`""
    
.EXAMPLE
    .\scripts\ocr_debug.ps1 -AddFix -Pattern "15'-6`"14-7`"" -Expected 226 -RunTests
    
.EXAMPLE
    .\scripts\ocr_debug.ps1 -Listings "15,26" -Verbose -SaveDetailed "debug_output.json"
    
.EXAMPLE
    .\scripts\ocr_debug.ps1 -Interactive
#>

param(
    [string]$Listings,
    [string]$Pattern,
    [switch]$AddFix,
    [int]$Expected,
    [string]$TestName,
    [switch]$RunTests,
    [string]$SaveLog,
    [string]$SaveDetailed,
    [switch]$Verbose,
    [switch]$Interactive,
    [string]$DebugPatternsFile,
    [string]$TestPatternFile,
    [string]$PatternFile
)

# Activate virtual environment
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Green
    & ".\venv\Scripts\Activate.ps1"
} elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Green
    & ".\.venv\Scripts\Activate.ps1"
} else {
    Write-Host "Warning: No virtual environment found" -ForegroundColor Yellow
}

# Build command arguments
$args = @()

if ($Listings) {
    $args += "--listings", $Listings
}

if ($Pattern) {
    $args += "--test-pattern", $Pattern
}

if ($AddFix) {
    $args += "--add-fix"
    if ($PatternFile) {
        $args += "--pattern-file", $PatternFile
    } elseif ($Pattern) {
        $args += "--pattern", $Pattern
    }
    if ($Expected) {
        $args += "--expected", $Expected
    }
    if ($TestName) {
        $args += "--test-name", $TestName
    }
    if ($RunTests) {
        $args += "--run-tests"
    }
}

if ($SaveLog) {
    $args += "--save-log", $SaveLog
}

if ($SaveDetailed) {
    $args += "--save-detailed", $SaveDetailed
}

if ($Verbose) {
    $args += "--verbose"
}

if ($DebugPatternsFile) {
    $args += "--debug-patterns-file", $DebugPatternsFile
}

if ($TestPatternFile) {
    $args += "--test-pattern-file", $TestPatternFile
}

if ($Interactive) {
    # No additional args needed for interactive mode
}

# Run the Python script
Write-Host "Running OCR Debug Workflow..." -ForegroundColor Green
Write-Host "Command: python tools/ocr/debug_workflow.py $($args -join ' ')" -ForegroundColor Cyan

try {
    python tools/ocr/debug_workflow.py @args
} catch {
    Write-Host "Error running OCR debug workflow: $_" -ForegroundColor Red
    exit 1
}
